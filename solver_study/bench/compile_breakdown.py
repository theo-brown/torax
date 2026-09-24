"""Compile-time breakdown of the structured Jacobian's factors.

Usage: python compile_breakdown.py [n_rho] [globals]
Times jax.jit(...).lower(x).compile() of the dense Jacobian, the structured
Jacobian as implemented in the working tree, and its individual factors.
"""
import copy
import functools
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import convertors
from torax._src.core_profiles import updaters
from torax._src.fvm import calc_coeffs
from torax._src.fvm import fvm_conversions
from torax._src.fvm import newton_raphson_solve_block
from torax._src.fvm import residual_and_loss
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import run_simulation
from torax._src.orchestration import step_function_processing
from torax._src.solver import structured_jacobian
from torax._src.torax_pydantic import model_config
from torax.examples import iterhybrid_predictor_corrector
import global_sources

jax.config.update('jax_enable_x64', True)
n_rho = int(sys.argv[1]) if len(sys.argv) > 1 else 50
with_globals = len(sys.argv) > 2 and sys.argv[2] == 'globals'


def problem():
  config = copy.deepcopy(iterhybrid_predictor_corrector.CONFIG)
  config['geometry']['n_rho'] = n_rho
  config['solver'] = dict(solver_type='newton_raphson', use_predictor_corrector=True,
                          n_corrector_steps=2, use_pereverzev=True)
  if with_globals:
    config['sources'].update(copy.deepcopy(global_sources.SOURCES))
    config['geometry'] = dict(global_sources.GEOMETRY, n_rho=n_rho)
  torax_config = model_config.ToraxConfig.from_dict(config)
  step_fn = run_simulation.make_step_fn(torax_config)
  state, _ = initial_state_lib.get_initial_state_and_post_processed_outputs(step_fn=step_fn)
  models = step_fn.solver.models
  runtime_params_t, geo_t, explicit_source_profiles, edge_outputs, pts = (
      step_function_processing.pre_step(input_state=state,
                                        runtime_params_provider=step_fn.runtime_params_provider,
                                        geometry_provider=step_fn.geometry_provider, models=models))
  dt = jnp.asarray(0.05)
  runtime_params_t_plus_dt, geo_t_plus_dt = build_runtime_params.get_consistent_runtime_params_and_geometry(
      t=state.t + dt, runtime_params_provider=step_fn.runtime_params_provider,
      geometry_provider=step_fn.geometry_provider, edge_outputs=edge_outputs,
      core_profiles=state.core_profiles)
  core_profiles_t = state.core_profiles
  core_profiles_t_plus_dt = updaters.provide_core_profiles_t_plus_dt(
      dt=dt, runtime_params_t=runtime_params_t, runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt, core_profiles_t=core_profiles_t)
  evolving_names = runtime_params_t.numerics.evolving_names
  x_old = convertors.core_profiles_to_solver_x_tuple(core_profiles_t, evolving_names)
  coeffs_callback = calc_coeffs.CoeffsCallback(models=models, evolving_names=evolving_names)
  coeffs_old = coeffs_callback(runtime_params_t, geo_t, core_profiles_t, prev_core_profiles=None,
                               dt=None, x=x_old, explicit_source_profiles=explicit_source_profiles,
                               explicit_call=True, pedestal_transition_state=pts)
  kwargs = dict(dt=dt, runtime_params_t_plus_dt=runtime_params_t_plus_dt, geo_t_plus_dt=geo_t_plus_dt,
                x_old=x_old, core_profiles_t=core_profiles_t, core_profiles_t_plus_dt=core_profiles_t_plus_dt,
                explicit_source_profiles=explicit_source_profiles, models=models, coeffs_old=coeffs_old,
                evolving_names=evolving_names, pedestal_transition_state=pts)
  residual_fun = functools.partial(residual_and_loss.theta_method_block_residual, **kwargs)
  return residual_fun, kwargs, fvm_conversions.cell_variable_tuple_to_vec(x_old)


def timed_compile(name, fn, *args):
  t0 = time.perf_counter()
  lowered = jax.jit(fn).lower(*args)
  t1 = time.perf_counter()
  compiled = lowered.compile()
  t2 = time.perf_counter()
  try:
    hlo_mb = len(compiled.as_text()) / 1e6
  except Exception:  # pylint: disable=broad-except
    hlo_mb = float('nan')
  out = compiled(*args)
  jax.block_until_ready(out)
  t3 = time.perf_counter()
  # run time (best of 5)
  best = 1e9
  for _ in range(5):
    ta = time.perf_counter(); jax.block_until_ready(compiled(*args)); best = min(best, time.perf_counter() - ta)
  print(f'{name:48s} lower {t1 - t0:6.2f} s  xla {t2 - t1:6.2f} s  hlo {hlo_mb:6.2f} MB  run {best * 1e3:8.2f} ms', flush=True)
  return dict(name=name, lower_s=t1 - t0, xla_s=t2 - t1, hlo_mb=hlo_mb, run_ms=best * 1e3)


residual_fun, kwargs, x = problem()
jac = newton_raphson_solve_block._structured_jacobian_fn(residual_fun=residual_fun, **kwargs)
kw = jac.keywords
residual_fn, raw_transport_fn = kw['residual_fn'], kw['raw_transport_fn']
source_globals_fn, smoothing_matrix, stencils = kw['source_globals_fn'], kw['smoothing_matrix'], kw['stencils']
n_state, n_faces = stencils.n_state, stencils.n_faces
colours_x, n_colours_x = structured_jacobian._colouring_x(stencils)
colours_h, n_colours_h = structured_jacobian._colouring_h(stencils)
seeds_x = structured_jacobian._seeds(colours_x, n_colours_x)
seeds_h = structured_jacobian._seeds(colours_h, n_colours_h)
raw = raw_transport_fn(x)
c = structured_jacobian.apply_smoothing(raw, smoothing_matrix, n_faces)
g = source_globals_fn(x) if source_globals_fn is not None else jnp.zeros((0,))
print(f'n_rho={n_rho} globals={with_globals} N={n_state} seeds x/h/c/g = {n_colours_x}/{n_colours_h}/8/{g.shape[0]}')
results = []
results.append(timed_compile('residual R(x) primal', residual_fun, x))
results.append(timed_compile('dense jacfwd(R)', jax.jacfwd(residual_fun), x))
results.append(timed_compile('structured Jacobian (working tree)', jac, x))
results.append(timed_compile('h(x) primal', raw_transport_fn, x))
results.append(timed_compile('dh/dx: vmap(jvp(h)) 16 seeds', lambda xx: structured_jacobian._batched_jvp(raw_transport_fn, xx, seeds_h), x))
results.append(timed_compile('h linearize + vmap(lin) 16 seeds', lambda xx: jax.vmap(jax.linearize(raw_transport_fn, xx)[1])(seeds_h), x))
results.append(timed_compile('G(x,c,g) primal', lambda xx: residual_fn(xx, c, g), x))
results.append(timed_compile('dG/dx: vmap(jvp(G)) x seeds', lambda xx: structured_jacobian._batched_jvp(lambda z: residual_fn(z, c, g), xx, seeds_x), x))
results.append(timed_compile('G linearize + vmap(lin) x seeds', lambda xx: jax.vmap(jax.linearize(residual_fn, xx, c, g)[1])(seeds_x, jnp.zeros((n_colours_x, c.shape[0])), jnp.zeros((n_colours_x, g.shape[0]))), x))
if source_globals_fn is not None:
  results.append(timed_compile('g(x) primal', source_globals_fn, x))
  k = g.shape[0]
  results.append(timed_compile('dg/dx: vmap(vjp(g)) k seeds', lambda xx: jax.vmap(lambda ct: jax.vjp(source_globals_fn, xx)[1](ct)[0])(jnp.eye(k)), x))
  def lin_t(xx):
    _, g_lin = jax.linearize(source_globals_fn, xx)
    g_vjp = jax.linear_transpose(g_lin, xx)
    return jax.vmap(lambda ct: g_vjp(ct)[0])(jnp.eye(k))
  results.append(timed_compile('g linearize + transpose k seeds', lin_t, x))
json.dump(results, open(f'compile_breakdown_n{n_rho}_{"globals" if with_globals else "plain"}.json', 'w'), indent=1)
print('DONE')
