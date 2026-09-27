"""Component micro-timings of the TORAX Newton step at a mid-sim state."""
import json, sys, time, functools
import numpy as np
import jax, jax.numpy as jnp
from torax.examples import iterhybrid_rampup
from torax._src.solver import jax_root_finding
from torax._src.fvm import newton_raphson_solve_block, enums, calc_coeffs, fvm_conversions, discrete_system
from torax._src.core_profiles import updaters
from torax._src.transport_model import transport_coefficients_builder
from torax._src.sources import source_profile_builders
from torax._src import tridiagonal
import bench_common as bc

n_rho = int(sys.argv[1]) if len(sys.argv) > 1 else 50
dt = float(sys.argv[2]) if len(sys.argv) > 2 else None
OUT = {'n_rho': n_rho}
p = bc.build_problem(iterhybrid_rampup.CONFIG, n_warm_steps=3, dt=dt, n_rho=n_rho)
N, n_ch, n_cells = p.n, len(p.evolving_names), p.n_rho
OUT.update(N=N, dt=float(p.dt), t=float(p.state.t))
print(f'N={N} n_rho={n_cells} dt={float(p.dt)} t={float(p.state.t)}')
x0 = p.x0
models = p.step_fn.solver.models

res = jax.jit(p.residual_fun)
jacf = jax.jit(jax.jacfwd(p.residual_fun))
jacr = jax.jit(jax.jacrev(p.residual_fun))
jvp1 = jax.jit(lambda x, v: jax.jvp(p.residual_fun, (x,), (v,))[1])
vjp1 = jax.jit(lambda x, v: jax.vjp(p.residual_fun, x)[1](v)[0])
def batched_jvp(k):
  def f(x, V):  # V: (k, N)
    return jax.vmap(lambda v: jax.jvp(p.residual_fun, (x,), (v,))[1])(V)
  return jax.jit(f)

T = {}
T['residual'] = bc.timeit(res, x0, n=10)
T['jacfwd_full'] = bc.timeit(jacf, x0, n=5)
T['jacrev_full'] = bc.timeit(jacr, x0, n=3)
v = jnp.ones_like(x0)
T['jvp_single'] = bc.timeit(jvp1, x0, v, n=10)
T['vjp_single'] = bc.timeit(vjp1, x0, v, n=10)
for k in [2, 4, 8, 16, 32, 64, 128, N]:
  if k > N: continue
  V = jnp.eye(N)[:k]
  T[f'batched_jvp_k{k}'] = bc.timeit(batched_jvp(k), x0, V, n=5)
J0 = jacf(x0)
r0 = res(x0)
solve = jax.jit(lambda A, b: jnp.linalg.solve(A, b))
T['dense_solve'] = bc.timeit(solve, J0, -r0, n=10)
lu = jax.jit(lambda A: jax.scipy.linalg.lu_factor(A))
T['dense_lu_factor'] = bc.timeit(lu, J0, n=10)
# block-tridiagonal Thomas solve on the Picard matrix (what the linear solver does)
M0, b0, lhs = p.picard_matrix(x0)
rhs2 = jnp.asarray(np.asarray(r0).reshape(n_ch, n_cells).T)
thomas = jax.jit(lambda L, r: tridiagonal.thomas_solve(L, r))
T['thomas_block_solve'] = bc.timeit(thomas, lhs, rhs2, n=10)

# --- JFNK-style Newton iteration in JAX: GMRES with exact JVPs, preconditioned by the Picard matrix ---
import jax.scipy.sparse.linalg as jsla
M0j = jnp.asarray(M0)
P_lu = jax.scipy.linalg.lu_factor(M0j)
def jfnk_iter(x, r, lu):
  A = lambda v: jax.jvp(p.residual_fun, (x,), (v,))[1]
  Mi = lambda v: jax.scipy.linalg.lu_solve(lu, v)
  d, info = jsla.gmres(A, -r, M=Mi, tol=1e-4, atol=0.0, restart=40, maxiter=1, solve_method='incremental')
  return d
jfnk_iter_j = jax.jit(jfnk_iter)
T['jfnk_gmres40_prec_solve'] = bc.timeit(jfnk_iter_j, x0, r0, P_lu, n=3)
def jfnk_iter20(x, r, lu):
  A = lambda v: jax.jvp(p.residual_fun, (x,), (v,))[1]
  Mi = lambda v: jax.scipy.linalg.lu_solve(lu, v)
  d, info = jsla.gmres(A, -r, M=Mi, tol=1e-4, atol=0.0, restart=20, maxiter=1, solve_method='incremental')
  return d
T['jfnk_gmres20_prec_solve'] = bc.timeit(jax.jit(jfnk_iter20), x0, r0, P_lu, n=3)
# cost of building the Picard matrix (dense) + LU
pm = jax.jit(lambda x: p.picard_matrix(x)[0])
T['picard_matrix_build'] = bc.timeit(pm, x0, n=5)
T['picard_lu'] = bc.timeit(lu, M0j, n=10)
# accuracy check of the gmres direction vs exact newton direction
d_exact = solve(J0, -r0)
d_g = jfnk_iter_j(x0, r0, P_lu)
OUT['jfnk_dir_rel_err'] = float(jnp.linalg.norm(d_g - d_exact) / jnp.linalg.norm(d_exact))
print('gmres(40, tol 1e-4, prec P) direction rel err vs exact:', OUT['jfnk_dir_rel_err'])

# components of one residual evaluation
def _cp(x):
  x_new_guess = fvm_conversions.vec_to_cell_variable_tuple(x, p.core_profiles_t_plus_dt, p.evolving_names)
  return updaters.update_core_profiles_during_step(
      x_new_guess, p.runtime_params_t_plus_dt, p.geo_t_plus_dt, p.core_profiles_t_plus_dt,
      prev_core_profiles=p.core_profiles_t, dt=p.dt, evolving_names=p.evolving_names)
cp_fn = jax.jit(_cp)
cp0 = cp_fn(x0)
T['update_core_profiles'] = bc.timeit(cp_fn, x0, n=10)
coeffs_fn = jax.jit(lambda cp: calc_coeffs.calc_coeffs(
    runtime_params=p.runtime_params_t_plus_dt, geo=p.geo_t_plus_dt, core_profiles=cp,
    explicit_source_profiles=p.explicit_source_profiles, models=models,
    evolving_names=p.evolving_names, use_pereverzev=False,
    pedestal_transition_state=p.pedestal_transition_state))
T['calc_coeffs_full'] = bc.timeit(coeffs_fn, cp0, n=10)
cond_fn = jax.jit(lambda cp: models.neoclassical_models.conductivity.calculate_conductivity(p.geo_t_plus_dt, cp))
T['conductivity'] = bc.timeit(cond_fn, cp0, n=10)
cond0 = cond_fn(cp0)
src_fn = jax.jit(lambda cp: source_profile_builders.build_source_profiles(
    source_models=models.source_models, neoclassical_models=models.neoclassical_models,
    runtime_params=p.runtime_params_t_plus_dt, geo=p.geo_t_plus_dt, core_profiles=cp,
    explicit=False, explicit_source_profiles=p.explicit_source_profiles, conductivity=cond0))
T['implicit_sources'] = bc.timeit(src_fn, cp0, n=10)
tr_fn = jax.jit(lambda cp: transport_coefficients_builder.calculate_all_transport_coeffs(
    transport_model=models.transport_model, neoclassical_models=models.neoclassical_models,
    internal_boundary_condition_model=models.internal_boundary_condition_model,
    runtime_params=p.runtime_params_t_plus_dt, geo=p.geo_t_plus_dt, core_profiles=cp,
    pedestal_transition_state=p.pedestal_transition_state, use_pereverzev=False))
T['transport_coeffs_all'] = bc.timeit(tr_fn, cp0, n=10)
# raw QLKNN transport model alone (core+pedestal combine incl smoothing)
pmo = p.pedestal_transition_state.pedestal_model_output
tm_fn = jax.jit(lambda cp: models.transport_model(p.runtime_params_t_plus_dt, p.geo_t_plus_dt, cp, pmo,
                                                  jnp.zeros_like(p.geo_t_plus_dt.rho_face_norm, dtype=bool)))
try:
  T['transport_model_only'] = bc.timeit(tm_fn, cp0, n=10)
except Exception as e:
  print('transport_model_only failed', e)
# residual assembly given coeffs
coeffs0 = coeffs_fn(cp0)
# the whole TORAX newton solve for this step
nr = functools.partial(jax_root_finding.root_newton_raphson, p.residual_fun, maxiter=30, tol=1e-5, coarse_tol=1e-2)
nr_cr = jax.jit(lambda x: nr(x0=x, use_jax_custom_root=True))
nr_nocr = jax.jit(lambda x: nr(x0=x, use_jax_custom_root=False))
nr_vmapls = jax.jit(lambda x: nr(x0=x, use_jax_custom_root=False, vmap_linesearch=True, max_linesearch_steps=8))
OUT['compile_newton_custom_root'] = bc.compile_time(nr_cr, x0)
OUT['compile_newton_no_custom_root'] = bc.compile_time(nr_nocr, x0)
OUT['compile_newton_vmap_ls8'] = bc.compile_time(nr_vmapls, x0)
xs, meta = nr_nocr(x0)
OUT['newton_iterations'] = int(meta.iterations)
T['newton_solve_custom_root'] = bc.timeit(nr_cr, x0, n=3)
T['newton_solve_no_custom_root'] = bc.timeit(nr_nocr, x0, n=3)
T['newton_solve_vmap_ls8'] = bc.timeit(nr_vmapls, x0, n=3)
# full step function (includes pre_step, finalize, post-processing)
sf = p.step_fn
T['full_step_fn'] = bc.timeit(lambda: sf(p.state, p.ppo), n=3)
# linear-solver step for comparison: predictor-corrector with 10 corrector steps
from torax._src.solver import predictor_corrector_method
from torax._src.core_profiles import convertors
x_new_guess = convertors.core_profiles_to_solver_x_tuple(p.core_profiles_t_plus_dt, p.evolving_names)
coeffs_exp_linear = p.coeffs_callback(p.runtime_params_t, p.geo_t, core_profiles=p.core_profiles_t, prev_core_profiles=None, dt=None, x=p.x_old, explicit_source_profiles=p.explicit_source_profiles, allow_pereverzev=True, explicit_call=True, pedestal_transition_state=p.pedestal_transition_state)
pc = jax.jit(lambda: predictor_corrector_method.predictor_corrector_method(
    dt=p.dt, runtime_params_t_plus_dt=p.runtime_params_t_plus_dt, geo_t_plus_dt=p.geo_t_plus_dt,
    x_old=p.x_old, x_new_guess=x_new_guess, core_profiles_t=p.core_profiles_t,
    core_profiles_t_plus_dt=p.core_profiles_t_plus_dt, coeffs_exp=coeffs_exp_linear,
    coeffs_callback=p.coeffs_callback, explicit_source_profiles=p.explicit_source_profiles,
    pedestal_transition_state=p.pedestal_transition_state))
T['predictor_corrector_10'] = bc.timeit(pc, n=5)

print('\n=== timings (best, median) in ms ===')
for k, (b, m) in T.items():
  print(f'{k:32s} {b*1e3:10.3f} {m*1e3:10.3f}')
print('compile times (s):', {k: round(v, 2) for k, v in OUT.items() if k.startswith('compile')})
print('newton iterations:', OUT['newton_iterations'])
print('ratio jacfwd/residual:', T['jacfwd_full'][0] / T['residual'][0], ' jacfwd/jvp_single:', T['jacfwd_full'][0] / T['jvp_single'][0])
OUT['timings_ms'] = {k: [b*1e3, m*1e3] for k, (b, m) in T.items()}
json.dump(OUT, open(f'timings_n{n_rho}_dt{float(p.dt):g}.json', 'w'), indent=1)
print('DONE')
