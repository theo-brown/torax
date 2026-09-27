"""Common harness: build a TORAX problem at a mid-simulation state and expose
residual / jacobian / picard-matrix callables for micro-benchmarks."""
import copy
import dataclasses
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

import torax
from torax._src.config import build_runtime_params
from torax._src.core_profiles import convertors
from torax._src.core_profiles import updaters
from torax._src.fvm import calc_coeffs
from torax._src.fvm import fvm_conversions
from torax._src.fvm import residual_and_loss
from torax._src.fvm import newton_raphson_solve_block
from torax._src.fvm import enums
from torax._src.orchestration import run_simulation as rs
from torax._src.orchestration import step_function_processing
from torax._src.solver import jax_root_finding
from torax._src.solver import predictor_corrector_method
from torax._src import tridiagonal


def timeit(fn, *args, n=5, warm=2):
  """Best-of-n wall time (seconds) after warm-up, blocking on outputs."""
  for _ in range(warm):
    jax.block_until_ready(fn(*args))
  ts = []
  for _ in range(n):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    ts.append(time.perf_counter() - t0)
  return min(ts), float(np.median(ts))


def compile_time(fn, *args):
  t0 = time.perf_counter()
  jax.block_until_ready(fn(*args))
  return time.perf_counter() - t0


@dataclasses.dataclass
class Problem:
  cfg: object
  step_fn: object
  state: object
  ppo: object
  dt: float
  runtime_params_t: object
  runtime_params_t_plus_dt: object
  geo_t: object
  geo_t_plus_dt: object
  core_profiles_t: object
  core_profiles_t_plus_dt: object
  explicit_source_profiles: object
  pedestal_transition_state: object
  edge_outputs: object
  evolving_names: tuple
  x_old: tuple
  coeffs_old: object
  coeffs_callback: object
  residual_fun: object  # x_vec -> residual vec
  x0: jax.Array  # initial guess vector (predictor-corrector linear guess)
  x_old_vec: jax.Array
  n_rho: int
  prev_x_vec: object = None  # x at the previous time level (for extrapolation)
  prev_dt: float = None

  @property
  def n(self):
    return int(self.x0.shape[0])

  def picard_matrix(self, x_vec):
    """Dense (I - scale*C(x)) matrix of the linearised theta-method system at x.

    This is exactly what the 'linear' solver factorises: the Jacobian with the
    transport coefficients and sources frozen at x (no dC/dx terms).
    Returns lhs (dense N x N in the *channel-major vector ordering*), and the
    vector b such that the residual is R(x) = lhs @ x + lhs_vec - rhs_result.
    """
    return _picard_matrix(self, x_vec)


def _picard_matrix(p: Problem, x_vec):
  x_new_guess = fvm_conversions.vec_to_cell_variable_tuple(
      x_vec, p.core_profiles_t_plus_dt, p.evolving_names
  )
  core_profiles_new = updaters.update_core_profiles_during_step(
      x_new_guess, p.runtime_params_t_plus_dt, p.geo_t_plus_dt,
      p.core_profiles_t_plus_dt, prev_core_profiles=p.core_profiles_t,
      dt=p.dt, evolving_names=p.evolving_names,
  )
  coeffs_new = calc_coeffs.calc_coeffs(
      runtime_params=p.runtime_params_t_plus_dt, geo=p.geo_t_plus_dt,
      core_profiles=core_profiles_new,
      explicit_source_profiles=p.explicit_source_profiles,
      models=p.step_fn.solver.models, evolving_names=p.evolving_names,
      use_pereverzev=False,
      pedestal_transition_state=p.pedestal_transition_state,
  )
  sp = p.runtime_params_t_plus_dt.solver
  lhs, lhs_vec, rhs, rhs_vec = residual_and_loss.theta_method_matrix_equation(
      dt=p.dt, x_old=p.x_old, x_new_guess=x_new_guess, coeffs_old=p.coeffs_old,
      coeffs_new=coeffs_new, theta_implicit=sp.theta_implicit,
      convection_dirichlet_mode=sp.convection_dirichlet_mode,
      convection_neumann_mode=sp.convection_neumann_mode,
  )
  if coeffs_new.has_internal_boundary_conditions:
    lhs, lhs_vec, rhs, rhs_vec = residual_and_loss.apply_internal_boundary_conditions(
        lhs, lhs_vec, rhs, rhs_vec,
        coeffs_new.internal_boundary_condition_mask,
        coeffs_new.internal_boundary_condition_target_vec,
    )
  n_cells = p.n_rho
  n_ch = len(p.evolving_names)
  dense = lhs.to_dense()  # ordering: cell-major (cell, channel)
  # permute to channel-major ordering used by the residual vector
  perm = np.arange(n_cells * n_ch).reshape(n_cells, n_ch).T.reshape(-1)
  # dense acts on cell-major vectors; residual uses channel-major x. Build
  # M such that M @ x_chmajor = (lhs @ x_cellmajor) reordered to channel-major
  M = dense[np.ix_(perm, perm)]
  x_old_array = fvm_conversions.cell_variable_tuple_to_array(p.x_old, axis=1)
  rhs_result = rhs.matvec(x_old_array) + rhs_vec
  b = (rhs_result - lhs_vec).T.reshape(-1)
  return M, b, lhs


def build_problem(config_dict, n_warm_steps=3, dt=None, n_rho=None,
                  solver_overrides=None, extra_overrides=None):
  cfg_dict = copy.deepcopy(config_dict)
  if n_rho is not None:
    cfg_dict['geometry']['n_rho'] = n_rho
  if solver_overrides:
    cfg_dict['solver'].update(solver_overrides)
  if extra_overrides:
    for k, v in extra_overrides.items():
      cfg_dict[k].update(v)
  cfg = torax.ToraxConfig.from_dict(cfg_dict)
  initial_state, ppo, step_fn = rs.prepare_simulation(cfg)
  state = initial_state
  prev_state = None
  for _ in range(n_warm_steps):
    prev_state = state
    state, ppo = step_fn(state, ppo)
    jax.block_until_ready(state)
    err = step_fn.check_for_errors(state, ppo)
    assert err == torax.SimError.NO_ERROR, err
  models = step_fn.solver.models
  (runtime_params_t, geo_t, explicit_source_profiles, edge_outputs,
   pedestal_transition_state) = step_function_processing.pre_step(
       input_state=state, runtime_params_provider=step_fn.runtime_params_provider,
       geometry_provider=step_fn.geometry_provider, models=models)
  if dt is None:
    dt = float(step_fn.time_step_calculator.next_dt(runtime_params_t, state))
  dt = jnp.asarray(dt)
  runtime_params_t_plus_dt, geo_t_plus_dt = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=state.t + dt, runtime_params_provider=step_fn.runtime_params_provider,
          geometry_provider=step_fn.geometry_provider, edge_outputs=edge_outputs,
          core_profiles=state.core_profiles))
  core_profiles_t = state.core_profiles
  core_profiles_t_plus_dt = updaters.provide_core_profiles_t_plus_dt(
      dt=dt, runtime_params_t=runtime_params_t,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt, core_profiles_t=core_profiles_t)
  evolving_names = runtime_params_t.numerics.evolving_names
  x_old = convertors.core_profiles_to_solver_x_tuple(core_profiles_t, evolving_names)
  coeffs_callback = calc_coeffs.CoeffsCallback(models=models, evolving_names=evolving_names)
  coeffs_old = coeffs_callback(
      runtime_params_t, geo_t, core_profiles_t, prev_core_profiles=None, dt=None,
      x=x_old, explicit_source_profiles=explicit_source_profiles,
      explicit_call=True, pedestal_transition_state=pedestal_transition_state)
  # linear initial guess (same as InitialGuessMode.LINEAR)
  coeffs_exp_linear = coeffs_callback(
      runtime_params_t, geo_t, core_profiles=core_profiles_t, prev_core_profiles=None,
      dt=None, x=x_old, explicit_source_profiles=explicit_source_profiles,
      allow_pereverzev=True, explicit_call=True,
      pedestal_transition_state=pedestal_transition_state)
  x_new_guess = convertors.core_profiles_to_solver_x_tuple(core_profiles_t_plus_dt, evolving_names)
  init_x_new = predictor_corrector_method.predictor_corrector_method(
      dt=dt, runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt, x_old=x_old, x_new_guess=x_new_guess,
      core_profiles_t=core_profiles_t, core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      coeffs_exp=coeffs_exp_linear, coeffs_callback=coeffs_callback,
      explicit_source_profiles=explicit_source_profiles,
      pedestal_transition_state=pedestal_transition_state)
  x0 = fvm_conversions.cell_variable_tuple_to_vec(init_x_new)
  x_old_vec = fvm_conversions.cell_variable_tuple_to_vec(x_old)
  residual_fun = functools.partial(
      residual_and_loss.theta_method_block_residual,
      dt=dt, runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt, x_old=x_old, core_profiles_t=core_profiles_t,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt, models=models,
      explicit_source_profiles=explicit_source_profiles, coeffs_old=coeffs_old,
      evolving_names=evolving_names,
      pedestal_transition_state=pedestal_transition_state)
  return Problem(
      cfg=cfg, step_fn=step_fn, state=state, ppo=ppo, dt=dt,
      runtime_params_t=runtime_params_t, runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t=geo_t, geo_t_plus_dt=geo_t_plus_dt, core_profiles_t=core_profiles_t,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      explicit_source_profiles=explicit_source_profiles,
      pedestal_transition_state=pedestal_transition_state, edge_outputs=edge_outputs,
      evolving_names=evolving_names, x_old=x_old, coeffs_old=coeffs_old,
      coeffs_callback=coeffs_callback, residual_fun=residual_fun, x0=x0,
      x_old_vec=x_old_vec, n_rho=int(geo_t.torax_mesh.nx),
      prev_x_vec=(None if prev_state is None else fvm_conversions.cell_variable_tuple_to_vec(
          convertors.core_profiles_to_solver_x_tuple(prev_state.core_profiles, evolving_names))),
      prev_dt=(None if prev_state is None else float(state.dt)))


def mean_abs(x):
  return float(jnp.mean(jnp.abs(x)))


def newton_generic(res_fn, jac_fn, x0, tol=1e-5, maxiter=30, jac_every=1,
                   fixed_jac=None, delta_reduction=0.5, tau_min=0.01,
                   sufficient_decrease=1e-4, max_ls=100, verbose=False,
                   band_truncate=None, n_ch=None, n_cells=None, ls=True):
  """Python-level Newton loop replicating jax_root_finding._body semantics.

  jac_fn(x) returns the dense matrix to use as the Newton matrix; called every
  `jac_every` iterations (chord/modified Newton otherwise), or never if
  fixed_jac is given. Returns dict with iteration counts and history.
  """
  x = x0
  r = res_fn(x)
  norm = mean_abs(r)
  hist = [norm]
  xhist = [x]
  n_res, n_jac, n_ls_extra = 1, 0, 0
  A = fixed_jac
  it = 0
  last_tau = 1.0
  taus = []
  while norm > tol and it < maxiter and last_tau > tau_min:
    if fixed_jac is None and (it % jac_every == 0 or A is None):
      A = jac_fn(x)
      n_jac += 1
      if band_truncate is not None:
        A = truncate_band(A, band_truncate, n_ch, n_cells)
    direction = jnp.linalg.solve(A, -r)
    max_abs_dir = float(jnp.max(jnp.abs(direction)))
    # backtracking line search identical to TORAX's accept_fn
    step = 1.0
    k = 0
    while True:
      x_trial = x + step * direction
      r_trial = res_fn(x_trial)
      n_res += 1
      trial_norm = mean_abs(r_trial)
      ok = (trial_norm <= (1.0 - sufficient_decrease * step) * norm) or (
          step * max_abs_dir <= 1e-7)
      ok = ok and not np.isnan(trial_norm)
      k += 1
      if ok or k >= max_ls or not ls:
        break
      step *= delta_reduction
      n_ls_extra += 1
    x, r, norm = x_trial, r_trial, trial_norm
    last_tau = step
    taus.append(step)
    it += 1
    hist.append(norm)
    xhist.append(x)
    if verbose:
      print(f'  it {it}: norm={norm:.3e} tau={step}')
  return dict(x=x, iterations=it, converged=bool(norm <= tol), final_norm=norm,
              n_res=n_res, n_jac=n_jac, n_ls_extra=n_ls_extra, hist=hist, taus=taus, xhist=xhist)


def truncate_band(A, k_cells, n_ch, n_cells):
  """Zero Jacobian entries coupling cells further apart than k_cells."""
  A = np.asarray(A)
  cell_of = np.arange(n_ch * n_cells) % n_cells  # channel-major ordering
  mask = np.abs(cell_of[:, None] - cell_of[None, :]) <= k_cells
  return jnp.asarray(A * mask)
