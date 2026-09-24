"""Temporal accuracy experiment: backward Euler (theta=1) vs Crank-Nicolson
(theta=0.5) vs a BDF2 prototype (built from the BE step by substituting an
effective x_old and dt), all with the Newton solver so the error is purely
temporal. Errors measured against a fine-dt backward-Euler reference."""
import copy, json, sys, time, dataclasses
import numpy as np
import jax, jax.numpy as jnp
import torax
from torax._src.config import build_runtime_params
from torax._src.core_profiles import updaters
from torax._src.orchestration import run_simulation as rs
from torax._src.orchestration import step_function_processing
from torax.examples import iterhybrid_predictor_corrector

T_FINAL = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
DTS = [float(a) for a in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0.2, 0.1, 0.05, 0.025, 0.0125]
DT_REF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0015625
METHODS = sys.argv[4].split(',') if len(sys.argv) > 4 else ['be', 'cn', 'bdf2']
OUT = {'t_final': T_FINAL, 'dts': DTS, 'dt_ref': DT_REF}


def make_cfg(theta, dt):
  c = copy.deepcopy(iterhybrid_predictor_corrector.CONFIG)
  c['numerics']['t_final'] = T_FINAL
  c['numerics']['fixed_dt'] = dt
  c['numerics']['adaptive_dt'] = False
  c['time_step_calculator'] = {'calculator_type': 'fixed'}
  c['solver'] = {'solver_type': 'newton_raphson', 'theta_implicit': theta,
                 'use_predictor_corrector': True, 'n_corrector_steps': 10,
                 'use_pereverzev': True, 'chi_pereverzev': 30, 'D_pereverzev': 15,
                 'residual_tol': 1e-7, 'residual_coarse_tol': 1e-2, 'n_max_iterations': 40}
  return torax.ToraxConfig.from_dict(c)


def profiles(state):
  cp = state.core_profiles
  return {k: np.asarray(getattr(cp, k).value) for k in ['T_i', 'T_e', 'n_e', 'psi']}


def run_theta(theta, dt):
  cfg = make_cfg(theta, dt)
  state, ppo, step_fn = rs.prepare_simulation(cfg)
  n = 0; stats = []
  t0 = time.perf_counter()
  while not step_fn.is_done(state.t):
    state, ppo = step_fn(state, ppo)
    err = step_fn.check_for_errors(state, ppo)
    stats.append((int(state.solver_numeric_outputs.inner_solver_iterations), int(state.solver_numeric_outputs.solver_error_state)))
    assert err == torax.SimError.NO_ERROR, err
    n += 1
  return profiles(state), n, stats, time.perf_counter() - t0, float(state.t)


def run_bdf2(dt):
  """BDF2 with constant step: x_{n+1} - (4/3) x_n + (1/3) x_{n-1} = (2/3) dt F(x_{n+1}).
  Implemented by calling the BE solver with x_old_eff = (4/3) x_n - (1/3) x_{n-1}
  and dt_eff = (2/3) dt. First step: BE (dt) as startup. The transient
  coefficient ratio tc_in_old/tc_in_new is evaluated at x_old_eff (approximation
  that is exact when the transient coefficient is constant in time)."""
  cfg = make_cfg(1.0, dt)
  state, ppo, step_fn = rs.prepare_simulation(cfg)
  solver = step_fn.solver
  models = solver.models
  rpp, gp = step_fn.runtime_params_provider, step_fn.geometry_provider
  prev_cp = None
  n = 0; stats = []
  t0 = time.perf_counter()
  while not step_fn.is_done(state.t):
    (runtime_params_t, geo_t, explicit_source_profiles, edge_outputs, pts) = step_function_processing.pre_step(
        input_state=state, runtime_params_provider=rpp, geometry_provider=gp, models=models)
    dt_ = jnp.asarray(min(dt, float(runtime_params_t.numerics.t_final - state.t)))
    runtime_params_t_plus_dt, geo_t_plus_dt = build_runtime_params.get_consistent_runtime_params_and_geometry(
        t=state.t + dt_, runtime_params_provider=rpp, geometry_provider=gp, edge_outputs=edge_outputs,
        core_profiles=state.core_profiles)
    cp_t = state.core_profiles
    core_profiles_t_plus_dt = updaters.provide_core_profiles_t_plus_dt(
        dt=dt_, runtime_params_t=runtime_params_t, runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t_plus_dt=geo_t_plus_dt, core_profiles_t=cp_t)
    if prev_cp is None or float(dt_) != dt:
      cp_eff, dt_eff = cp_t, dt_
    else:
      repl = {}
      for k in ['T_i', 'T_e', 'n_e', 'psi']:
        cv = getattr(cp_t, k); cvp = getattr(prev_cp, k)
        repl[k] = dataclasses.replace(cv, value=(4.0/3.0) * cv.value - (1.0/3.0) * cvp.value)
      cp_eff = dataclasses.replace(cp_t, **repl)
      dt_eff = dt_ * (2.0 / 3.0)
    x_new, sno = solver(t=state.t, dt=dt_eff, runtime_params_t=runtime_params_t,
                        runtime_params_t_plus_dt=runtime_params_t_plus_dt, geo_t=geo_t,
                        geo_t_plus_dt=geo_t_plus_dt, core_profiles_t=cp_eff,
                        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
                        explicit_source_profiles=explicit_source_profiles, pedestal_transition_state=pts)
    stats.append((int(sno.inner_solver_iterations), int(sno.solver_error_state)))
    assert int(sno.solver_error_state) != 1, 'BDF2 newton failed'
    new_state, ppo = step_function_processing.finalize_outputs(
        t=state.t, dt=dt_, x_new=x_new, solver_numeric_outputs=sno,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt, geometry_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=cp_t, core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        explicit_source_profiles=explicit_source_profiles, edge_outputs=edge_outputs, models=models,
        evolving_names=runtime_params_t.numerics.evolving_names, input_post_processed_outputs=ppo,
        time_step_calculator_state_t=state.time_step_calculator_state, pedestal_transition_state=pts)
    prev_cp = cp_t
    state = new_state
    n += 1
  return profiles(state), n, stats, time.perf_counter() - t0, float(state.t)


def err(a, b):
  return {k: float(np.max(np.abs(a[k] - b[k])) / np.max(np.abs(b[k]))) for k in b}

print('reference run (BE, dt=%g)...' % DT_REF)
ref, nref, sref, tref, tend = run_theta(1.0, DT_REF)
print(f'  steps={nref} wall={tref:.1f}s t_end={tend}')
OUT['ref'] = dict(steps=nref, wall=tref)
# a second reference with 2x dt to estimate the reference's own error
ref2, n2, s2, t2, _ = run_theta(1.0, 2 * DT_REF)
OUT['ref_self_err'] = err(ref2, ref)
print('  BE(2*dt_ref) vs BE(dt_ref) max rel err:', OUT['ref_self_err'])
for m in METHODS:
  OUT[m] = {}
  for dt in DTS:
    try:
      if m == 'be': pr, n, st, w, te = run_theta(1.0, dt)
      elif m == 'cn': pr, n, st, w, te = run_theta(0.5, dt)
      else: pr, n, st, w, te = run_bdf2(dt)
      e = err(pr, ref)
      OUT[m][str(dt)] = dict(steps=n, wall=w, err=e, newton_its=[s[0] for s in st], errstates=[s[1] for s in st])
      print(f'{m:5s} dt={dt:<8g} steps={n:4d} wall={w:6.1f}s newton_its(mean)={np.mean([s[0] for s in st]):.1f} err={ {k: '%.2e' % v for k, v in e.items()} }')
    except Exception as ex:
      OUT[m][str(dt)] = dict(error=str(ex))
      print(f'{m} dt={dt} FAILED: {ex}')
    json.dump(OUT, open(f'time_accuracy_T{T_FINAL:g}.json', 'w'), indent=1)
print('DONE')
