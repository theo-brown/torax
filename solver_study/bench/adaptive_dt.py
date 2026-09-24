"""Step-size control experiment: TORAX's chi-based heuristic vs a local-error
controlled backward Euler (CVODE/PETSc-style: LTE estimated from the
difference between the linear-extrapolation predictor and the corrected
solution, elementary controller with safety factor, growth/shrink clips and
step rejection). Same physics/solver; errors vs fine-dt reference."""
import copy, json, sys, time, dataclasses
import numpy as np
import jax, jax.numpy as jnp
import torax
from torax._src.orchestration import run_simulation as rs
from torax.examples import iterhybrid_predictor_corrector

T_FINAL = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
DT_REF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0015625
TOLS = [float(a) for a in sys.argv[3].split(',')] if len(sys.argv) > 3 else [3e-2, 1e-2, 3e-3, 1e-3]
OUT = {'t_final': T_FINAL, 'dt_ref': DT_REF}
KEYS = ['T_i', 'T_e', 'n_e', 'psi']

def make_cfg(dt_mode, dt=None, prefactor=None):
  c = copy.deepcopy(iterhybrid_predictor_corrector.CONFIG)
  c['numerics']['t_final'] = T_FINAL
  c['numerics']['adaptive_dt'] = False
  c['solver'] = {'solver_type': 'newton_raphson', 'use_predictor_corrector': True, 'n_corrector_steps': 10,
                 'use_pereverzev': True, 'chi_pereverzev': 30, 'D_pereverzev': 15,
                 'residual_tol': 1e-7, 'residual_coarse_tol': 1e-2, 'n_max_iterations': 40}
  if dt_mode == 'fixed':
    c['numerics']['fixed_dt'] = dt
    c['time_step_calculator'] = {'calculator_type': 'fixed'}
  else:
    c['time_step_calculator'] = {'calculator_type': 'chi'}
    if prefactor is not None:
      c['numerics']['chi_timestep_prefactor'] = prefactor
  return torax.ToraxConfig.from_dict(c)

def profiles(state):
  return {k: np.asarray(getattr(state.core_profiles, k).value) for k in KEYS}

def err(a, b):
  return {k: float(np.max(np.abs(a[k] - b[k])) / np.max(np.abs(b[k]))) for k in b}

def run_cfg(cfg):
  state, ppo, step_fn = rs.prepare_simulation(cfg)
  n = 0; its = []; dts = []
  while not step_fn.is_done(state.t):
    state, ppo = step_fn(state, ppo)
    assert step_fn.check_for_errors(state, ppo) == torax.SimError.NO_ERROR
    n += 1; its.append(int(state.solver_numeric_outputs.inner_solver_iterations)); dts.append(float(state.dt))
  return profiles(state), n, its, dts

def wrms(d, x, rtol, atol=1e-3):
  w = 1.0 / (rtol * np.abs(x) + atol)
  return float(np.sqrt(np.mean((d * w) ** 2)))

def run_error_controlled(rtol, dt0=0.01, dt_max=0.5, dt_min=1e-4, safety=0.9, grow=2.0, shrink=0.2):
  cfg = make_cfg('fixed', dt=dt_max)  # fixed_dt = dt_max; the actual dt is imposed through max_dt
  state, ppo, step_fn = rs.prepare_simulation(cfg)
  dt = dt0; prev = None  # (x_prev, dt_prev)
  n = 0; rej = 0; its = []; dts = []
  x_cur = np.concatenate([profiles(state)[k] for k in KEYS])
  t_final = float(cfg.numerics.t_final)
  while float(state.t) < t_final - 1e-9:
    dt = min(dt, t_final - float(state.t))
    new_state, new_ppo = step_fn(state, ppo, max_dt=jnp.asarray(dt))
    assert step_fn.check_for_errors(new_state, new_ppo) == torax.SimError.NO_ERROR
    x_new = np.concatenate([profiles(new_state)[k] for k in KEYS])
    dt_taken = float(new_state.dt)
    if prev is None:
      # no history: accept, and use a conservative estimate from the change
      est = None
    else:
      x_prev, dt_prev = prev
      x_pred = x_cur + (dt_taken / dt_prev) * (x_cur - x_prev)
      # LTE of BE ~ (corrector - predictor) / (1 + dt/dt_prev)
      lte = (x_new - x_pred) / (1.0 + dt_taken / dt_prev)
      est = wrms(lte, x_new, rtol)
    if est is not None and est > 1.0:
      rej += 1
      dt = max(dt_taken * max(shrink, safety * est ** (-0.5)), dt_min)
      if dt_taken <= dt_min * 1.01:
        pass
      else:
        continue  # reject and retry from the same state
    # accept
    prev = (x_cur, dt_taken); x_cur = x_new; state, ppo = new_state, new_ppo
    n += 1; its.append(int(state.solver_numeric_outputs.inner_solver_iterations)); dts.append(dt_taken)
    if est is None:
      dt = dt_taken
    else:
      dt = dt_taken * min(grow, max(shrink, safety * max(est, 1e-10) ** (-0.5)))
    dt = min(dt, dt_max)
  return profiles(state), n, its, dts, rej

print('reference...')
t0 = time.time(); ref, nref, _, _ = run_cfg(make_cfg('fixed', dt=DT_REF)); print(f'  ref steps={nref} wall={time.time()-t0:.0f}s')
# TORAX chi-based heuristic with different prefactors
OUT['chi'] = {}
SKIP_SWEEPS = len(sys.argv) > 4 and sys.argv[4] == 'errctrl_only'
for pf in ([] if SKIP_SWEEPS else [10, 30, 50, 100, 300]):
  try:
    pr, n, its, dts = run_cfg(make_cfg('chi', prefactor=pf))
    e = err(pr, ref)
    OUT['chi'][str(pf)] = dict(steps=n, newton_its=int(np.sum(its)), dt_min=min(dts), dt_med=float(np.median(dts)), dt_max=max(dts), err=e)
    print(f'chi prefactor={pf:<4} steps={n:4d} newton_its={int(np.sum(its)):4d} dt[min/med/max]={min(dts):.3g}/{np.median(dts):.3g}/{max(dts):.3g} maxerr={max(e.values()):.2e} {e}')
  except Exception as ex:
    print('chi', pf, 'FAILED', ex)
OUT['fixed'] = {}
for dt in ([] if SKIP_SWEEPS else [0.2, 0.1, 0.05, 0.025]):
  pr, n, its, dts = run_cfg(make_cfg('fixed', dt=dt))
  e = err(pr, ref)
  OUT['fixed'][str(dt)] = dict(steps=n, newton_its=int(np.sum(its)), err=e)
  print(f'fixed dt={dt:<6} steps={n:4d} newton_its={int(np.sum(its)):4d} maxerr={max(e.values()):.2e} {e}')
OUT['errctrl'] = {}
for rtol in TOLS:
  try:
    pr, n, its, dts, rej = run_error_controlled(rtol)
    e = err(pr, ref)
    OUT['errctrl'][str(rtol)] = dict(steps=n, rejected=rej, newton_its=int(np.sum(its)), dt_min=min(dts), dt_med=float(np.median(dts)), dt_max=max(dts), err=e, dts=dts)
    print('   dt sequence:', ' '.join('%.3g' % d for d in dts))
    print(f'errctrl rtol={rtol:<6g} steps={n:4d} (+{rej} rejected) newton_its={int(np.sum(its)):4d} dt[min/med/max]={min(dts):.3g}/{np.median(dts):.3g}/{max(dts):.3g} maxerr={max(e.values()):.2e} {e}')
  except Exception as ex:
    import traceback; traceback.print_exc()
    print('errctrl', rtol, 'FAILED', ex)
  json.dump(OUT, open(f'adaptive_dt_T{T_FINAL:g}' + ('_errctrl' if SKIP_SWEEPS else '') + '.json', 'w'), indent=1)
print('DONE')
