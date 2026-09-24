"""Same physics, same fixed dt sequence: how far is the linear (Picard) solver
from the converged Newton solution after a 2 s simulation?"""
import copy, json, sys
import numpy as np
import torax
from torax._src.orchestration import run_simulation as rs
from torax.examples import iterhybrid_predictor_corrector

T_FINAL = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
DT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
KEYS = ['T_i', 'T_e', 'n_e', 'psi']

def run(solver):
  c = copy.deepcopy(iterhybrid_predictor_corrector.CONFIG)
  c['numerics']['t_final'] = T_FINAL; c['numerics']['fixed_dt'] = DT; c['numerics']['adaptive_dt'] = False
  c['time_step_calculator'] = {'calculator_type': 'fixed'}
  c['solver'].update(solver)
  cfg = torax.ToraxConfig.from_dict(c)
  state, ppo, step_fn = rs.prepare_simulation(cfg)
  n = 0
  while not step_fn.is_done(state.t):
    state, ppo = step_fn(state, ppo); n += 1
    assert step_fn.check_for_errors(state, ppo) == torax.SimError.NO_ERROR
  return {k: np.asarray(getattr(state.core_profiles, k).value) for k in KEYS}, n

ref, n = run({'solver_type': 'newton_raphson', 'residual_tol': 1e-7, 'n_max_iterations': 40})
print('newton (tol 1e-7): steps', n)
OUT = {}
for label, s in [('newton_default_tol', {'solver_type': 'newton_raphson'}),
                 ('linear_1corr', {'solver_type': 'linear', 'use_predictor_corrector': True, 'n_corrector_steps': 1}),
                 ('linear_3corr', {'solver_type': 'linear', 'use_predictor_corrector': True, 'n_corrector_steps': 3}),
                 ('linear_10corr', {'solver_type': 'linear', 'use_predictor_corrector': True, 'n_corrector_steps': 10}),
                 ('linear_30corr', {'solver_type': 'linear', 'use_predictor_corrector': True, 'n_corrector_steps': 30}),
                 ('linear_nopc', {'solver_type': 'linear', 'use_predictor_corrector': False})]:
  pr, n = run(s)
  e = {k: float(np.max(np.abs(pr[k] - ref[k])) / np.max(np.abs(ref[k]))) for k in KEYS}
  OUT[label] = dict(steps=n, err=e)
  print(f'{label:20s} steps={n} max rel diff vs converged newton: ' + ' '.join(f'{k}={v:.2e}' for k, v in e.items()))
json.dump(OUT, open(f'linear_vs_newton_dt{DT:g}.json', 'w'), indent=1)
print('DONE')
