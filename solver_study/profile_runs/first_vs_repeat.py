"""Wall time of the first and of repeated torax.run_simulation calls.

python first_vs_repeat.py CASE N_RHO MODE [RUNS]
CASE: rampup | rampup_global | predictor_corrector. Each run builds the
ToraxConfig and calls torax.run_simulation, as a user script would; the XLA
compilations during each run are counted with jax.monitoring.
"""
import copy
import json
import os
import sys
import time

t_process = time.perf_counter()
import jax  # pylint: disable=g-import-not-at-top
import torax  # pylint: disable=g-import-not-at-top

import_s = time.perf_counter() - t_process
COMPILES = []


def _listener(event, duration, **unused_kwargs):
  if event == '/jax/core/compile/backend_compile_duration':
    COMPILES.append(duration)


jax.monitoring.register_event_duration_secs_listener(_listener)

STRESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stress')
# Dummy ToricNN weights (the file is written by importing stress/harness.py).
TORIC = os.path.join(STRESS, 'toric_nn.json')


def build_config(case, n_rho, mode):
  if case == 'rampup_global' and not os.path.exists(TORIC):
    sys.path.insert(0, STRESS)
    import harness  # pylint: disable=g-import-not-at-top,unused-import
  if case in ('rampup', 'rampup_global'):
    from torax.examples import iterhybrid_rampup as example  # pylint: disable=g-import-not-at-top
    config = copy.deepcopy(example.CONFIG)
    config['geometry']['n_rho'] = n_rho
    config['solver']['jacobian_mode'] = mode
    if case == 'rampup_global':
      config['geometry'] = {'geometry_type': 'circular', 'n_rho': n_rho}
      config['sources'].update({
          'cyclotron_radiation': {},
          'impurity_radiation': {
              'model_name': 'P_in_scaled_flat_profile',
              'fraction_P_heating': 0.1,
          },
          'icrh': {
              'model_path': TORIC,
              'P_total': 10e6,
              'wall_inner': 1.24,
              'wall_outer': 2.43,
          },
      })
  elif case == 'predictor_corrector':
    from torax.examples import iterhybrid_predictor_corrector as example  # pylint: disable=g-import-not-at-top
    config = copy.deepcopy(example.CONFIG)
    config['numerics']['t_final'] = 1.0
    config['geometry']['n_rho'] = n_rho
    config['solver'] = dict(
        config['solver'], solver_type='newton_raphson', jacobian_mode=mode
    )
  else:
    raise ValueError(case)
  return config


def main():
  case, n_rho, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
  runs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
  config = build_config(case, n_rho, mode)
  results = []
  for run in range(1, runs + 1):
    n_compiles = len(COMPILES)
    t0 = time.perf_counter()
    torax_config = torax.ToraxConfig.from_dict(copy.deepcopy(config))
    _, history = torax.run_simulation(torax_config, progress_bar=False)
    wall = time.perf_counter() - t0
    results.append(dict(
        run=run,
        wall_s=wall,
        compiles=len(COMPILES) - n_compiles,
        compile_s=sum(COMPILES[n_compiles:]),
        steps=len(history.times) - 1,
        t_final=float(history.times[-1]),
        T_e0=float(history.core_profiles[-1].T_e.value[0]),
    ))
    print('RUN ' + json.dumps(results[-1]), flush=True)
  print('RESULT ' + json.dumps(dict(
      case=case, n_rho=n_rho, mode=mode, import_s=import_s,
      process_s=time.perf_counter() - t_process, runs=results,
  )), flush=True)


if __name__ == '__main__':
  main()
