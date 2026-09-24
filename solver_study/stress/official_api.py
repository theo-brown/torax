"""Time torax.run_simulation (the public entry point) for one mode.

python official_api.py MODE N_RHO T_FINAL [wrap]
"""
import copy
import sys
import time

import torax
from torax.examples import iterhybrid_rampup

MODE, N_RHO, T_FINAL = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
WRAP = len(sys.argv) > 4 and sys.argv[4] == 'wrap'


def run():
  cfg = copy.deepcopy(iterhybrid_rampup.CONFIG)
  cfg['solver']['jacobian_mode'] = MODE
  cfg['geometry']['n_rho'] = N_RHO
  cfg['numerics']['t_final'] = T_FINAL
  t0 = time.perf_counter()
  _, history = torax.run_simulation(torax.ToraxConfig.from_dict(cfg),
                                    progress_bar=False)
  wall = time.perf_counter() - t0
  print(f'RESULT mode={MODE} n_rho={N_RHO} t_final={T_FINAL} wrap={WRAP} '
        f'wall={wall:.1f}s steps={len(history.times) - 1} '
        f'T_e0={float(history.core_profiles[-1].T_e.value[0]):.12g}', flush=True)


def main():
  run()


if WRAP:
  main()
else:
  cfg = copy.deepcopy(iterhybrid_rampup.CONFIG)
  cfg['solver']['jacobian_mode'] = MODE
  cfg['geometry']['n_rho'] = N_RHO
  cfg['numerics']['t_final'] = T_FINAL
  t0 = time.perf_counter()
  _, history = torax.run_simulation(torax.ToraxConfig.from_dict(cfg),
                                    progress_bar=False)
  wall = time.perf_counter() - t0
  print(f'RESULT mode={MODE} n_rho={N_RHO} t_final={T_FINAL} wrap={WRAP} '
        f'wall={wall:.1f}s steps={len(history.times) - 1} '
        f'T_e0={float(history.core_profiles[-1].T_e.value[0]):.12g}', flush=True)
