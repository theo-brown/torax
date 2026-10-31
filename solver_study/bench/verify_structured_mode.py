"""End-to-end check of jacobian_mode='structured' vs 'dense' on real steps."""
import copy, os, sys, time
import numpy as np
import jax
import torax
from torax.examples import iterhybrid_rampup, iterhybrid_predictor_corrector
import bench_common as bc

which = sys.argv[1] if len(sys.argv) > 1 else 'rampup'
n_rho = int(sys.argv[2]) if len(sys.argv) > 2 else 50
with_global_sources = len(sys.argv) > 3 and sys.argv[3] == 'globals'
CONFIG = copy.deepcopy(iterhybrid_rampup.CONFIG if which == 'rampup' else iterhybrid_predictor_corrector.CONFIG)
if with_global_sources:
  # Cyclotron radiation, constant-fraction impurity radiation and ToricNN ICRH
  # (dummy surrogate): the sources with global (integral) state dependences.
  import global_sources
  CONFIG['sources'].update(copy.deepcopy(global_sources.SOURCES))
  CONFIG['geometry'] = dict(global_sources.GEOMETRY, n_rho=n_rho)
for mode in ['dense', 'structured']:
  over = {'jacobian_mode': mode}
  if which != 'rampup': over['solver_type'] = 'newton_raphson'
  p = bc.build_problem(CONFIG, n_warm_steps=3, n_rho=n_rho, solver_overrides=over)
  sf = p.step_fn
  t0 = time.perf_counter(); out = sf(p.state, p.ppo); jax.block_until_ready(out); tc = time.perf_counter() - t0
  st, ppo = out
  tw = bc.timeit(lambda: sf(p.state, p.ppo), n=5)[0]
  cp = st.core_profiles
  vals = {k: np.asarray(getattr(cp, k).value) for k in ['T_i', 'T_e', 'n_e', 'psi']}
  print(f'{which} n_rho={n_rho} globals={with_global_sources} mode={mode}: compile {tc:.1f}s, step {tw*1e3:.1f} ms, inner its {int(st.solver_numeric_outputs.inner_solver_iterations)}, err state {int(st.solver_numeric_outputs.solver_error_state)}')
  if mode == 'dense': ref = vals
  else:
    print('  max rel diff vs dense:', {k: '%.1e' % (np.max(np.abs(vals[k] - ref[k])) / np.max(np.abs(ref[k]))) for k in vals})
print('DONE')
