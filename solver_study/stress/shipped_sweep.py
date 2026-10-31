"""Structured vs jacfwd Jacobian for one shipped config, forced to Newton.

python shipped_sweep.py MODULE [N_RHO]
Prints the worst entry error relative to its row norm at x_old and at 5% noise.
"""
import copy
import importlib
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import numpy as np
import harness
from torax._src.solver import structured_jacobian as sj

module, n_rho = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16
config = copy.deepcopy(importlib.import_module(module).CONFIG)
config['solver'] = dict(config.get('solver', {}), solver_type='newton_raphson')
if 'face_centers' not in config['geometry']:
  config['geometry']['n_rho'] = n_rho
try:
  kwargs, _ = harness.solve_block_kwargs(config)
except Exception as e:  # pylint: disable=broad-except
  print(f'SKIP {module}: {type(e).__name__}: {str(e).splitlines()[0][:100]}')
  sys.exit(0)
residual_fun = harness.residual_fun_from_kwargs(kwargs)
structured = jax.jit(sj.jacobian_fn(residual_fun))
dense = jax.jit(jax.jacfwd(residual_fun))
errors = []
for scale, seed in ((0.0, 0), (0.05, 1)):
  x = harness.perturbed_x(kwargs, scale, seed)
  js, jd = np.asarray(structured(x)), np.asarray(dense(x))
  ok = np.isfinite(js) & np.isfinite(jd)
  if not ok.all():
    errors.append(f'non-finite (structured {int((~np.isfinite(js)).sum())}, dense {int((~np.isfinite(jd)).sum())})')
    continue
  row = np.max(np.abs(js - jd).max(axis=1) / np.maximum(np.linalg.norm(jd, axis=1), 1e-300))
  errors.append(f'{row:.1e}')
print(f'RESULT {module.split(".")[-1]}: worst row x_old {errors[0]}, 5% noise {errors[1]}')
