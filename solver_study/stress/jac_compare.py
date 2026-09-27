"""Structured vs jacfwd Jacobian for a named variant: python jac_compare.py VARIANT N_RHO

Prints the relative Frobenius error and the largest entry error relative to its
row norm, at x_old and at 5% and 20% multiplicative noise.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import numpy as np
import harness
import variants
from torax._src.solver import structured_jacobian as sj

name, n_rho = sys.argv[1], int(sys.argv[2])
config = variants.variant(name, n_rho)
kwargs, _ = harness.solve_block_kwargs(config)
residual_fun = harness.residual_fun_from_kwargs(kwargs)
structured = jax.jit(sj.jacobian_fn(residual_fun))
dense = jax.jit(jax.jacfwd(residual_fun))
for scale, seed in ((0.0, 0), (0.05, 1), (0.2, 2)):
  x = harness.perturbed_x(kwargs, scale, seed)
  js, jd = np.asarray(structured(x)), np.asarray(dense(x))
  fro = np.linalg.norm(js - jd) / np.linalg.norm(jd)
  row = np.max(np.max(np.abs(js - jd), axis=1) / np.linalg.norm(jd, axis=1))
  both = np.isfinite(js) & np.isfinite(jd)
  nan_note = f' (nan entries: structured {int((~np.isfinite(js)).sum())}, dense {int((~np.isfinite(jd)).sum())}, same positions {bool(np.array_equal(~np.isfinite(js), ~np.isfinite(jd)))})' if not both.all() else ''
  if not both.all():
    js, jd = np.where(both, js, 0), np.where(both, jd, 0)
    fro = np.linalg.norm(js - jd) / np.linalg.norm(jd)
    row = np.max(np.max(np.abs(js - jd), axis=1) / np.maximum(np.linalg.norm(jd, axis=1), 1e-300))
  print(f'{name} n_rho={n_rho} noise={scale}: frobenius {fro:.1e}, worst row {row:.1e}{nan_note}', flush=True)
