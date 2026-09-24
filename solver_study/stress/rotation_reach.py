"""Measures the stencil of the raw transport coefficients with rotation.

python rotation_reach.py MODE N_RHO
Prints, per cell-minus-face offset, the largest |d h / d x| relative to the
largest entry of that coefficient, over three states.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import numpy as np
import harness

mode, n_rho = sys.argv[1], int(sys.argv[2])
config = harness.base_config(n_rho)
config['transport']['core_transport_models']['qlknn']['rotation_mode'] = mode
kwargs, _ = harness.solve_block_kwargs(config)
residual_fun = harness.residual_fun_from_kwargs(kwargs)
cap, jac_fn = harness.capture(residual_fun)
jax.eval_shape(jac_fn, harness.perturbed_x(kwargs, 0.0))  # fills cap


@jax.jit
def dh(x):
  q = cap['state_globals'](x)  # h does not depend on q differentiably.
  return jax.jacfwd(lambda x: cap['raw_transport'](x, q))(x)


n_cells, n_ch = cap['n_cells'], cap['n_channels']
n_faces = n_cells + 1
worst = {}
for scale, seed in ((0.0, 0), (0.05, 1), (0.2, 2)):
  m = np.asarray(dh(harness.perturbed_x(kwargs, scale, seed)))
  for coef in range(4):
    block = m[coef * n_faces:(coef + 1) * n_faces]
    top = np.abs(block).max()
    if top == 0:
      continue
    for f in range(n_faces):
      for j in range(n_ch * n_cells):
        v = abs(block[f, j]) / top
        off = j % n_cells - f
        worst[off] = max(worst.get(off, 0.0), v)
print(f'mode={mode} n_rho={n_rho} channels={n_ch}')
for off in sorted(worst):
  print(f'  offset {off:+d}: {worst[off]:.2e}')
