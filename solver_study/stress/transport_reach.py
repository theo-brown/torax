"""Stencil of the raw turbulent transport coefficients.

python transport_reach.py MODEL GRID N_RHO
MODEL: qlknn | qlknn_rotation | tglfnn_rotation; GRID: uniform | skewed
(face_centers = u + 0.1 u (1 - u)). Prints, per cell-minus-face offset, the
largest |dh/dx| relative to the largest entry of its coefficient, at the
initial state and at 5% noise, and every entry outside the stencil that the
structured Jacobian uses. TGLFNN-UKAEA with rotation has NaN in the rows of
faces 1 and 2 (an infinite partial derivative near the axis turns every column
of those rows into NaN in forward mode, in both Jacobian modes), so rows that
are not finite are skipped.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax  # noqa: E402
import numpy as np  # noqa: E402
import harness  # noqa: E402
from torax.tests.test_data import test_iterhybrid_predictor_corrector_rotation as qlknn_rotation  # noqa: E402
from torax.tests.test_data import test_iterhybrid_predictor_corrector_tglfnn_ukaea_rotation as tglfnn_rotation  # noqa: E402

model, grid, n_rho = sys.argv[1], sys.argv[2], int(sys.argv[3])
if model == 'qlknn':
  config = harness.base_config(n_rho)
else:
  config = copy.deepcopy(
      {'qlknn_rotation': qlknn_rotation, 'tglfnn_rotation': tglfnn_rotation}[
          model
      ].CONFIG
  )
  config['solver'] = dict(config['solver'], solver_type='newton_raphson')
  config['geometry']['n_rho'] = n_rho
  if model == 'qlknn_rotation':
    qlknn = config['transport']['core_transport_models']['qlknn']
    qlknn['rotation_mode'] = 'full_radius'
if grid == 'skewed':
  u = np.linspace(0.0, 1.0, n_rho + 1)
  config['geometry'].pop('n_rho')
  config['geometry']['face_centers'] = u + 0.1 * u * (1 - u)
kwargs, _ = harness.solve_block_kwargs(config)
cap, jac_fn = harness.capture(harness.residual_fun_from_kwargs(kwargs))
jax.eval_shape(jac_fn, harness.perturbed_x(kwargs, 0.0))  # fills cap


@jax.jit
def dh(x):
  q = cap['state_globals'](x)
  return jax.jacfwd(lambda x: cap['raw_transport'](x, q))(x)


n_cells = cap['n_cells']
n_faces = n_cells + 1
print(f'{model} {grid} n_rho={n_rho} stencil used: {cap["h_reach"]}')
left, right = cap['h_reach']
worst, outside = {}, {}
for scale, seed in ((0.0, 0), (0.05, 1)):
  m = np.asarray(dh(harness.perturbed_x(kwargs, scale, seed)))
  finite_rows = np.isfinite(m).all(axis=1)
  print(f'  noise {scale}: faces with non-finite rows: '
        f'{sorted(set(int(r % n_faces) for r in np.where(~finite_rows)[0]))}')
  for coef in range(4):
    block = m[coef * n_faces:(coef + 1) * n_faces]
    ok = np.isfinite(block).all(axis=1)
    top = np.abs(block[ok]).max() if ok.any() else 0.0
    if top == 0:
      continue
    for f in np.where(ok)[0]:
      for j in range(m.shape[1]):
        off = j % n_cells - f
        v = abs(block[f, j]) / top
        worst[off] = max(worst.get(off, 0.0), v)
        if (off < -left or off > right) and v > 1e-12:
          key = (int(f), off)
          outside[key] = max(outside.get(key, 0.0), v)
for off in sorted(worst):
  if worst[off] > 0:
    print(f'  offset {off:+d}: {worst[off]:.1e}')
for (f, off), v in sorted(outside.items()):
  print(f'  outside the stencil: face {f} offset {off:+d}: {v:.1e}')
