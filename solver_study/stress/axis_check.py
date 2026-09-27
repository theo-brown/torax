"""White-box check of the near-axis coupling of the residual.

python axis_check.py GRID N_RHO MIN_RHO_NORM
GRID: uniform | skewed (face_centers = u + 0.1 u (1 - u)) | axis_refined
(face_centers = u**1.5). With the ohmic source, whose heating uses the
extrapolated current density. Prints (via harness.analyse) the entries of dG/dx
(transport coefficients and globals frozen) outside the band |i - j| <= 2 and
the four psi columns that the assembly colours separately, and the error of
the structured Jacobian against jax.jacfwd.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import harness  # noqa: E402

grid, n_rho, min_rho_norm = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
config = harness.base_config(n_rho)
config['sources']['ohmic'] = {}
config['numerics']['min_rho_norm'] = min_rho_norm
u = np.linspace(0.0, 1.0, n_rho + 1)
if grid != 'uniform':
  config['geometry'].pop('n_rho')
  config['geometry']['face_centers'] = {
      'skewed': u + 0.1 * u * (1 - u),
      'axis_refined': u**1.5,
  }[grid]
kwargs, _ = harness.solve_block_kwargs(config)
residual_fun = harness.residual_fun_from_kwargs(kwargs)
harness.analyse(
    residual_fun,
    harness.perturbed_x(kwargs, 0.05, 1),
    tag=f'{grid} n_rho={n_rho} min_rho_norm={min_rho_norm}',
)
