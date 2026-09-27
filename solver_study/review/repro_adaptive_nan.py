"""ADAPTIVE_TRANSPORT pedestal switched off: structured vs dense (REPORT 3.14).

Before the fix of 3.14 (1), 768 of the 1024 entries of the structured Jacobian
were NaN while jax.jacfwd was finite, the TORAX_ERRORS_ENABLED check did not
fire, and the structured Newton solve failed where the dense one converged in
six iterations. Both are now finite and equal to round-off, and both solves
take the same iterations to the same state.

  python repro_adaptive_nan.py
"""
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stress')
)
import harness  # pylint: disable=g-import-not-at-top
import jax  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top
from torax._src import jax_utils  # pylint: disable=g-import-not-at-top
from torax._src.fvm import fvm_conversions  # pylint: disable=g-import-not-at-top
from torax._src.fvm import newton_raphson_solve_block  # pylint: disable=g-import-not-at-top
from torax._src.solver import structured_jacobian as sj  # pylint: disable=g-import-not-at-top

config = harness.base_config(8)
config['pedestal'] = {
    'model_name': 'set_T_ped_n_ped',
    'mode': 'ADAPTIVE_TRANSPORT',
    'set_pedestal': False,  # e.g. an L-mode phase: {0: False, 1: True}
    'T_i_ped': 4.5, 'T_e_ped': 4.5, 'n_e_ped': 0.62e20,
    'rho_norm_ped_top': 0.9,
}
kwargs, _ = harness.solve_block_kwargs(config)
residual_fun = harness.residual_fun_from_kwargs(kwargs)
x = fvm_conversions.cell_variable_tuple_to_vec(kwargs['x_old'])

dense = np.asarray(jax.jit(jax.jacfwd(residual_fun))(x))
with jax_utils.enable_errors(True):
  structured = np.asarray(jax.jit(sj.jacobian_fn(residual_fun))(x))
print('dense non-finite:', int((~np.isfinite(dense)).sum()), 'of', dense.size)
print('structured non-finite (errors enabled, no exception):',
      int((~np.isfinite(structured)).sum()), 'of', structured.size)
ok = np.isfinite(structured)
print('max |structured - dense| on finite entries:',
      float(np.abs(structured[ok] - dense[ok]).max()))

for mode in ('dense', 'structured'):
  x_new, out = newton_raphson_solve_block.newton_raphson_solve_block(
      jacobian_mode=mode, **kwargs)
  print(f'{mode}: solver_error_state={int(out.solver_error_state)}'
        f' iterations={int(out.inner_solver_iterations)}'
        f' T_e[0]={float(x_new[1].value[0]):.6f}')
