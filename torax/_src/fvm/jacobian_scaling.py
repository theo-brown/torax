# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Row/column equilibration scales for the theta-method residual.

The theta-method Jacobian mixes channels with very different physical scales
(temperatures in keV, poloidal flux in Wb, scaled density) and rows whose
magnitudes vary over several orders of magnitude between quiescent regions
and regions with stiff transport (measured condition numbers of 1e5-1e8 on
representative cases). This module computes diagonal scalings that
equilibrate the system solved by iterative solvers:

  - Column scales: per-channel, from the magnitude of the state at the start
    of the time step, so that scaled state entries are O(1).
  - Row scales: elementwise, from the row maxima of the Jacobian evaluated at
    the initial guess (a one-step Ruiz equilibration). Cheaper proxies that
    avoid a Jacobian evaluation (e.g. the diagonal of the assembled
    theta-method LHS matrix) were measured to miss the rows dominated by
    stiff transport-derivative contributions, and recovered almost none of
    the achievable conditioning improvement.

Solving the equivalent scaled system R_hat(x_hat) = D_r R(D_c x_hat) = 0
leaves the root unchanged but improves the conditioning of the Jacobian seen
by the solver (D_r J D_c, measured 10-50x better on representative cases) and
makes residual tolerances, Armijo tests and step-size thresholds
scale-invariant across machines and unit conventions.
"""

import jax
import jax.numpy as jnp
from torax._src.fvm import cell_variable

# Floors avoiding division by zero for identically-zero channels or
# structurally-zero Jacobian rows.
_MIN_COL_SCALE = 1e-8
_MIN_ROW_MAX = 1e-8


def compute_column_scales(
    x_old: tuple[cell_variable.CellVariable, ...],
) -> jax.Array:
  """Returns per-channel column scales for the flattened state vector.

  Args:
    x_old: Tuple of CellVariables at the start of the time step (in solver
      scaling).

  Returns:
    Positive vector of length N = num_channels * n_cells (channel-major,
    matching fvm_conversions.cell_variable_tuple_to_vec): per-channel
    max|x_old|.
  """
  return jnp.concatenate([
      jnp.full_like(
          var.value,
          jnp.maximum(jnp.max(jnp.abs(var.value)), _MIN_COL_SCALE),
      )
      for var in x_old
  ])


def compute_row_scales(
    jacobian: jax.Array,
    col_scale: jax.Array,
) -> jax.Array:
  """Returns row scales equilibrating the column-scaled Jacobian.

  Args:
    jacobian: The residual Jacobian evaluated at the initial guess, shape
      (N, N).
    col_scale: Column scales from compute_column_scales.

  Returns:
    Positive vector of length N: the reciprocal row maxima of
    |jacobian| * col_scale, so that every row of the scaled Jacobian
    row_scale * J * col_scale has unit infinity-norm.
  """
  row_max = jnp.max(jnp.abs(jacobian) * col_scale[None, :], axis=1)
  return 1.0 / jnp.maximum(row_max, _MIN_ROW_MAX)
