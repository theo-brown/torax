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

"""The `make_diffusion_terms` function.

Builds the diffusion terms of the discrete matrix equation.
"""

import chex
from jax import numpy as jnp
import jaxtyping as jt
from torax._src import array_typing
from torax._src import tridiagonal
from torax._src.fvm import cell_variable


def make_diffusion_terms(
    d_face: array_typing.FloatVectorFace, var: cell_variable.CellVariable
) -> tuple[tridiagonal.TriDiagonal, array_typing.FloatVectorCell]:
  """Makes the terms of the matrix equation derived from the diffusion term.

  The diffusion term is of the form
  (partial / partial x) D partial x / partial x

  Single-channel wrapper around `make_multichannel_diffusion_terms`.

  Args:
    d_face: Diffusivity coefficient on faces.
    var: CellVariable (to define geometry and boundary conditions)

  Returns:
    mat: Tridiagonal matrix of coefficients on u
    c: Vector of terms not dependent on u
  """
  diagonal, above, below, vec = make_multichannel_diffusion_terms(
      d_face[:, jnp.newaxis], (var,)
  )
  return (
      tridiagonal.TriDiagonal(diagonal[:, 0], above[:, 0], below[:, 0]),
      vec[:, 0],
  )


def make_multichannel_diffusion_terms(
    d_face: jt.Float[array_typing.Array, 'face channel'],
    variables: tuple[cell_variable.CellVariable, ...],
) -> tuple[
    jt.Float[array_typing.Array, 'cell channel'],
    jt.Float[array_typing.Array, 'cell-1 channel'],
    jt.Float[array_typing.Array, 'cell-1 channel'],
    jt.Float[array_typing.Array, 'cell channel'],
]:
  """Builds the diffusion terms for every channel in one batched pass.

  Channels are independent in the diffusion operator, so this returns the three
  scalar diagonals per channel rather than a block matrix. All channels are
  assumed to live on the same mesh, which lets the interior stencil be evaluated
  once for all of them; only the boundary rows, whose constraint *type* differs
  per channel, are assembled channel by channel.

  Args:
    d_face: Diffusivity coefficients on faces, shape (num_faces, num_channels).
    variables: One CellVariable per channel, supplying the boundary conditions.
      All must share a mesh.

  Returns:
    diagonal, above, below: The three diagonals, each (num_cells, num_channels)
      or (num_cells - 1, num_channels).
    vec: Terms not dependent on u, shape (num_cells, num_channels).
  """
  var = variables[0]
  for other in variables[1:]:
    if other.face_centers.shape != var.face_centers.shape:
      raise ValueError(
          'All channels must share a mesh, got face_centers of shape '
          f'{other.face_centers.shape} and {var.face_centers.shape}.'
      )

  # Start by using the formula for the interior rows everywhere.
  dx = var.cell_widths
  cell_spacings = jnp.concat([dx[:1], var.cell_spacings, dx[-1:]])
  # Fill in the inner diagonal.
  face_flux_right = d_face[1:] / cell_spacings[1:, jnp.newaxis]
  face_flux_left = d_face[:-1] / cell_spacings[:-1, jnp.newaxis]
  diag = (-face_flux_right - face_flux_left) / dx[:, jnp.newaxis]

  off = d_face[1:-1] / var.cell_spacings[:, jnp.newaxis]
  # Divide by different cell widths for the upper and lower diagonals.
  upper_off = off / dx[:-1, jnp.newaxis]
  lower_off = off / dx[1:, jnp.newaxis]

  vec = jnp.zeros_like(diag)

  if vec.shape[0] < 2:
    raise NotImplementedError(
        'We do not support the case where a single cell'
        ' is affected by both boundary conditions.'
    )

  # Boundary rows need to be special-cased. The choice between the Dirichlet
  # and Neumann form is a per-channel Python branch, so these rows are the one
  # part that cannot be batched.
  left_diag, left_vec, right_diag, right_vec = [], [], [], []
  for channel, var_i in enumerate(variables):
    d_face_i = d_face[:, channel]
    # These checks are redundant with CellVariable.__post_init__, but including
    # them here for readability because they're in important part of the logic
    # of this function.
    chex.assert_exactly_one_is_none(
        var_i.left_face_grad_constraint, var_i.left_face_constraint
    )
    chex.assert_exactly_one_is_none(
        var_i.right_face_grad_constraint, var_i.right_face_constraint
    )

    if var_i.left_face_constraint is not None:
      # Left face Dirichlet condition.
      denom_left = cell_spacings[0] * dx[0]
      denom_right = cell_spacings[1] * dx[0]
      left_diag.append(
          -2 * d_face_i[0] / denom_left - d_face_i[1] / denom_right
      )
      left_vec.append(
          2 * d_face_i[0] * var_i.left_face_constraint / denom_left
      )
    else:
      # Left face gradient condition.
      denom_right = cell_spacings[1] * dx[0]
      left_diag.append(-d_face_i[1] / denom_right)
      left_vec.append(
          -d_face_i[0] * var_i.left_face_grad_constraint / dx[0]
      )

    if var_i.right_face_constraint is not None:
      # Right face Dirichlet condition.
      denom_left = cell_spacings[-2] * dx[-1]
      denom_right = cell_spacings[-1] * dx[-1]
      right_diag.append(
          -2 * d_face_i[-1] / denom_right - d_face_i[-2] / denom_left
      )
      right_vec.append(
          2 * d_face_i[-1] * var_i.right_face_constraint / denom_right
      )
    else:
      # Right face gradient condition.
      denom_left = cell_spacings[-2] * dx[-1]
      right_diag.append(-d_face_i[-2] / denom_left)
      right_vec.append(
          d_face_i[-1] * var_i.right_face_grad_constraint / dx[-1]
      )

  diag = diag.at[0].set(jnp.stack(left_diag))
  diag = diag.at[-1].set(jnp.stack(right_diag))
  vec = vec.at[0].set(jnp.stack(left_vec))
  vec = vec.at[-1].set(jnp.stack(right_vec))

  return diag, upper_off, lower_off, vec
