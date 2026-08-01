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

"""The `make_convection_terms` function.

Builds the convection terms of the discrete matrix equation.
"""

import chex
import jax
from jax import numpy as jnp
import jaxtyping as jt
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import tridiagonal
from torax._src.fvm import cell_variable


# TODO(b/469726859): Once non-uniform grid is supported add in testing.
def make_convection_terms(
    v_face: jax.Array,
    d_face: jax.Array,
    var: cell_variable.CellVariable,
    dirichlet_mode: str = 'ghost',
    neumann_mode: str = 'ghost',
) -> tuple[tridiagonal.TriDiagonal, jax.Array]:
  """Makes the terms of the matrix equation derived from the convection term.

  The convection term of the differential equation is of the form
  - (partial / partial r) v u

  Args:
    v_face: Convection coefficient on faces.
    d_face: Diffusion coefficient on faces. The relative strength of convection
      to diffusion is used to weight the contribution of neighboring cells when
      calculating face values of u.
    var: CellVariable to define mesh and boundary conditions.
    dirichlet_mode: The strategy to use to handle Dirichlet boundary conditions.
      The default is 'ghost', which has superior stability. 'ghost' -> Boundary
      face values are inferred by constructing a ghost cell then alpha weighting
      cells 'direct' -> Boundary face values are read directly from constraints
      'semi-implicit' -> Matches FiPy. Boundary face values are alpha weighted
      with the constraint value specifying the value of the "other" cell:
      x_{boundary_face} = alpha x_{last_cell} + (1 - alpha) BC
    neumann_mode: Which strategy to use to handle Neumann boundary conditions.
      The default is `ghost`, which has superior stability. 'ghost' -> Boundary
      face values are inferred by constructing a ghost cell then alpha weighting
      cells. 'semi-implicit' -> Matches FiPy. Boundary face values are alpha
      weighted, with the (1 - alpha) weight applied to the external face value
      rather than to a ghost cell.

  Returns:
    mat: Tridiagonal matrix of coefficients on u
    c: Vector of terms not dependent on u
  """
  diagonal, above, below, vec = make_multichannel_convection_terms(
      v_face[:, jnp.newaxis],
      d_face[:, jnp.newaxis],
      (var,),
      dirichlet_mode=dirichlet_mode,
      neumann_mode=neumann_mode,
  )
  return (
      tridiagonal.TriDiagonal(
          diagonal=diagonal[:, 0], above=above[:, 0], below=below[:, 0]
      ),
      vec[:, 0],
  )


def make_multichannel_convection_terms(
    v_face: jt.Float[array_typing.Array, 'face channel'],
    d_face: jt.Float[array_typing.Array, 'face channel'],
    variables: tuple[cell_variable.CellVariable, ...],
    dirichlet_mode: str = 'ghost',
    neumann_mode: str = 'ghost',
) -> tuple[
    jt.Float[array_typing.Array, 'cell channel'],
    jt.Float[array_typing.Array, 'cell-1 channel'],
    jt.Float[array_typing.Array, 'cell-1 channel'],
    jt.Float[array_typing.Array, 'cell channel'],
]:
  """Builds the convection terms for every channel in one batched pass.

  As with the diffusion terms, channels are independent, so this returns three
  scalar diagonals per channel. All channels are assumed to share a mesh; the
  Péclet weighting — by far the heaviest part of the assembly — is therefore
  evaluated once for all channels, and only the boundary rows are built
  channel by channel.

  Args:
    v_face: Convection coefficients on faces, shape (num_faces, num_channels).
    d_face: Diffusion coefficients on faces, same shape. See
      `make_convection_terms` for how these weight neighbouring cells.
    variables: One CellVariable per channel, supplying the boundary conditions.
      All must share a mesh.
    dirichlet_mode: See `make_convection_terms`.
    neumann_mode: See `make_convection_terms`.

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

  # Alpha weighting calculated using power law scheme described in
  # https://www.ctcms.nist.gov/fipy/documentation/numerical/scheme.html

  # Avoid divide by zero
  eps = 1e-20
  is_neg = d_face < 0.0
  nonzero_sign = jnp.ones_like(is_neg) - 2 * is_neg
  d_face = nonzero_sign * jnp.maximum(eps, jnp.abs(d_face))

  # FiPy uses half mesh width at the boundaries
  half = jnp.array([0.5], dtype=jax_utils.get_dtype())
  ones = jnp.ones(v_face.shape[0] - 2, dtype=jax_utils.get_dtype())
  scale = jnp.concatenate((half, ones, half))[:, jnp.newaxis]

  cell_spacings = jnp.concat(
      [var.cell_widths[:1], var.cell_spacings, var.cell_widths[-1:]]
  )
  ratio = scale * cell_spacings[:, jnp.newaxis] * v_face / d_face

  # left_peclet[i] gives the Péclet number of cell i's left face
  left_peclet = -ratio[:-1]
  right_peclet = ratio[1:]

  def peclet_to_alpha(p):
    eps = 1e-3
    p = jnp.where(jnp.abs(p) < eps, eps, p)

    alpha_pg10 = (p - 1) / p
    alpha_p0to10 = ((p - 1) + (1 - p / 10) ** 5) / p
    # FiPy doc has a typo on the next line, where we use a + the doc has a
    # -, which is clearly a mistake since it makes the function
    # discontinuous and negative
    alpha_pneg10to0 = ((1 + p / 10) ** 5 - 1) / p
    alpha_plneg10 = -1 / p

    alpha = 0.5 * jnp.ones_like(p)
    alpha = jnp.where(p > 10.0, alpha_pg10, alpha)
    alpha = jnp.where(jnp.logical_and(10.0 >= p, p > eps), alpha_p0to10, alpha)
    alpha = jnp.where(
        jnp.logical_and(-eps > p, p >= -10), alpha_pneg10to0, alpha
    )
    alpha = jnp.where(p < -10.0, alpha_plneg10, alpha)

    return alpha

  left_alpha = peclet_to_alpha(left_peclet)
  right_alpha = peclet_to_alpha(right_peclet)

  left_v = v_face[:-1]
  right_v = v_face[1:]

  cell_widths = var.cell_widths[:, jnp.newaxis]
  diag = (left_alpha * left_v - right_alpha * right_v) / cell_widths
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
    v_i = v_face[:, channel]
    left_alpha_i = left_alpha[:, channel]
    right_alpha_i = right_alpha[:, channel]
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
      # Dirichlet condition at leftmost face
      match dirichlet_mode:
        case 'ghost':
          diag_left_face = (
              v_i[0] * (2.0 * left_alpha_i[0] - 1.0)
              - v_i[1] * right_alpha_i[0]
          ) / cell_spacings[0]
          vec_left_face = (
              2.0
              * v_i[0]
              * (1.0 - left_alpha_i[0])
              * var_i.left_face_constraint
          ) / cell_spacings[0]
        case 'direct':
          vec_left_face = (
              v_i[0] * var_i.left_face_constraint / cell_spacings[0]
          )
          diag_left_face = -v_i[1] * right_alpha_i[0] / cell_spacings[0]
        case 'semi-implicit':
          vec_left_face = (
              v_i[0] * (1.0 - left_alpha_i[0]) * var_i.left_face_constraint
          ) / cell_spacings[0]
          diag_left_face = diag[0, channel]
        case _:
          raise ValueError(dirichlet_mode)
    else:
      # Gradient boundary condition at leftmost face
      diag_left_face = (
          v_i[0] - right_alpha_i[0] * v_i[1]
      ) / cell_spacings[0]
      vec_left_face = (
          -v_i[0] * (1.0 - left_alpha_i[0]) * var_i.left_face_grad_constraint
      )
      match neumann_mode:
        case 'ghost':
          pass  # no adjustment needed
        case 'semi-implicit':
          vec_left_face /= 2.0
        case _:
          raise ValueError(neumann_mode)

    if var_i.right_face_constraint is not None:
      # Dirichlet condition at rightmost face
      match dirichlet_mode:
        case 'ghost':
          diag_right_face = (
              v_i[-2] * left_alpha_i[-1]
              + v_i[-1] * (1.0 - 2.0 * right_alpha_i[-1])
          ) / cell_spacings[-1]
          vec_right_face = (
              -2.0
              * v_i[-1]
              * (1.0 - right_alpha_i[-1])
              * var_i.right_face_constraint
          ) / cell_spacings[-1]
        case 'direct':
          diag_right_face = v_i[-2] * left_alpha_i[-1] / cell_spacings[-1]
          vec_right_face = (
              -v_i[-1] * var_i.right_face_constraint / cell_spacings[-1]
          )
        case 'semi-implicit':
          diag_right_face = diag[-1, channel]
          vec_right_face = (
              -(
                  v_i[-1]
                  * (1.0 - right_alpha_i[-1])
                  * var_i.right_face_constraint
              )
              / cell_spacings[-1]
          )
        case _:
          raise ValueError(dirichlet_mode)
    else:
      # Gradient boundary condition at rightmost face
      diag_right_face = (
          -(v_i[-1] - v_i[-2] * left_alpha_i[-1]) / cell_spacings[-1]
      )
      vec_right_face = (
          -v_i[-1]
          * (1.0 - right_alpha_i[-1])
          * var_i.right_face_grad_constraint
      )
      match neumann_mode:
        case 'ghost':
          pass  # no adjustment needed
        case 'semi-implicit':
          vec_right_face /= 2.0
        case _:
          raise ValueError(neumann_mode)

    left_diag.append(diag_left_face)
    left_vec.append(vec_left_face)
    right_diag.append(diag_right_face)
    right_vec.append(vec_right_face)

  diag = diag.at[0].set(jnp.stack(left_diag))
  diag = diag.at[-1].set(jnp.stack(right_diag))
  vec = vec.at[0].set(jnp.stack(left_vec))
  vec = vec.at[-1].set(jnp.stack(right_vec))

  above = -(1.0 - right_alpha) * right_v / cell_widths
  below = (1.0 - left_alpha) * left_v / cell_widths

  return diag, above[:-1], below[1:], vec
