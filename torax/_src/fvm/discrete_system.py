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
"""Functionality for building discrete linear systems.

This file is expected to be used mostly internally by `fvm` itself.

The functionality here is for constructing a description of one discrete
time step of a PDE in terms of a linear equation. In practice, the
actual expressive power of the resulting Jax expression may still be
nonlinear because the coefficients of this linear equation are Jax
expressions, not just numeric values, so nonlinear solvers like
newton_raphson_solve_block can capture nonlinear dynamics even when
each step is expressed using a matrix multiply.
"""

from typing import TypeAlias

import jax
from jax import numpy as jnp
from torax._src import tridiagonal
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import convection_terms
from torax._src.fvm import diffusion_terms

AuxiliaryOutput: TypeAlias = block_1d_coeffs.AuxiliaryOutput
Block1DCoeffs: TypeAlias = block_1d_coeffs.Block1DCoeffs


def calc_c(
    x: tuple[cell_variable.CellVariable, ...],
    coeffs: Block1DCoeffs,
    convection_dirichlet_mode: str = 'ghost',
    convection_neumann_mode: str = 'ghost',
) -> tuple[tridiagonal.BlockTriDiagonal, jax.Array]:
  """Calculate banded blocks and vector c such that F = C x + c.

  Returns the block-tridiagonal representation of C. The matrix structure comes
  from the 1D FVM stencil: each cell couples to itself and its two neighbors.

  Args:
    x: Tuple containing CellVariables for each channel. This function uses only
      their shape and their boundary conditions, not their values.
    coeffs: Coefficients defining the differential equation.
    convection_dirichlet_mode: See docstring of the `convection_terms` function,
      `dirichlet_mode` argument.
    convection_neumann_mode: See docstring of the `convection_terms` function,
      `neumann_mode` argument.

  Returns:
    A tuple of (c_matrix, c_forcing) where:
      c_matrix: BlockTriDiagonal with sub/main/super-diagonal blocks.
      c_forcing: An array with the terms arising from explicit sources and
        boundary conditions.
  """

  d_face = coeffs.d_face
  v_face = coeffs.v_face
  source_mat_cell = coeffs.source_mat_cell
  source_cell = coeffs.source_cell

  num_cells = x[0].value.shape[0]
  num_channels = len(x)
  for _, x_i in enumerate(x):
    if x_i.value.shape != (num_cells,):
      raise ValueError(
          f'Expected each x channel to have shape ({num_cells},) '
          f'but got {x_i.value.shape}.'
      )

  def stack_channels(values, fallback_like):
    """Stacks per-channel arrays on a trailing channel axis, zeroing Nones.

    Missing channels take the shape of the first channel that is present, so
    face-grid and cell-grid tuples both round-trip correctly; `fallback_like`
    is only consulted when every channel is None.
    """
    template = next((v for v in values if v is not None), fallback_like)
    return jnp.stack(
        [jnp.zeros_like(template) if v is None else v for v in values], axis=-1
    )

  # The channels share a mesh, so diffusion and convection are assembled once
  # for all of them on (face, channel) arrays. The three diagonals are
  # accumulated in the compact (cell, channel) form and expanded into (C, C)
  # blocks only once, at the end.
  d_face_stacked = (
      None if d_face is None else stack_channels(d_face, x[0].value)
  )
  v_face_stacked = (
      None if v_face is None else stack_channels(v_face, x[0].value)
  )

  if d_face_stacked is None:
    diagonal = jnp.zeros((num_cells, num_channels))
    above = jnp.zeros((num_cells - 1, num_channels))
    below = jnp.zeros((num_cells - 1, num_channels))
    c_forcing = jnp.zeros((num_cells, num_channels))
  else:
    diagonal, above, below, c_forcing = (
        diffusion_terms.make_multichannel_diffusion_terms(d_face_stacked, x)
    )

  if v_face_stacked is not None:
    # Resolve diffusion to zeros if it is not specified
    d_for_convection = (
        jnp.zeros_like(v_face_stacked)
        if d_face_stacked is None
        else d_face_stacked
    )
    conv_diagonal, conv_above, conv_below, conv_forcing = (
        convection_terms.make_multichannel_convection_terms(
            v_face_stacked,
            d_for_convection,
            x,
            dirichlet_mode=convection_dirichlet_mode,
            neumann_mode=convection_neumann_mode,
        )
    )
    diagonal += conv_diagonal
    above += conv_above
    below += conv_below
    c_forcing += conv_forcing

  c_matrix = tridiagonal.BlockTriDiagonal.from_channel_diagonals(
      diagonal=diagonal, above=above, below=below
  )

  # Add implicit source terms. These are the only terms that couple channels,
  # so they are the only contribution that is not block-diagonal.
  if source_mat_cell is not None:
    source_block = jnp.stack(
        [stack_channels(row, x[0].value) for row in source_mat_cell], axis=1
    )
    c_matrix = tridiagonal.BlockTriDiagonal(
        lower=c_matrix.lower,
        diagonal=c_matrix.diagonal + source_block,
        upper=c_matrix.upper,
    )

  # Add explicit source terms
  if source_cell is not None:
    c_forcing += stack_channels(source_cell, x[0].value)

  return c_matrix, c_forcing
