# Copyright 2026 DeepMind Technologies Limited
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
"""Declared Jacobian sparsity pattern and probe coloring for the Newton solver.

The Jacobian of the theta-method residual couples (cell, channel) unknowns
through paths that are all known from the config and geometry before any
simulation step is taken:

1. FVM stencil: the residual's block-tridiagonal matvec couples each cell to
   its immediate neighbours, with dense channel-channel blocks.
2. Local coefficient dependence: transport and source coefficients at a face
   or cell depend on the profiles in a small radial halo (face gradients read
   one cell to each side; quantities derived from second derivatives reach
   one further).
3. Smoothing: turbulent transport coefficients are convolved with the
   combined transport model's clipped-Gaussian smoothing matrix, which gives
   the heat and particle equations long-range dependence on all channels
   inside the smoothing zones. The pattern reuses the actual smoothing matrix
   built by the transport model, so it cannot drift from the physics
   implementation.

The declared pattern feeds a column coloring (Curtis-Powell-Reid seed
compression): columns that share no residual row are probed together, so the
full Jacobian is recovered from one batched Jacobian-vector product per color
instead of one forward tangent per unknown.

The pattern is a *contract*: an entry outside it is assumed to be zero for
every state the solver visits. A superset costs extra probes but is always
correct; a missed entry silently corrupts the reconstruction. Use
`verification_error` to check the contract at runtime with a single extra
tangent evaluation.
"""

import jax
import jax.numpy as jnp
import numpy as np

# Channels whose equations contain turbulent (smoothed) transport
# coefficients. The psi equation's coefficients (conductivity, bootstrap
# current) are not smoothed, which keeps its rows local.
_TURBULENT_CHANNELS = frozenset({'T_i', 'T_e', 'n_e'})

# Radial halo (in cells) of the local coefficient dependence: face gradients
# read one cell to each side, and shear-like quantities built from second
# derivatives reach one further.
_LOCAL_HALO_WIDTH = 2

# Cells adjacent to the domain boundaries use one-sided and axis-regularised
# derivative stencils (e.g. on-axis current and q from psi), which reach
# deeper into the domain than the interior halo. Within this many cells of
# either boundary, all-to-all coupling is declared (see `build_pattern`).
_BOUNDARY_STENCIL_WIDTH = 5


def build_pattern(
    num_cells: int,
    evolving_names: tuple[str, ...],
    smoothing_matrix: jax.Array | np.ndarray | None,
    halo_width: int = _LOCAL_HALO_WIDTH,
) -> np.ndarray:
  """Builds the declared Jacobian sparsity pattern from config-level structure.

  Args:
    num_cells: Number of radial cells (n_rho).
    evolving_names: Names of the evolving channels, in solver order. The
      solver vector is channel-major: entry `c * num_cells + i` is channel c
      at cell i.
    smoothing_matrix: The (num_faces, num_faces) smoothing matrix used by the
      combined transport model, or None when no smoothing is configured. Only
      its nonzero pattern is used. Entries outside the clipped Gaussian
      support are exactly zero, so no thresholding is applied.
    halo_width: Radial halo (in cells) of local coefficient dependence.

  Returns:
    Boolean array of shape (size, size) with size = num_cells *
    len(evolving_names), True where the Jacobian may be nonzero. Layout
    matches the solver vector on both axes.
  """
  num_channels = len(evolving_names)
  size = num_cells * num_channels

  cell = np.arange(num_cells)
  offset = np.abs(cell[:, None] - cell[None, :])

  # Path 1 + 2: matvec stencil (width 1) composed with the coefficient halo.
  # All channel pairs couple locally: sources and derived profiles (n_i,
  # charge states, conductivity, exchange terms) mix every channel within the
  # halo.
  local_cells = offset <= 1 + halo_width

  # Boundary stencils: near the axis and the edge, derivative stencils are
  # one-sided and reach deeper than the interior halo (measured example: the
  # on-axis T_e equation depends on psi four cells inward through the
  # axis-regularised current). Declare the boundary corner blocks densely;
  # they are tiny, so the extra probes are negligible.
  corner = _BOUNDARY_STENCIL_WIDTH + halo_width + 1
  near_axis = cell < min(corner, num_cells)
  near_edge = cell >= max(num_cells - corner, 0)
  local_cells = (
      local_cells
      | (near_axis[:, None] & near_axis[None, :])
      | (near_edge[:, None] & near_edge[None, :])
  )

  # Path 3: row cell i reads transport coefficients at faces {i, i+1}; a
  # smoothed coefficient at face f gathers raw coefficients from every face g
  # with S[f, g] != 0; the raw coefficient at face g depends on cells within
  # the halo of its face gradient. Composed as boolean products on the
  # face/cell incidence maps.
  if smoothing_matrix is not None:
    smoothing_pattern = np.asarray(smoothing_matrix) != 0.0
    num_faces = num_cells + 1
    if smoothing_pattern.shape != (num_faces, num_faces):
      raise ValueError(
          f'smoothing_matrix has shape {smoothing_pattern.shape}, expected'
          f' {(num_faces, num_faces)}.'
      )
    # cell_to_face[i, f]: cell i's equation reads face f.
    face = np.arange(num_faces)
    cell_to_face = (face[None, :] == cell[:, None]) | (
        face[None, :] == cell[:, None] + 1
    )
    # face_to_cell[g, j]: the raw coefficient at face g depends on cell j.
    # The face gradient reads cells g-1 and g; the halo widens this reach.
    face_to_cell = np.abs(
        (face[:, None] - 0.5) - cell[None, :]
    ) <= 0.5 + halo_width
    smoothed_cells = (cell_to_face @ smoothing_pattern @ face_to_cell) > 0
  else:
    smoothed_cells = np.zeros((num_cells, num_cells), dtype=bool)

  pattern = np.zeros((size, size), dtype=bool)
  for row_channel, row_name in enumerate(evolving_names):
    rows = slice(row_channel * num_cells, (row_channel + 1) * num_cells)
    cells = local_cells
    if row_name in _TURBULENT_CHANNELS:
      cells = cells | smoothed_cells
    for col_channel in range(num_channels):
      cols = slice(col_channel * num_cells, (col_channel + 1) * num_cells)
      # Transport coefficients depend on every channel (temperatures,
      # density, and q/shear through psi), so the smoothing path applies to
      # all input channels of a turbulent row.
      pattern[rows, cols] = cells
  return pattern


def color_columns(pattern: np.ndarray) -> np.ndarray:
  """Greedy distance-2 coloring of the pattern's columns.

  Two columns receive different colors when any residual row depends on both,
  so all columns of one color can be probed with a single tangent vector and
  unambiguously scattered back into the Jacobian.

  Args:
    pattern: Boolean (size, size) sparsity pattern.

  Returns:
    Integer array of shape (size,) assigning each column a color in
    [0, num_colors).
  """
  incidence = pattern.astype(np.int64)
  conflicts = (incidence.T @ incidence) > 0
  # Color densest columns first: they conflict the most and constrain the
  # coloring, which keeps the greedy color count close to the graph's lower
  # bound.
  order = np.argsort(-incidence.sum(axis=0), kind='stable')
  colors = np.full(pattern.shape[1], -1, dtype=np.int64)
  for column in order:
    neighbor_colors = colors[conflicts[column]]
    used = set(neighbor_colors[neighbor_colors >= 0].tolist())
    color = 0
    while color in used:
      color += 1
    colors[column] = color
  return colors


def build_seeds_and_scatter(
    pattern: np.ndarray, colors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Builds the probe seed matrix and the row-wise scatter map.

  Args:
    pattern: Boolean (size, size) sparsity pattern.
    colors: Column color assignment from `color_columns`.

  Returns:
    A tuple of (seeds, scatter_columns):
      seeds: (num_colors, size) float seed matrix; row k is the probe tangent
        for color k (the indicator vector of that color's columns).
      scatter_columns: (size, num_colors) integer map; entry (row, k) is the
        column index the batched product's (k, row) entry belongs to, or
        `size` when no column of color k appears in that row (used as an
        out-of-range drop index for jnp scatters).
  """
  size = pattern.shape[0]
  num_colors = int(colors.max()) + 1
  seeds = np.zeros((num_colors, size))
  seeds[colors, np.arange(size)] = 1.0

  scatter_columns = np.full((size, num_colors), size, dtype=np.int64)
  rows, cols = np.nonzero(pattern)
  # By construction of the coloring, each (row, color) pair has at most one
  # pattern column, so this assignment never collides.
  scatter_columns[rows, colors[cols]] = cols
  return seeds, scatter_columns


def reconstruct_dense(
    batched_products: jax.Array, scatter_columns: np.ndarray
) -> jax.Array:
  """Scatters batched Jacobian-probe products into a dense Jacobian.

  Args:
    batched_products: (num_colors, size) array with row k equal to J @
      seeds[k], e.g. from one vmapped tangent evaluation.
    scatter_columns: Scatter map from `build_seeds_and_scatter`.

  Returns:
    Dense (size, size) Jacobian with declared-zero entries equal to zero.
  """
  size = scatter_columns.shape[0]
  # Scatter row-wise: entry (row, k) of the products belongs to column
  # scatter_columns[row, k]. Out-of-band products carry the drop index
  # `size` and are discarded by the out-of-bounds scatter mode.
  return (
      jnp.zeros((size, size + 1), dtype=batched_products.dtype)
      .at[jnp.arange(size)[:, None], jnp.asarray(scatter_columns)]
      .set(batched_products.T, mode='drop')[:, :size]
  )


def verification_error(
    jvp_fun,
    reconstructed: jax.Array,
    probe: jax.Array,
) -> jax.Array:
  """Checks the pattern contract with one extra tangent evaluation.

  If the declared pattern covers every true Jacobian entry, the reconstructed
  matrix reproduces any Jacobian-vector product exactly (up to roundoff). A
  significant mismatch means the pattern missed a coupling and the
  reconstruction is silently aliased.

  Args:
    jvp_fun: Function computing the true Jacobian-vector product at the
      current state, e.g. from `jax.linearize`.
    reconstructed: Jacobian from `reconstruct_dense`.
    probe: Verification tangent. Any vector outside the seed set works; a
      random vector is the standard choice.

  Returns:
    Scalar max-abs mismatch between the true and reconstructed products,
    normalised by the max-abs of the true product.
  """
  true_product = jvp_fun(probe)
  scale = jnp.maximum(jnp.max(jnp.abs(true_product)), 1e-300)
  return jnp.max(jnp.abs(reconstructed @ probe - true_product)) / scale
