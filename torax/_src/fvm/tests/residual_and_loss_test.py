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
"""Tests for `residual_and_loss`."""

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax._src.fvm import residual_and_loss


class VectorLayoutTest(parameterized.TestCase):
  """The solver vector and the tridiagonal operators use different layouts."""

  @parameterized.parameters([(1, 5), (2, 5), (4, 9)])
  def test_round_trip(self, num_channels, num_cells):
    rng = np.random.default_rng(0)
    vec = jnp.asarray(rng.normal(size=num_channels * num_cells))

    array = residual_and_loss.residual_vec_to_cell_channel_array(
        vec, num_channels
    )
    self.assertEqual(array.shape, (num_cells, num_channels))
    np.testing.assert_array_equal(
        residual_and_loss.cell_channel_array_to_residual_vec(array), vec
    )

  @parameterized.parameters([(1, 5), (2, 5), (4, 9)])
  def test_round_trip_from_array(self, num_channels, num_cells):
    rng = np.random.default_rng(1)
    array = jnp.asarray(rng.normal(size=(num_cells, num_channels)))
    vec = residual_and_loss.cell_channel_array_to_residual_vec(array)
    np.testing.assert_array_equal(
        residual_and_loss.residual_vec_to_cell_channel_array(
            vec, num_channels
        ),
        array,
    )

  def test_index_convention_matches_solver_vector(self):
    """The solver vector is channel-major: channel c occupies block c."""
    num_cells, num_channels = 7, 3
    rng = np.random.default_rng(2)
    array = jnp.asarray(rng.normal(size=(num_cells, num_channels)))
    vec = residual_and_loss.cell_channel_array_to_residual_vec(array)
    for c in range(num_channels):
      np.testing.assert_array_equal(
          vec[c * num_cells:(c + 1) * num_cells], array[:, c]
      )

  def test_layout_matches_cell_variable_tuple_to_vec(self):
    """The layout must agree with how the solver builds its initial guess."""
    num_cells, num_channels = 6, 2
    rng = np.random.default_rng(3)
    array = jnp.asarray(rng.normal(size=(num_cells, num_channels)))
    # cell_variable_tuple_to_vec concatenates channels, and
    # cell_variable_tuple_to_array(axis=1) stacks them into (cells, channels).
    concatenated = jnp.concatenate([array[:, c] for c in range(num_channels)])
    np.testing.assert_array_equal(
        residual_and_loss.cell_channel_array_to_residual_vec(array),
        concatenated,
    )


if __name__ == '__main__':
  absltest.main()
