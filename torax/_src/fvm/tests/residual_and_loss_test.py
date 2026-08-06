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

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import tridiagonal
from torax._src.config import build_runtime_params
from torax._src.core_profiles import convertors
from torax._src.core_profiles import initialization
from torax._src.fvm import calc_coeffs
from torax._src.fvm import fvm_conversions
from torax._src.fvm import residual_and_loss
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.sources import source_profile_builders
from torax._src.test_utils import default_sources
from torax._src.torax_pydantic import model_config


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


def _build_test_case(num_cells: int = 10):
  """Builds a small multi-channel case with state-dependent physics."""
  torax_config = model_config.ToraxConfig.from_dict(
      dict(
          numerics=dict(
              evolve_ion_heat=True,
              evolve_electron_heat=True,
              evolve_density=True,
              evolve_current=True,
          ),
          plasma_composition=dict(),
          profile_conditions=dict(),
          geometry=dict(geometry_type='circular', n_rho=num_cells),
          pedestal=dict(),
          sources=default_sources.get_default_source_config(),
          solver=dict(use_predictor_corrector=False, theta_implicit=1.0),
          transport=dict(
              model_name='combined',
              transport_models=[
                  dict(model_name='constant', chi_i=1.0),
              ],
              chi_min=0,
          ),
          time_step_calculator=dict(),
      )
  )
  models = torax_config.build_models()
  runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
      torax_config
  )(t=torax_config.numerics.t_initial)
  geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
  core_profiles = initialization.initial_core_profiles(
      runtime_params,
      geo,
      source_models=models.source_models,
      neoclassical_models=models.neoclassical_models,
  )
  evolving_names = ('T_i', 'T_e', 'n_e', 'psi')
  explicit_source_profiles = source_profile_builders.build_source_profiles(
      source_models=models.source_models,
      neoclassical_models=models.neoclassical_models,
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      explicit=True,
  )
  pedestal_transition_state = (
      pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode()
  )
  coeffs_old = calc_coeffs.calc_coeffs(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      models=models,
      explicit_source_profiles=explicit_source_profiles,
      evolving_names=evolving_names,
      use_pereverzev=False,
      pedestal_transition_state=pedestal_transition_state,
  )
  # The solver works in scaled units, so go through the same convertor the
  # solver uses rather than reading the profiles directly.
  x_old = convertors.core_profiles_to_solver_x_tuple(
      core_profiles, evolving_names
  )
  kwargs = dict(
      dt=jnp.array(0.05),
      runtime_params_t_plus_dt=runtime_params,
      geo_t_plus_dt=geo,
      x_old=x_old,
      core_profiles_t=core_profiles,
      core_profiles_t_plus_dt=core_profiles,
      explicit_source_profiles=explicit_source_profiles,
      models=models,
      coeffs_old=coeffs_old,
      evolving_names=evolving_names,
      pedestal_transition_state=pedestal_transition_state,
  )
  x_vec = fvm_conversions.cell_variable_tuple_to_vec(x_old)
  return kwargs, x_vec, len(evolving_names)


def _make_preconditioner(x_vec, num_channels, **kwargs):
  """Returns a callable applying lhs^-1 in the flat solver layout."""
  _, lhs = residual_and_loss.theta_method_block_residual_with_operator(
      x_new_guess_vec=x_vec, **kwargs
  )
  block = lhs

  def apply_inverse(v):
    solution = block.solve(
        residual_and_loss.residual_vec_to_cell_channel_array(v, num_channels),
        solver_type=tridiagonal.SolverType.THOMAS,
    )
    return residual_and_loss.cell_channel_array_to_residual_vec(solution)

  return apply_inverse


class ResidualWithOperatorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update('jax_enable_x64', True)

  def test_residual_matches_plain_residual(self):
    """Returning the operator must not perturb the residual itself."""
    kwargs, x_vec, _ = _build_test_case()
    plain = residual_and_loss.theta_method_block_residual(
        x_new_guess_vec=x_vec, **kwargs
    )
    with_operator, _ = (
        residual_and_loss.theta_method_block_residual_with_operator(
            x_new_guess_vec=x_vec, **kwargs
        )
    )
    np.testing.assert_array_equal(plain, with_operator)

  def test_preconditioner_round_trip(self):
    """M(M^-1 v) == v in the flat solver layout."""
    kwargs, x_vec, num_channels = _build_test_case()
    _, lhs = residual_and_loss.theta_method_block_residual_with_operator(
        x_new_guess_vec=x_vec, **kwargs
    )
    apply_inverse = _make_preconditioner(x_vec, num_channels, **kwargs)

    def apply_forward(v):
      product = lhs.matvec(
          residual_and_loss.residual_vec_to_cell_channel_array(
              v, num_channels
          )
      )
      return residual_and_loss.cell_channel_array_to_residual_vec(product)

    rng = np.random.default_rng(4)
    v = jnp.asarray(rng.normal(size=x_vec.shape))
    np.testing.assert_allclose(apply_forward(apply_inverse(v)), v, atol=1e-8)
    np.testing.assert_allclose(apply_inverse(apply_forward(v)), v, atol=1e-8)

  def test_operator_clusters_the_jacobian_spectrum(self):
    """M^-1 J must have its spectrum tightly clustered around 1.

    This is the property a Jacobian-free Krylov solver relies on: GMRES
    converges in a handful of iterations exactly when the preconditioned
    spectrum is clustered. Note that M is *not* the Jacobian -- the sources
    depend on the state, so J - M is a genuinely large term (mainly d(Qei)/dn_e
    in the temperature rows) -- yet it is a small enough perturbation in the
    right sense for the clustering to hold.

    This is also a sharp check on the vector layout: `test_wrong_layout_does_
    not_cluster` below shows the clustering is destroyed if the flat vector is
    unpacked with the wrong convention.
    """
    kwargs, x_vec, num_channels = _build_test_case()
    residual_fun = functools.partial(
        residual_and_loss.theta_method_block_residual, **kwargs
    )
    jacobian = jax.jacfwd(residual_fun)(x_vec)
    apply_inverse = _make_preconditioner(x_vec, num_channels, **kwargs)

    preconditioned = jax.vmap(apply_inverse, in_axes=1, out_axes=1)(jacobian)
    eigenvalues = np.linalg.eigvals(np.asarray(preconditioned))
    self.assertLess(float(np.abs(eigenvalues - 1.0).max()), 0.1)

    # The unpreconditioned Jacobian is nothing like as well clustered.
    raw_eigenvalues = np.linalg.eigvals(np.asarray(jacobian))
    self.assertGreater(float(np.abs(raw_eigenvalues - 1.0).max()), 1.0)

  def test_wrong_layout_does_not_cluster(self):
    """Guards the layout convention itself.

    If the flat solver vector were unpacked as (num_cells, num_channels)
    directly instead of transposing, the preconditioner would apply the wrong
    matrix rows to the wrong unknowns. The test above would then be vacuous, so
    check here that the wrong convention really is detectably wrong.
    """
    kwargs, x_vec, num_channels = _build_test_case()
    residual_fun = functools.partial(
        residual_and_loss.theta_method_block_residual, **kwargs
    )
    jacobian = jax.jacfwd(residual_fun)(x_vec)
    _, lhs = residual_and_loss.theta_method_block_residual_with_operator(
        x_new_guess_vec=x_vec, **kwargs
    )

    def apply_inverse_wrong(v):
      solution = lhs.solve(
          v.reshape(-1, num_channels),
          solver_type=tridiagonal.SolverType.THOMAS,
      )
      return solution.reshape(-1)

    preconditioned = jax.vmap(apply_inverse_wrong, in_axes=1, out_axes=1)(
        jacobian
    )
    eigenvalues = np.linalg.eigvals(np.asarray(preconditioned))
    self.assertGreater(float(np.abs(eigenvalues - 1.0).max()), 0.5)


if __name__ == '__main__':
  absltest.main()
