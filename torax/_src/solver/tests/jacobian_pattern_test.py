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
"""Validates the declared Jacobian pattern against the true Jacobian."""

import functools

from absl.testing import absltest
import jax
from jax import numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import convertors
from torax._src.core_profiles import initialization
from torax._src.fvm import calc_coeffs
from torax._src.fvm import fvm_conversions
from torax._src.fvm import residual_and_loss
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.solver import jacobian_pattern
from torax._src.sources import source_profile_builders
from torax._src.test_utils import default_sources
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import combined

# Entries below this fraction of their row's largest entry are treated as
# numerical noise when extracting the measured pattern.
_MEASURED_PATTERN_RTOL = 1e-12


def _build_test_case(num_cells: int = 25):
  """A multi-channel case with state-dependent, smoothed turbulent transport."""
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
                  dict(model_name='qlknn'),
              ],
              chi_min=0.05,
              smoothing_zones=[
                  dict(rho_min=0.2, rho_max=0.8, smoothing_width=0.1),
              ],
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
  x_old = convertors.core_profiles_to_solver_x_tuple(
      core_profiles, evolving_names
  )
  residual_fun = functools.partial(
      residual_and_loss.theta_method_block_residual,
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
  smoothing_matrix = combined._build_smoothing_matrix(  # pylint: disable=protected-access
      runtime_params.transport,
      runtime_params,
      geo,
      pedestal_transition_state.pedestal_model_output,
  )
  x_vec = fvm_conversions.cell_variable_tuple_to_vec(x_old)
  return residual_fun, x_vec, evolving_names, num_cells, smoothing_matrix


class JacobianPatternTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    jax.config.update('jax_enable_x64', True)
    (
        cls.residual_fun,
        cls.x_vec,
        cls.evolving_names,
        cls.num_cells,
        cls.smoothing_matrix,
    ) = _build_test_case()
    cls.jacobian = np.asarray(jax.jacfwd(cls.residual_fun)(cls.x_vec))
    cls.declared = jacobian_pattern.build_pattern(
        cls.num_cells, cls.evolving_names, cls.smoothing_matrix
    )

  def _measured_pattern(self):
    abs_jacobian = np.abs(self.jacobian)
    row_max = np.maximum(abs_jacobian.max(axis=1, keepdims=True), 1e-300)
    return abs_jacobian > _MEASURED_PATTERN_RTOL * row_max

  def test_declared_pattern_covers_measured_pattern(self):
    """Every true Jacobian entry must lie inside the declared pattern."""
    missed = self._measured_pattern() & ~self.declared
    self.assertEqual(
        int(missed.sum()),
        0,
        msg=f'{missed.sum()} measured entries outside the declared pattern',
    )

  def test_declared_pattern_is_meaningfully_sparse(self):
    """Guards against the pattern degenerating to (nearly) dense."""
    self.assertLess(self.declared.mean(), 0.5)

  def test_coloring_is_valid_and_compresses(self):
    colors = jacobian_pattern.color_columns(self.declared)
    num_colors = int(colors.max()) + 1
    self.assertLess(num_colors, self.declared.shape[0])
    # No two same-colored columns may share a row.
    incidence = self.declared.astype(np.int64)
    conflicts = (incidence.T @ incidence) > 0
    np.fill_diagonal(conflicts, False)
    cols_a, cols_b = np.nonzero(conflicts)
    self.assertTrue(np.all(colors[cols_a] != colors[cols_b]))

  def test_colored_probing_reconstructs_exact_jacobian(self):
    colors = jacobian_pattern.color_columns(self.declared)
    seeds, scatter_columns = jacobian_pattern.build_seeds_and_scatter(
        self.declared, colors
    )
    _, jvp_fun = jax.linearize(self.residual_fun, self.x_vec)
    products = jax.vmap(jvp_fun)(jnp.asarray(seeds))
    reconstructed = np.asarray(
        jacobian_pattern.reconstruct_dense(products, scatter_columns)
    )
    scale = np.abs(self.jacobian).max()
    np.testing.assert_allclose(
        reconstructed, self.jacobian, atol=1e-9 * scale, rtol=0.0
    )
    # The reconstructed matrix must give the exact Newton direction.
    residual = np.asarray(self.residual_fun(self.x_vec))
    exact = np.linalg.solve(self.jacobian, -residual)
    probed = np.linalg.solve(reconstructed, -residual)
    np.testing.assert_allclose(probed, exact, rtol=1e-8)

  def test_verification_probe_passes_on_declared_pattern(self):
    colors = jacobian_pattern.color_columns(self.declared)
    seeds, scatter_columns = jacobian_pattern.build_seeds_and_scatter(
        self.declared, colors
    )
    _, jvp_fun = jax.linearize(self.residual_fun, self.x_vec)
    products = jax.vmap(jvp_fun)(jnp.asarray(seeds))
    reconstructed = jacobian_pattern.reconstruct_dense(
        products, scatter_columns
    )
    probe = jax.random.normal(jax.random.PRNGKey(0), self.x_vec.shape)
    error = jacobian_pattern.verification_error(
        jvp_fun, reconstructed, probe
    )
    self.assertLess(float(error), 1e-10)

  def test_boundary_corner_blocks_are_declared(self):
    """One-sided boundary stencils reach deeper than the interior halo.

    Regression check for a measured coupling of the on-axis T_e equation to
    psi four cells inward (through the axis-regularised current), which lies
    outside the interior band.
    """
    pattern = jacobian_pattern.build_pattern(
        self.num_cells, self.evolving_names, smoothing_matrix=None
    )
    t_e = self.evolving_names.index('T_e')
    psi = self.evolving_names.index('psi')
    self.assertTrue(pattern[t_e * self.num_cells, psi * self.num_cells + 4])
    self.assertTrue(
        pattern[
            (t_e + 1) * self.num_cells - 1, (psi + 1) * self.num_cells - 5
        ]
    )

  def test_verification_probe_catches_missing_smoothing_path(self):
    """A pattern that omits the smoothing coupling must fail verification."""
    banded_only = jacobian_pattern.build_pattern(
        self.num_cells, self.evolving_names, smoothing_matrix=None
    )
    colors = jacobian_pattern.color_columns(banded_only)
    seeds, scatter_columns = jacobian_pattern.build_seeds_and_scatter(
        banded_only, colors
    )
    _, jvp_fun = jax.linearize(self.residual_fun, self.x_vec)
    products = jax.vmap(jvp_fun)(jnp.asarray(seeds))
    reconstructed = jacobian_pattern.reconstruct_dense(
        products, scatter_columns
    )
    probe = jax.random.normal(jax.random.PRNGKey(0), self.x_vec.shape)
    error = jacobian_pattern.verification_error(
        jvp_fun, reconstructed, probe
    )
    self.assertGreater(float(error), 1e-4)


if __name__ == '__main__':
  absltest.main()
