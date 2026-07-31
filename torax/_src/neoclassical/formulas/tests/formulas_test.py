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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.neoclassical.formulas import formulas
from torax._src.physics import collisions
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name

_N_RHO = 10
_A_TOL = 1e-6
_R_TOL = 1e-6


class FormulasTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torax_config = model_config.ToraxConfig.from_dict({
        'profile_conditions': {
            'Ip': 15e6,
            'current_profile_nu': 3,
            'n_e_nbar_is_fGW': True,
            'normalize_n_e_to_nbar': True,
            'nbar': 0.85,
            'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
        },
        'numerics': {},
        'plasma_composition': {
            'Z_eff': 2.0,
        },
        'geometry': {
            'geometry_type': 'chease',
            'Ip_from_parameters': False,
            'n_rho': _N_RHO,
        },
        'transport': {},
        'solver': {},
        'pedestal': {},
        'sources': {},
    })

    params_provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params, self.geo = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=torax_config.numerics.t_initial,
            runtime_params_provider=params_provider,
            geometry_provider=torax_config.geometry.build_provider,
            is_initialization=True,
        )
    )
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    self.core_profiles = initialization.initial_core_profiles(
        runtime_params,
        self.geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    log_lambda_ei = collisions.calculate_log_lambda_ei(
        self.core_profiles.T_e.face_value(), self.core_profiles.n_e.face_value()  # pyrefly: ignore[bad-argument-type]
    )
    self.nu_e_star = formulas.calculate_nu_e_star(
        q=self.core_profiles.q_face,
        geo=self.geo,
        n_e=self.core_profiles.n_e.face_value(),  # pyrefly: ignore[bad-argument-type]
        T_e=self.core_profiles.T_e.face_value(),  # pyrefly: ignore[bad-argument-type]
        Z_eff=self.core_profiles.Z_eff_face,
        log_lambda_ei=log_lambda_ei,
    )

  def test_calculate_sauter_trapped_fraction_positive_triangularity(self):
    result = formulas.calculate_sauter_trapped_fraction(
        epsilon=np.array(0.1), delta=np.array(0.2)
    )
    expected = 0.4362384616678634
    np.testing.assert_allclose(result, expected)

  def test_calculate_sauter_trapped_fraction_negative_triangularity(self):
    result = formulas.calculate_sauter_trapped_fraction(
        epsilon=np.array(0.1), delta=np.array(-0.2)
    )
    expected = 0.45134158459680895
    np.testing.assert_allclose(result, expected)

  def test_calculate_bounce_averaged_trapped_fraction_uniform_B_is_zero(self):
    """A surface with uniform |B| has no magnetic wells, so f_t = 0."""
    theta = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    B_0 = 2.5
    B = np.full_like(theta, B_0)
    trapped_fraction = formulas.calculate_bounce_averaged_trapped_fraction(
        B=B,
        dl_over_Bp=np.ones_like(theta),
        flux_surf_avg_B2=B_0**2,
    )
    np.testing.assert_allclose(trapped_fraction, 0.0, atol=1e-3)

  @parameterized.parameters(0.001, 0.01)
  def test_calculate_bounce_averaged_trapped_fraction_large_aspect_ratio(
      self, epsilon: float
  ):
    """Compares against the analytic circular large-aspect-ratio limit.

    For a large-aspect-ratio circular flux surface with B = B_0/(1 + eps*cos
    (theta)), the effective trapped fraction approaches 1.46*sqrt(eps) as
    eps -> 0. See e.g. Y.R. Lin-Liu and R.L. Miller, Phys. Plasmas 2, 1666
    (1995), Eq. 8.
    """
    theta = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    B = 1.0 / (1.0 + epsilon * np.cos(theta))
    dl_over_Bp = np.ones_like(theta)
    flux_surf_avg_B2 = np.mean(B**2)
    trapped_fraction = formulas.calculate_bounce_averaged_trapped_fraction(
        B=B,
        dl_over_Bp=dl_over_Bp,
        flux_surf_avg_B2=flux_surf_avg_B2,
    )
    np.testing.assert_allclose(
        trapped_fraction, 1.46 * np.sqrt(epsilon), rtol=0.02
    )

  def test_calculate_bounce_averaged_trapped_fraction_is_physical(self):
    """f_t lies in [0, 1] and grows with the depth of the magnetic well."""
    theta = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    trapped_fractions = []
    for epsilon in [0.05, 0.1, 0.2, 0.3]:
      B = 1.0 / (1.0 + epsilon * np.cos(theta))
      trapped_fraction = formulas.calculate_bounce_averaged_trapped_fraction(
          B=B,
          dl_over_Bp=np.ones_like(theta),
          flux_surf_avg_B2=np.mean(B**2),
      )
      trapped_fractions.append(float(trapped_fraction))
    trapped_fractions = np.array(trapped_fractions)
    with self.subTest('within_physical_bounds'):
      self.assertTrue(np.all(trapped_fractions >= 0.0))
      self.assertTrue(np.all(trapped_fractions <= 1.0))
    with self.subTest('increases_with_epsilon'):
      self.assertTrue(np.all(np.diff(trapped_fractions) > 0.0))

  def test_calculate_poloidal_velocity_values_are_correct(self):
    poloidal_velocity = formulas.calculate_poloidal_velocity(
        T_i=self.core_profiles.T_i,
        n_i=self.core_profiles.n_i.face_value(),
        q=self.core_profiles.q_face,
        Z_eff=self.core_profiles.Z_eff_face,
        Z_i=self.core_profiles.Z_i_face,
        B_tor=np.ones_like(self.geo.rho_face_norm),
        B_total_squared=np.ones_like(self.geo.rho_face_norm),
        geo=self.geo,
    )
    np.testing.assert_allclose(
        _POLOIDAL_VELOCITY_EXPECTED,
        poloidal_velocity.face_value(),
        atol=_A_TOL,
        rtol=_R_TOL,
    )


# Reference values from running test code in a notebook.
# The test thus does not directly test the implementation, but rather
# guards against unexpected modifications.
# If a change is expected to theese reference values, the new values can b
# copied/pasted from the logs of a failing test.
_POLOIDAL_VELOCITY_EXPECTED = np.array([
    -1485.871716,
    -2507.496827,
    -3933.755809,
    -4537.621566,
    -4854.858931,
    -5031.592012,
    -5073.608117,
    -4858.248803,
    -3559.941551,
    3265.428187,
    18579.094079,
])

if __name__ == '__main__':
  absltest.main()
