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
from typing import ClassVar

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax._src import constants
from torax._src import math_utils
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.sources import metis_nbi_source
from torax._src.sources.tests import test_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

_INJECTOR_CONFIG = {
    'P_total': 20e6,
    'beam_energy': 500.0,
    'beam_mass': 2.0,
    'tangency_radius': 5.0,
}


def _build_test_inputs(nbi_config):
  """Builds config, geometry, runtime params and core profiles for tests."""
  config = default_configs.get_default_config_dict()
  config['sources'] = {'nbi': nbi_config}
  torax_config = model_config.ToraxConfig.from_dict(config)
  geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
  runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
      torax_config
  )(t=torax_config.numerics.t_initial)
  source_models = torax_config.sources.build_models()
  neoclassical_models = torax_config.neoclassical.build_models()
  core_profiles = initialization.initial_core_profiles(
      runtime_params=runtime_params,
      geo=geo,
      source_models=source_models,
      neoclassical_models=neoclassical_models,
  )
  return torax_config, geo, runtime_params, source_models, core_profiles


def _get_nbi_value(geo, runtime_params, source_models, core_profiles):
  return source_models.standard_sources['nbi'].get_value(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      calculated_source_profiles=None,
      conductivity=None,
  )


class MetisNBISourceTest(test_lib.SourceTestCase):

  source_name: ClassVar[str] = 'nbi'
  source_config_class: ClassVar[type] = metis_nbi_source.MetisNBISourceConfig
  model_name: ClassVar[str] = 'metis_nbi'

  def test_config_defaults_to_metis_model(self):
    _, _, runtime_params, _, _ = _build_test_inputs(
        {'injectors': [_INJECTOR_CONFIG]}
    )
    nbi_params = runtime_params.sources['nbi']
    self.assertIsInstance(nbi_params, metis_nbi_source.RuntimeParams)
    np.testing.assert_allclose(nbi_params.beam_energy, [500.0])

  def test_source_values_on_the_cell_grid(self):
    _, geo, runtime_params, source_models, core_profiles = _build_test_inputs(
        {'injectors': [_INJECTOR_CONFIG]}
    )
    value = _get_nbi_value(geo, runtime_params, source_models, core_profiles)
    # (ion heat, electron heat, current, particles).
    self.assertLen(value, 4)
    for profile in value:
      self.assertEqual(profile.shape, geo.rho.shape)
      self.assertFalse(jnp.any(jnp.isnan(profile)))


class SuzukiCrossSectionTest(parameterized.TestCase):

  def test_matches_hand_computed_reference(self):
    # Hand-computed from the Suzuki 1998 Table 2 fit (D beam at 200 keV/amu,
    # n_e = 5e19 m^-3, T_e = 2 keV, no impurities): sigma = 1.441e-20 m^2.
    sigma = metis_nbi_source.suzuki_beam_stopping_cross_section(
        beam_energy=400.0,
        beam_mass=2.0,
        n_e=jnp.array([5e19]),
        T_e=jnp.array([2.0]),
        n_impurity=jnp.array([0.0]),
        Z_impurity=jnp.array([6.0]),
    )
    np.testing.assert_allclose(sigma, 1.441e-20, rtol=1e-3)

  def test_decreases_with_beam_energy(self):
    sigmas = [
        metis_nbi_source.suzuki_beam_stopping_cross_section(
            beam_energy=E,
            beam_mass=1.0,
            n_e=jnp.array([5e19]),
            T_e=jnp.array([2.0]),
            n_impurity=jnp.array([0.0]),
            Z_impurity=jnp.array([6.0]),
        )[0]
        for E in [150.0, 300.0, 600.0, 1200.0]
    ]
    np.testing.assert_array_less(np.diff(np.array(sigmas)), 0.0)

  def test_impurities_increase_stopping(self):
    kwargs = dict(
        beam_energy=500.0,
        beam_mass=2.0,
        n_e=jnp.array([5e19]),
        T_e=jnp.array([2.0]),
        Z_impurity=jnp.array([6.0]),
    )
    sigma_clean = metis_nbi_source.suzuki_beam_stopping_cross_section(
        n_impurity=jnp.array([0.0]), **kwargs
    )
    sigma_carbon = metis_nbi_source.suzuki_beam_stopping_cross_section(
        n_impurity=jnp.array([1e18]), **kwargs
    )
    self.assertGreater(sigma_carbon[0], sigma_clean[0])


class WessonIonFractionTest(parameterized.TestCase):

  def test_limits(self):
    # E_b << E_c: all power to ions; E_b >> E_c: all power to electrons.
    frac_slow = metis_nbi_source.wesson_ion_heating_fraction(
        1.0, jnp.array([100.0])
    )
    frac_fast = metis_nbi_source.wesson_ion_heating_fraction(
        1000.0, jnp.array([1.0])
    )
    np.testing.assert_allclose(frac_slow, 1.0, atol=1e-3)
    np.testing.assert_allclose(frac_fast, 0.0, atol=1e-2)

  def test_monotonically_decreasing_in_energy_ratio(self):
    critical_energy = jnp.array([100.0])
    fracs = [
        metis_nbi_source.wesson_ion_heating_fraction(E_b, critical_energy)[0]
        for E_b in [10.0, 50.0, 100.0, 500.0, 1000.0]
    ]
    np.testing.assert_array_less(np.diff(np.array(fracs)), 0.0)


class BeamDepositionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    _, self.geo, self.runtime_params, self.source_models, self.core_profiles = (
        _build_test_inputs({'injectors': [_INJECTOR_CONFIG]})
    )
    self.sigma_face = metis_nbi_source.suzuki_beam_stopping_cross_section(
        beam_energy=500.0,
        beam_mass=2.0,
        n_e=self.core_profiles.n_e.face_value(),
        T_e=self.core_profiles.T_e.face_value(),
        n_impurity=self.core_profiles.n_impurity.face_value(),
        Z_impurity=self.core_profiles.Z_impurity_face,
    )

  def test_deposited_fraction_matches_shine_through(self):
    birth_density, _, shine = metis_nbi_source.calc_beam_deposition(
        self.geo,
        self.core_profiles.n_e.face_value(),
        self.sigma_face,
        tangency_radius=5.0,
    )
    absorbed = math_utils.volume_integration(birth_density, self.geo)
    np.testing.assert_allclose(absorbed, 1.0 - shine, rtol=1e-9)
    self.assertGreaterEqual(float(jnp.min(birth_density)), 0.0)

  def test_shine_through_decreases_with_density(self):
    n_e_face = self.core_profiles.n_e.face_value()
    # A transparent (low-density) plasma must have higher shine-through.
    _, _, shine_low = metis_nbi_source.calc_beam_deposition(
        self.geo, 0.01 * n_e_face, self.sigma_face, tangency_radius=5.0
    )
    _, _, shine_high = metis_nbi_source.calc_beam_deposition(
        self.geo, n_e_face, self.sigma_face, tangency_radius=5.0
    )
    self.assertGreater(float(shine_low), float(shine_high))
    self.assertLessEqual(float(shine_low), 1.0)

  def test_no_deposition_when_beam_misses_plasma(self):
    # Tangency radius outside the outboard LCFS radius.
    tangency_radius = float(self.geo.R_out_face[-1]) + 0.5
    birth_density, _, shine = metis_nbi_source.calc_beam_deposition(
        self.geo,
        self.core_profiles.n_e.face_value(),
        self.sigma_face,
        tangency_radius=tangency_radius,
    )
    np.testing.assert_allclose(shine, 1.0)
    np.testing.assert_allclose(birth_density, 0.0)

  def test_pitch_is_bounded(self):
    _, pitch, _ = metis_nbi_source.calc_beam_deposition(
        self.geo,
        self.core_profiles.n_e.face_value(),
        self.sigma_face,
        tangency_radius=5.0,
    )
    self.assertGreaterEqual(float(jnp.min(pitch)), 0.0)
    self.assertLessEqual(float(jnp.max(pitch)), 1.0)

  def test_perpendicular_injection_has_zero_pitch(self):
    _, pitch, _ = metis_nbi_source.calc_beam_deposition(
        self.geo,
        self.core_profiles.n_e.face_value(),
        self.sigma_face,
        tangency_radius=0.0,
    )
    np.testing.assert_allclose(pitch, 0.0)


class NBISourceValuesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    _, self.geo, self.runtime_params, self.source_models, self.core_profiles = (
        _build_test_inputs({'injectors': [_INJECTOR_CONFIG]})
    )
    self.p_ion, self.p_el, self.j_cd, self.s_particle = _get_nbi_value(
        self.geo, self.runtime_params, self.source_models, self.core_profiles
    )

  def test_heat_source_conserves_absorbed_power(self):
    sigma_face = metis_nbi_source.suzuki_beam_stopping_cross_section(
        beam_energy=500.0,
        beam_mass=2.0,
        n_e=self.core_profiles.n_e.face_value(),
        T_e=self.core_profiles.T_e.face_value(),
        n_impurity=self.core_profiles.n_impurity.face_value(),
        Z_impurity=self.core_profiles.Z_impurity_face,
    )
    _, _, shine = metis_nbi_source.calc_beam_deposition(
        self.geo,
        self.core_profiles.n_e.face_value(),
        sigma_face,
        tangency_radius=5.0,
    )
    total_absorbed = math_utils.volume_integration(
        self.p_ion + self.p_el, self.geo
    )
    np.testing.assert_allclose(total_absorbed, 20e6 * (1.0 - shine), rtol=1e-9)
    self.assertGreaterEqual(float(jnp.min(self.p_ion)), 0.0)
    self.assertGreaterEqual(float(jnp.min(self.p_el)), 0.0)

  def test_particle_source_consistent_with_heating(self):
    beam_energy_J = 500.0 * constants.CONSTANTS.keV_to_J
    np.testing.assert_allclose(
        self.s_particle, (self.p_ion + self.p_el) / beam_energy_J, rtol=1e-9
    )

  def test_current_drive_is_co_current(self):
    self.assertGreater(
        float(math_utils.area_integration(self.j_cd, self.geo)), 0.0
    )
    self.assertGreaterEqual(float(jnp.min(self.j_cd)), 0.0)

  def test_two_identical_injectors_double_all_profiles(self):
    _, geo, runtime_params, source_models, core_profiles = _build_test_inputs(
        {'injectors': [_INJECTOR_CONFIG, _INJECTOR_CONFIG]}
    )
    values_two = _get_nbi_value(
        geo, runtime_params, source_models, core_profiles
    )
    for profile_one, profile_two in zip(
        (self.p_ion, self.p_el, self.j_cd, self.s_particle), values_two
    ):
      np.testing.assert_allclose(profile_two, 2.0 * profile_one, rtol=1e-9)

  def test_balanced_injection_cancels_current_but_not_heating(self):
    counter_injector = dict(_INJECTOR_CONFIG, counter_injection=True)
    _, geo, runtime_params, source_models, core_profiles = _build_test_inputs(
        {'injectors': [_INJECTOR_CONFIG, counter_injector]}
    )
    p_ion, p_el, j_cd, _ = _get_nbi_value(
        geo, runtime_params, source_models, core_profiles
    )
    np.testing.assert_allclose(j_cd, 0.0, atol=1e-10)
    np.testing.assert_allclose(p_ion + p_el, 2.0 * (self.p_ion + self.p_el))

  def test_counter_injection_flips_current_sign(self):
    counter_injector = dict(_INJECTOR_CONFIG, counter_injection=True)
    _, geo, runtime_params, source_models, core_profiles = _build_test_inputs(
        {'injectors': [counter_injector]}
    )
    _, _, j_counter, _ = _get_nbi_value(
        geo, runtime_params, source_models, core_profiles
    )
    np.testing.assert_allclose(j_counter, -self.j_cd, rtol=1e-9)

  def test_perpendicular_injection_drives_no_current(self):
    perpendicular_injector = dict(_INJECTOR_CONFIG, tangency_radius=0.0)
    _, geo, runtime_params, source_models, core_profiles = _build_test_inputs(
        {'injectors': [perpendicular_injector]}
    )
    _, _, j_cd, _ = _get_nbi_value(
        geo, runtime_params, source_models, core_profiles
    )
    np.testing.assert_allclose(j_cd, 0.0, atol=1e-12)

  def test_time_varying_injector_parameters(self):
    time_varying_injector = dict(
        _INJECTOR_CONFIG, P_total={0.0: 0.0, 10.0: 20e6}
    )
    torax_config, _, _, _, _ = _build_test_inputs(
        {'injectors': [time_varying_injector]}
    )
    nbi_config = torax_config.sources.nbi
    np.testing.assert_allclose(
        nbi_config.build_runtime_params(t=5.0).P_total, [10e6]
    )


if __name__ == '__main__':
  absltest.main()
