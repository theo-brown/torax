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
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import constants
from torax._src import math_utils
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.sources import metis_nbi_source
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources.tests import test_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# Register the METIS NBI models once for all tests in this module. The
# registration is additive, so it does not affect other test modules.
metis_nbi_source.register_metis_nbi_sources()

_NBI_SOURCE_CONFIG = {
    'model_name': 'metis_nbi',
    'P_total': 20e6,
    'beam_energy': 500.0,
    'beam_mass': 2.0,
    'tangency_radius': 5.0,
}


def _build_test_inputs(sources_config):
  """Builds config, geometry, runtime params and core profiles for tests."""
  config = default_configs.get_default_config_dict()
  config['sources'] = sources_config
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


class MetisNBIHeatSourceTest(test_lib.MultipleProfileSourceTestCase):

  source_name: ClassVar[str] = 'generic_heat'
  source_config_class: ClassVar[type] = (
      metis_nbi_source.MetisNBIHeatSourceConfig
  )
  model_name: ClassVar[str] = 'metis_nbi'


class MetisNBICurrentSourceTest(test_lib.SingleProfileSourceTestCase):

  source_name: ClassVar[str] = 'generic_current'
  source_config_class: ClassVar[type] = (
      metis_nbi_source.MetisNBICurrentSourceConfig
  )
  model_name: ClassVar[str] = 'metis_nbi'


class MetisNBIParticleSourceTest(test_lib.SingleProfileSourceTestCase):

  source_name: ClassVar[str] = 'generic_particle'
  source_config_class: ClassVar[type] = (
      metis_nbi_source.MetisNBIParticleSourceConfig
  )
  model_name: ClassVar[str] = 'metis_nbi'


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
        _build_test_inputs({'generic_heat': dict(_NBI_SOURCE_CONFIG)})
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


class MetisNBISourceValuesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    sources_config = {
        'generic_heat': dict(_NBI_SOURCE_CONFIG),
        'generic_current': dict(_NBI_SOURCE_CONFIG),
        'generic_particle': dict(_NBI_SOURCE_CONFIG),
    }
    _, self.geo, self.runtime_params, self.source_models, self.core_profiles = (
        _build_test_inputs(sources_config)
    )

  def _get_value(self, source_name):
    return self.source_models.standard_sources[source_name].get_value(
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )

  def test_heat_source_conserves_absorbed_power(self):
    ion_heat, el_heat = self._get_value('generic_heat')
    source_params = self.runtime_params.sources['generic_heat']
    sigma_face = metis_nbi_source.suzuki_beam_stopping_cross_section(
        beam_energy=source_params.beam_energy,
        beam_mass=source_params.beam_mass,
        n_e=self.core_profiles.n_e.face_value(),
        T_e=self.core_profiles.T_e.face_value(),
        n_impurity=self.core_profiles.n_impurity.face_value(),
        Z_impurity=self.core_profiles.Z_impurity_face,
    )
    _, _, shine = metis_nbi_source.calc_beam_deposition(
        self.geo,
        self.core_profiles.n_e.face_value(),
        sigma_face,
        source_params.tangency_radius,
    )
    total_absorbed = math_utils.volume_integration(ion_heat + el_heat, self.geo)
    np.testing.assert_allclose(
        total_absorbed, source_params.P_total * (1.0 - shine), rtol=1e-9
    )
    self.assertGreaterEqual(float(jnp.min(ion_heat)), 0.0)
    self.assertGreaterEqual(float(jnp.min(el_heat)), 0.0)

  def test_particle_source_consistent_with_heating(self):
    ion_heat, el_heat = self._get_value('generic_heat')
    (particle_source,) = self._get_value('generic_particle')
    source_params = self.runtime_params.sources['generic_particle']
    beam_energy_J = source_params.beam_energy * constants.CONSTANTS.keV_to_J
    np.testing.assert_allclose(
        particle_source, (ion_heat + el_heat) / beam_energy_J, rtol=1e-9
    )

  def test_current_drive_is_co_current_and_flips_with_counter_injection(self):
    (j_co,) = self._get_value('generic_current')
    self.assertGreater(float(math_utils.area_integration(j_co, self.geo)), 0.0)
    self.assertGreaterEqual(float(jnp.min(j_co)), 0.0)

    # Rebuild with counter-injection: the driven current changes sign.
    counter_config = dict(_NBI_SOURCE_CONFIG, counter_injection=True)
    _, geo, runtime_params, source_models, core_profiles = _build_test_inputs(
        {'generic_current': counter_config}
    )
    (j_counter,) = source_models.standard_sources['generic_current'].get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )
    np.testing.assert_allclose(j_counter, -j_co, rtol=1e-9)

  def test_perpendicular_injection_drives_no_current(self):
    perpendicular_config = dict(_NBI_SOURCE_CONFIG, tangency_radius=0.0)
    _, geo, runtime_params, source_models, core_profiles = _build_test_inputs(
        {'generic_current': perpendicular_config}
    )
    (j_cd,) = source_models.standard_sources['generic_current'].get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )
    np.testing.assert_allclose(j_cd, 0.0, atol=1e-12)


class RegistrationTest(parameterized.TestCase):

  def test_registered_configs_are_built_from_dict(self):
    sources_config = {
        'generic_heat': dict(_NBI_SOURCE_CONFIG),
        'generic_current': dict(_NBI_SOURCE_CONFIG, counter_injection=True),
        'generic_particle': dict(_NBI_SOURCE_CONFIG),
    }
    torax_config, _, runtime_params, _, _ = _build_test_inputs(sources_config)
    self.assertIsInstance(
        torax_config.sources.generic_heat,
        metis_nbi_source.MetisNBIHeatSourceConfig,
    )
    self.assertIsInstance(
        torax_config.sources.generic_current,
        metis_nbi_source.MetisNBICurrentSourceConfig,
    )
    self.assertIsInstance(
        torax_config.sources.generic_particle,
        metis_nbi_source.MetisNBIParticleSourceConfig,
    )
    heat_params = runtime_params.sources['generic_heat']
    self.assertIsInstance(heat_params, metis_nbi_source.RuntimeParams)
    self.assertEqual(heat_params.beam_energy, 500.0)
    current_params = runtime_params.sources['generic_current']
    self.assertIsInstance(
        current_params, metis_nbi_source.CurrentDriveRuntimeParams
    )
    self.assertEqual(current_params.current_drive_sign, -1.0)

  def test_default_models_still_work_after_registration(self):
    config = default_configs.get_default_config_dict()
    config['sources'] = {'generic_heat': {}}
    torax_config = model_config.ToraxConfig.from_dict(config)
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    self.assertIsInstance(
        runtime_params.sources['generic_heat'],
        runtime_params_lib.RuntimeParams,
    )


if __name__ == '__main__':
  absltest.main()
