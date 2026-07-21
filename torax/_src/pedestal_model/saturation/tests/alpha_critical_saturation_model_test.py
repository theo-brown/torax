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

"""Tests for the alpha-critical saturation model."""

import dataclasses

from absl.testing import absltest
from jax import numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model.saturation import alpha_critical_saturation_model
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class AlphaCriticalSaturationModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'model_name': 'set_T_ped_n_ped',
        'set_pedestal': True,
        'mode': 'ADAPTIVE_TRANSPORT',
        'saturation_model': {'model_name': 'alpha_critical'},
    }
    self.torax_config = model_config.ToraxConfig.from_dict(config)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        self.torax_config
    )
    self.runtime_params = provider(t=0.0)
    self.geo = self.torax_config.geometry.build_provider(t=0.0)
    self.core_profiles = initialization.initial_core_profiles(
        self.runtime_params,
        self.geo,
        self.torax_config.sources.build_models(),
        self.torax_config.neoclassical.build_models(),
    )
    self.saturation_model = (
        alpha_critical_saturation_model.AlphaCriticalSaturationModel()
    )
    self.pedestal_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=jnp.array(0.9),
        T_i_ped=jnp.array(5.0),
        T_e_ped=jnp.array(5.0),
        n_e_ped=jnp.array(0.7e20),
    )

  def _call(self, core_profiles, pedestal_output=None):
    return self.saturation_model(
        self.runtime_params,
        self.geo,
        core_profiles,
        pedestal_output if pedestal_output is not None else (
            self.pedestal_output
        ),
    )

  def _scaled_profiles(self, scale):
    def scaled(var):
      return dataclasses.replace(
          var,
          value=var.value * scale,
          right_face_constraint=var.right_face_constraint * scale,
      )

    return dataclasses.replace(
        self.core_profiles,
        T_i=scaled(self.core_profiles.T_i),
        T_e=scaled(self.core_profiles.T_e),
    )

  def test_stable_gradients_give_unit_multipliers(self):
    """Far below the ballooning boundary, no saturation."""
    multipliers = self._call(self._scaled_profiles(1e-3))
    np.testing.assert_allclose(multipliers.chi_e_multiplier, 1.0)
    np.testing.assert_allclose(multipliers.chi_i_multiplier, 1.0)
    np.testing.assert_allclose(multipliers.D_e_multiplier, 1.0)
    np.testing.assert_allclose(multipliers.v_e_multiplier, 1.0)

  def test_supercritical_alpha_activates_all_channels_except_pinch(self):
    """Above the boundary, chi_i, chi_e and D_e rise together; v does not."""
    multipliers = self._call(self._scaled_profiles(100.0))
    self.assertGreater(float(multipliers.chi_e_multiplier), 1.0)
    np.testing.assert_allclose(
        multipliers.chi_i_multiplier, multipliers.chi_e_multiplier
    )
    np.testing.assert_allclose(
        multipliers.D_e_multiplier, multipliers.chi_e_multiplier
    )
    np.testing.assert_allclose(multipliers.v_e_multiplier, 1.0)

  def test_pedestal_targets_are_ignored(self):
    """T_ped/n_ped targets do not affect the alpha-critical response."""
    hot = self._scaled_profiles(100.0)
    multipliers_a = self._call(hot)
    other_targets = dataclasses.replace(
        self.pedestal_output,
        T_i_ped=jnp.array(0.1),
        T_e_ped=jnp.array(0.1),
        n_e_ped=jnp.array(1e18),
    )
    multipliers_b = self._call(hot, pedestal_output=other_targets)
    np.testing.assert_allclose(
        multipliers_a.chi_e_multiplier, multipliers_b.chi_e_multiplier
    )

  def test_empty_pedestal_region_is_inert(self):
    """The rho_norm_ped_top=inf fallback output yields unit multipliers."""
    fallback_output = dataclasses.replace(
        self.pedestal_output, rho_norm_ped_top=jnp.array(jnp.inf)
    )
    multipliers = self._call(
        self._scaled_profiles(100.0), pedestal_output=fallback_output
    )
    np.testing.assert_allclose(multipliers.chi_e_multiplier, 1.0)
    np.testing.assert_allclose(multipliers.v_e_multiplier, 1.0)


if __name__ == '__main__':
  absltest.main()
