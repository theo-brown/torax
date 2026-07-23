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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model.saturation import profile_value_saturation_model
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class FromPedestalModelSaturationModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    self.torax_config = model_config.ToraxConfig.from_dict(config)
    self.provider = build_runtime_params.RuntimeParamsProvider.from_config(
        self.torax_config
    )
    self.runtime_params = self.provider(t=0.0)
    self.geo = self.torax_config.geometry.build_provider(t=0.0)
    self.source_models = self.torax_config.sources.build_models()
    self.neoclassical_models = self.torax_config.neoclassical.build_models()
    self.core_profiles = initialization.initial_core_profiles(
        self.runtime_params,
        self.geo,
        self.source_models,
        self.neoclassical_models,
    )

  def _saturation_fraction(self, pedestal_output, core_profiles=None):
    """Runs the saturation model on the given pedestal output."""
    saturation_model = (
        profile_value_saturation_model.ProfileValueSaturationModel()
    )
    return saturation_model(
        self.runtime_params,
        self.geo,
        core_profiles if core_profiles is not None else self.core_profiles,
        pedestal_output,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='above_target',
          # T_current >> T_target -> saturation fraction ~1 (saturation active).
          T_target_over_T_current=1e-1,
      ),
      dict(
          testcase_name='below_target',
          # T_current << T_target -> saturation fraction ~0 (no saturation).
          T_target_over_T_current=1e1,
      ),
  )
  def test_temperature_saturation_fraction(
      self,
      T_target_over_T_current,
  ):
    # For this test, we put the pedestal top at the last grid point.
    ped_top_idx = -1
    current_T_e_ped = self.core_profiles.T_e.face_value()[ped_top_idx]  # pyrefly: ignore[bad-index]

    saturation_fraction = self._saturation_fraction(
        pedestal_model_output.PedestalModelOutput(
            rho_norm_ped_top=self.geo.rho_face[ped_top_idx],
            T_i_ped=1.0,
            T_e_ped=current_T_e_ped * T_target_over_T_current,
            n_e_ped=1.0,
        )
    )

    if T_target_over_T_current > 1.0:
      # Below target: the channel is closed (no saturation).
      self.assertLess(float(saturation_fraction.chi_e_saturation_fraction), 0.01)
    else:
      # Above target: the channel is fully open (saturation active).
      self.assertGreater(float(saturation_fraction.chi_e_saturation_fraction), 0.99)
    # The saturation fraction is the bounded response of the relative target
    # deviation.
    saturation_params = self.runtime_params.pedestal.saturation
    np.testing.assert_allclose(
        saturation_fraction.chi_e_saturation_fraction,
        jax.nn.sigmoid(
            (1.0 / T_target_over_T_current - 1.0 - saturation_params.offset)
            / saturation_params.response_width
        ),
        rtol=1e-6,
    )

  def test_particle_channel_aliased_to_electron_heat_channel(self):
    """The particle channel currently follows the electron heat channel."""
    ped_top_idx = -1
    current_T_e_ped = self.core_profiles.T_e.face_value()[ped_top_idx]  # pyrefly: ignore[bad-index]

    saturation_fraction = self._saturation_fraction(
        pedestal_model_output.PedestalModelOutput(
            rho_norm_ped_top=self.geo.rho_face[ped_top_idx],
            T_i_ped=1.0,
            T_e_ped=current_T_e_ped * 0.5,
            n_e_ped=1.0,
        )
    )
    np.testing.assert_allclose(
        saturation_fraction.D_e_saturation_fraction,
        saturation_fraction.chi_e_saturation_fraction,
    )


if __name__ == '__main__':
  absltest.main()
