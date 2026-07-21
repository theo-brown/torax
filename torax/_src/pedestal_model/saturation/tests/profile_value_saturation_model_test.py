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
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
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

  @parameterized.named_parameters(
      dict(
          testcase_name='active',
          # T_current >> T_target -> saturation is active.
          T_target_over_T_current=1e-3,
      ),
      dict(
          testcase_name='inactive',
          # T_current << T_target -> no saturation.
          T_target_over_T_current=1e3,
      ),
  )
  def test_saturation_multiplier(
      self,
      T_target_over_T_current,
  ):
    saturation_model = (
        profile_value_saturation_model.ProfileValueSaturationModel()
    )

    # For this test, we put the pedestal top at the last grid point.
    ped_top_idx = -1
    current_T_e_ped = self.core_profiles.T_e.face_value()[ped_top_idx]  # pyrefly: ignore[bad-index]
    current_n_e_ped = self.core_profiles.n_e.face_value()[ped_top_idx]  # pyrefly: ignore[bad-index]

    # Construct a pedestal output that is asking for a pedestal with
    # target temperature. The density target is set far above the current
    # density so the density channel stays inactive in this test.
    pedestal_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=self.geo.rho_face[ped_top_idx],
        T_i_ped=1.0,
        T_e_ped=current_T_e_ped * T_target_over_T_current,
        n_e_ped=current_n_e_ped * 1e3,
    )

    transport_multipliers = saturation_model(
        self.runtime_params,
        self.geo,
        self.core_profiles,
        pedestal_output,
    )

    if T_target_over_T_current > 1.0:
      # If the target temperature is above the current temperature, we expect
      # the multiplier to be equal to 1.0 - the pedestal is not saturated.
      np.testing.assert_allclose(transport_multipliers.chi_e_multiplier, 1.0)
    else:
      # If the target temperature is below the current temperature, we expect
      # the multiplier to be greater than 1.0 - the pedestal is saturated.
      self.assertGreater(transport_multipliers.chi_e_multiplier, 1.0)

  @parameterized.named_parameters(
      dict(
          testcase_name='active',
          # n_current >> n_target -> density saturation is active.
          n_target_over_n_current=1e-3,
      ),
      dict(
          testcase_name='inactive',
          # n_current << n_target -> no density saturation.
          n_target_over_n_current=1e3,
      ),
  )
  def test_density_saturation_multiplier(self, n_target_over_n_current):
    """The particle diffusivity channel is driven by the n_e deviation."""
    saturation_model = (
        profile_value_saturation_model.ProfileValueSaturationModel()
    )
    ped_top_idx = -1
    current_T_e_ped = self.core_profiles.T_e.face_value()[ped_top_idx]  # pyrefly: ignore[bad-index]
    current_T_i_ped = self.core_profiles.T_i.face_value()[ped_top_idx]  # pyrefly: ignore[bad-index]
    current_n_e_ped = self.core_profiles.n_e.face_value()[ped_top_idx]  # pyrefly: ignore[bad-index]

    # Temperature targets far above current values: heat channels inactive.
    pedestal_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=self.geo.rho_face[ped_top_idx],
        T_i_ped=current_T_i_ped * 1e3,
        T_e_ped=current_T_e_ped * 1e3,
        n_e_ped=current_n_e_ped * n_target_over_n_current,
    )

    transport_multipliers = saturation_model(
        self.runtime_params,
        self.geo,
        self.core_profiles,
        pedestal_output,
    )

    with self.subTest('heat_channels_unaffected'):
      np.testing.assert_allclose(transport_multipliers.chi_e_multiplier, 1.0)
      np.testing.assert_allclose(transport_multipliers.chi_i_multiplier, 1.0)
    with self.subTest('pinch_never_increased_by_saturation'):
      np.testing.assert_allclose(transport_multipliers.v_e_multiplier, 1.0)
    if n_target_over_n_current > 1.0:
      np.testing.assert_allclose(transport_multipliers.D_e_multiplier, 1.0)
    else:
      self.assertGreater(transport_multipliers.D_e_multiplier, 1.0)

  def test_density_saturation_senses_pedestal_region_maximum(self):
    """Density pileup inside the pedestal region activates the feedback.

    With edge fueling and strongly suppressed D, density can pile up at
    interior pedestal cells while the ped-top value is still below target.
    The density channel senses the (smooth) maximum over the pedestal region,
    so such pileup raises D_e_multiplier even when the ped-top point value
    alone would not.
    """
    saturation_model = (
        profile_value_saturation_model.ProfileValueSaturationModel()
    )
    # Pedestal top a few cells inside the boundary.
    ped_top_idx = -4
    current_T_e_ped = self.core_profiles.T_e.face_value()[ped_top_idx]  # pyrefly: ignore[bad-index]
    current_T_i_ped = self.core_profiles.T_i.face_value()[ped_top_idx]  # pyrefly: ignore[bad-index]
    current_n_e_ped = self.core_profiles.n_e.face_value()[ped_top_idx]  # pyrefly: ignore[bad-index]

    # Target above the ped-top value: point sampling alone would be inactive.
    pedestal_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=self.geo.rho_face_norm[ped_top_idx],
        T_i_ped=current_T_i_ped * 1e3,
        T_e_ped=current_T_e_ped * 1e3,
        n_e_ped=current_n_e_ped * 2.0,
    )

    with self.subTest('no_pileup_is_inactive'):
      multipliers = saturation_model(
          self.runtime_params, self.geo, self.core_profiles, pedestal_output
      )
      np.testing.assert_allclose(multipliers.D_e_multiplier, 1.0)

    with self.subTest('interior_pileup_activates'):
      # Create a density spike at the second-to-last cell, well above target.
      spiked_value = self.core_profiles.n_e.value.at[-2].mul(10.0)
      spiked_core_profiles = dataclasses.replace(
          self.core_profiles,
          n_e=dataclasses.replace(self.core_profiles.n_e, value=spiked_value),
      )
      multipliers = saturation_model(
          self.runtime_params, self.geo, spiked_core_profiles, pedestal_output
      )
      self.assertGreater(multipliers.D_e_multiplier, 1.0)
      # Heat channels remain unaffected by the density spike.
      np.testing.assert_allclose(multipliers.chi_e_multiplier, 1.0)
      np.testing.assert_allclose(multipliers.chi_i_multiplier, 1.0)

  def test_channels_are_decoupled(self):
    """Regression test for the old aliasing of D_e/v_e to chi_e.

    A temperature overshoot must not raise the particle diffusivity (that was
    the mechanism that flushed the density pedestal), and the pinch must not
    be raised by saturation at all.
    """
    saturation_model = (
        profile_value_saturation_model.ProfileValueSaturationModel()
    )
    ped_top_idx = -1
    current_T_e_ped = self.core_profiles.T_e.face_value()[ped_top_idx]  # pyrefly: ignore[bad-index]
    current_T_i_ped = self.core_profiles.T_i.face_value()[ped_top_idx]  # pyrefly: ignore[bad-index]
    current_n_e_ped = self.core_profiles.n_e.face_value()[ped_top_idx]  # pyrefly: ignore[bad-index]

    # Temperatures far above target (heat saturation active), density far
    # below target (density saturation inactive).
    pedestal_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=self.geo.rho_face[ped_top_idx],
        T_i_ped=current_T_i_ped * 1e-3,
        T_e_ped=current_T_e_ped * 1e-3,
        n_e_ped=current_n_e_ped * 1e3,
    )

    transport_multipliers = saturation_model(
        self.runtime_params,
        self.geo,
        self.core_profiles,
        pedestal_output,
    )

    self.assertGreater(transport_multipliers.chi_e_multiplier, 1.0)
    self.assertGreater(transport_multipliers.chi_i_multiplier, 1.0)
    np.testing.assert_allclose(transport_multipliers.D_e_multiplier, 1.0)
    np.testing.assert_allclose(transport_multipliers.v_e_multiplier, 1.0)


if __name__ == '__main__':
  absltest.main()
