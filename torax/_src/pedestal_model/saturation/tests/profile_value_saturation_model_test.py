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

  def _ped_output(
      self,
      ped_top_idx,
      T_i_factor,
      T_e_factor,
      n_e_factor,
      rho_norm_ped_top=None,
  ):
    """Pedestal output with targets scaled from the current profile values.

    Each target is the current face value at ped_top_idx times its factor:
    a factor >> 1 puts the target far above the current value (channel
    closed), a factor << 1 far below it (channel open).
    """
    if rho_norm_ped_top is None:
      rho_norm_ped_top = self.geo.rho_face[ped_top_idx]
    return pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=rho_norm_ped_top,
        T_i_ped=self.core_profiles.T_i.face_value()[ped_top_idx] * T_i_factor,  # pyrefly: ignore[bad-index]
        T_e_ped=self.core_profiles.T_e.face_value()[ped_top_idx] * T_e_factor,  # pyrefly: ignore[bad-index]
        n_e_ped=self.core_profiles.n_e.face_value()[ped_top_idx] * n_e_factor,  # pyrefly: ignore[bad-index]
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='above_target',
          # T_current >> T_target -> saturation fraction ~1 (transport opens).
          T_target_over_T_current=1e-1,
      ),
      dict(
          testcase_name='below_target',
          # T_current << T_target -> saturation fraction ~0 (transport suppressed).
          T_target_over_T_current=1e1,
      ),
  )
  def test_temperature_saturation_fraction(
      self,
      T_target_over_T_current,
  ):
    # Pedestal top at the last grid point. The T_i and n_e targets are set
    # far above their current values so those channels stay closed.
    saturation_fraction = self._saturation_fraction(
        self._ped_output(
            ped_top_idx=-1,
            T_i_factor=1e3,
            T_e_factor=T_target_over_T_current,
            n_e_factor=1e3,
        )
    )

    if T_target_over_T_current > 1.0:
      # Below target: the channel is closed (transport stays suppressed).
      self.assertLess(float(saturation_fraction.chi_e_saturation_fraction), 0.01)
    else:
      # Above target: the channel is fully open (transport opens up).
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

  @parameterized.named_parameters(
      dict(
          testcase_name='above_target',
          # n_current >> n_target -> particle channel open.
          n_target_over_n_current=1e-1,
      ),
      dict(
          testcase_name='below_target',
          # n_current << n_target -> particle channel closed.
          n_target_over_n_current=1e1,
      ),
  )
  def test_density_saturation_fraction(self, n_target_over_n_current):
    """The particle diffusivity channel is driven by the n_e deviation."""
    # Temperature targets far above current values: heat channels closed.
    saturation_fraction = self._saturation_fraction(
        self._ped_output(
            ped_top_idx=-1,
            T_i_factor=1e3,
            T_e_factor=1e3,
            n_e_factor=n_target_over_n_current,
        )
    )

    with self.subTest('heat_channels_unaffected'):
      self.assertLess(float(saturation_fraction.chi_e_saturation_fraction), 0.01)
      self.assertLess(float(saturation_fraction.chi_i_saturation_fraction), 0.01)
    if n_target_over_n_current > 1.0:
      self.assertLess(float(saturation_fraction.D_e_saturation_fraction), 0.01)
    else:
      self.assertGreater(float(saturation_fraction.D_e_saturation_fraction), 0.99)

  def test_density_saturation_fraction_senses_pedestal_region_maximum(self):
    """Density pileup inside the pedestal region activates the feedback.

    With edge fueling and strongly suppressed D, density can pile up at
    interior pedestal cells while the ped-top value is still below target.
    The density channel senses the (smooth) maximum over the pedestal region,
    so such pileup opens the particle channel even when the ped-top point
    value alone would not.
    """
    # Pedestal top a few cells inside the boundary, with the density target
    # above the ped-top value: point sampling alone would be inactive.
    ped_top_idx = -4
    pedestal_output = self._ped_output(
        ped_top_idx=ped_top_idx,
        T_i_factor=1e3,
        T_e_factor=1e3,
        n_e_factor=2.0,
        rho_norm_ped_top=self.geo.rho_face_norm[ped_top_idx],
    )

    with self.subTest('no_pileup_is_below_target'):
      saturation_fraction = self._saturation_fraction(pedestal_output)
      self.assertLess(float(saturation_fraction.D_e_saturation_fraction), 0.01)

    with self.subTest('interior_pileup_activates'):
      # Create a density spike at the second-to-last cell, well above target.
      spiked_value = self.core_profiles.n_e.value.at[-2].mul(10.0)
      spiked_core_profiles = dataclasses.replace(
          self.core_profiles,
          n_e=dataclasses.replace(self.core_profiles.n_e, value=spiked_value),
      )
      saturation_fraction = self._saturation_fraction(pedestal_output, spiked_core_profiles)
      self.assertGreater(float(saturation_fraction.D_e_saturation_fraction), 0.5)
      # Heat channels remain unaffected by the density spike.
      self.assertLess(float(saturation_fraction.chi_e_saturation_fraction), 0.01)
      self.assertLess(float(saturation_fraction.chi_i_saturation_fraction), 0.01)

  def test_channels_are_decoupled(self):
    """Regression test for the old aliasing of D_e to chi_e.

    A temperature overshoot must not open the particle diffusivity channel
    (that was the mechanism that flushed the density pedestal).
    """
    # Temperatures far above target (heat channels open), density far
    # below target (particle channel closed).
    saturation_fraction = self._saturation_fraction(
        self._ped_output(
            ped_top_idx=-1,
            T_i_factor=1e-3,
            T_e_factor=1e-3,
            n_e_factor=1e3,
        )
    )

    self.assertGreater(float(saturation_fraction.chi_e_saturation_fraction), 0.99)
    self.assertGreater(float(saturation_fraction.chi_i_saturation_fraction), 0.99)
    self.assertLess(float(saturation_fraction.D_e_saturation_fraction), 0.01)


if __name__ == '__main__':
  absltest.main()
