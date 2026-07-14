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
import numpy as np
import pydantic
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model.saturation import alpha_crit_saturation_model
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import quasilinear_transport_model

# pylint: disable=invalid-name


def _get_config():
  config = default_configs.get_default_config_dict()
  config['pedestal'] = {
      'model_name': 'set_P_ped_n_ped',
      'saturation_model': {'model_name': 'alpha_crit', 'alpha_crit': 1.0},
  }
  return config


class AlphaCritSaturationModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.torax_config = model_config.ToraxConfig.from_dict(_get_config())
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

  def _compute_reference_alpha_max(self, rho_norm_ped_top):
    """Reference max alpha over the pedestal region, mirroring production."""
    normalized_logarithmic_gradients = quasilinear_transport_model.NormalizedLogarithmicGradients.from_profiles(
        core_profiles=self.core_profiles,
        radial_coordinate=self.geo.r_mid,
        radial_face_coordinate=self.geo.r_mid_face,
        reference_length=self.geo.R_major,
    )
    alpha_face = quasilinear_transport_model.calculate_alpha(
        core_profiles=self.core_profiles,
        q=self.core_profiles.q_face,
        reference_magnetic_field=self.geo.B_0,
        normalized_logarithmic_gradients=normalized_logarithmic_gradients,
    )
    in_region = np.array(self.geo.rho_face_norm) >= rho_norm_ped_top
    return np.max(np.where(in_region, np.array(alpha_face), -np.inf))

  @parameterized.named_parameters(
      dict(
          testcase_name='active',
          # alpha_crit << alpha_ped -> saturation is active.
          alpha_crit_over_alpha=1e-3,
      ),
      dict(
          testcase_name='inactive',
          # alpha_crit >> alpha_ped -> no saturation.
          alpha_crit_over_alpha=1e3,
      ),
  )
  def test_saturation_multiplier(self, alpha_crit_over_alpha):
    ped_top_idx = -2
    rho_norm_ped_top = self.geo.rho_face_norm[ped_top_idx]
    reference_alpha_max = self._compute_reference_alpha_max(rho_norm_ped_top)

    config = _get_config()
    config['pedestal']['saturation_model']['alpha_crit'] = float(
        reference_alpha_max * alpha_crit_over_alpha
    )
    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)

    saturation_model = alpha_crit_saturation_model.AlphaCritSaturationModel()

    # T_i_ped/T_e_ped/n_e_ped are not used by this saturation model.
    pedestal_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=rho_norm_ped_top,
        T_i_ped=1.0,
        T_e_ped=1.0,
        n_e_ped=1.0,
    )

    transport_multipliers = saturation_model(
        runtime_params,
        self.geo,
        self.core_profiles,
        pedestal_output,
    )

    if alpha_crit_over_alpha > 1.0:
      np.testing.assert_allclose(transport_multipliers.chi_e_multiplier, 1.0)
    else:
      self.assertGreater(transport_multipliers.chi_e_multiplier, 1.0)
    np.testing.assert_allclose(
        transport_multipliers.chi_e_multiplier,
        transport_multipliers.chi_i_multiplier,
    )
    np.testing.assert_allclose(
        transport_multipliers.chi_e_multiplier,
        transport_multipliers.D_e_multiplier,
    )
    np.testing.assert_allclose(
        transport_multipliers.chi_e_multiplier,
        transport_multipliers.v_e_multiplier,
    )

  def test_ignores_pedestal_output_targets(self):
    """The multiplier must not depend on T_i_ped/T_e_ped/n_e_ped."""
    ped_top_idx = -2
    rho_norm_ped_top = self.geo.rho_face_norm[ped_top_idx]
    saturation_model = alpha_crit_saturation_model.AlphaCritSaturationModel()

    output_1 = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=rho_norm_ped_top,
        T_i_ped=1.0,
        T_e_ped=1.0,
        n_e_ped=1.0,
    )
    output_2 = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=rho_norm_ped_top,
        T_i_ped=100.0,
        T_e_ped=100.0,
        n_e_ped=1e21,
    )

    multipliers_1 = saturation_model(
        self.runtime_params, self.geo, self.core_profiles, output_1
    )
    multipliers_2 = saturation_model(
        self.runtime_params, self.geo, self.core_profiles, output_2
    )
    np.testing.assert_allclose(
        multipliers_1.chi_e_multiplier, multipliers_2.chi_e_multiplier
    )

  def test_compatible_with_set_T_ped_n_ped(self):
    # set_T_ped_n_ped provides rho_norm_ped_top, the only quantity this
    # saturation model uses from the pedestal model, so it should build
    # successfully (T_i_ped/T_e_ped/n_e_ped are unused).
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'model_name': 'set_T_ped_n_ped',
        'saturation_model': {'model_name': 'alpha_crit', 'alpha_crit': 1.0},
    }
    model_config.ToraxConfig.from_dict(config)

  def test_incompatible_with_no_pedestal(self):
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'model_name': 'no_pedestal',
        'saturation_model': {'model_name': 'alpha_crit', 'alpha_crit': 1.0},
    }
    with self.assertRaises(pydantic.ValidationError):
      model_config.ToraxConfig.from_dict(config)

  def test_triggers_on_max_alpha_in_region_not_just_boundary(self):
    """The trigger must use the max alpha in the region, not just the value
    at rho_norm_ped_top.

    Regression test: an earlier version of this model only checked alpha at
    the single grid face nearest rho_norm_ped_top, which let the gradient
    run away anywhere further into the pedestal region (towards the
    separatrix) without ever triggering saturation.
    """
    ped_top_idx = -2  # Leaves exactly one further face (-1) in the region.
    rho_norm_ped_top = self.geo.rho_face_norm[ped_top_idx]

    normalized_logarithmic_gradients = quasilinear_transport_model.NormalizedLogarithmicGradients.from_profiles(
        core_profiles=self.core_profiles,
        radial_coordinate=self.geo.r_mid,
        radial_face_coordinate=self.geo.r_mid_face,
        reference_length=self.geo.R_major,
    )
    alpha_face = np.array(
        quasilinear_transport_model.calculate_alpha(
            core_profiles=self.core_profiles,
            q=self.core_profiles.q_face,
            reference_magnetic_field=self.geo.B_0,
            normalized_logarithmic_gradients=normalized_logarithmic_gradients,
        )
    )
    alpha_at_boundary = alpha_face[ped_top_idx]
    alpha_further_in = alpha_face[-1]
    # Pick alpha_crit strictly between the two, so the boundary-only check
    # would say "not saturated" but the max-over-region check must trigger.
    self.assertNotEqual(alpha_at_boundary, alpha_further_in)
    alpha_crit = float((alpha_at_boundary + alpha_further_in) / 2)
    self.assertLess(min(alpha_at_boundary, alpha_further_in), alpha_crit)
    self.assertGreater(max(alpha_at_boundary, alpha_further_in), alpha_crit)

    config = _get_config()
    config['pedestal']['rho_norm_ped_top'] = float(rho_norm_ped_top)
    config['pedestal']['saturation_model']['alpha_crit'] = alpha_crit
    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)

    saturation_model = alpha_crit_saturation_model.AlphaCritSaturationModel()
    pedestal_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=rho_norm_ped_top,
        T_i_ped=1.0,
        T_e_ped=1.0,
        n_e_ped=1.0,
    )
    transport_multipliers = saturation_model(
        runtime_params, self.geo, self.core_profiles, pedestal_output
    )
    self.assertGreater(transport_multipliers.chi_e_multiplier, 1.0)


if __name__ == '__main__':
  absltest.main()
