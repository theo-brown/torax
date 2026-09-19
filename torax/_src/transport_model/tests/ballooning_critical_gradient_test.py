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

"""Tests for the ballooning critical gradient transport model."""

import dataclasses

from absl.testing import absltest
from jax import numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class BallooningCriticalGradientTransportModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    config['transport'] = {'model_name': 'ballooning_CGM'}
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
    self.transport_model = (
        self.torax_config.transport.build_transport_model()
    )
    self.pedestal_model_output = pedestal_model_output_lib.PedestalModelOutput(
        rho_norm_ped_top=jnp.inf,
        T_i_ped=0.0,
        T_e_ped=0.0,
        n_e_ped=0.0,
    )

  def _call(self, core_profiles):
    return self.transport_model(
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=core_profiles,
        pedestal_model_output=self.pedestal_model_output,
    )

  def test_output_is_finite_and_bounded(self):
    coeffs = self._call(self.core_profiles)
    transport = self.runtime_params.transport
    for name in ('chi_face_ion', 'chi_face_el', 'd_face_el'):
      value = getattr(coeffs, name)
      with self.subTest(name):
        self.assertTrue(bool(jnp.all(jnp.isfinite(value))))
        self.assertTrue(bool(jnp.all(value > 0.0)))
        # Bounded by ceiling + floor (before base-class clipping).
        self.assertTrue(
            bool(
                jnp.all(
                    value
                    <= transport.chi_ceiling + transport.chi_floor + 1e-9  # pyrefly: ignore[missing-attribute]
                )
            )
        )
    np.testing.assert_allclose(coeffs.v_face_el, 0.0)

  def test_shallow_gradients_give_floor_transport(self):
    """With profiles scaled far down, alpha << alpha_crit -> floor only."""
    scale = 1e-3

    def scaled(var):
      return dataclasses.replace(
          var,
          value=var.value * scale,
          right_face_constraint=var.right_face_constraint * scale,
      )

    cold_profiles = dataclasses.replace(
        self.core_profiles,
        T_i=scaled(self.core_profiles.T_i),
        T_e=scaled(self.core_profiles.T_e),
    )
    coeffs = self._call(cold_profiles)
    transport = self.runtime_params.transport
    # Transport should sit near the floors: the residual sigmoid tail at
    # alpha=0 contributes at most ceiling * sigmoid(-alpha_crit/width), which
    # is floor-sized for the defaults, and far below the ceiling.
    self.assertTrue(
        bool(
            jnp.all(coeffs.chi_face_ion <= 3.0 * transport.chi_floor)  # pyrefly: ignore[missing-attribute]
        )
    )
    self.assertTrue(
        bool(jnp.all(coeffs.d_face_el <= 3.0 * transport.D_e_floor))  # pyrefly: ignore[missing-attribute]
    )
    self.assertLess(
        float(jnp.max(coeffs.chi_face_ion)),
        0.05 * float(transport.chi_ceiling),  # pyrefly: ignore[missing-attribute]
    )

  def test_transport_increases_with_pressure_gradient(self):
    """Steepening the pressure profile monotonically increases transport."""
    coeffs_base = self._call(self.core_profiles)

    def scaled(var, s):
      return dataclasses.replace(
          var,
          value=var.value * s,
          right_face_constraint=var.right_face_constraint * s,
      )

    hot_profiles = dataclasses.replace(
        self.core_profiles,
        T_i=scaled(self.core_profiles.T_i, 5.0),
        T_e=scaled(self.core_profiles.T_e, 5.0),
    )
    coeffs_hot = self._call(hot_profiles)
    self.assertGreater(
        float(jnp.max(coeffs_hot.chi_face_ion)),
        float(jnp.max(coeffs_base.chi_face_ion)),
    )
    # The response is bounded even for very steep gradients.
    transport = self.runtime_params.transport
    self.assertLessEqual(
        float(jnp.max(coeffs_hot.chi_face_ion)),
        float(transport.chi_ceiling + transport.chi_floor) + 1e-9,  # pyrefly: ignore[missing-attribute]
    )

  def test_higher_critical_alpha_reduces_transport(self):
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'ballooning_CGM',
        'alpha_crit_multiplier': 10.0,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    coeffs_stable = torax_config.transport.build_transport_model()(
        runtime_params=runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        pedestal_model_output=self.pedestal_model_output,
    )
    coeffs_default = self._call(self.core_profiles)
    self.assertLessEqual(
        float(jnp.max(coeffs_stable.chi_face_ion)),
        float(jnp.max(coeffs_default.chi_face_ion)) + 1e-12,
    )


if __name__ == '__main__':
  absltest.main()
