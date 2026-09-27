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

"""Tests for transport_coefficients_builder."""

import dataclasses
import functools

from absl.testing import absltest
import chex
import jax.numpy as jnp
import numpy as np
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import pereverzev
from torax._src.transport_model import transport_coefficients_builder
from torax._src.transport_model import transport_coeffs


class TransportCoefficientsBuilderTest(absltest.TestCase):

  def test_adaptive_pedestal_scales_turbulent_and_pereverzev_only(self):
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'model_name': 'set_T_ped_n_ped',
        'set_pedestal': True,
        'mode': 'ADAPTIVE_TRANSPORT',
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=0.0)
    geo = torax_config.geometry.build_provider(t=0.0)
    models = torax_config.build_models()
    core_profiles = initialization.initial_core_profiles(
        runtime_params, geo, models.source_models, models.neoclassical_models
    )
    pedestal_model_output = pedestal_model_output_lib.PedestalModelOutput(
        rho_norm_ped_top=jnp.array(0.5),
        T_i_ped=jnp.array(1.0),
        T_e_ped=jnp.array(1.0),
        n_e_ped=jnp.array(1e19),
        transport_multipliers=pedestal_model_output_lib.TransportMultipliers(
            chi_e_multiplier=jnp.array(0.2),
            chi_i_multiplier=jnp.array(0.3),
            D_e_multiplier=jnp.array(0.4),
            v_e_multiplier=jnp.array(0.5),
        ),
    )

    core_transport = transport_coefficients_builder.calculate_all_transport_coeffs(
        models.transport_model,
        models.neoclassical_models,
        models.internal_boundary_condition_model,
        runtime_params,
        geo,
        core_profiles,
        dataclasses.replace(
            pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode(),
            pedestal_model_output=pedestal_model_output,
        ),
        use_pereverzev=True,
    )

    # In ADAPTIVE_TRANSPORT mode there are no internal boundary conditions.
    two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
    turbulent = models.transport_model(
        runtime_params,
        geo,
        core_profiles,
        pedestal_model_output,
        two_point_mask,
    )
    scale = functools.partial(
        pedestal_model_output.scale_transport_coeffs,
        geo=geo,
        pedestal_runtime_params=runtime_params.pedestal,
    )
    turbulent_total = scale(turbulent.total)
    neoclassical = models.neoclassical_models.transport(
        runtime_params, geo, core_profiles
    )
    pereverzev_coeffs = scale(
        pereverzev.calculate_pereverzev_transport(
            runtime_params, geo, core_profiles, two_point_mask
        )
    )
    self.assertFalse(
        np.allclose(turbulent_total.chi_face_el, turbulent.total.chi_face_el)
    )
    # The coefficients of the individual models and the neoclassical ones are
    # not scaled.
    chex.assert_trees_all_close(
        core_transport,
        state.CoreTransport(
            total=transport_coeffs.sum_transport_coeffs(
                turbulent_total, neoclassical, pereverzev_coeffs
            ),
            turbulent=dataclasses.replace(turbulent, total=turbulent_total),
            neoclassical=neoclassical,
            pereverzev=pereverzev_coeffs,
        ),
    )


if __name__ == '__main__':
  absltest.main()
