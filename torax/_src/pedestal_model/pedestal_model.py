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

"""The PedestalModel abstract base class.

The pedestal model calculates quantities relevant to the pedestal.
"""

import abc
import dataclasses

import jax
import jax.numpy as jnp
from torax._src import jax_utils
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.pedestal_model.formation import base as formation_base
from torax._src.pedestal_model.saturation import base as saturation_base
from torax._src.sources import source_profiles as source_profiles_lib

# pylint: disable=invalid-name
# Using physics notation naming convention


@dataclasses.dataclass(frozen=True, eq=False)
class PedestalModel(static_dataclass.StaticDataclass, abc.ABC):
  """Calculates temperature and density of the pedestal."""

  formation_model: formation_base.FormationModel
  saturation_model: saturation_base.SaturationModel

  def compute_transport_multipliers(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
      pedestal_output: pedestal_model_output.PedestalModelOutput,
  ) -> pedestal_model_output.TransportMultipliers:
    """Computes transport multipliers from formation and saturation models.

    Both the formation model's H-mode fraction and the saturation model's
    per-channel saturation fractions are bounded in (0, 1). Each is mapped to
    a multiplier by the same affine remap between 1 (channel unaffected) and
    that model's base_multiplier (channel fully affected); the two
    multipliers are then combined multiplicatively.
    """

    H_mode_fraction = self.formation_model(
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        pedestal_transition_state,
    )
    saturation_fractions = self.saturation_model(
        runtime_params, geo, core_profiles, pedestal_output
    )

    def to_multiplier(fraction, base_multiplier):
      return (1.0 - fraction) + fraction * base_multiplier

    formation_base_multiplier = runtime_params.pedestal.formation.base_multiplier
    decrease_multiplier = to_multiplier(H_mode_fraction, formation_base_multiplier)
    transport_decrease_multiplier = pedestal_model_output.TransportMultipliers(
        chi_e_multiplier=decrease_multiplier,
        chi_i_multiplier=decrease_multiplier,
        D_e_multiplier=decrease_multiplier,
        v_e_multiplier=decrease_multiplier,
    )

    saturation_base_multiplier = runtime_params.pedestal.saturation.base_multiplier
    # The pinch follows the electron heat channel's saturation fraction.
    chi_e_increase_multiplier = to_multiplier(
        saturation_fractions.chi_e_saturation_fraction, saturation_base_multiplier
    )
    transport_increase_multiplier = pedestal_model_output.TransportMultipliers(
        chi_e_multiplier=chi_e_increase_multiplier,
        chi_i_multiplier=to_multiplier(
            saturation_fractions.chi_i_saturation_fraction,
            saturation_base_multiplier,
        ),
        D_e_multiplier=to_multiplier(
            saturation_fractions.D_e_saturation_fraction,
            saturation_base_multiplier,
        ),
        v_e_multiplier=chi_e_increase_multiplier,
    )

    # Combine via exp(log) for numerical stability, as multipliers can
    # be very small or large.
    return jax.tree.map(
        lambda x, y: jnp.exp(jnp.log(x) + jnp.log(y)),
        transport_decrease_multiplier,
        transport_increase_multiplier,
    )

  def _evaluate_pedestal(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
  ) -> pedestal_model_output.PedestalModelOutput:
    pedestal_output = self._call_implementation(
        runtime_params, geo, core_profiles, pedestal_transition_state,
    )

    # If in ADAPTIVE_TRANSPORT mode, calculate the transport multipliers based
    # on the formation and saturation models.
    if (
        runtime_params.pedestal.mode
        == pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT
    ):
      transport_multipliers = self.compute_transport_multipliers(
          runtime_params,
          geo,
          core_profiles,
          source_profiles,
          pedestal_transition_state,
          pedestal_output,
      )
      pedestal_output = dataclasses.replace(
          pedestal_output, transport_multipliers=transport_multipliers
      )

    return pedestal_output

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
  ) -> pedestal_model_output.PedestalModelOutput:
    return jax.lax.cond(
        runtime_params.pedestal.set_pedestal,
        self._evaluate_pedestal,
        lambda runtime_params, geo, core_profiles, source_profiles, pedestal_transition_state: pedestal_model_output.PedestalModelOutput(
            rho_norm_ped_top=jnp.array(
                jnp.inf, dtype=jax_utils.get_dtype()
            ),
            T_i_ped=jnp.array(0.0, dtype=jax_utils.get_dtype()),
            T_e_ped=jnp.array(0.0, dtype=jax_utils.get_dtype()),
            n_e_ped=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        ),
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        pedestal_transition_state,
    )

  @abc.abstractmethod
  def _call_implementation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
  ) -> pedestal_model_output.PedestalModelOutput:
    """Calculate the pedestal properties."""
