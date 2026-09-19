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

"""Power scaling pedestal formation model."""

import dataclasses
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.pedestal_model.formation import base
from torax._src.physics import scaling_laws
from torax._src.sources import source_profiles as source_profiles_lib

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PowerScalingFormationRuntimeParams(
    pedestal_runtime_params_lib.FormationRuntimeParams
):
  """Runtime params for power scaling pedestal formation models."""

  P_LH_prefactor: array_typing.FloatScalar = 1.0


def calculate_P_SOL_total(
    internal_plasma_energy: state.PlasmaInternalEnergy,
    core_sources: source_profiles_lib.SourceProfiles,
    geo: geometry.Geometry,
    include_dW_dt: bool = True,
) -> jax.Array:
  """Calculates the total power out of the separatrix.

  Args:
    internal_plasma_energy: Internal plasma energy state.
    core_sources: Source profiles.
    geo: Geometry.
    include_dW_dt: If True, subtracts dW/dt from P_heat to get P_SOL. If
      False, returns P_heat (total heating power) without the dW/dt correction.

  Returns:
    P_SOL or P_heat total [W].
  """
  P_heat_e = sum(
      math_utils.volume_integration(source, geo)
      for source in core_sources.T_e.values()
  )
  P_heat_i = sum(
      math_utils.volume_integration(source, geo)
      for source in core_sources.T_i.values()
  )
  P_heat_total = P_heat_e + P_heat_i
  if not include_dW_dt:
    return P_heat_total  # pyrefly: ignore[bad-return]
  return P_heat_total - internal_plasma_energy.dW_thermal_dt_smoothed  # pyrefly: ignore[bad-return]


@dataclasses.dataclass(frozen=True, eq=False)
class PowerScalingFormationModel(base.FormationModel):
  """Pedestal formation based on P_SOL and P_LH thresholds.

  Returns the barrier fraction g = sigmoid(sharpness * ((P_SOL - P_LH) / P_LH
  - offset)), the blend weight between L-mode transport (g=0) and the barrier
  transport branch (g=1).

  Attributes:
    scaling_law: The scaling law to use for pedestal formation.
    divertor_configuration: The divertor configuration. Only used for the
      Delabie scaling law.
  """

  scaling_law: scaling_laws.PLHScalingLaw
  divertor_configuration: scaling_laws.DivertorConfiguration = (
      scaling_laws.DivertorConfiguration.HT
  )

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      core_sources: source_profiles_lib.SourceProfiles,
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
  ) -> array_typing.FloatScalar:
    """Calculates the barrier fraction based on P_SOL and P_LH."""
    assert isinstance(
        runtime_params.pedestal.formation, PowerScalingFormationRuntimeParams
    )

    P_SOL_total = calculate_P_SOL_total(
        core_profiles.internal_plasma_energy,  # pyrefly: ignore[bad-argument-type]
        core_sources,
        geo,
        include_dW_dt=runtime_params.pedestal.include_dW_dt_in_P_SOL,
    )

    P_LH, _ = scaling_laws.calculate_P_LH(
        geo,
        core_profiles,
        scaling_law=self.scaling_law,
        divertor_configuration=self.divertor_configuration,
    )

    rescaled_P_LH = P_LH * runtime_params.pedestal.formation.P_LH_prefactor

    # Apply hysteresis: in H-mode, use a lower effective P_LH threshold,
    # making it harder to transition back to L-mode.
    effective_P_LH = jnp.where(
        pedestal_transition_state.confinement_mode
        == pedestal_transition_state_lib.ConfinementMode.H_MODE,
        rescaled_P_LH * runtime_params.pedestal.P_LH_hysteresis_factor,
        rescaled_P_LH,
    )

    # Calculate the barrier fraction g.
    # If P_SOL > effective_P_LH, g tends to 1.0 (barrier fully formed).
    # If P_SOL < effective_P_LH, g tends to 0.0 (L-mode transport).
    sharpness = runtime_params.pedestal.formation.sharpness
    offset = runtime_params.pedestal.formation.offset
    normalized_deviation = (P_SOL_total - effective_P_LH) / effective_P_LH
    shifted_deviation = normalized_deviation - offset
    return jax.nn.sigmoid(shifted_deviation * sharpness)
