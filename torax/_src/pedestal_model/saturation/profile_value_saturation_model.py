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

"""Saturation model based on deviation from pedestal model."""

import dataclasses
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model.saturation import base

# pylint: disable=invalid-name

# Sharpness of the smooth maximum (logsumexp) used to sense the peak density
# over the pedestal region, in units of the normalized density n_e / n_e_ped.
# The softmax overestimates the true maximum by ~log(n_cells)/sharpness in
# normalized units (~1% here), and is C-infinity for the Newton solver.
_SOFTMAX_SHARPNESS: float = 100.0


@dataclasses.dataclass(frozen=True, eq=False)
class ProfileValueSaturationModel(base.SaturationModel):
  """Saturation model based on values of the profiles at the pedestal top."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_output: pedestal_model_output.PedestalModelOutput,
  ) -> array_typing.FloatScalar:
    """Calculates transport increase multipliers."""
    # Get the current profile values at the pedestal top.
    # Interpolating to get the values at exactly rho_norm_ped_top is difficult,
    # as the gradients in the pedestal and the core are very different and are
    # going to be varying a lot during a solve step. Instead, we take the values
    # at the nearest grid point, which is more stable.
    rho_norm_face_ped_top_idx = jnp.argmin(
        jnp.abs(geo.rho_face_norm - pedestal_output.rho_norm_ped_top)
    )
    current_T_e_ped_top = core_profiles.T_e.face_value()[  # pyrefly: ignore[bad-index]
        rho_norm_face_ped_top_idx
    ]
    current_T_i_ped_top = core_profiles.T_i.face_value()[  # pyrefly: ignore[bad-index]
        rho_norm_face_ped_top_idx
    ]
    # The density channel senses the *maximum* n_e over the pedestal region
    # rather than the ped-top point value. With edge particle fueling and
    # strongly suppressed D, density can pile up at interior pedestal cells
    # while the ped-top value is still below target; sensing the region
    # maximum lets the feedback react to such pileup anywhere in the pedestal.
    # A softmax (logsumexp) is used to keep the residual smooth for the
    # Newton solver. For a well-formed (monotonically decreasing) pedestal the
    # maximum coincides with the ped-top value.
    n_e_face = core_profiles.n_e.face_value()
    pedestal_region_mask = (
        geo.rho_face_norm >= pedestal_output.rho_norm_ped_top
    )
    # Empty mask (e.g. rho_norm_ped_top=inf fallback) falls back to the
    # ped-top point sample.
    n_e_scale = jnp.maximum(
        pedestal_output.n_e_ped, constants.CONSTANTS.eps
    )
    masked_scaled_n_e = jnp.where(
        pedestal_region_mask,
        _SOFTMAX_SHARPNESS * n_e_face / n_e_scale,
        -jnp.inf,
    )
    current_n_e_ped_top = jnp.where(
        jnp.any(pedestal_region_mask),
        jax.nn.logsumexp(masked_scaled_n_e)
        * n_e_scale
        / _SOFTMAX_SHARPNESS,
        n_e_face[rho_norm_face_ped_top_idx],  # pyrefly: ignore[bad-index]
    )

    saturation = runtime_params.pedestal.saturation

    # Compute the multipliers based on the deviation from the pedestal model.
    # Each channel is driven by its own profile deviation.
    chi_e_multiplier = self._calculate_multiplier(
        current_T_e_ped_top,
        pedestal_output.T_e_ped,
        saturation.steepness,
        saturation.offset,
        saturation.base_multiplier,
    )
    chi_i_multiplier = self._calculate_multiplier(
        current_T_i_ped_top,
        pedestal_output.T_i_ped,
        saturation.steepness,
        saturation.offset,
        saturation.base_multiplier,
    )
    D_e_multiplier = self._calculate_multiplier(
        current_n_e_ped_top,
        pedestal_output.n_e_ped,
        saturation.density_steepness,
        saturation.density_offset,
        saturation.density_base_multiplier,
    )

    return pedestal_model_output.TransportMultipliers(  # pyrefly: ignore[bad-return]
        chi_e_multiplier=chi_e_multiplier,
        chi_i_multiplier=chi_i_multiplier,
        D_e_multiplier=D_e_multiplier,
        # The pinch is deliberately not increased by saturation (it is scaled
        # by the formation multiplier only). The steady-state density profile
        # shape is set by the ratio v/D, so raising D alone shifts that ratio
        # and regulates the pedestal density height; scaling D and v together
        # would preserve the shape and only change the relaxation timescale,
        # providing no height control.
        v_e_multiplier=jnp.array(1.0, dtype=jax_utils.get_dtype()),
    )

  def _calculate_multiplier(
      self,
      current: array_typing.FloatScalar,
      target: array_typing.FloatScalar,
      steepness: array_typing.FloatScalar,
      offset: array_typing.FloatScalar,
      base_multiplier: array_typing.FloatScalar,
  ) -> array_typing.FloatScalar:
    """Calculates the transport increase multiplier.

    If current >> target, multiplier -> infinity.
    If current << target, multiplier -> 1.

    Args:
      current: The current value of the profile at the pedestal top.
      target: The target value of the profile at the pedestal top.
      steepness: Steepness of the softplus saturation response.
      offset: Dimensionless offset of the saturation window.
      base_multiplier: Base value of the transport increase.

    Returns:
      The transport increase multiplier.
    """
    # Guard against zero targets (e.g. the fallback pedestal output used when
    # set_pedestal is False has all-zero targets); without this the deviation
    # is inf and can poison downstream jnp.where branches with NaNs under
    # differentiation, even though the multiplier itself is masked out.
    safe_target = jnp.maximum(target, constants.CONSTANTS.eps)
    normalized_deviation = (current - safe_target) / safe_target - offset
    transport_multiplier = 1 + base_multiplier * jax.nn.softplus(
        normalized_deviation * steepness
    )
    return transport_multiplier
