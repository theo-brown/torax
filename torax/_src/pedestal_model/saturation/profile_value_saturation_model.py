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
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model.saturation import base

# pylint: disable=invalid-name

# Sharpness of the logsumexp smooth maximum used to sense the peak density
# over the pedestal region, in units of n_e / n_e_ped. It overestimates the
# true maximum by ~log(n_cells)/sharpness (~1% here) and is smooth for the
# Newton solver.
_SOFTMAX_SHARPNESS: float = 100.0


@dataclasses.dataclass(frozen=True, eq=False)
class ProfileValueSaturationModel(base.SaturationModel):
  """Target-based saturation: proximity is current / target - 1 per channel.

  The heat channels sense T_e and T_i at the pedestal-top face against the
  pedestal model's targets; the particle channel senses the (smooth) maximum
  of n_e over the pedestal region against n_e_ped. Suitable for pedestal
  models that prescribe specific pedestal-top values, e.g. EPED-style
  predictions of T_e_ped.
  """

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_output: pedestal_model_output.PedestalModelOutput,
  ) -> base.SaturationFraction:
    """Calculates the saturation fraction of each channel from its deviation."""
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
    # The density channel senses the smooth (logsumexp) maximum of n_e over
    # the pedestal region rather than the ped-top point value: with edge
    # fueling against suppressed D, density can pile up at interior cells
    # while the ped-top value is still below target. For a monotonically
    # decreasing pedestal the maximum coincides with the ped-top value.
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

    saturation_params = runtime_params.pedestal.saturation

    def saturation_fraction(current, target, offset, response_width):
      return base.bounded_response(
          _relative_deviation(current, target), offset, response_width
      )

    return base.SaturationFraction(
        chi_i_saturation_fraction=saturation_fraction(
            current_T_i_ped_top,
            pedestal_output.T_i_ped,
            saturation_params.offset,
            saturation_params.response_width,
        ),
        chi_e_saturation_fraction=saturation_fraction(
            current_T_e_ped_top,
            pedestal_output.T_e_ped,
            saturation_params.offset,
            saturation_params.response_width,
        ),
        D_e_saturation_fraction=saturation_fraction(
            current_n_e_ped_top,
            pedestal_output.n_e_ped,
            saturation_params.density_offset,
            saturation_params.density_response_width,
        ),
    )


def _relative_deviation(
    current: array_typing.FloatScalar,
    target: array_typing.FloatScalar,
) -> array_typing.FloatScalar:
  """Returns current / target - 1, guarding against zero targets.

  The set_pedestal=False fallback output carries all-zero targets; an inf
  deviation there can poison downstream jnp.where branches with NaNs under
  differentiation, even though the resulting saturation fraction is masked
  out.
  """
  safe_target = jnp.maximum(target, constants.CONSTANTS.eps)
  return (current - safe_target) / safe_target
