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


@dataclasses.dataclass(frozen=True, eq=False)
class ProfileValueSaturationModel(base.SaturationModel):
  """Target-based saturation fraction: sigmoid of current/target - 1.

  The heat channels sense T_e and T_i at the pedestal-top face against the
  pedestal model's targets. Suitable for pedestal models that prescribe
  specific pedestal-top values, e.g. EPED-style predictions of T_e_ped.
  """

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_output: pedestal_model_output.PedestalModelOutput,
  ) -> base.SaturationFraction:
    """Calculates the saturation fraction of each channel from its target deviation."""
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

    saturation_params = runtime_params.pedestal.saturation

    def saturation_fraction(current, target):
      # Guard against zero targets (e.g. the set_pedestal=False fallback
      # output carries all-zero targets); without this the deviation is inf
      # and can poison downstream jnp.where branches with NaNs under
      # differentiation, even though the resulting fraction is masked out.
      safe_target = jnp.maximum(target, constants.CONSTANTS.eps)
      deviation = (current - safe_target) / safe_target - saturation_params.offset
      return jax.nn.sigmoid(deviation * saturation_params.steepness)

    chi_e_saturation_fraction = saturation_fraction(
        current_T_e_ped_top, pedestal_output.T_e_ped
    )
    return base.SaturationFraction(
        chi_i_saturation_fraction=saturation_fraction(
            current_T_i_ped_top, pedestal_output.T_i_ped
        ),
        chi_e_saturation_fraction=chi_e_saturation_fraction,
        # TODO(b/487920703): set the particle channel saturation fraction
        # based on n_e_ped. In testing, we found this could be unstable.
        D_e_saturation_fraction=chi_e_saturation_fraction,
    )
