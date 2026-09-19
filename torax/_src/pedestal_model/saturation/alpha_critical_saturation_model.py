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

"""Saturation signal based on the ballooning critical pressure gradient.

Instead of regulating the pedestal towards user-prescribed T_ped/n_ped
targets, this saturation model senses proximity of the s-alpha normalized
pressure gradient alpha to its critical value
alpha_crit = c_alpha * max(s, s_min), an approximation of the ideal
ballooning first-stability boundary. Combined with a power-scaling formation
model (P_SOL vs P_LH trigger), this yields an adaptive-transport pedestal
whose L-H transition timing is empirical (scaling-law based) while the
pedestal height emerges from MHD stability physics.
"""

import dataclasses

import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.pedestal_model.saturation import base
from torax._src.physics import formulas

# pylint: disable=invalid-name

# Sharpness of the smooth maximum (logsumexp) used to sense the peak
# alpha/alpha_crit ratio over the pedestal region, in normalized units.
_SOFTMAX_SHARPNESS: float = 100.0


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class AlphaCriticalSaturationRuntimeParams(
    pedestal_runtime_params_lib.SaturationRuntimeParams
):
  """Runtime params for the alpha-critical saturation model."""

  alpha_crit_multiplier: array_typing.FloatScalar = 0.6
  s_min: array_typing.FloatScalar = 0.5


@dataclasses.dataclass(frozen=True, eq=False)
class AlphaCriticalSaturationModel(base.SaturationModel):
  """Saturation signal from proximity to the ballooning stability boundary."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_output: pedestal_model_output.PedestalModelOutput,
  ) -> pedestal_model_output.BarrierSignals:
    """Calculates the proximity-to-stability-boundary signal.

    The sensed signal is the smooth (logsumexp) maximum of alpha/alpha_crit
    over the pedestal region, so that the boundary being approached anywhere
    inside the pedestal activates the response. The proximity signal is

      x = max(alpha / alpha_crit) - 1

    (zero at the boundary, negative inside it), and is shared by both heat
    channels and the particle diffusivity: alpha involves the total pressure
    gradient, so all channels relieve it.

    Args:
      runtime_params: Runtime parameters.
      geo: Geometry.
      core_profiles: Core profiles (current solver iterate when called from
        within the solver residual).
      pedestal_output: Output of the pedestal model implementation; only
        rho_norm_ped_top is used, to define the pedestal region. Any
        T_ped/n_e_ped targets it carries are ignored by this model.

    Returns:
      The per-channel proximity-to-limit signals.
    """
    saturation = runtime_params.pedestal.saturation
    # Required for pytype.
    assert isinstance(saturation, AlphaCriticalSaturationRuntimeParams)

    alpha = formulas.calc_ballooning_alpha_face(geo, core_profiles)
    alpha_crit = saturation.alpha_crit_multiplier * jnp.maximum(
        core_profiles.s_face, saturation.s_min
    )
    ratio = alpha / jnp.maximum(alpha_crit, constants.CONSTANTS.eps)

    # Smooth maximum of the ratio over the pedestal region. An empty mask
    # (e.g. the rho_norm_ped_top=inf fallback output) yields a fully stable
    # signal (ratio 0 -> signal -1).
    pedestal_region_mask = (
        geo.rho_face_norm >= pedestal_output.rho_norm_ped_top
    )
    masked_scaled_ratio = jnp.where(
        pedestal_region_mask, _SOFTMAX_SHARPNESS * ratio, -jnp.inf
    )
    max_ratio = jnp.where(
        jnp.any(pedestal_region_mask),
        jax.nn.logsumexp(masked_scaled_ratio) / _SOFTMAX_SHARPNESS,
        0.0,
    )

    signal = max_ratio - 1.0
    return pedestal_model_output.BarrierSignals(
        chi_i_signal=signal,
        chi_e_signal=signal,
        D_e_signal=signal,
    )
