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
"""Saturation model based on a ballooning-stability limit (alpha_crit)."""

import dataclasses

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.pedestal_model.saturation import base
from torax._src.transport_model import quasilinear_transport_model

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class AlphaCritSaturationRuntimeParams(
    pedestal_runtime_params_lib.SaturationRuntimeParams
):
  """Runtime params for the AlphaCritSaturationModel."""

  alpha_crit: array_typing.FloatScalar


@dataclasses.dataclass(frozen=True, eq=False)
class AlphaCritSaturationModel(base.SaturationModel):
  r"""Saturation model based on a local ballooning-stability limit.

  Increases pedestal-region transport once the local infinite-n ideal
  ballooning mode normalized pressure gradient ``alpha``, evaluated at the
  pedestal top from the *current* (evolving) core profiles, exceeds a
  user-provided critical value ``alpha_crit``. This uses the same alpha_MHD
  convention (reference length :math:`R_{major}`, radial coordinate the
  midplane minor radius) as
  ``quasilinear_transport_model.calculate_alpha``, which is used by the
  QuaLiKiz- and TGLF-based transport models.

  ``alpha_crit`` is not computed by TORAX -- it must be supplied externally
  (e.g. from a local ideal ballooning mode stability calculation).

  Only compatible with the ``set_P_ped_n_ped`` pedestal model (for the
  ``rho_norm_ped_top`` and composition it provides). The pedestal model's own
  ``P_ped`` is not used here -- this saturation model determines when the
  pedestal stops growing purely from the alpha vs. alpha_crit comparison, not
  from any target pressure/temperature.
  """

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_output: pedestal_model_output.PedestalModelOutput,
  ) -> pedestal_model_output.TransportMultipliers:
    saturation_params = runtime_params.pedestal.saturation
    assert isinstance(saturation_params, AlphaCritSaturationRuntimeParams)

    normalized_logarithmic_gradients = quasilinear_transport_model.NormalizedLogarithmicGradients.from_profiles(
        core_profiles=core_profiles,
        radial_coordinate=geo.r_mid,
        radial_face_coordinate=geo.r_mid_face,
        reference_length=geo.R_major,
    )
    alpha_face = quasilinear_transport_model.calculate_alpha(
        core_profiles=core_profiles,
        q=core_profiles.q_face,
        reference_magnetic_field=geo.B_0,
        normalized_logarithmic_gradients=normalized_logarithmic_gradients,
    )

    # As in ProfileValueSaturationModel, use the nearest grid point rather
    # than interpolating: alpha is itself a gradient-based quantity, and is
    # noisy under interpolation near the pedestal top where gradients change
    # rapidly.
    rho_norm_face_ped_top_idx = jnp.argmin(
        jnp.abs(geo.rho_face_norm - pedestal_output.rho_norm_ped_top)
    )
    alpha_ped = alpha_face[rho_norm_face_ped_top_idx]  # pyrefly: ignore[bad-index]

    alpha_crit = saturation_params.alpha_crit
    normalized_deviation = (
        (alpha_ped - alpha_crit) / alpha_crit - saturation_params.offset
    )
    transport_multiplier = 1 + saturation_params.base_multiplier * jax.nn.softplus(
        normalized_deviation * saturation_params.steepness
    )

    return pedestal_model_output.TransportMultipliers(  # pyrefly: ignore[bad-return]
        chi_e_multiplier=transport_multiplier,
        chi_i_multiplier=transport_multiplier,
        D_e_multiplier=transport_multiplier,
        v_e_multiplier=transport_multiplier,
    )
