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

"""Dataclass representing runtime parameter inputs to the pedestal models."""

import dataclasses
import enum

import jax
from torax._src import array_typing

# pylint: disable=invalid-name


@enum.unique
class Mode(enum.Enum):
  """Defines how the pedestal is generated."""

  # The pedestal is set by modifying the transport coefficients.
  ADAPTIVE_TRANSPORT = "ADAPTIVE_TRANSPORT"

  # The pedestal is set by modifying the state equations and directly setting
  # internal Dirichlet boundary conditions at a grid point or partial profile.
  INTERNAL_BOUNDARY_CONDITION = "INTERNAL_BOUNDARY_CONDITION"


@enum.unique
class PedestalProfileForm(enum.StrEnum):
  """Controls the shape of internal boundary conditions in the pedestal region.

  Attributes:
    SET_AT_PED_TOP: Pedestal values are pinned at a single grid cell nearest to
      rho_norm_ped_top.
    MTANH: A smooth modified-tanh (mtanh) profile is applied across the pedestal
      region from the pedestal top to the separatrix, following the mtanh
      parameterization of Snyder et al., Nucl. Fusion 44 (2004) 320.
  """

  SET_AT_PED_TOP = "SET_AT_PED_TOP"
  MTANH = "MTANH"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FormationRuntimeParams:
  """Runtime params for pedestal formation models.

  Attributes:
    sharpness: Sharpness of the sigmoid mapping the formation trigger
      to the H-mode fraction g in [0, 1].
    offset: Dimensionless offset of the formation window.
  """

  sharpness: array_typing.FloatScalar
  offset: array_typing.FloatScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SaturationRuntimeParams:
  """Runtime params for pedestal saturation models.

  The shared response maps each proximity-to-limit value x to a
  saturation fraction r = sigmoid((x - offset) / response_width); the heat
  channels
  (chi_i, chi_e) use (offset, response_width) and the particle diffusivity
  channel (D_e) uses (density_offset, density_response_width).

  Attributes:
    offset: Proximity value at which the heat channel saturation fraction
      reaches 0.5 (r = 0.5).
    response_width: Width of the heat channel saturation response in proximity
      units. Smaller values regulate more tightly but stiffen the implicit
      solve.
    density_offset: As `offset`, for the particle diffusivity channel.
    density_response_width: As `response_width`, for the particle diffusivity
      channel.
  """

  offset: array_typing.FloatScalar
  response_width: array_typing.FloatScalar
  density_offset: array_typing.FloatScalar
  density_response_width: array_typing.FloatScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  """Input params for the pedestal model.

  Attributes:
    set_pedestal: Whether to use the pedestal model and set the pedestal.
    mode: Defines how the pedestal is generated.
    use_formation_model_with_internal_boundary_condition: When True and mode is
      INTERNAL_BOUNDARY_CONDITION, enables state-dependent L-H transitions based
      on P_SOL vs P_LH comparison. The formation model is used to check the
      transition condition. When False, INTERNAL_BOUNDARY_CONDITION mode always
      applies the pedestal values directly, whenever set_pedestal is True.
    transition_time_width: Duration of the L-H or H-L transition ramp [s].
      During a transition, pedestal values are linearly interpolated between
      L-mode and H-mode values over this time window. Only used when
      use_formation_model_with_internal_boundary_condition is True.
    P_LH_hysteresis_factor: Hysteresis factor for H-L back transitions. When
      checking for an H-L transition, the L-H threshold power P_LH is multiplied
      by this factor, i.e. the back transition occurs when P_SOL < P_LH *
      P_LH_hysteresis_factor. A value less than 1 means that the plasma must
      lose more power to transition back to L-mode than was required to enter
      H-mode, which is the experimentally observed behavior. Must be in [0, 1].
      Only used when use_formation_model_with_internal_boundary_condition is
      True.
    include_dW_dt_in_P_SOL: Whether to include the dW/dt term in the P_SOL
      calculation used for comparing against P_LH. When False (default), uses
      P_heat (total auxiliary + Ohmic power - sinks) instead of P_SOL = P_heat -
      dW/dt. Excluding dW/dt avoids possible spurious dithering during
      transients.
    explicit_pedestal: When True (default), the pedestal model is evaluated once
      per timestep before the solver loop and its output (T_ped, n_ped,
      rho_ped_top) is frozen during Newton iterations. When False, the pedestal
      model is re-evaluated every Newton iteration, coupling pedestal physics to
      the implicit solve. Note: for ADAPTIVE_TRANSPORT mode, the transport blend
      is always re-evaluated implicitly with current profiles regardless of
      this setting, since the saturation feedback loop requires implicit
      coupling.
    pedestal_profile_form: Controls the shape of internal boundary conditions in
      the pedestal region. SET_AT_PED_TOP (default) pins pedestal values at a
      single grid cell nearest to rho_norm_ped_top. MTANH applies a smooth
      modified-tanh profile between the pedestal top and the separatrix, with
      the mtanh width Δ derived from rho_norm_ped_top via the ψ_N(ρ) mapping in
      core_profiles.
    formation: Runtime params for the formation model.
    saturation: Runtime params for the saturation model.
    chi_H_mode_max: Heat diffusivity of the H-mode transport branch at full
      saturation fraction (r = 1) [m^2/s].
    D_e_H_mode_max: Particle diffusivity of the H-mode transport branch at full
      saturation fraction (r = 1) [m^2/s].
    chi_H_mode_min: Heat diffusivity of the H-mode transport branch at zero
      saturation fraction (r = 0) [m^2/s].
    D_e_H_mode_min: Particle diffusivity of the H-mode transport branch at zero
      saturation fraction (r = 0) [m^2/s].
    pedestal_top_smoothing_width: Width of the smoothing kernel at the pedestal
      top.
  """

  set_pedestal: array_typing.BoolScalar
  mode: Mode = dataclasses.field(metadata={"static": True})
  use_formation_model_with_internal_boundary_condition: bool = (
      dataclasses.field(metadata={"static": True})
  )
  transition_time_width: array_typing.FloatScalar
  P_LH_hysteresis_factor: array_typing.FloatScalar
  include_dW_dt_in_P_SOL: bool = dataclasses.field(metadata={"static": True})
  explicit_pedestal: bool = dataclasses.field(metadata={"static": True})
  pedestal_profile_form: PedestalProfileForm = dataclasses.field(
      metadata={"static": True}
  )
  formation: FormationRuntimeParams
  saturation: SaturationRuntimeParams
  chi_H_mode_max: array_typing.FloatScalar
  D_e_H_mode_max: array_typing.FloatScalar
  chi_H_mode_min: array_typing.FloatScalar
  D_e_H_mode_min: array_typing.FloatScalar
  pedestal_top_smoothing_width: array_typing.FloatScalar
