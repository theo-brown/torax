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

"""Pydantic config for Pedestal."""

import abc
import copy
from typing import Annotated, Any, Literal, TypeAlias
import chex
import pydantic
from torax._src import array_typing
from torax._src.pedestal_model import no_pedestal
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import runtime_params
from torax._src.pedestal_model import set_pped_tpedratio_nped
from torax._src.pedestal_model import set_tped_nped
from torax._src.pedestal_model.formation import power_scaling_formation_model
from torax._src.pedestal_model.saturation import alpha_critical_saturation_model
from torax._src.pedestal_model.saturation import profile_value_saturation_model
from torax._src.physics import scaling_laws
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions

# pylint: disable=invalid-name


class PowerScalingFormation(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Configuration for power scaling formation model.

  This formation model raises the barrier fraction g (the blend weight
  between L-mode transport and the barrier transport branch) when P_SOL >
  P_LH, where P_LH is calculated from an appropriate scaling law.

  The formula is
    g = sigmoid(sharpness * [x - offset]),
  where x is the normalized power excess
    (P_SOL - P_LH * P_LH_prefactor) / (P_LH * P_LH_prefactor).

  Attributes:
    sharpness: Scaling factor applied to the argument of the sigmoid function,
      setting the sharpness of the smooth formation window. Decrease for a
      smoother formation, which may be more numerically stable but forms the
      barrier more gradually around the threshold.
    offset: Bias applied to the argument of the sigmoid function, setting the
      dimensionless offset of the formation window. Increase to start formation
      at a higher P_SOL.
    P_LH_prefactor: Dimensionless multiplier for P_LH. Increase to scale up
      P_LH, and therefore start the L-H transition at a higher P_SOL.
  """

  sharpness: pydantic.PositiveFloat = 100.0
  offset: Annotated[
      array_typing.FloatScalar, pydantic.Field(ge=-10.0, le=10.0)
  ] = 0.0
  P_LH_prefactor: pydantic.PositiveFloat = 1.0

  @abc.abstractmethod
  def build_formation_model(
      self,
  ) -> power_scaling_formation_model.PowerScalingFormationModel:
    """Builds the formation model."""

  @abc.abstractmethod
  def build_runtime_params(
      self, t: chex.Numeric
  ) -> power_scaling_formation_model.PowerScalingFormationRuntimeParams:
    """Builds the runtime params."""


class MartinScalingFormation(PowerScalingFormation):
  """Configuration for Martin scaling formation model.

  This formation model triggers a reduction in pedestal transport when P_SOL >
  P_LH, where P_LH is calculated from the Martin scaling law. See
  `PowerScalingFormation` for more details.
  """

  model_name: Annotated[
      Literal["martin_scaling"], torax_pydantic.JAX_STATIC
  ] = "martin_scaling"

  def build_formation_model(
      self,
  ) -> power_scaling_formation_model.PowerScalingFormationModel:
    return power_scaling_formation_model.PowerScalingFormationModel(
        scaling_law=scaling_laws.PLHScalingLaw.MARTIN,
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> power_scaling_formation_model.PowerScalingFormationRuntimeParams:
    del t
    return power_scaling_formation_model.PowerScalingFormationRuntimeParams(
        sharpness=self.sharpness,
        offset=self.offset,
        P_LH_prefactor=self.P_LH_prefactor,
    )


class DelabieScalingFormation(PowerScalingFormation):
  """Configuration for Delabie scaling formation model.

  This formation model triggers a reduction in pedestal transport when P_SOL >
  P_LH, where P_LH is calculated from the Delabie scaling law. See
  `PowerScalingFormation` for more details.
  """

  model_name: Annotated[
      Literal["delabie_scaling"], torax_pydantic.JAX_STATIC
  ] = "delabie_scaling"

  divertor_configuration: Annotated[
      scaling_laws.DivertorConfiguration, torax_pydantic.JAX_STATIC
  ] = scaling_laws.DivertorConfiguration.HT

  def build_formation_model(
      self,
  ) -> power_scaling_formation_model.PowerScalingFormationModel:
    return power_scaling_formation_model.PowerScalingFormationModel(
        scaling_law=scaling_laws.PLHScalingLaw.DELABIE,
        divertor_configuration=self.divertor_configuration,
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> power_scaling_formation_model.PowerScalingFormationRuntimeParams:
    del t
    return power_scaling_formation_model.PowerScalingFormationRuntimeParams(
        sharpness=self.sharpness,
        offset=self.offset,
        P_LH_prefactor=self.P_LH_prefactor,
    )


class ProfileValueSaturation(torax_pydantic.BaseModelFrozen):
  """Configuration for ProfileValueSaturation model.

  This is the target-based saturation signal: each transport channel's
  proximity-to-limit signal is the relative deviation of the sensed profile
  value from the target requested by the pedestal model implementation,
    x = current / target - 1.
  The heat channels chi_e and chi_i sense T_e and T_i at the pedestal-top
  face against T_e_ped and T_i_ped; the particle diffusivity channel D_e
  senses the (smooth) maximum of n_e over the pedestal region against
  n_e_ped, so that density pileup anywhere inside the pedestal (e.g. from
  edge fueling against suppressed transport) activates the response. This
  supports pedestal models that ask for specific pedestal-top values, e.g.
  EPED-style predictions of T_e_ped.

  The signals are mapped to barrier openness by the shared bounded response
    r = sigmoid((x - offset) / response_width),
  so the regulated profile value settles within a band of roughly
  +/- response_width (relative) around target * (1 + offset), at the point
  where the barrier transport carries the incoming flux. The particle pinch
  v_e has no saturation signal (within the barrier branch it is suppressed
  to zero): the steady-state density profile shape is set by the ratio v/D,
  so raising D alone shifts that ratio and lets the feedback regulate the
  pedestal density height, whereas raising D and v together would only
  change the relaxation timescale.

  Note that saturation is one-sided within the barrier branch: it can raise
  transport at most to the local raw (turbulent) transport level, i.e.
  throttle the pedestal at the target. In particular, the achieved pedestal
  density cannot exceed what the edge particle fueling (and any inward pinch)
  can sustain; with insufficient fueling n_e_ped is not reached. Conversely,
  if the incoming flux exceeds what that cap can exhaust, the profile settles
  above the target with the barrier fully open.

  Attributes:
    offset: Dimensionless offset of the heat channel saturation response: the
      relative deviation from target at which the barrier is half open.
    response_width: Width of the heat channel saturation response, in units of
      relative deviation from target. Decrease for tighter regulation of the
      pedestal-top values, at the cost of a steeper (stiffer) transport
      response for the solver.
    density_offset: As `offset`, for the particle diffusivity channel driven
      by the n_e deviation from n_e_ped.
    density_response_width: As `response_width`, for the particle diffusivity
      channel.
  """

  model_name: Annotated[Literal["profile_value"], torax_pydantic.JAX_STATIC] = (
      "profile_value"
  )
  offset: Annotated[
      array_typing.FloatScalar, pydantic.Field(ge=-10.0, le=10.0)
  ] = 0.0
  response_width: pydantic.PositiveFloat = 0.05
  density_offset: Annotated[
      array_typing.FloatScalar, pydantic.Field(ge=-10.0, le=10.0)
  ] = 0.0
  density_response_width: pydantic.PositiveFloat = 0.05

  def build_saturation_model(
      self,
  ) -> profile_value_saturation_model.ProfileValueSaturationModel:
    return profile_value_saturation_model.ProfileValueSaturationModel()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.SaturationRuntimeParams:
    del t
    return runtime_params.SaturationRuntimeParams(
        offset=self.offset,
        response_width=self.response_width,
        density_offset=self.density_offset,
        density_response_width=self.density_response_width,
    )


class AlphaCriticalSaturation(torax_pydantic.BaseModelFrozen):
  """Configuration for the alpha-critical saturation model.

  This is the stability-based saturation signal: it senses proximity of the
  s-alpha normalized pressure gradient alpha to its critical value
  alpha_crit = alpha_crit_multiplier * max(s, s_min), an approximation of the
  ideal ballooning first-stability boundary. The signal is
    x = max(alpha / alpha_crit) - 1,
  where the maximum is the smooth maximum over the pedestal region, and is
  shared by both heat channels and the particle diffusivity (alpha involves
  the total pressure gradient, so all channels relieve it); the pinch has no
  saturation signal.

  Combined with a power-scaling formation model, the L-H transition timing
  remains empirical (P_SOL vs P_LH scaling law) while the pedestal height
  emerges from MHD stability physics: any T_ped/n_e_ped targets carried by
  the pedestal model are ignored by this saturation model (only
  rho_norm_ped_top is used, defining the pedestal region).

  The signal is mapped to barrier openness by the shared bounded response
    r = sigmoid((x - offset) / response_width),
  so the pedestal pressure gradient settles within roughly a
  +/- response_width (relative) band around the stability boundary.

  Attributes:
    offset: Dimensionless offset of the saturation response in
      alpha / alpha_crit: the relative excess over the boundary at which the
      barrier is half open.
    response_width: Width of the saturation response, in units of relative
      deviation of alpha / alpha_crit from unity. Decrease to hold the
      gradient closer to the boundary, at the cost of a steeper (stiffer)
      transport response for the solver.
    alpha_crit_multiplier: Prefactor c_alpha of the s-alpha critical gradient
      alpha_crit = c_alpha * max(s, s_min).
    s_min: Magnetic shear floor in the critical alpha, avoiding a vanishing
      stability limit near zero shear.
  """

  model_name: Annotated[
      Literal["alpha_critical"], torax_pydantic.JAX_STATIC
  ] = "alpha_critical"
  offset: Annotated[
      array_typing.FloatScalar, pydantic.Field(ge=-10.0, le=10.0)
  ] = 0.0
  response_width: pydantic.PositiveFloat = 0.05
  alpha_crit_multiplier: pydantic.PositiveFloat = 0.6
  s_min: pydantic.PositiveFloat = 0.5

  def build_saturation_model(
      self,
  ) -> alpha_critical_saturation_model.AlphaCriticalSaturationModel:
    return alpha_critical_saturation_model.AlphaCriticalSaturationModel()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> alpha_critical_saturation_model.AlphaCriticalSaturationRuntimeParams:
    del t
    return alpha_critical_saturation_model.AlphaCriticalSaturationRuntimeParams(
        offset=self.offset,
        response_width=self.response_width,
        # The particle diffusivity channel shares the alpha-based signal and
        # response in this model.
        density_offset=self.offset,
        density_response_width=self.response_width,
        alpha_crit_multiplier=self.alpha_crit_multiplier,
        s_min=self.s_min,
    )


# For new formation and saturation models, add to these TypeAliases via Union.
FormationConfig: TypeAlias = DelabieScalingFormation | MartinScalingFormation
SaturationConfig: TypeAlias = ProfileValueSaturation | AlphaCriticalSaturation


class BasePedestal(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base class for pedestal models.

  Attributes:
    set_pedestal: Whether to use the pedestal model and set the pedestal. Can be
      time varying.
    mode: Defines how the pedestal is generated. Set to ADAPTIVE_TRANSPORT to
      set the pedestal by modifying the transport coefficients in the pedestal
      region, allowing the pedestal to self-consistently evolve. Set to
      INTERNAL_BOUNDARY_CONDITION to set the pedestal by adding a source/sink
      term at the pedestal top, forcing the pedestal top values to be as
      prescribed. use_formation_model_with_internal_boundary_condition: When
      True and mode is INTERNAL_BOUNDARY_CONDITION, enables state-dependent L-H
      transitions based on P_SOL vs P_LH comparison. When False,
      INTERNAL_BOUNDARY_CONDITION mode always applies the prescribed pedestal
      values (legacy behavior). Ignored when mode is ADAPTIVE_TRANSPORT.
    transition_time_width: Duration of the L-H or H-L transition ramp [s].
      During a transition, pedestal values are linearly interpolated between
      L-mode baseline and H-mode target values over this time window. Only used
      when use_formation_model_with_internal_boundary_condition is True.
    P_LH_hysteresis_factor: Hysteresis factor for H-L back transitions. When
      checking for an H-L transition, the L-H threshold power P_LH is multiplied
      by this factor, i.e. the back transition occurs when P_SOL < P_LH *
      P_LH_hysteresis_factor. A value less than 1 means that the plasma must
      lose more power to transition back to L-mode than was required to enter
      H-mode, which is the experimentally observed behavior. Must be in [0, 1].
      Only applicable when use_formation_model_with_internal_boundary_condition
      is True.
    include_dW_dt_in_P_SOL: Whether to include the dW/dt term in the P_SOL
      calculation used for comparing against P_LH. When False (default), uses
      P_heat (total auxiliary + Ohmic power - sinks) instead of P_SOL = P_heat -
      dW/dt. Excluding dW/dt avoids unphysical dithering during transients.
    formation_model: Configuration for the pedestal formation model.
    saturation_model: Configuration for the pedestal saturation model.

  The ADAPTIVE_TRANSPORT barrier branch has no free cap/residual parameters:
  at zero saturation openness (r = 0) each channel's transport sits at the
  local neoclassical level (chi_neo_i / chi_neo_e / D_neo_e, already computed
  by the neoclassical model, floored at `constants.CONSTANTS.eps` to avoid a
  vanishing diffusivity under full suppression); at full openness (r = 1) it
  reverts to whatever the turbulent transport model already predicts for the
  current profile at that face. Both bounds are therefore local, machine- and
  scenario-scaled quantities the existing models already compute, rather than
  hand-picked absolute diffusivities.
  """

  set_pedestal: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(False)
  )
  mode: Annotated[runtime_params.Mode, torax_pydantic.JAX_STATIC] = (
      runtime_params.Mode.INTERNAL_BOUNDARY_CONDITION
  )
  use_formation_model_with_internal_boundary_condition: Annotated[
      bool, torax_pydantic.JAX_STATIC
  ] = False
  transition_time_width: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.5)
  )
  P_LH_hysteresis_factor: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.8)
  )
  include_dW_dt_in_P_SOL: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  explicit_pedestal: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  pedestal_profile_form: Annotated[
      runtime_params.PedestalProfileForm, torax_pydantic.JAX_STATIC
  ] = runtime_params.PedestalProfileForm.SET_AT_PED_TOP
  formation_model: FormationConfig = torax_pydantic.ValidatedDefault(
      MartinScalingFormation()
  )
  saturation_model: SaturationConfig = torax_pydantic.ValidatedDefault(
      ProfileValueSaturation()
  )
  pedestal_top_smoothing_width: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.02)
  )

  @pydantic.model_validator(mode="before")
  @classmethod
  def _defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
    configurable_data = copy.deepcopy(data)
    if "formation_model" not in configurable_data:
      configurable_data["formation_model"] = {"model_name": "martin_scaling"}
    if "saturation_model" not in configurable_data:
      configurable_data["saturation_model"] = {"model_name": "profile_value"}
    # Set default model names.
    if "model_name" not in configurable_data["formation_model"]:
      configurable_data["formation_model"]["model_name"] = "martin_scaling"
    if "model_name" not in configurable_data["saturation_model"]:
      configurable_data["saturation_model"]["model_name"] = "profile_value"

    return configurable_data

  @pydantic.model_validator(mode="after")
  def _check_source_mode(self) -> typing_extensions.Self:
    if (
        self.use_formation_model_with_internal_boundary_condition
        and self.mode != runtime_params.Mode.INTERNAL_BOUNDARY_CONDITION
    ):
      raise ValueError(
          "use_formation_model_with_internal_boundary_condition can only be"
          " True when mode is INTERNAL_BOUNDARY_CONDITION"
      )
    if (
        self.use_formation_model_with_internal_boundary_condition
        and not isinstance(
            self.formation_model,
            PowerScalingFormation,
        )
    ):
      raise ValueError(
          "use_formation_model_with_internal_boundary_condition can only be"
          " True when formation_model is PowerScalingFormationModel"
      )
    return self

  @abc.abstractmethod
  def build_pedestal_model(self) -> pedestal_model.PedestalModel:
    """Builds the pedestal model."""

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.RuntimeParams:
    """Builds the runtime params."""
    return runtime_params.RuntimeParams(
        set_pedestal=self.set_pedestal.get_value(t),
        mode=self.mode,
        use_formation_model_with_internal_boundary_condition=self.use_formation_model_with_internal_boundary_condition,
        transition_time_width=self.transition_time_width.get_value(t),
        P_LH_hysteresis_factor=self.P_LH_hysteresis_factor.get_value(t),
        include_dW_dt_in_P_SOL=self.include_dW_dt_in_P_SOL,
        explicit_pedestal=self.explicit_pedestal,
        pedestal_profile_form=self.pedestal_profile_form,
        formation=self.formation_model.build_runtime_params(t),
        saturation=self.saturation_model.build_runtime_params(t),
        pedestal_top_smoothing_width=self.pedestal_top_smoothing_width.get_value(
            t
        ),
    )


class SetPpedTpedRatioNped(BasePedestal):
  """Model for direct specification of pressure, temperature ratio, and density.

  Attributes:
    P_ped: The plasma pressure at the pedestal [Pa].
    P_ped_multiplier: Multiplier for the pedestal pressure (mostly used for
      sensitivity analysis) [dimensionless].
    n_e_ped: The electron density at the pedestal [m^-3] or fGW.
    n_e_ped_is_fGW: Whether the electron density at the pedestal is in units of
      fGW.
    T_i_T_e_ratio: Ratio of the ion and electron temperature at the pedestal
      [dimensionless].
    rho_norm_ped_top: The location of the pedestal top.
  """

  model_name: Annotated[
      Literal["set_P_ped_n_ped"], torax_pydantic.JAX_STATIC
  ] = "set_P_ped_n_ped"
  P_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(1e5)
  P_ped_multiplier: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  n_e_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.7e20
  )
  n_e_ped_is_fGW: bool = False
  T_i_T_e_ratio: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  rho_norm_ped_top: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.91)
  )

  def build_pedestal_model(
      self,
  ) -> (
      set_pped_tpedratio_nped.SetPressureTemperatureRatioAndDensityPedestalModel
  ):
    return set_pped_tpedratio_nped.SetPressureTemperatureRatioAndDensityPedestalModel(
        formation_model=self.formation_model.build_formation_model(),
        saturation_model=self.saturation_model.build_saturation_model(),
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> set_pped_tpedratio_nped.RuntimeParams:
    base_runtime_params = super().build_runtime_params(t)
    return set_pped_tpedratio_nped.RuntimeParams(
        set_pedestal=base_runtime_params.set_pedestal,
        mode=base_runtime_params.mode,
        use_formation_model_with_internal_boundary_condition=base_runtime_params.use_formation_model_with_internal_boundary_condition,
        transition_time_width=base_runtime_params.transition_time_width,
        P_LH_hysteresis_factor=base_runtime_params.P_LH_hysteresis_factor,
        include_dW_dt_in_P_SOL=base_runtime_params.include_dW_dt_in_P_SOL,
        explicit_pedestal=base_runtime_params.explicit_pedestal,
        pedestal_profile_form=base_runtime_params.pedestal_profile_form,
        P_ped=self.P_ped.get_value(t),
        P_ped_multiplier=self.P_ped_multiplier.get_value(t),
        n_e_ped=self.n_e_ped.get_value(t),
        n_e_ped_is_fGW=self.n_e_ped_is_fGW,
        T_i_T_e_ratio=self.T_i_T_e_ratio.get_value(t),
        rho_norm_ped_top=self.rho_norm_ped_top.get_value(t),
        formation=base_runtime_params.formation,
        saturation=base_runtime_params.saturation,
        pedestal_top_smoothing_width=self.pedestal_top_smoothing_width.get_value(
            t
        ),
    )


class SetTpedNped(BasePedestal):
  """A basic version of the pedestal model that uses direct specification.

  Attributes:
    n_e_ped: The electron density at the pedestal [m^-3] or fGW.
    n_e_ped_is_fGW: Whether the electron density at the pedestal is in units of
      fGW.
    T_i_ped: Ion temperature at the pedestal [keV].
    T_e_ped: Electron temperature at the pedestal [keV].
    rho_norm_ped_top: The location of the pedestal top.
  """

  model_name: Annotated[
      Literal["set_T_ped_n_ped"], torax_pydantic.JAX_STATIC
  ] = "set_T_ped_n_ped"
  n_e_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.7e20
  )
  n_e_ped_is_fGW: bool = False
  # TODO(b/434175938): Consider extending to TimeVaryingArray
  T_i_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      5.0
  )
  T_e_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      5.0
  )
  # TODO(b/517487852): Add pedestal height multiplier.
  rho_norm_ped_top: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.91)
  )

  def build_pedestal_model(
      self,
  ) -> set_tped_nped.SetTemperatureDensityPedestalModel:
    return set_tped_nped.SetTemperatureDensityPedestalModel(
        formation_model=self.formation_model.build_formation_model(),
        saturation_model=self.saturation_model.build_saturation_model(),
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> set_tped_nped.RuntimeParams:
    base_runtime_params = super().build_runtime_params(t)
    return set_tped_nped.RuntimeParams(
        set_pedestal=base_runtime_params.set_pedestal,
        mode=base_runtime_params.mode,
        use_formation_model_with_internal_boundary_condition=base_runtime_params.use_formation_model_with_internal_boundary_condition,
        transition_time_width=base_runtime_params.transition_time_width,
        P_LH_hysteresis_factor=base_runtime_params.P_LH_hysteresis_factor,
        include_dW_dt_in_P_SOL=base_runtime_params.include_dW_dt_in_P_SOL,
        explicit_pedestal=base_runtime_params.explicit_pedestal,
        pedestal_profile_form=base_runtime_params.pedestal_profile_form,
        n_e_ped=self.n_e_ped.get_value(t),
        n_e_ped_is_fGW=self.n_e_ped_is_fGW,
        T_i_ped=self.T_i_ped.get_value(t),
        T_e_ped=self.T_e_ped.get_value(t),
        rho_norm_ped_top=self.rho_norm_ped_top.get_value(t),
        formation=base_runtime_params.formation,
        saturation=base_runtime_params.saturation,
        pedestal_top_smoothing_width=self.pedestal_top_smoothing_width.get_value(
            t
        ),
    )


class NoPedestal(BasePedestal):
  """A pedestal model for when there is no pedestal.

  Note that setting `set_pedestal` to True with a NoPedestal model is the
  equivalent of setting it to False.
  """

  model_name: Annotated[Literal["no_pedestal"], torax_pydantic.JAX_STATIC] = (
      "no_pedestal"
  )

  def build_pedestal_model(
      self,
  ) -> no_pedestal.NoPedestal:
    return no_pedestal.NoPedestal(
        formation_model=self.formation_model.build_formation_model(),
        saturation_model=self.saturation_model.build_saturation_model(),
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.RuntimeParams:
    base_runtime_params = super().build_runtime_params(t)
    return runtime_params.RuntimeParams(
        set_pedestal=base_runtime_params.set_pedestal,
        mode=base_runtime_params.mode,
        use_formation_model_with_internal_boundary_condition=base_runtime_params.use_formation_model_with_internal_boundary_condition,
        transition_time_width=base_runtime_params.transition_time_width,
        P_LH_hysteresis_factor=base_runtime_params.P_LH_hysteresis_factor,
        include_dW_dt_in_P_SOL=base_runtime_params.include_dW_dt_in_P_SOL,
        explicit_pedestal=base_runtime_params.explicit_pedestal,
        pedestal_profile_form=base_runtime_params.pedestal_profile_form,
        formation=base_runtime_params.formation,
        saturation=base_runtime_params.saturation,
        pedestal_top_smoothing_width=self.pedestal_top_smoothing_width.get_value(
            t
        ),
    )


PedestalConfig = SetPpedTpedRatioNped | SetTpedNped | NoPedestal
