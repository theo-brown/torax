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
"""Pedestal model with height set by a ballooning-stability limit."""

import dataclasses

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.physics import formulas
from typing_extensions import override


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(pedestal_runtime_params_lib.RuntimeParams):
  """Runtime params for the SetAlphaCritDensityPedestalModel."""

  alpha_crit: array_typing.FloatScalar
  n_e_ped: array_typing.FloatScalar
  T_i_T_e_ratio: array_typing.FloatScalar
  rho_norm_ped_top: array_typing.FloatScalar
  n_e_ped_is_fGW: array_typing.BoolScalar


@dataclasses.dataclass(frozen=True, eq=False)
class SetAlphaCritDensityPedestalModel(pedestal_model.PedestalModel):
  r"""Pedestal model with height set by an infinite-n ballooning mode limit.

  The pedestal-top pressure is set to the value at which the local
  infinite-n ideal ballooning mode normalized pressure gradient, evaluated
  between the pedestal top and the separatrix, reaches a user-provided
  critical value ``alpha_crit``:

  .. math::

    \alpha_{crit} = \frac{2 \mu_0 R_{major} q_{ped}^2}{B_0^2}
      \frac{p_{ped} - p_{sep}}{r_{sep} - r_{ped}}

  where :math:`q_{ped}` is the safety factor at the pedestal top, :math:`r`
  is the midplane minor radius, and :math:`p_{ped}`, :math:`p_{sep}` are the
  total (electron + ion + impurity) pressures at the pedestal top and
  separatrix. This uses the same alpha_MHD convention (reference length
  :math:`R_{major}`, radial coordinate the midplane minor radius) as
  ``quasilinear_transport_model.calculate_alpha``, which is used by the
  QuaLiKiz- and TGLF-based transport models.

  ``alpha_crit`` is not computed by TORAX -- it must be supplied externally
  (e.g. from a local ideal ballooning mode stability calculation).
  """

  @override
  def _call_implementation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
  ) -> pedestal_model_output.PedestalModelOutput:
    assert isinstance(runtime_params.pedestal, RuntimeParams)
    consts = constants.CONSTANTS

    # Convert n_e_ped to reference units.
    # Ip in MA. a_minor in m. nGW in m^-3.
    nGW = (
        runtime_params.profile_conditions.Ip
        / 1e6  # Convert to MA.
        / (jnp.pi * geo.a_minor**2)
        * 1e20
    )
    n_e_ped = jnp.where(
        runtime_params.pedestal.n_e_ped_is_fGW,
        runtime_params.pedestal.n_e_ped * nGW,
        runtime_params.pedestal.n_e_ped,
    )

    rho_norm_ped_top = runtime_params.pedestal.rho_norm_ped_top
    temperature_ratio = runtime_params.pedestal.T_i_T_e_ratio

    # Composition (Z_eff, Z_i, Z_impurity) at the pedestal top. Used both for
    # the pedestal-top pressure and, as an approximation, for the separatrix
    # pressure, since the impurity/main-ion mix is not separately resolved at
    # the separatrix boundary condition.
    ped_idx = jnp.abs(geo.rho_norm - rho_norm_ped_top).argmin()
    Z_eff_ped = jnp.take(core_profiles.Z_eff, ped_idx)
    Z_i_ped = jnp.take(core_profiles.Z_i, ped_idx)
    Z_impurity_ped = jnp.take(core_profiles.Z_impurity, ped_idx)
    dilution_factor_ped = jnp.where(
        Z_eff_ped == 1.0,
        1.0,
        formulas.calculate_main_ion_dilution_factor(
            Z_i_ped, Z_impurity_ped, Z_eff_ped
        ),
    )
    # Guard against division by zero when Z_impurity_ped is zero or
    # degenerate (pure plasma case).
    safe_Z_impurity_ped = jnp.where(Z_eff_ped == 1.0, 1.0, Z_impurity_ped)

    # Separatrix total pressure P = P_e + P_i + P_imp, from the existing
    # boundary condition densities/temperatures.
    n_e_sep = runtime_params.profile_conditions.n_e_right_bc
    T_e_sep = runtime_params.profile_conditions.T_e_right_bc
    T_i_sep = runtime_params.profile_conditions.T_i_right_bc
    n_i_sep = dilution_factor_ped * n_e_sep
    n_impurity_sep = jnp.where(
        Z_eff_ped == 1.0,
        0.0,
        (n_e_sep - Z_i_ped * n_i_sep) / safe_Z_impurity_ped,
    )
    p_sep = (
        n_e_sep * T_e_sep + n_i_sep * T_i_sep + n_impurity_sep * T_i_sep
    ) * consts.keV_to_J

    # Safety factor and midplane minor radius at the pedestal top and at the
    # separatrix, used to evaluate the local ballooning parameter alpha.
    # Interpolated (rather than snapped to the nearest face grid point) so
    # that r_sep - r_ped does not spuriously vanish on coarse grids.
    q_ped = jnp.interp(rho_norm_ped_top, geo.rho_face_norm, core_profiles.q_face)
    r_ped = jnp.interp(rho_norm_ped_top, geo.rho_face_norm, geo.r_mid_face)
    r_sep = geo.r_mid_face[-1]

    # Solve for the pedestal-top pressure at which alpha == alpha_crit:
    #   alpha_crit = (2 * mu_0 * R_major * q_ped^2 / B_0^2)
    #                * (p_ped - p_sep) / (r_sep - r_ped)
    alpha_crit = runtime_params.pedestal.alpha_crit
    p_ped = p_sep + alpha_crit * geo.B_0**2 * (r_sep - r_ped) / (
        2 * consts.mu_0 * geo.R_major * q_ped**2
    )

    # Calculate T_e_ped given n_e_ped, T_i_T_e_ratio and p_ped.
    # P = P_e + P_i + P_imp = T_e*n_e + T_i*n_i + T_i*n_imp, with
    # T_i = ratio * T_e.
    n_i_ped = dilution_factor_ped * n_e_ped
    n_impurity_ped = jnp.where(
        Z_eff_ped == 1.0,
        0.0,
        (n_e_ped - Z_i_ped * n_i_ped) / safe_Z_impurity_ped,
    )
    T_e_ped = (
        p_ped
        / (
            n_e_ped  # Electron pressure contribution.
            + temperature_ratio * n_i_ped  # Ion pressure contribution.
            + temperature_ratio
            * n_impurity_ped  # Impurity pressure contribution.
        )
        / consts.keV_to_J
    )
    T_i_ped = temperature_ratio * T_e_ped

    return pedestal_model_output.PedestalModelOutput(
        n_e_ped=n_e_ped,
        T_i_ped=T_i_ped,
        T_e_ped=T_e_ped,
        rho_norm_ped_top=rho_norm_ped_top,
    )
