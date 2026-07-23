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

"""Output of the pedestal model."""

import dataclasses
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src import state
from torax._src.geometry import geometry
from torax._src.internal_boundary_conditions import internal_boundary_conditions as internal_boundary_conditions_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.pedestal_model.saturation import base as saturation_base

# pylint: disable=invalid-name


def _build_smoothing_matrix(
    rho_face_norm: array_typing.FloatVectorFace,
    rho_norm_ped_top: array_typing.FloatScalar,
    smoothing_width: array_typing.FloatScalar,
    n_sigma: float = 2.0,
) -> jax.Array:
  """Builds a smoothing matrix for the pedestal top."""
  # Gaussian kernel with sigma = smoothing_width.
  kernel = jnp.exp(
      -jnp.log(2)
      * (rho_face_norm[:, jnp.newaxis] - rho_face_norm) ** 2
      / (smoothing_width**2 + constants.CONSTANTS.eps)
  )
  # Smoothing matrix is only non-identity within n_sigma of the pedestal top.
  mask = jnp.abs(rho_face_norm - rho_norm_ped_top) < (n_sigma * smoothing_width)
  # Zero out restricted columns so active points don't read from them
  masked_kernel = jnp.where(mask, kernel, 0.0)
  # Replace restricted rows with identity so they are unmodified (pass-through)
  smoothing_matrix = jnp.where(
      mask[:, jnp.newaxis], masked_kernel, jnp.eye(kernel.shape[0])
  )
  # Normalize the smoothing matrix
  smoothing_matrix /= jnp.sum(smoothing_matrix, axis=1, keepdims=True)
  # Remove small values
  smoothing_matrix = jnp.where(smoothing_matrix < 1e-3, 0.0, smoothing_matrix)
  # Re-normalize
  smoothing_matrix /= jnp.sum(smoothing_matrix, axis=1, keepdims=True)
  return smoothing_matrix


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalModelOutput:
  """Output of a PedestalModel.

  Attributes:
    rho_norm_ped_top: The requested location of the pedestal top in rho_norm,
      not quantized to either the cell or face grid.
    T_i_ped: The ion temperature at the pedestal top in keV.
    T_e_ped: The electron temperature at the pedestal top in keV.
    n_e_ped: The electron density at the pedestal top in m^-3.
    H_mode_fraction: Blend weight g in [0, 1] between the unmodified
      transport (g=0, L-mode) and the H-mode transport branch (g=1, fully
      established H-mode edge). Only used if the pedestal is in
      ADAPTIVE_TRANSPORT mode; see `modify_core_transport`.
    saturation_fraction: Per-channel saturation fraction r in (0, 1) from the
      saturation model. Only used if the pedestal is in ADAPTIVE_TRANSPORT
      mode; see `modify_core_transport`.
  """

  rho_norm_ped_top: array_typing.FloatScalar
  T_i_ped: array_typing.FloatScalar
  T_e_ped: array_typing.FloatScalar
  n_e_ped: array_typing.FloatScalar
  H_mode_fraction: array_typing.FloatScalar = dataclasses.field(
      default_factory=lambda: jnp.array(0.0, dtype=jax_utils.get_dtype())
  )
  saturation_fraction: saturation_base.SaturationFraction = dataclasses.field(
      default_factory=saturation_base.SaturationFraction.default
  )

  def to_internal_boundary_conditions(
      self,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles | None = None,
      pedestal_profile_form: pedestal_runtime_params_lib.PedestalProfileForm = (
          pedestal_runtime_params_lib.PedestalProfileForm.SET_AT_PED_TOP
      ),
  ) -> internal_boundary_conditions_lib.InternalBoundaryConditions:
    """Convert the pedestal model output to internal boundary conditions.

    When pedestal_profile_form is MTANH and core_profiles is provided, generates
    an mtanh-shaped profile across the pedestal region using the formula:
      a(ψ) = a_sep + a₀·[tanh(1) - tanh(2(ψ - ψ_mid)/Δ)]
    where a denotes either T_i, T_e, or n_e, and Δ is derived from
    rho_norm_ped_top via the ψ_N(ρ) mapping:
      ψ_top = ψ_N(rho_norm_ped_top), Δ = (1 - ψ_top) / 1.5

    When pedestal_profile_form is SET_AT_PED_TOP, falls back to a single-point
    mask at the nearest cell to rho_norm_ped_top.

    Args:
      geo: Geometry object for the grid.
      core_profiles: Core profiles, needed for ψ_N mapping and separatrix values
        when using mtanh profiles.
      pedestal_profile_form: Controls the shape of the pedestal profile.

    Returns:
      Internal boundary conditions for T_i, T_e, n_e.
    """
    match pedestal_profile_form:
      case pedestal_runtime_params_lib.PedestalProfileForm.MTANH:
        if core_profiles is None:
          raise ValueError(
              "core_profiles must be provided when pedestal_profile_form"
              " is MTANH."
          )
        return self._tanh_internal_boundary_conditions(geo, core_profiles)
      case pedestal_runtime_params_lib.PedestalProfileForm.SET_AT_PED_TOP:
        # Single-point mask: pin values at the nearest cell to ped top.
        rho_norm_ped_top_idx = jnp.argmin(
            jnp.abs(geo.rho_norm - self.rho_norm_ped_top)
        )
        pedestal_mask = (
            jnp.zeros_like(geo.rho, dtype=bool)
            .at[rho_norm_ped_top_idx]
            .set(True)
        )
        return internal_boundary_conditions_lib.InternalBoundaryConditions(
            T_i=jnp.where(pedestal_mask, self.T_i_ped, 0.0),
            T_e=jnp.where(pedestal_mask, self.T_e_ped, 0.0),
            n_e=jnp.where(pedestal_mask, self.n_e_ped, 0.0),
        )

  def _tanh_internal_boundary_conditions(
      self,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> internal_boundary_conditions_lib.InternalBoundaryConditions:
    """Compute mtanh-shaped internal boundary conditions.

    The mtanh width Δ is derived from rho_norm_ped_top using the ψ_N(ρ)
    mapping. In the mtanh geometry, the pedestal top is at ψ_top = 1 - 1.5Δ,
    so Δ = (1 - ψ_top) / 1.5.

    The profile is then:
      q(ψ) = q_sep + a₀·[tanh(1) - tanh(2(ψ - ψ_mid)/Δ)]
    where:
      ψ_mid = 1 - Δ/2 (center of the tanh)
      a₀ = (q_top - q_sep) / (tanh(1) + tanh(2))

    Values are only applied for cells at or beyond ρ_ped_top (the pedestal
    region). Core cells get 0.0 (no IBC contribution).

    Args:
      geo: Geometry object for the grid.
      core_profiles: Core profiles for ψ_N mapping and separatrix values.

    Returns:
      Internal boundary conditions with mtanh-shaped profiles for T_i, T_e, n_e.
    """
    # Get ψ_N at each cell grid point.
    psi_face = core_profiles.psi.face_value()
    psi_norm_cell = (core_profiles.psi.value - psi_face[0]) / (  # pyrefly: ignore[bad-index]
        psi_face[-1] - psi_face[0]  # pyrefly: ignore[bad-index]
    )

    # Derive Δ from rho_norm_ped_top via ψ_N mapping.
    # Use psi at the nearest cell to rho_ped_top (not interpolated) so that
    # the mtanh formula evaluates to exactly q_top at that cell.
    rho_norm_ped_top_idx = jnp.argmin(
        jnp.abs(geo.rho_norm - self.rho_norm_ped_top)
    )
    psi_top = psi_norm_cell[rho_norm_ped_top_idx]
    delta = (1.0 - psi_top) / 1.5
    psi_mid = 1.0 - delta / 2.0

    # Separatrix values from the rightmost face of core_profiles.
    T_i_sep = core_profiles.T_i.right_face_value
    T_e_sep = core_profiles.T_e.right_face_value
    n_e_sep = core_profiles.n_e.right_face_value

    # Pedestal region mask: cells at or beyond rho_norm_ped_top.
    ped_mask = geo.rho_norm >= self.rho_norm_ped_top

    def _mtanh_profile(val_top, val_sep):
      """Evaluate mtanh for one quantity."""
      tanh1 = jnp.tanh(1.0)
      tanh2 = jnp.tanh(2.0)
      a0 = (val_top - val_sep) / (tanh1 + tanh2)
      profile = val_sep + a0 * (
          tanh1 - jnp.tanh(2.0 * (psi_norm_cell - psi_mid) / delta)
      )
      return jnp.where(ped_mask, profile, 0.0)

    return internal_boundary_conditions_lib.InternalBoundaryConditions(
        T_i=_mtanh_profile(self.T_i_ped, T_i_sep),
        T_e=_mtanh_profile(self.T_e_ped, T_e_sep),
        n_e=_mtanh_profile(self.n_e_ped, n_e_sep),
    )

  def modify_core_transport(
      self,
      core_transport: state.CoreTransport,
      geo: geometry.Geometry,
      pedestal_runtime_params: pedestal_runtime_params_lib.RuntimeParams,
  ) -> state.CoreTransport:
    """Modify transport coefficients in the entire pedestal region.

    Each transport channel (chi_i, chi_e, D_e, and the pinch v_e) blends its
    raw (L-mode) coefficient with a bounded H-mode branch:

      coeff_ped = (1 - g) * coeff_raw + g * (residual + r * (cap - residual))

    where g is the H-mode fraction (formation) and r is the channel's
    saturation fraction (saturation response). Since g and r are bounded in
    [0, 1] and the blend is linear in each, the transport sensitivity to the
    saturation response is bounded by (cap - residual) / (4 * response_width)
    -- required by the Newton-Raphson solver, which re-evaluates the
    transport blend inside its residual.

    Each channel's diagnostic turbulence breakdowns (Bohm, ITG, ...) and
    Pereverzev-Corrigan terms track the same total coefficient, so they are
    scaled by the channel's relative suppression factor (1 - g) + g * r
    instead of being floored/capped themselves; a Pereverzev
    diffusion/convection pair shares one factor, preserving its exact flux
    cancellation. Neoclassical transport is not affected.

    Args:
      core_transport: The core transport coefficients to modify.
      geo: The geometry of the torus.
      pedestal_runtime_params: The runtime parameters of the pedestal model.

    Returns:
      The modified core transport coefficients.
    """
    # We are using the face grid here, since transport coefficients are
    # applied on the face grid.

    # TODO(b/485147781):  In the case where we have a CombinedTransportModel
    # with a pedestal transport model specified, we are currently scaling
    # all the coefficients in the pedestal region, whereas we should be only
    # scaling the turbulent coeffs and leaving the pedestal coeffs alone.
    pedestal_active_mask_face = geo.rho_face_norm > self.rho_norm_ped_top

    smoothing_matrix = _build_smoothing_matrix(
        geo.rho_face_norm,
        self.rho_norm_ped_top,
        pedestal_runtime_params.pedestal_top_smoothing_width,
    )

    def smooth(
        blended: array_typing.FloatVectorFace, raw: array_typing.FloatVectorFace
    ) -> array_typing.FloatVectorFace:
      """Restricts a blend to the pedestal region and smooths its edge."""
      blended = jnp.where(pedestal_active_mask_face, blended, raw)
      return jnp.dot(smoothing_matrix, blended)

    g = self.H_mode_fraction

    def blend_channel(
        total_field: str,
        component_fields: tuple[str, ...],
        r: array_typing.FloatScalar,
        residual: array_typing.FloatScalar,
        cap: array_typing.FloatScalar,
    ) -> dict[str, array_typing.FloatVectorFace]:
      """Blends one transport channel's total field and diagnostic breakdowns.

      Args:
        total_field: Name of the CoreTransport field solved for in the state
          equations (e.g. 'chi_face_ion').
        component_fields: Names of the diagnostic turbulence breakdowns
          (Bohm, ITG, ...) and Pereverzev-Corrigan terms for this channel.
        r: This channel's saturation fraction, in (0, 1).
        residual: H-mode transport at r = 0 (the floor of the H-mode branch).
        cap: H-mode transport at r = 1 (the ceiling of the H-mode branch).

      Returns:
        A field name -> blended value mapping for `dataclasses.replace`.
      """
      raw_total = getattr(core_transport, total_field)
      h_mode_value = residual + r * (cap - residual)
      blended_total = (1.0 - g) * raw_total + g * h_mode_value

      suppression_factor = (1.0 - g) + g * r
      updates = {total_field: smooth(blended_total, raw_total)}
      for field in component_fields:
        raw = getattr(core_transport, field)
        # Diagnostic breakdowns are only populated by some transport models
        # (e.g. bohm/gyrobohm terms are QLKNN-specific); fields the active
        # transport model doesn't produce stay None and are left untouched.
        if raw is not None:
          updates[field] = smooth(raw * suppression_factor, raw)
      return updates

    saturation = self.saturation_fraction
    params = pedestal_runtime_params
    updates = {}
    updates.update(blend_channel(
        total_field="chi_face_ion",
        component_fields=(
            "chi_face_ion_bohm",
            "chi_face_ion_gyrobohm",
            "chi_face_ion_itg",
            "chi_face_ion_tem",
            "chi_face_ion_pereverzev",
            "full_v_heat_face_ion_pereverzev",
        ),
        r=saturation.chi_i_saturation_fraction,
        residual=params.chi_H_mode_min,
        cap=params.chi_H_mode_max,
    ))
    updates.update(blend_channel(
        total_field="chi_face_el",
        component_fields=(
            "chi_face_el_bohm",
            "chi_face_el_gyrobohm",
            "chi_face_el_itg",
            "chi_face_el_tem",
            "chi_face_el_etg",
            "chi_face_el_pereverzev",
            "full_v_heat_face_el_pereverzev",
        ),
        r=saturation.chi_e_saturation_fraction,
        residual=params.chi_H_mode_min,
        cap=params.chi_H_mode_max,
    ))
    updates.update(blend_channel(
        total_field="d_face_el",
        # The Pereverzev particle pair (d_face_el_pereverzev,
        # v_face_el_pereverzev) are a diffusion/convection pair that must
        # share one factor to cancel exactly; that shared factor is D_e's r,
        # so the convection member is included here rather than under the
        # pinch channel below.
        component_fields=(
            "d_face_el_itg",
            "d_face_el_tem",
            "d_face_el_pereverzev",
            "v_face_el_pereverzev",
        ),
        r=saturation.D_e_saturation_fraction,
        residual=params.D_e_H_mode_min,
        cap=params.D_e_H_mode_max,
    ))
    updates.update(blend_channel(
        total_field="v_face_el",
        component_fields=("v_face_el_itg", "v_face_el_tem"),
        # The turbulent pinch is suppressed to zero within the H-mode branch
        # and has no saturation fraction of its own (raising D alone shifts
        # the v/D ratio and controls the pedestal density height; see
        # `saturation.base.SaturationFraction.D_e_saturation_fraction`);
        # residual = cap = 0 makes the total collapse to
        # (1 - g) * coeff_raw for any r.
        r=0.0,
        residual=0.0,
        cap=0.0,
    ))

    return dataclasses.replace(core_transport, **updates)
