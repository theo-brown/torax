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

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class BarrierSignals:
  """Per-channel proximity-to-limit signals from a saturation model.

  Each signal x is dimensionless and normalized so that x = 0 at the limit
  the saturation model regulates towards (a prescribed pedestal-top target,
  a stability boundary, ...), x < 0 below the limit and x > 0 beyond it.
  The signals are mapped to barrier openness by the shared bounded response
  r = sigmoid((x - offset) / response_width) in
  `PedestalModel.compute_barrier_state`; saturation models only choose the
  signal, not the response shape.

  Attributes:
    chi_i_signal: Signal driving the ion heat channel.
    chi_e_signal: Signal driving the electron heat channel.
    D_e_signal: Signal driving the particle diffusivity channel. There is
      deliberately no pinch signal: the steady-state density profile shape is
      set by the ratio v/D, so raising D alone shifts that ratio and regulates
      the pedestal density height, whereas raising D and v together would
      preserve the shape and provide no height control.
  """

  chi_i_signal: array_typing.FloatScalar
  chi_e_signal: array_typing.FloatScalar
  D_e_signal: array_typing.FloatScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class BarrierState:
  """State of the edge transport barrier for ADAPTIVE_TRANSPORT mode.

  The pedestal-region transport is a convex blend between the unmodified
  (L-mode) transport and a bounded barrier transport branch:

    coeff_ped = (1 - g) * coeff_raw + g * (residual + r * (cap - residual))

  where g is the barrier fraction set by the formation model and r is the
  per-channel barrier openness set by the saturation response. The formation
  and saturation signals act on different axes (branch selection vs position
  within the barrier branch), so neither ever needs to undo the other, and
  both are bounded in [0, 1]: the solver residual has no large-gain product
  terms.

  Neither residual nor cap is a free parameter: residual is the local
  neoclassical transport level (already computed independently by the
  neoclassical model), and cap is the local raw (turbulent) transport level
  coeff_raw itself, i.e. a fully open barrier (r = 1) reverts exactly to
  whatever the turbulence model already predicts for the current profile at
  that face. Both bounds are therefore machine- and profile-scaled quantities
  the existing models already compute, rather than hand-picked absolute
  diffusivities.

  Attributes:
    barrier_fraction: Blend weight g in [0, 1] between the unmodified
      transport (g=0, L-mode) and the barrier transport branch (g=1, fully
      formed edge barrier).
    chi_i_openness: Barrier openness r in (0, 1) for the ion heat channel:
      0 = suppressed to the residual level, 1 = opened up to the cap.
    chi_e_openness: As above, for the electron heat channel.
    D_e_openness: As above, for the particle diffusivity channel. The pinch
      has no openness; within the barrier branch it is suppressed to zero
      (see `BarrierSignals.D_e_signal` for the v/D rationale).
  """

  barrier_fraction: array_typing.FloatScalar
  chi_i_openness: array_typing.FloatScalar
  chi_e_openness: array_typing.FloatScalar
  D_e_openness: array_typing.FloatScalar

  @classmethod
  def default(cls):
    """No barrier: transport coefficients are unmodified."""
    return cls(
        barrier_fraction=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        chi_i_openness=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        chi_e_openness=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        D_e_openness=jnp.array(0.0, dtype=jax_utils.get_dtype()),
    )


# Explicit classification of CoreTransport fields for the ADAPTIVE_TRANSPORT
# barrier blend. Fields not listed here (e.g. neoclassical transport) are left
# unmodified. The total fields are the coefficients used in the state
# equations and receive the full barrier blend (with the residual floor and
# cap); the component fields are diagnostic breakdowns from the turbulent
# transport models and are scaled by the channel's relative suppression
# factor only (no floor: the residual barrier transport is not turbulence).
_CHI_I_TOTAL_FIELD = 'chi_face_ion'
_CHI_E_TOTAL_FIELD = 'chi_face_el'
_D_E_TOTAL_FIELD = 'd_face_el'
_V_E_TOTAL_FIELD = 'v_face_el'
_CHI_I_COMPONENT_FIELDS = frozenset({
    'chi_face_ion_bohm',
    'chi_face_ion_gyrobohm',
    'chi_face_ion_itg',
    'chi_face_ion_tem',
})
_CHI_E_COMPONENT_FIELDS = frozenset({
    'chi_face_el_bohm',
    'chi_face_el_gyrobohm',
    'chi_face_el_itg',
    'chi_face_el_tem',
    'chi_face_el_etg',
})
_D_E_COMPONENT_FIELDS = frozenset({
    'd_face_el_itg',
    'd_face_el_tem',
})
_V_E_COMPONENT_FIELDS = frozenset({
    'v_face_el_itg',
    'v_face_el_tem',
})
# Pereverzev-Corrigan terms come in diffusion/convection pairs whose fluxes
# cancel exactly at the current profile. Each pair must be scaled by a single
# shared factor so the cancellation is preserved.
_PEREVERZEV_HEAT_ION_FIELDS = frozenset({
    'chi_face_ion_pereverzev',
    'full_v_heat_face_ion_pereverzev',
})
_PEREVERZEV_HEAT_EL_FIELDS = frozenset({
    'chi_face_el_pereverzev',
    'full_v_heat_face_el_pereverzev',
})
_PEREVERZEV_PARTICLE_FIELDS = frozenset({
    'd_face_el_pereverzev',
    'v_face_el_pereverzev',
})


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
    barrier_state: State of the edge transport barrier (blend weight and
      per-channel openness). Only used if the pedestal is in
      ADAPTIVE_TRANSPORT mode.
  """

  rho_norm_ped_top: array_typing.FloatScalar
  T_i_ped: array_typing.FloatScalar
  T_e_ped: array_typing.FloatScalar
  n_e_ped: array_typing.FloatScalar
  barrier_state: BarrierState = dataclasses.field(
      default_factory=BarrierState.default
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

    Blends the turbulent transport coefficients in the pedestal region between
    the unmodified (L-mode) value and a bounded barrier transport branch,
    according to the barrier state:

      coeff_ped = (1 - g) * coeff_raw + g * (residual + r * (cap - residual))

    where g is the barrier fraction (formation), r is the per-channel barrier
    openness (saturation response), residual is the local neoclassical
    transport level (chi_neo_i / chi_neo_e / D_neo_e, floored at
    constants.CONSTANTS.eps; zero for the pinch) and cap is
    max(coeff_raw, residual): the local raw (turbulent) transport level for
    that channel, i.e. a fully open barrier reverts exactly to whatever the
    turbulence model already predicts for the current profile. Neither bound
    is a hand-picked constant: both are local, machine- and profile-scaled
    quantities the neoclassical and turbulent transport models already
    compute, so the barrier's throttling authority is not something that
    needs re-tuning per device or scenario. Transport coefficients from
    neoclassical and pedestal transport models are not otherwise affected.

    Both g and r are bounded in [0, 1] and the blend is linear in each, so the
    modified coefficients are smooth functions of the profiles (through the
    formation and saturation sigmoids) with bounded sensitivity: the maximum
    transport change per unit signal is (cap - residual) / (4 * width). This
    is required for the Newton-Raphson solver, which re-evaluates the barrier
    state inside the solver residual. Diagnostic turbulence components (Bohm,
    ITG, ...) are scaled by the channel's relative suppression factor
    (1 - g) + g * r, without the residual floor (the residual barrier
    transport is not turbulence). Pereverzev-Corrigan diffusion/convection
    pairs are scaled by the same shared per-channel factor so their mutual
    flux cancellation is preserved exactly.

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

    barrier = self.barrier_state
    g = barrier.barrier_fraction

    def barrier_blend(coeff, openness, residual):
      """Blend between the raw coefficient and the barrier branch.

      The cap is max(coeff, residual): the local raw (turbulent) transport
      level for this channel, so a fully open barrier (openness=1) reverts
      exactly to the turbulence model's own prediction rather than an
      externally chosen constant. Clamping at residual guards against the
      (rare) case where the raw level dips below the neoclassical floor,
      which would otherwise invert the sign of (cap - residual).
      """
      cap = jnp.maximum(coeff, residual)
      barrier_coeff = residual + openness * (cap - residual)
      return (1.0 - g) * coeff + g * barrier_coeff

    def suppression_factor(openness):
      """Relative factor for diagnostic components and Pereverzev pairs."""
      return (1.0 - g) + g * openness

    def modify_coeff(
        path: jax.tree_util.KeyPath, coeff: array_typing.FloatVectorFace
    ) -> array_typing.FloatVectorFace:
      """Apply the barrier blend to transport coefficients in the pedestal."""
      # Get the variable name of the leaf.
      key = path[-1]
      name = key.name if hasattr(key, "name") else str(key).lstrip(".")

      if name == _CHI_I_TOTAL_FIELD:
        residual = jnp.maximum(
            core_transport.chi_neo_i, constants.CONSTANTS.eps
        )
        modified_coeff = barrier_blend(coeff, barrier.chi_i_openness, residual)
      elif name == _CHI_E_TOTAL_FIELD:
        residual = jnp.maximum(
            core_transport.chi_neo_e, constants.CONSTANTS.eps
        )
        modified_coeff = barrier_blend(coeff, barrier.chi_e_openness, residual)
      elif name == _D_E_TOTAL_FIELD:
        # The residual floor is the local neoclassical particle diffusivity,
        # ensuring fueling deposited in the pedestal region has a finite
        # transport channel even under full suppression.
        residual = jnp.maximum(
            core_transport.D_neo_e, constants.CONSTANTS.eps
        )
        modified_coeff = barrier_blend(coeff, barrier.D_e_openness, residual)
      elif name == _V_E_TOTAL_FIELD:
        # The turbulent pinch is suppressed to zero within the barrier branch
        # and has no saturation openness (see BarrierSignals.D_e_signal for
        # the v/D height-control rationale).
        modified_coeff = (1.0 - g) * coeff
      elif name in _CHI_I_COMPONENT_FIELDS:
        modified_coeff = coeff * suppression_factor(barrier.chi_i_openness)
      elif name in _CHI_E_COMPONENT_FIELDS:
        modified_coeff = coeff * suppression_factor(barrier.chi_e_openness)
      elif name in _D_E_COMPONENT_FIELDS:
        modified_coeff = coeff * suppression_factor(barrier.D_e_openness)
      elif name in _V_E_COMPONENT_FIELDS:
        modified_coeff = (1.0 - g) * coeff
      elif name in _PEREVERZEV_HEAT_ION_FIELDS:
        modified_coeff = coeff * suppression_factor(barrier.chi_i_openness)
      elif name in _PEREVERZEV_HEAT_EL_FIELDS:
        modified_coeff = coeff * suppression_factor(barrier.chi_e_openness)
      elif name in _PEREVERZEV_PARTICLE_FIELDS:
        # Both members of the particle pair share the same factor so the
        # v/D ratio (and hence the Pereverzev flux cancellation) is unchanged.
        modified_coeff = coeff * suppression_factor(barrier.D_e_openness)
      else:
        # Neoclassical transport and any unclassified coefficients are not
        # affected by an ADAPTIVE_TRANSPORT pedestal model.
        return coeff

      # Only modify the coefficients in the pedestal region.
      modified_coeff = jnp.where(
          pedestal_active_mask_face, modified_coeff, coeff
      )

      # Apply smoothing to the pedestal top
      modified_coeff = jnp.dot(smoothing_matrix, modified_coeff)

      return modified_coeff

    return jax.tree_util.tree_map_with_path(modify_coeff, core_transport)
