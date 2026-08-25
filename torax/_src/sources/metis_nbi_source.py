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

"""METIS neutral beam injection (NBI) source models.

Standalone reimplementation of the NBI model of the METIS integrated tokamak
simulator [J.F. Artaud et al., Nucl. Fusion 58 (2018) 105001], following the
METIS source code (zerod/zicd0.m, zerod/z0nbipath.m, zerod/z0suzuki_crx.m,
zerod/zfract0.m).

The model consists of:
  * A pencil-beam attenuation calculation along a tangential midplane chord
    through the flux surfaces, giving the fast-ion birth profile and the
    shine-through fraction.
  * The Suzuki beam-stopping cross-section [S. Suzuki et al., Plasma Phys.
    Control. Fusion 40 (1998) 2097], including the impurity correction.
  * The Wesson/Stix critical-energy formula for the ion/electron split of the
    deposited power [Wesson, Tokamaks, 2nd edition, p. 227].
  * Neutral beam current drive following L.-G. Eriksson's slowing-down
    average, with Lin-Liu & Hilton electron shielding [Phys. Plasmas 4 (1997)
    4179] and trapped-ion suppression.

This module is not part of the default TORAX source schema. To use it, call
`register_metis_nbi_sources()` before building a `ToraxConfig`, then select
the model by setting `model_name='metis_nbi'` in the `generic_heat`,
`generic_current` and/or `generic_particle` source configs.

Simplifications relative to full METIS: a single injector; a zero-width pencil
beam (no horizontal/vertical beam extent sampling and no vertical offset); no
first-orbit losses; no fast-ion accumulation correction to the stopping
cross-section; steady-state slowing down (deposited power is thermalized
instantaneously); no Doppler shift of the injection energy from plasma
rotation.
"""

import dataclasses
from typing import Annotated, Literal

import chex
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.neoclassical.formulas import formulas as neoclassical_formulas
from torax._src.sources import base
from torax._src.sources import generic_current_source
from torax._src.sources import generic_ion_el_heat_source
from torax._src.sources import generic_particle_source
from torax._src.sources import register_model
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name

MODEL_FUNCTION_NAME: str = 'metis_nbi'

# Number of points for the numerical integral in the NBCD slowing-down
# average (zicd0.m uses linspace(0,1,101)).
_NBCD_INTEGRAL_POINTS: int = 101

# Floor on the critical energies [keV] (METIS zicd0.m: max(30, ...) in eV).
_MIN_CRITICAL_ENERGY_KEV: float = 0.03

# Suzuki 1998 beam-stopping fit coefficients, transcribed from METIS
# zerod/z0suzuki_crx.m. Columns are the beam species [H, D, T].
# High-energy table (Table 2, 100 <= E/A <= 10000 keV/amu).
_SUZUKI_A_HIGH = np.array([
    [1.27e1, 1.41e1, 1.27e1],
    [1.25, 1.11, 1.26],
    [4.52e-1, 4.08e-1, 4.49e-1],
    [1.05e-2, 1.05e-2, 1.05e-2],
    [5.47e-1, 5.47e-1, 5.47e-1],
    [-1.02e-1, -4.03e-2, -5.77e-3],
    [3.60e-1, 3.45e-1, 3.36e-1],
    [-2.98e-2, -2.88e-2, -2.82e-2],
    [-9.59e-2, -9.71e-2, -9.74e-2],
    [4.21e-3, 4.74e-3, 4.87e-3],
])
# Low-energy table (Table 3, 10 <= E/A < 100 keV/amu).
_SUZUKI_A_LOW = np.array([
    [-5.29e1, -6.79e1, -7.42e1],
    [-1.36, -1.22, -1.18],
    [7.19e-2, 8.14e-2, 8.43e-2],
    [1.37e-2, 1.39e-2, 1.39e-2],
    [4.54e-1, 4.54e-1, 4.53e-1],
    [4.03e-1, 4.65e-1, 4.91e-1],
    [-2.20e-1, -2.73e-1, -2.94e-1],
    [6.66e-2, 7.51e-2, 7.88e-2],
    [-6.77e-2, -6.30e-2, -6.12e-2],
    [-1.48e-3, -5.08e-4, -1.85e-4],
])
# Impurity correction coefficients B_ijk. Columns are the impurity species
# [He, Li, Be, B, C, N, O, Fe]; rows are (i,j,k) in the order
# (111, 112, 121, 122, 211, 212, 221, 222, 311, 312, 321, 322), multiplying
# eps^(i-1) * ln(N)^(j-1) * U^(k-1).
_SUZUKI_IMPURITY_Z = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 26.0])
_SUZUKI_B_HIGH = np.array([
    [2.31e-1, -4.41e-1, -6.13e-1, -7.32e-1, -1.00, -1.00, -9.89e-1, -1.93e-1],
    [3.43e-1, 1.29e-1, 5.52e-2, 1.83e-2, -2.55e-2, -4.15e-2, -4.98e-2, 5.03e-3],
    [
        -1.85e-1,
        -1.70e-1,
        -1.67e-1,
        -1.55e-1,
        -1.25e-1,
        -9.76e-2,
        -6.36e-2,
        1.06e-1,
    ],
    [
        -1.62e-2,
        -1.62e-2,
        -1.59e-2,
        -1.72e-2,
        -1.42e-2,
        -1.10e-2,
        -7.75e-3,
        1.25e-2,
    ],
    [1.05e-1, 2.77e-1, 3.04e-1, 3.21e-1, 3.88e-1, 3.70e-1, 3.52e-1, 2.36e-3],
    [-7.03e-2, -1.56e-2, 1.54e-3, 9.46e-3, 2.06e-2, 2.28e-2, 2.34e-2, -7.41e-3],
    [5.31e-2, 4.66e-2, 4.36e-2, 3.97e-2, 2.97e-2, 2.15e-2, 1.17e-2, -3.88e-2],
    [3.42e-3, 3.79e-3, 3.78e-3, 4.20e-3, 3.26e-3, 2.33e-3, 1.42e-3, -4.60e-3],
    [
        -8.38e-3,
        -1.93e-2,
        -2.01e-2,
        -2.04e-2,
        -2.46e-2,
        -2.22e-2,
        -2.04e-2,
        9.39e-3,
    ],
    [
        4.15e-3,
        7.53e-4,
        -2.16e-4,
        -6.19e-4,
        -1.31e-3,
        -1.33e-3,
        -1.29e-3,
        1.58e-3,
    ],
    [
        -3.35e-3,
        -2.86e-3,
        -2.51e-3,
        -2.24e-3,
        -1.48e-3,
        -9.25e-4,
        -2.75e-4,
        3.32e-3,
    ],
    [
        -2.21e-4,
        -2.39e-4,
        -2.27e-4,
        -2.54e-4,
        -1.80e-4,
        -1.15e-4,
        -5.45e-5,
        3.88e-4,
    ],
])
_SUZUKI_B_LOW = np.array([
    [-7.92e-1, 1.12e-1, 1.12e-1, 1.22e-1, 1.58e-1, 1.34e-1, 1.07e-1, -4.65e-5],
    [4.20e-2, 4.95e-2, 4.95e-2, 5.27e-2, 5.54e-2, 5.24e-2, 4.31e-2, -7.29e-4],
    [
        5.30e-2,
        1.16e-2,
        1.16e-2,
        -4.30e-4,
        -4.31e-3,
        -4.69e-3,
        -2.83e-3,
        -3.10e-3,
    ],
    [
        -1.39e-2,
        -2.86e-3,
        -2.86e-3,
        -3.18e-3,
        -3.35e-3,
        -3.06e-3,
        -1.82e-3,
        -5.17e-4,
    ],
    [
        3.01e-1,
        -1.49e-1,
        -1.49e-1,
        -1.51e-1,
        -1.55e-1,
        -1.30e-1,
        -1.06e-1,
        -1.34e-2,
    ],
    [
        -2.64e-2,
        -3.31e-2,
        -3.31e-2,
        -3.64e-2,
        -3.74e-2,
        -3.52e-2,
        -2.89e-2,
        -5.06e-5,
    ],
    [-2.99e-2, -4.26e-3, -4.26e-3, 3.43e-3, 5.37e-3, 5.31e-3, 3.87e-3, 2.87e-3],
    [6.07e-3, 9.80e-4, 9.80e-4, 1.51e-3, 1.74e-3, 1.64e-3, 9.10e-4, 2.74e-4],
    [2.72e-4, 4.47e-2, 4.47e-2, 4.20e-2, 3.88e-2, 3.31e-2, 2.77e-2, 5.04e-3],
    [6.11e-3, 6.52e-3, 6.52e-3, 6.92e-3, 6.83e-3, 6.35e-3, 5.27e-3, 2.13e-4],
    [
        3.47e-3,
        -3.56e-4,
        -3.56e-4,
        -1.41e-3,
        -1.60e-3,
        -1.51e-3,
        -1.23e-3,
        -6.65e-4,
    ],
    [
        -9.19e-4,
        -2.03e-4,
        -2.03e-4,
        -2.90e-4,
        -3.22e-4,
        -3.04e-4,
        -1.89e-4,
        -5.83e-5,
    ],
])


def suzuki_beam_stopping_cross_section(
    beam_energy: array_typing.FloatScalar,
    beam_mass: array_typing.FloatScalar,
    n_e: array_typing.FloatVector,
    T_e: array_typing.FloatVector,
    n_impurity: array_typing.FloatVector,
    Z_impurity: array_typing.FloatVector,
) -> array_typing.FloatVector:
  """Suzuki 1998 beam-stopping cross-section for hydrogenic beams.

  Follows METIS zerod/z0suzuki_crx.m. The hydrogenic contribution is
  interpolated between the H/D/T fit columns according to the beam mass, and
  the low/high-energy tables are selected at E/A = 100 keV/amu. The impurity
  correction uses the bundled TORAX impurity, with the fit coefficients
  interpolated in the impurity charge.

  Args:
    beam_energy: Beam injection energy [keV].
    beam_mass: Beam species mass [amu], in [1, 3].
    n_e: Electron density [m^-3].
    T_e: Electron temperature [keV].
    n_impurity: Bundled impurity density [m^-3].
    Z_impurity: Bundled impurity charge [dimensionless].

  Returns:
    Beam-stopping cross-section [m^2], same shape as n_e.
  """
  E_per_A = beam_energy / beam_mass  # [keV/amu]
  N = n_e / 1e19
  U = jnp.log(T_e)
  log_N = jnp.log(N)

  # Interpolation weights over the [H, D, T] fit columns.
  w_H = jnp.clip(2.0 - beam_mass, 0.0, 1.0)
  w_T = jnp.clip(beam_mass - 2.0, 0.0, 1.0)
  species_weights = jnp.stack([w_H, 1.0 - w_H - w_T, w_T])

  def sigma_hydrogenic(A_table: np.ndarray, E: jax.Array) -> jax.Array:
    """Hydrogenic stopping cross-section [cm^2] from a Suzuki A-table."""
    A = jnp.asarray(A_table) @ species_weights
    eps_log = jnp.log(E)
    density_factor = 1.0 + (1.0 - jnp.exp(-A[3] * N)) ** A[4] * (
        A[5] + A[6] * eps_log + A[7] * eps_log**2
    )
    return (
        A[0]
        * 1e-16
        / E
        * (1.0 + A[1] * eps_log + A[2] * eps_log**2)
        * density_factor
        * (1.0 + A[8] * U + A[9] * U**2)
    )

  def impurity_sum(B_table: np.ndarray, E: jax.Array) -> jax.Array:
    """Impurity term of the stopping cross-section (dimensionless)."""
    # Interpolate each fit coefficient in the impurity charge.
    B = jax.vmap(jnp.interp, in_axes=(None, None, 0))(
        Z_impurity, jnp.asarray(_SUZUKI_IMPURITY_Z), jnp.asarray(B_table)
    )
    eps_log = jnp.log(E)
    S_z = (
        B[0]
        + B[1] * U
        + B[2] * log_N
        + B[3] * log_N * U
        + eps_log * (B[4] + B[5] * U + B[6] * log_N + B[7] * log_N * U)
        + eps_log**2 * (B[8] + B[9] * U + B[10] * log_N + B[11] * log_N * U)
    )
    return n_impurity * Z_impurity * (Z_impurity - 1.0) * S_z / n_e

  E_low = jnp.clip(E_per_A, 10.0, 100.0)
  E_high = jnp.clip(E_per_A, 100.0, 1e4)
  sigma_cm2 = jnp.where(
      E_per_A < 100.0,
      sigma_hydrogenic(_SUZUKI_A_LOW, E_low)
      * (1.0 + impurity_sum(_SUZUKI_B_LOW, E_low)),
      sigma_hydrogenic(_SUZUKI_A_HIGH, E_high)
      * (1.0 + impurity_sum(_SUZUKI_B_HIGH, E_high)),
  )
  return 1e-4 * sigma_cm2  # [cm^2] -> [m^2]


def calc_beam_deposition(
    geo: geometry.Geometry,
    n_e_face: array_typing.FloatVectorFace,
    sigma_stop_face: array_typing.FloatVectorFace,
    tangency_radius: array_typing.FloatScalar,
) -> tuple[
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
    array_typing.FloatScalar,
]:
  """Pencil-beam attenuation along a tangential midplane chord.

  Follows METIS zerod/z0nbipath.m: the beam is a straight chord in the
  midplane with tangency radius `tangency_radius`, entering the plasma at the
  outboard LCFS. The surviving beam fraction is I(l) = exp(-int n_e
  sigma_stop dl), and the fraction ionized between two flux-surface crossings
  is deposited in the corresponding flux-surface shell. Flux surfaces are
  located by their inboard/outboard midplane radii, so the Shafranov shift is
  included (as in METIS via `Raxe`).

  Args:
    geo: Torus geometry.
    n_e_face: Electron density on the face grid [m^-3].
    sigma_stop_face: Beam-stopping cross-section on the face grid [m^2].
    tangency_radius: Beam tangency radius [m]. Zero corresponds to
      perpendicular injection through the machine axis.

  Returns:
    birth_density: Fast-ion birth profile on the cell grid, normalized such
      that its volume integral is the absorbed power fraction (1 - shine)
      [m^-3].
    pitch: Birth-averaged beam pitch v_par/v on the cell grid [dimensionless].
    shine_through: Fraction of the injected power leaving the plasma
      [dimensionless].
  """
  n_face = geo.rho_face_norm.shape[0]
  n_cells = n_face - 1

  # The beam crosses each flux surface up to four times: twice on the way in
  # (outboard then inboard midplane radii) and twice, mirrored, on the way
  # out. Ordered by path length, the crossing radii are:
  r_inward = jnp.concatenate([geo.R_out_face[::-1], geo.R_in_face[1:]])
  r_seq = jnp.concatenate([r_inward, r_inward[::-1]])
  # Face index at each crossing, and cell index of each path segment
  # (static, shape-only quantities).
  idx_inward = np.concatenate(
      [np.arange(n_face - 1, -1, -1), np.arange(1, n_face)]
  )
  face_idx = np.concatenate([idx_inward, idx_inward[::-1]])
  cells_inward = np.concatenate(
      [np.arange(n_cells - 1, -1, -1), np.arange(n_cells)]
  )
  # The segment joining the two passes crosses the central hole (vacuum) when
  # the tangency radius is inside the inboard LCFS radius; it deposits
  # nothing (its optical depth is zeroed below) so its cell index is a dummy.
  segment_cells = np.concatenate([cells_inward, [0], cells_inward[::-1]])
  vacuum_segment = 2 * n_face - 2

  # Path length of each crossing, measured from the entry point. For a
  # midplane chord with tangency radius R_t, the distance between the
  # tangency point and the crossing of the cylinder of radius r is
  # sqrt(r^2 - R_t^2). Crossings with r < R_t are never reached and collapse
  # onto the tangency point, contributing zero path length.
  half_chord = jnp.sqrt(jnp.clip(r_seq**2 - tangency_radius**2, 0.0))
  half_chord_entry = half_chord[0]
  n_crossings = 2 * n_face - 1
  l_seq = jnp.concatenate([
      half_chord_entry - half_chord[:n_crossings],
      half_chord_entry + half_chord[n_crossings:],
  ])

  # Trapezoidal optical depth per segment and surviving beam fraction.
  attenuation_coeff = (n_e_face * sigma_stop_face)[face_idx]
  dl = jnp.diff(l_seq)
  dtau = 0.5 * (attenuation_coeff[:-1] + attenuation_coeff[1:]) * dl
  dtau = dtau.at[vacuum_segment].set(0.0)
  survival = jnp.exp(-jnp.concatenate([jnp.zeros(1), jnp.cumsum(dtau)]))
  shine_through = survival[-1]

  # Power fraction ionized in each segment, mapped onto flux-surface shells.
  dP = survival[:-1] - survival[1:]
  birth_frac = jnp.zeros(n_cells).at[segment_cells].add(dP)
  shell_volume = geo.vpr * geo.drho_norm
  birth_density = birth_frac / shell_volume

  # Birth pitch v_par/v = R_t / R at the segment midpoint, averaged over the
  # power deposited in each cell.
  l_mid = 0.5 * (l_seq[:-1] + l_seq[1:])
  R_mid = jnp.sqrt(tangency_radius**2 + (l_mid - half_chord_entry) ** 2)
  pitch_seg = tangency_radius / jnp.clip(R_mid, constants.CONSTANTS.eps)
  pitch = jnp.zeros(n_cells).at[segment_cells].add(pitch_seg * dP) / jnp.clip(
      birth_frac, constants.CONSTANTS.eps
  )
  return birth_density, pitch, shine_through


def wesson_ion_heating_fraction(
    beam_energy: array_typing.FloatScalar,
    critical_energy: array_typing.FloatVector,
) -> array_typing.FloatVector:
  """Fraction of the fast-ion power transferred to thermal ions.

  Wesson, Tokamaks 2nd edition p. 227 (METIS zerod/zfract0.m).

  Args:
    beam_energy: Beam injection energy [keV].
    critical_energy: Critical energy [keV] at which the fast ions heat ions
      and electrons at equal rates.

  Returns:
    Ion heating fraction in [0, 1].
  """
  x = jnp.clip(
      beam_energy / jnp.clip(critical_energy, constants.CONSTANTS.eps), 1e-3
  )
  sqrt_x = jnp.sqrt(x)
  return (
      (1.0 / 3.0) * jnp.log((1.0 - sqrt_x + x) / (1.0 + sqrt_x) ** 2)
      + 2.0
      / jnp.sqrt(3.0)
      * (jnp.arctan((2.0 * sqrt_x - 1.0) / jnp.sqrt(3.0)) + jnp.pi / 6.0)
  ) / x


def _critical_energies_and_slowing_time(
    core_profiles: state.CoreProfiles,
    beam_mass: array_typing.FloatScalar,
) -> tuple[
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
    array_typing.FloatVectorCell,
]:
  """METIS critical energies and Spitzer slowing-down time (zicd0.m).

  Args:
    core_profiles: Core plasma profiles.
    beam_mass: Beam species mass [amu].

  Returns:
    E_c_slow: Critical energy associated with the critical velocity v_c,
      used for the ion/electron split [keV].
    E_gamma: Critical energy associated with the Stix velocity v_gamma, used
      for the NBCD slowing-down exponent [keV].
    tau_s: Spitzer fast-ion slowing-down time on electrons [s].
  """
  T_e = core_profiles.T_e.value  # [keV]
  n_e = core_profiles.n_e.value  # [m^-3]
  # sum_j n_j Z_j^2 / A_j / n_e over thermal ions (main ion + bundled
  # impurity), the `fact` term in zicd0.m.
  fact = (
      core_profiles.n_i.value * core_profiles.Z_i**2 / core_profiles.A_i
      + core_profiles.n_impurity.value
      * core_profiles.Z_impurity**2
      / core_profiles.A_impurity
  ) / n_e
  E_c_slow = jnp.maximum(
      _MIN_CRITICAL_ENERGY_KEV,
      14.8 * T_e * (beam_mass**1.5 * fact) ** (2.0 / 3.0),
  )
  E_gamma = jnp.maximum(
      _MIN_CRITICAL_ENERGY_KEV,
      14.8
      * T_e
      * (2.0 * jnp.sqrt(beam_mass) * core_profiles.Z_eff) ** (2.0 / 3.0),
  )
  # NRL formulary slowing-down time, with T_e in eV and n_e in cm^-3.
  log_lambda = 15.2 - 0.5 * jnp.log(n_e / 1e20) + jnp.log(T_e)
  tau_s = 6.27e8 * beam_mass * (T_e * 1e3) ** 1.5 / ((n_e / 1e6) * log_lambda)
  return E_c_slow, E_gamma, tau_s


def _absorbed_power_density(
    source_params: 'RuntimeParams',
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> tuple[array_typing.FloatVectorCell, array_typing.FloatVectorCell]:
  """Absorbed NBI power density [W/m^3] and birth pitch on the cell grid."""
  sigma_stop_face = suzuki_beam_stopping_cross_section(
      source_params.beam_energy,
      source_params.beam_mass,
      core_profiles.n_e.face_value(),
      core_profiles.T_e.face_value(),
      core_profiles.n_impurity.face_value(),
      core_profiles.Z_impurity_face,
  )
  birth_density, pitch, _ = calc_beam_deposition(
      geo,
      core_profiles.n_e.face_value(),
      sigma_stop_face,
      source_params.tangency_radius,
  )
  return source_params.P_total * birth_density, pitch


def calc_nbi_heating(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[array_typing.FloatVectorCell, array_typing.FloatVectorCell]:
  """Returns the METIS NBI (ion, electron) heating power densities [W/m^3]."""
  source_params = runtime_params.sources[source_name]
  assert isinstance(source_params, RuntimeParams)
  p_dep, _ = _absorbed_power_density(source_params, geo, core_profiles)
  E_c_slow, _, _ = _critical_energies_and_slowing_time(
      core_profiles, source_params.beam_mass
  )
  frac_ion = wesson_ion_heating_fraction(source_params.beam_energy, E_c_slow)
  return p_dep * frac_ion, p_dep * (1.0 - frac_ion)


def calc_nbi_particle_source(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[array_typing.FloatVectorCell, ...]:
  """Returns the METIS NBI fueling source [particles/(m^3 s)]."""
  source_params = runtime_params.sources[source_name]
  assert isinstance(source_params, RuntimeParams)
  p_dep, _ = _absorbed_power_density(source_params, geo, core_profiles)
  beam_energy_J = source_params.beam_energy * constants.CONSTANTS.keV_to_J
  return (p_dep / beam_energy_J,)


def calc_nbi_current(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[array_typing.FloatVectorCell, ...]:
  """Returns the METIS neutral beam driven current density [A/m^2].

  Follows zicd0.m: the fast-ion current is j = e * S * tau_s * <v_par>, with
  the slowing-down averaged velocity from L.-G. Eriksson, reduced by the
  Lin-Liu & Hilton electron shielding factor and by a trapped-ion
  suppression factor. The result approximates the flux-surface averaged
  parallel current density <j.B>/B_0 used by the psi equation.
  """
  source_params = runtime_params.sources[source_name]
  assert isinstance(source_params, CurrentDriveRuntimeParams)
  p_dep, pitch = _absorbed_power_density(source_params, geo, core_profiles)
  E_c_slow, E_gamma, tau_s = _critical_energies_and_slowing_time(
      core_profiles, source_params.beam_mass
  )

  # Fast-ion birth rate [m^-3 s^-1].
  beam_energy_J = source_params.beam_energy * constants.CONSTANTS.keV_to_J
  S_birth = p_dep / beam_energy_J

  # Velocities associated with the injection and critical energies. METIS
  # uses the proton mass; m_amu differs by < 1%.
  beam_mass_kg = source_params.beam_mass * constants.CONSTANTS.m_amu
  v0 = jnp.sqrt(2.0 * beam_energy_J / beam_mass_kg)
  v_c = jnp.sqrt(2.0 * E_c_slow * constants.CONSTANTS.keV_to_J / beam_mass_kg)
  v_gamma = jnp.sqrt(
      2.0 * E_gamma * constants.CONSTANTS.keV_to_J / beam_mass_kg
  )

  # Slowing-down average of the parallel velocity (zicd0.m, following
  # L.-G. Eriksson): <v> = v_c ((v0^3+v_c^3)/v0^3)^(ev-1)
  #                        * int_0^1 (v0/v_c) [u^3/(1+u^3)]^ev dlambda,
  # with u = (v0/v_c) lambda and ev = 1 + 2 v_gamma^3 / (3 v_c^3).
  ev = 1.0 + 2.0 * v_gamma**3 / (3.0 * v_c**3)
  lam = jnp.linspace(0.0, 1.0, _NBCD_INTEGRAL_POINTS)
  u = (v0 / v_c)[:, jnp.newaxis] * lam[jnp.newaxis, :]
  integrand = (v0 / v_c)[:, jnp.newaxis] * (u**3 / (1.0 + u**3)) ** (
      ev[:, jnp.newaxis]
  )
  integral = jnp.trapezoid(integrand, lam, axis=-1)
  v_mean = jnp.minimum(
      v_c * ((v0**3 + v_c**3) / v0**3) ** (ev - 1.0) * integral, v0
  )
  j_fast = constants.CONSTANTS.q_e * S_birth * tau_s * pitch * v_mean

  # Electron shielding: Lin-Liu & Hilton, Phys. Plasmas 4 (1997) 4179
  # (the METIS option.e_shielding analytic branch).
  Z_eff = core_profiles.Z_eff
  f_trap = geometry.face_to_cell(neoclassical_formulas.calculate_f_trap(geo))
  xt = f_trap / (1.0 - f_trap)
  D = (
      1.414 * Z_eff
      + Z_eff**2
      + xt * (0.754 + 2.657 * Z_eff + 2.0 * Z_eff**2)
      + xt**2 * (0.348 + 1.243 * Z_eff + Z_eff**2)
  )
  G_Z = (
      xt
      * (
          (0.754 + 2.21 * Z_eff + Z_eff**2)
          + xt * (0.348 + 1.243 * Z_eff + Z_eff**2)
      )
      / D
  )
  shielding = 1.0 - (1.0 - G_Z) / Z_eff

  # Trapped fast-ion suppression (zicd0.m): the driven current vanishes
  # where the birth pitch is inside the trapped cone mu_trap.
  mu_trap = jnp.sqrt(2.0 * geo.epsilon / (1.0 + geo.epsilon))
  fi_trap = jnp.minimum(1.0, 1.0 + jnp.tanh(10.0 * (pitch - mu_trap)))

  j_cd = (
      source_params.current_drive_sign
      * source_params.current_drive_multiplier
      * shielding
      * j_fast
      * fi_trap
  )
  return (j_cd,)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(sources_runtime_params_lib.RuntimeParams):
  """Runtime parameters shared by the METIS NBI source models."""

  P_total: array_typing.FloatScalar
  beam_energy: array_typing.FloatScalar
  beam_mass: array_typing.FloatScalar
  tangency_radius: array_typing.FloatScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class CurrentDriveRuntimeParams(RuntimeParams):
  """Runtime parameters for the METIS NBI current drive model."""

  current_drive_multiplier: array_typing.FloatScalar
  current_drive_sign: array_typing.FloatScalar


class _MetisNBIConfigBase(base.SourceModelBase):
  """Base config for the METIS NBI source models.

  Attributes:
    P_total: Injected neutral beam power [W], before shine-through losses.
    beam_energy: Beam injection energy [keV] (METIS option.einj; METIS models
      a single full-energy beam component).
    beam_mass: Beam species mass [amu], in [1, 3] (hydrogenic beams).
    tangency_radius: Beam tangency radius [m] (METIS option.rtang). Zero
      corresponds to perpendicular injection aimed at the machine axis.
  """

  model_name: Annotated[Literal['metis_nbi'], torax_pydantic.JAX_STATIC] = (
      'metis_nbi'
  )
  P_total: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      10e6
  )
  beam_energy: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1e3)
  )
  beam_mass: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(2.0)
  )
  tangency_radius: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  mode: Annotated[
      sources_runtime_params_lib.Mode, torax_pydantic.JAX_STATIC
  ] = sources_runtime_params_lib.Mode.MODEL_BASED

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    return RuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
        mode=self.mode,
        is_explicit=self.is_explicit,
        P_total=self.P_total.get_value(t),
        beam_energy=self.beam_energy.get_value(t),
        beam_mass=self.beam_mass.get_value(t),
        tangency_radius=self.tangency_radius.get_value(t),
    )


class MetisNBIHeatSourceConfig(_MetisNBIConfigBase):
  """METIS NBI ion and electron heating model.

  Register against the 'generic_heat' source.
  """

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return calc_nbi_heating

  def build_source(
      self,
  ) -> generic_ion_el_heat_source.GenericIonElectronHeatSource:
    return generic_ion_el_heat_source.GenericIonElectronHeatSource(
        model_func=self.model_func
    )


class MetisNBIParticleSourceConfig(_MetisNBIConfigBase):
  """METIS NBI fueling model.

  Register against the 'generic_particle' source.
  """

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return calc_nbi_particle_source

  def build_source(self) -> generic_particle_source.GenericParticleSource:
    return generic_particle_source.GenericParticleSource(
        model_func=self.model_func
    )


class MetisNBICurrentSourceConfig(_MetisNBIConfigBase):
  """METIS neutral beam current drive model.

  Register against the 'generic_current' source.

  Attributes:
    current_drive_multiplier: Multiplier on the NBCD efficiency (METIS
      option.nbicdmul).
    counter_injection: If True, the beam is injected counter to the plasma
      current and the driven current is negative (METIS sign(angle_nbi)).
  """

  current_drive_multiplier: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  counter_injection: Annotated[bool, torax_pydantic.JAX_STATIC] = False

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return calc_nbi_current

  def build_source(self) -> generic_current_source.GenericCurrentSource:
    return generic_current_source.GenericCurrentSource(
        model_func=self.model_func
    )

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> CurrentDriveRuntimeParams:
    return CurrentDriveRuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
        mode=self.mode,
        is_explicit=self.is_explicit,
        P_total=self.P_total.get_value(t),
        beam_energy=self.beam_energy.get_value(t),
        beam_mass=self.beam_mass.get_value(t),
        tangency_radius=self.tangency_radius.get_value(t),
        current_drive_multiplier=self.current_drive_multiplier.get_value(t),
        current_drive_sign=-1.0 if self.counter_injection else 1.0,
    )


def register_metis_nbi_sources():
  """Registers the METIS NBI models with the TORAX source config schema.

  Must be called before building a `ToraxConfig`. After registration, the
  models are selected with `model_name='metis_nbi'` in the `generic_heat`,
  `generic_current` and/or `generic_particle` source configs.
  """
  register_model.register_source_model_config(
      MetisNBIHeatSourceConfig, 'generic_heat'
  )
  register_model.register_source_model_config(
      MetisNBICurrentSourceConfig, 'generic_current'
  )
  register_model.register_source_model_config(
      MetisNBIParticleSourceConfig, 'generic_particle'
  )
