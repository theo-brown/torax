# Copyright 2025 DeepMind Technologies Limited
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

"""STEP SPP-001 'ECHD' Power Plant Scenario.

This is a *fully non-inductive* flat-top (steady state) scenario for the STEP
tokamak. Key points of interest:
- Zero loop voltage boundary condition on current equation, signifying no use of
the central solenoid. Relies on high bootstrap fraction (≈ 0.9) to achieve
target plasma current, which is achievable thanks to high beta (≈ 4.5).
- Bohm-gyrobohm transport tuned to give the desired H98y (≈ 1.1).
- Pellet fuelling tuned to give the desired Greenwald fraction (≈ 0.95).
- Loading profiles, sources, and geometry from IMAS.

Tuning of this scenario should be done as follows:
  1. Disable all transport equations and set ECCD efficiency to get target A/W
    efficiency (≈ 12.5).
  2. Enable current and heat transport equations and adjust `bgb_multiplier` to
    achieve H98 ≈ 1.1.
  3. Enable all transport equations and adjust pellet fuelling rate to achieve
    fGW ≈ 0.95.

Based on:
 1. T.A. Brown, F.J. Casson et al., "OpenSTEP: public data release of the STEP
  Prototype Powerplant scenario SPP-001" (2025). United Kingdom Atomic Energy
  Authority. DOI: 10.14468/07jt-s540. URL: https://github.com/ukaea/OpenSTEP.
 2. Tholerus, E., et al. "Flat-top plasma operational space of the STEP power
  plant." Nuclear Fusion 64.10 (2024): 106030. DOI: 10.1088/1741-4326/ad6ea2.
"""

import imas
import numpy as np
from torax._src import path_utils
from torax._src.imas_tools.input import core_profiles
from torax._src.imas_tools.input import loader

# Load IDSs
path = (
    path_utils.torax_path()
    / "data"
    / "third_party"
    / "imas"
    / "STEP_SPP_001_ECHD_ftop.nc"
)
equilibrium_ids = loader.load_imas_data(str(path), "equilibrium")
core_profiles_ids = loader.load_imas_data(str(path), "core_profiles")
core_sources_ids = loader.load_imas_data(str(path), "core_sources")

# Replace Ip from profile conditions with Ip from equilibrium
# TODO(b/323504363): can this be handled within the IMAS loader? e.g. if Ip
# is not present in profile_conditions, pull from equilibrium automatically.
profile_conditions_from_ids = core_profiles.profile_conditions_from_IMAS(
    core_profiles_ids,
)
equilibrium_xr = imas.util.to_xarray(equilibrium_ids)
profile_conditions_from_ids["Ip"] = equilibrium_xr[
    "time_slice.global_quantities.ip"
][0].item()

# Extract the source information
# As this is a steady-state scenario, we only need the first time slice
core_sources_xr = imas.util.to_xarray(core_sources_ids)
ec_idx = np.where(core_sources_xr["source.identifier.name"] == "ec")[0].item()
rho_norm_ec = core_sources_xr["source.profiles_1d.grid.rho_tor_norm"][ec_idx][0]
electron_heating_ec = core_sources_xr["source.profiles_1d.electrons.energy"][
    ec_idx
][0]

# Set BgB multiplier to achieve desired confinement
# Lower -> better confinement
bgb_multiplier = 0.15


CONFIG = {
    "plasma_composition": core_profiles.plasma_composition_from_IMAS(
        core_profiles_ids
    ),
    "profile_conditions": (
        profile_conditions_from_ids
        | {
            "use_v_loop_lcfs_boundary_condition": True,
            "v_loop_lcfs": 0.0,
        }
    ),
    "geometry": {
        "geometry_type": "IMAS",
        "imas_filepath": "STEP_SPP_001_ECHD_ftop.nc",
        "n_rho": 100,
    },
    "pedestal": {
        # Ballooning-stability-limited pedestal: rather than pinning the
        # pedestal top to a preset temperature/density (set via
        # ADAPTIVE_TRANSPORT instead of INTERNAL_BOUNDARY_CONDITION), the
        # pedestal-region transport is decreased once P_SOL > P_LH (the
        # formation model) and then increased once the local infinite-n
        # ideal ballooning parameter alpha exceeds alpha_crit (the
        # saturation model), letting the pedestal self-organize to marginal
        # ballooning stability. T_i_ped/T_e_ped/n_e_ped below are therefore
        # unused (only rho_norm_ped_top matters to the alpha_crit
        # saturation model) and are left at their defaults.
        # alpha_crit is not reported in [2] (which uses a separate
        # empirical Europed pedestal-height scaling, eq 13-14, not
        # alpha_crit).
        # WIP/known limitation: the local alpha at rho=0.95 implied by the
        # pedestal this scenario was originally tuned to (T_i_ped=4.0,
        # T_e_ped=5.0 keV, n_e_ped=6e19 m^-3) is ~87 -- a physically
        # motivated value, not a grid artifact (it reflects a real
        # transport-barrier step in the gradient at the pedestal top).
        # However, using alpha_crit=87 with the saturation model's default
        # gain (base_multiplier=1e6, steepness=100) makes the solver
        # numerically stiff/unstable (near-discontinuous transport
        # response to a self-referential local gradient signal). alpha_crit
        # is set to 15 below instead, which converges cleanly but only
        # reaches T_e_ped ~ 0.7 keV, T_i_ped ~ 0.8 keV -- well short of the
        # scenario's original ~5 keV pedestal, so H98/fusion-performance
        # will not match the docstring's tuning targets. Reaching the
        # original target with alpha_crit~87 likely needs a much gentler
        # saturation gain (lower base_multiplier/steepness) than the
        # defaults tuned for ProfileValueSaturation's use case.
        "model_name": "set_T_ped_n_ped",
        "set_pedestal": True,
        "mode": "ADAPTIVE_TRANSPORT",
        "rho_norm_ped_top": 0.95,
        "saturation_model": {
            "model_name": "alpha_crit",
            "alpha_crit": 15.0,
        },
    },
    "sources": {
        # Physics-based sources
        # Note: Bremsstrahlung is not included in [1,2]
        "ohmic": {},
        "fusion": {},
        "ei_exchange": {},
        "impurity_radiation": {
            "model_name": "P_in_scaled_flat_profile",
            "fraction_P_heating": 0.7,
        },
        # Actuators
        "ecrh": {
            "extra_prescribed_power_density": (
                rho_norm_ec.values,
                np.clip(electron_heating_ec.values, a_min=0, a_max=None),
            ),
            # Tuned to match A/W efficiency in [2] Table 5
            "current_drive_efficiency": 0.14,
        },
        "pellet": {
            # TODO(b/323504363): load from IDS?
            "pellet_deposition_location": 0.8,  # from [2] sec 3.4
            "pellet_width": 0.17,  # from [2] sec 3.4
            "S_total": 3e21,  # [s^-1], tuned to get desired fGW in Table 5
        },
    },
    "transport": {
        "model_name": "bohm-gyrobohm",
        # BgB settings from [2] sec 3.3
        # Tuning factor to achieve desired confinement
        "chi_e_bohm_multiplier": bgb_multiplier,
        "chi_i_bohm_multiplier": bgb_multiplier,
        "chi_e_gyrobohm_multiplier": bgb_multiplier,
        "chi_i_gyrobohm_multiplier": bgb_multiplier,
        # Base coefficients
        "chi_e_bohm_coeff": 0.01 * 2e-4,
        "chi_e_gyrobohm_coeff": 50 * 5e-6,
        "chi_i_bohm_coeff": 0.001 * 2e-4,
        "chi_i_gyrobohm_coeff": 1.0 * 5e-6,
        "D_face_c1": 1,
        "D_face_c2": 0.3,
        "V_face_coeff": -0.1,
        # Clipping
        "chi_min": 0.15,
        "chi_max": 100.0,
        "D_e_min": 1e-3,
        "D_e_max": 100.0,
        "V_e_min": -50.0,
        "V_e_max": 50.0,
        # Smoothing
        "smooth_everywhere": True,
        "smoothing_width": 0.05,
    },
    "neoclassical": {
        "bootstrap_current": {"model_name": "redl"},
        "transport": {
            "model_name": "angioni_sauter",
            "use_shaing_ion_correction": True,
        },
    },
    "numerics": {
        "t_initial": 0.0,
        "t_final": 400.0,
        "fixed_dt": 10.0,
        "min_dt": 1e-3,
        "dt_reduction_factor": 2.0,
        # Current diffusion time in STEP plasmas is very long, so artificially
        # boost the resistivity to decrease simulation time
        "resistivity_multiplier": 10.0,
        "evolve_current": True,
        "evolve_ion_heat": True,
        "evolve_electron_heat": True,
        "evolve_density": True,
    },
    "solver": {
        "solver_type": "newton_raphson",
        "use_pereverzev": True,
    },
    "time_step_calculator": {
        "calculator_type": "fixed",
    },
}
