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

"""test_psi_heat_dens with QLKNN transport, for the convergence-order study.

This is `test_psi_heat_dens` with exactly one thing changed: the constant
transport model is replaced by QLKNN. Its only purpose is to attribute the
observed time-integration order to the transport model rather than to the rest
of the physics, so nothing else may differ between this config and the
constant-transport one.
"""

CONFIG = {
    'profile_conditions': {
        'nbar': 0.85,
        'n_e_right_bc': 0.5e20,
        'current_profile_nu': 0,
    },
    'numerics': {
        'evolve_ion_heat': True,
        'evolve_electron_heat': True,
        'evolve_density': True,
        'evolve_current': True,
        'resistivity_multiplier': 100,
        't_final': 2,
    },
    'plasma_composition': {},
    'geometry': {
        'geometry_type': 'circular',
    },
    'sources': {
        'generic_heat': {},
        'ei_exchange': {},
        'generic_current': {},
    },
    'neoclassical': {
        'bootstrap_current': {},
    },
    'pedestal': {'model_name': 'set_T_ped_n_ped', 'set_pedestal': False},
    'transport': {
        'model_name': 'qlknn',
    },
    'solver': {
        'solver_type': 'newton_raphson',
        'use_predictor_corrector': True,
        'use_pereverzev': True,
    },
    'time_step_calculator': {
        'calculator_type': 'fixed',
    },
}
