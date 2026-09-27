"""Runs short whole simulations and dumps every output array.

Usage: python sims.py OUT.npz CASE
Runs unchanged against the base (via run_old.py) and the branch.
"""

import copy
import os
import sys
import time

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

import torax  # pylint: disable=g-import-not-at-top
from torax._src.sources import register_model
from torax.experimental import gas_puff_feedback_source
from torax.tests.test_data import test_iterhybrid_lh_transition
from torax.tests.test_data import test_iterhybrid_lh_transition_internal_boundary_condition as lh_ibc
from torax.tests.test_data import test_iterhybrid_predictor_corrector
from torax.tests.test_data import test_iterhybrid_predictor_corrector_constant_fraction_impurity_radiation as constfrac
from torax.tests.test_data import test_iterhybrid_predictor_corrector_cyclotron as cyclotron

HERE = os.path.dirname(os.path.abspath(__file__))
TORIC = os.path.join(HERE, 'toricnn.json')
N_RHO = int(os.environ.get('N_RHO', '25'))

register_model.register_source_model_config(
    gas_puff_feedback_source.GasPuffFeedbackSourceConfig, 'gas_puff'
)

NR = dict(
    solver_type='newton_raphson',
    use_predictor_corrector=True,
    n_corrector_steps=1,
    use_pereverzev=True,
)


def case_config(case):
  if case.startswith('cyclotron'):
    config = copy.deepcopy(cyclotron.CONFIG)
  elif case.startswith('constfrac'):
    config = copy.deepcopy(constfrac.CONFIG)
  elif case.startswith('lh_ibc'):
    config = copy.deepcopy(lh_ibc.CONFIG)
  elif case.startswith('lh_adaptive'):
    config = copy.deepcopy(test_iterhybrid_lh_transition.CONFIG)
  else:
    config = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)
  config['geometry']['n_rho'] = N_RHO
  config['numerics']['t_final'] = 2.0
  match case:
    case 'cyclotron_linear' | 'constfrac_linear':
      pass
    case 'cyclotron_nr' | 'constfrac_nr':
      config['solver'] = dict(NR)
    case 'lh_ibc_native' | 'lh_adaptive_native':
      config['numerics']['t_final'] = 8.0
    case 'lh_ibc_highP' | 'lh_adaptive_highP':
      config['sources']['generic_heat']['P_total'] = 8e7
      config['numerics']['t_final'] = 6.0
    case 'lh_ibc_highP_mtanh':
      config['sources']['generic_heat']['P_total'] = 8e7
      config['numerics']['t_final'] = 6.0
      config['pedestal']['pedestal_profile_form'] = 'MTANH'
    case 'mtanh_implicit':
      config['solver'] = dict(NR)
      config['pedestal'] = {
          'model_name': 'set_P_ped_n_ped',
          'set_pedestal': True,
          'P_ped': 9e4,
          'n_e_ped': 0.62e20,
          'rho_norm_ped_top': 0.9,
          'explicit_pedestal': False,
          'pedestal_profile_form': 'MTANH',
      }
      faces = np.linspace(0.0, 1.0, N_RHO + 1)
      config['geometry'].pop('n_rho')
      config['geometry']['face_centers'] = faces + 0.1 * faces * (1 - faces)
      config['numerics']['min_rho_norm'] = 0.1
    case 'bpp_nr' | 'bpp_linear':
      if case == 'bpp_nr':
        config['solver'] = dict(NR)
      config['pedestal']['set_pedestal'] = False
      config['profile_conditions']['internal_boundary_conditions'] = {
          'model_name': 'beta_poloidal_prime',
          'rho_norm_edge': 0.85,
          'n_e_edge': 0.4e20,
          'beta_poloidal_prime': 0.3,
          'Ti_Te_ratio': 1.0,
      }
    case 'global_sources_nr':
      config['solver'] = dict(NR)
      config['sources'].update({
          'cyclotron_radiation': {},
          'impurity_radiation': {
              'model_name': 'P_in_scaled_flat_profile',
              'fraction_P_heating': 0.1,
          },
          'icrh': {'model_path': TORIC, 'P_total': 10e6},
      })
      config['geometry'] = {'geometry_type': 'circular', 'n_rho': N_RHO}
      qlknn = config['transport']['core_transport_models']['qlknn']
      qlknn['rotation_mode'] = 'half_radius'
    case 'gaspuff_fb_nr':
      config['solver'] = dict(NR)
      config['sources']['gas_puff'] = {
          'model_name': 'feedback',
          'feedback_gain': 10.0,
          'target_average_n_e': 1e20,
          'S_feedforward': 1e21,
      }
    case 'adaptive_implicit_nr':
      config['solver'] = dict(NR)
      config['pedestal'] = {
          'model_name': 'set_T_ped_n_ped',
          'set_pedestal': True,
          'mode': 'ADAPTIVE_TRANSPORT',
          'explicit_pedestal': False,
          'T_i_ped': 4.5,
          'T_e_ped': 4.5,
          'n_e_ped': 0.62e20,
          'rho_norm_ped_top': 0.9,
          'formation_model': {'model_name': 'martin_scaling'},
          'saturation_model': {'model_name': 'profile_value'},
      }
      config['transport']['pedestal_transport_models'] = {
          'prescribed': {
              'model_name': 'prescribed',
              'chi_i': 1.0,
              'chi_e': 1.0,
              'D_e': 0.2,
              'V_e': 0.1,
          },
      }
      config['sources']['generic_heat']['P_total'] = 1.5e8
    case _:
      raise ValueError(case)
  return config


def main():
  out_path, case = sys.argv[1], sys.argv[2]
  config = case_config(case)
  torax_config = torax.ToraxConfig.from_dict(config)
  t0 = time.time()
  dt, history = torax.run_simulation(torax_config, progress_bar=False)
  wall = time.time() - t0
  out = {}
  for node in dt.subtree:
    ds = node.to_dataset(inherit=False)
    for name, var in list(ds.data_vars.items()) + list(ds.coords.items()):
      arr = np.asarray(var.values)
      if arr.dtype.kind in 'fciub':
        out[f'{node.path}/{name}'] = arr
  np.savez(out_path, **out)
  with open(out_path + '.config.json', 'w') as f:
    f.write(dt.attrs.get('config', ''))
  n_steps = int(np.asarray(dt['time']).size)
  print(f'RESULT case={case} steps={n_steps} wall={wall:.1f}s sim_error='
        f'{getattr(history, "sim_error", getattr(history, "_sim_error", None))} arrays={len(out)} torax={torax.__file__}')


if __name__ == '__main__':
  main()
