"""Config variants for the white-box experiments."""
import copy
from harness import base_config


def variant(name, n_rho=10):
  c = base_config(n_rho)
  if name == 'baseline':
    pass
  elif name == 'mtanh':
    c['pedestal']['pedestal_profile_form'] = 'MTANH'
  elif name == 'bpp_ibc':
    c['pedestal']['set_pedestal'] = False
    c['profile_conditions']['internal_boundary_conditions'] = {
        'model_name': 'beta_poloidal_prime',
        'rho_norm_edge': 0.85,
        'n_e_edge': 0.4e20,
        'beta_poloidal_prime': 0.3,
        'Ti_Te_ratio': 1.0,
    }
  elif name == 'qlknn_rot':
    c['transport']['core_transport_models']['qlknn']['rotation_mode'] = 'full_radius'
  elif name == 'qlknn_rot_half':
    c['transport']['core_transport_models']['qlknn']['rotation_mode'] = 'half_radius'
  elif name == 'no_density':
    c['numerics']['evolve_density'] = False
  elif name == 'te_only':
    c['numerics']['evolve_density'] = False
    c['numerics']['evolve_current'] = False
    c['numerics']['evolve_ion_heat'] = False
  elif name == 'vloop':
    c['profile_conditions']['use_v_loop_lcfs_boundary_condition'] = True
    c['profile_conditions']['v_loop_lcfs'] = 0.1
  elif name == 'no_pedestal':
    c['pedestal']['set_pedestal'] = False
  else:
    raise ValueError(name)
  return c


def rich(n_rho, which):
  """Configs stacking many state-dependent terms that must stay local."""
  c = base_config(n_rho)
  if which == 'neo':
    c['neoclassical'] = {
        'bootstrap_current': {'model_name': 'redl', 'bootstrap_multiplier': 1.0},
        'transport': {'model_name': 'angioni_sauter'},
    }
    c['solver']['theta_implicit'] = 0.5
    c['numerics']['min_rho_norm'] = 0.05
    c['sources']['ohmic'] = {}
  elif which == 'models':
    c['transport'] = {
        'core_transport_models': {
            'bgb': {'model_name': 'bohm-gyrobohm', 'rho_max': 0.5},
            'cgm': {'model_name': 'CGM', 'rho_min': 0.5, 'rho_max': 0.9},
        },
        'pedestal_transport_models': {
            'ped': {'model_name': 'bohm-gyrobohm'},
        },
        'smoothing_width': 0.1,
        'chi_min': 0.05, 'chi_max': 100, 'D_e_min': 0.05,
    }
    c['sources'].update({
        'ecrh': {'gaussian_location': 0.35, 'gaussian_width': 0.1, 'P_total': 10e6},
        'bremsstrahlung': {},
        'impurity_radiation': {'model_name': 'mavrin_fit'},
        'ohmic': {},
    })
    c['plasma_composition']['impurity'] = {
        'impurity_mode': 'n_e_ratios_Z_eff',
        'species': {'Ne': None, 'W': 1e-5},
    }
  elif name_is_known(which):
    pass
  else:
    raise ValueError(which)
  return c


def name_is_known(which):
  return False
