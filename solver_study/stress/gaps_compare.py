"""Structured vs jacfwd Jacobian for the configurations that were rejected.

python gaps_compare.py CASE N_RHO
CASE: baseline | global_rotation | implicit_mtanh_grid | beta_pol |
      lh_transition | lh_transition_implicit | prescribed_timedependent_ne |
      rotation_grid | axis_XXX (min_rho_norm = XXX / 100)
Prints the relative Frobenius error and the largest entry error relative to its
row norm, at x_old and at 5% and 20% multiplicative noise.
"""
import copy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import numpy as np
import harness
from torax._src.solver import structured_jacobian as sj


def build(case, n_rho):
  if case == 'baseline':
    return harness.base_config(n_rho)
  if case == 'global_rotation':
    c = harness.base_config(n_rho, global_sources=True)
    c['transport']['core_transport_models']['qlknn']['rotation_mode'] = 'half_radius'
    return c
  if case == 'implicit_mtanh_grid':
    c = harness.base_config(n_rho)
    c['pedestal'] = {
        'model_name': 'set_P_ped_n_ped', 'set_pedestal': True,
        'P_ped': 9e4, 'n_e_ped': 0.62e20, 'T_i_T_e_ratio': 1.0,
        'rho_norm_ped_top': 0.9, 'explicit_pedestal': False,
        'pedestal_profile_form': 'MTANH',
    }
    u = np.linspace(0.0, 1.0, n_rho + 1)
    c['geometry'].pop('n_rho', None)
    c['geometry']['face_centers'] = u + 0.1 * u * (1 - u)
    c['numerics']['min_rho_norm'] = 0.1
    return c
  if case == 'beta_pol':
    c = harness.base_config(n_rho)
    c['pedestal']['set_pedestal'] = False
    c['profile_conditions']['internal_boundary_conditions'] = {
        'model_name': 'beta_poloidal_prime', 'rho_norm_edge': 0.85,
        'n_e_edge': 0.4e20, 'beta_poloidal_prime': 0.3, 'Ti_Te_ratio': 1.0,
    }
    return c
  if case == 'rotation_grid':
    c = build('implicit_mtanh_grid', n_rho)
    c['transport']['core_transport_models']['qlknn']['rotation_mode'] = 'full_radius'
    return c
  if case.startswith('axis_'):  # axis_020: min_rho_norm = 0.2
    c = harness.base_config(n_rho)
    c['sources']['ohmic'] = {}
    c['numerics']['min_rho_norm'] = int(case[len('axis_'):]) / 100
    return c
  if case == 'lh_transition_implicit':
    c = build('lh_transition', n_rho)
    c['pedestal']['explicit_pedestal'] = False
    return c
  if case in ('lh_transition', 'prescribed_timedependent_ne'):
    name = {'lh_transition': 'test_iterhybrid_lh_transition',
            'prescribed_timedependent_ne': 'test_prescribed_timedependent_ne'}[case]
    c = copy.deepcopy(__import__(f'torax.tests.test_data.{name}', fromlist=['CONFIG']).CONFIG)
    c['geometry']['n_rho'] = n_rho
    c['solver'] = dict(c.get('solver', {}), solver_type='newton_raphson')
    return c
  raise ValueError(case)


def main():
  case, n_rho = sys.argv[1], int(sys.argv[2])
  kwargs, _ = harness.solve_block_kwargs(build(case, n_rho))
  residual_fun = harness.residual_fun_from_kwargs(kwargs)
  structured = jax.jit(sj.jacobian_fn(residual_fun))
  dense = jax.jit(jax.jacfwd(residual_fun))
  for scale, seed in ((0.0, 0), (0.05, 1), (0.2, 2)):
    x = harness.perturbed_x(kwargs, scale, seed)
    js, jd = np.asarray(structured(x)), np.asarray(dense(x))
    ok = np.isfinite(js) & np.isfinite(jd)
    note = '' if ok.all() else f' (non-finite: structured {int((~np.isfinite(js)).sum())}, dense {int((~np.isfinite(jd)).sum())})'
    js, jd = np.where(ok, js, 0.0), np.where(ok, jd, 0.0)
    fro = np.linalg.norm(js - jd) / np.linalg.norm(jd)
    row = np.max(np.abs(js - jd).max(axis=1) / np.maximum(np.linalg.norm(jd, axis=1), 1e-300))
    print(f'{case} n_rho={n_rho} noise={scale}: frobenius {fro:.1e}, worst row {row:.1e}{note}', flush=True)


if __name__ == '__main__':
  main()
