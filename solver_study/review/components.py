"""Evaluates the refactored components on fixed states; dumps every array.

Usage: python components.py OUT.npz VARIANT [VARIANT ...]
Runs unchanged against the base (via run_old.py) and the branch. Uses only APIs
that exist in both.
"""

import copy
import dataclasses
import functools
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)

import torax  # pylint: disable=g-import-not-at-top
from torax._src.config import build_runtime_params
from torax._src.core_profiles import convertors
from torax._src.core_profiles import updaters
from torax._src.fvm import calc_coeffs
from torax._src.fvm import enums
from torax._src.fvm import fvm_conversions
from torax._src.fvm import newton_raphson_solve_block
from torax._src.fvm import residual_and_loss
from torax._src.internal_boundary_conditions import builder as ibc_builder
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import run_simulation
from torax._src.orchestration import step_function_processing
from torax._src.pedestal_model import pedestal_transition_state as pts_lib
from torax._src.sources import register_model
from torax._src.sources import source_profile_builders
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import transport_coefficients_builder
from torax.examples import iterhybrid_predictor_corrector
from torax.experimental import gas_puff_feedback_source
from torax.tests.test_data import test_iterhybrid_lh_transition
from torax.tests.test_data import test_iterhybrid_lh_transition_internal_boundary_condition as lh_ibc

HERE = os.path.dirname(os.path.abspath(__file__))
TORIC = os.path.join(HERE, 'toricnn.json')
N_RHO = int(os.environ.get('N_RHO', '20'))

register_model.register_source_model_config(
    gas_puff_feedback_source.GasPuffFeedbackSourceConfig, 'gas_puff'
)


def _nr_solver():
  return dict(
      solver_type='newton_raphson',
      use_predictor_corrector=True,
      n_corrector_steps=2,
      use_pereverzev=True,
  )


def config_for(variant):
  config = copy.deepcopy(iterhybrid_predictor_corrector.CONFIG)
  config['geometry']['n_rho'] = N_RHO
  config['solver'] = _nr_solver()
  force_mode = None
  match variant:
    case 'local':
      pass
    case 'cyclotron':
      config['sources']['cyclotron_radiation'] = {}
    case 'constfrac':
      config['sources']['impurity_radiation'] = {
          'model_name': 'P_in_scaled_flat_profile',
          'fraction_P_heating': 0.1,
      }
    case 'icrh':
      config['sources']['icrh'] = {'model_path': TORIC, 'P_total': 10e6}
      config['geometry'] = {'geometry_type': 'circular', 'n_rho': N_RHO}
    case 'icrh_minority':
      config['sources']['icrh'] = {
          'model_path': TORIC,
          'P_total': 10e6,
          'minority_species': 'He3',
          'minority_concentration': None,
      }
      config['plasma_composition']['impurity'] = {
          'impurity_mode': 'fractions',
          'species': {'Ne': 0.9, 'He3': 0.1},
      }
      config['geometry'] = {'geometry_type': 'circular', 'n_rho': N_RHO}
    case 'gaspuff_fb':
      config['sources']['gas_puff'] = {
          'model_name': 'feedback',
          'feedback_gain': 10.0,
          'target_average_n_e': 1e20,
          'S_feedforward': 1e21,
      }
    case 'gaspuff_fb_volume':
      config['sources']['gas_puff'] = {
          'model_name': 'feedback',
          'feedback_gain': 10.0,
          'target_average_n_e': 1e20,
          'S_feedforward': 1e21,
          'average_type': 'volume',
      }
    case 'global_sources':
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
    case 'implicit_pedestal_mtanh':
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
    case 'explicit_mtanh':
      config['pedestal']['pedestal_profile_form'] = 'MTANH'
    case 'adaptive_L' | 'adaptive_H' | 'adaptive_H_implicit':
      config['pedestal'] = {
          'model_name': 'set_T_ped_n_ped',
          'set_pedestal': True,
          'mode': 'ADAPTIVE_TRANSPORT',
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
      config['sources']['generic_heat']['P_total'] = (
          2e6 if variant == 'adaptive_L' else 2e8
      )
      if variant == 'adaptive_H_implicit':
        config['pedestal']['explicit_pedestal'] = False
    case 'lh_adaptive_L' | 'lh_adaptive_H':
      config = copy.deepcopy(test_iterhybrid_lh_transition.CONFIG)
      config['geometry']['n_rho'] = N_RHO
      config['sources']['generic_heat']['P_total'] = (
          2e6 if variant == 'lh_adaptive_L' else 2e8
      )
    case 'lh_ibc_L' | 'lh_ibc_H' | 'lh_ibc_trans' | 'lh_ibc_mtanh_H':
      config = copy.deepcopy(lh_ibc.CONFIG)
      config['geometry']['n_rho'] = N_RHO
      force_mode = {
          'lh_ibc_L': None,
          'lh_ibc_H': pts_lib.ConfinementMode.H_MODE,
          'lh_ibc_trans': pts_lib.ConfinementMode.TRANSITIONING_TO_H_MODE,
          'lh_ibc_mtanh_H': pts_lib.ConfinementMode.TRANSITIONING_TO_H_MODE,
      }[variant]
      if variant == 'lh_ibc_mtanh_H':
        config['pedestal']['pedestal_profile_form'] = 'MTANH'
    case 'bpp':
      config['pedestal']['set_pedestal'] = False
      config['profile_conditions']['internal_boundary_conditions'] = {
          'model_name': 'beta_poloidal_prime',
          'rho_norm_edge': 0.85,
          'n_e_edge': 0.4e20,
          'beta_poloidal_prime': 0.3,
          'Ti_Te_ratio': 1.0,
      }
    case 'bpp_fgw':
      config['pedestal']['set_pedestal'] = False
      config['profile_conditions']['internal_boundary_conditions'] = {
          'model_name': 'beta_poloidal_prime',
          'rho_norm_edge': 0.9,
          'n_e_edge': 0.3,
          'n_e_is_fGW': True,
          'beta_poloidal_prime': 0.5,
          'Ti_Te_ratio': 1.2,
      }
    case 'prescribed_ibc':
      config['pedestal']['set_pedestal'] = False
      config['profile_conditions']['internal_boundary_conditions'] = {
          'model_name': 'prescribed',
          'T_e': {0.0: {0.9: 4.0}},
          'T_i': {0.0: {0.9: 4.0}},
      }
    case _:
      raise ValueError(variant)
  return config, force_mode


def flatten(prefix, tree, out):
  leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
  for path, leaf in leaves:
    if leaf is None:
      continue
    key = prefix + jax.tree_util.keystr(path)
    try:
      out[key] = np.asarray(leaf)
    except Exception:  # pylint: disable=broad-except
      pass


def run_variant(variant, out):
  config, force_mode = config_for(variant)
  torax_config = model_config.ToraxConfig.from_dict(config)
  step_fn = run_simulation.make_step_fn(torax_config)
  state, _ = initial_state_lib.get_initial_state_and_post_processed_outputs(
      step_fn=step_fn
  )
  models = step_fn.solver.models
  runtime_params_t, geo_t, esp, edge_outputs, pts = (
      step_function_processing.pre_step(
          input_state=state,
          runtime_params_provider=step_fn.runtime_params_provider,
          geometry_provider=step_fn.geometry_provider,
          models=models,
      )
  )
  dt = jnp.asarray(0.05)
  if force_mode is not None:
    pts = dataclasses.replace(
        pts,
        confinement_mode=jnp.asarray(int(force_mode), dtype=jnp.int32),
        transition_start_time=jnp.asarray(state.t - 1.0),
        T_i_ped_L_mode=jnp.asarray(1.0),
        T_e_ped_L_mode=jnp.asarray(1.2),
        n_e_ped_L_mode=jnp.asarray(0.3e20),
    )
  rp1, geo1 = build_runtime_params.get_consistent_runtime_params_and_geometry(
      t=state.t + dt,
      runtime_params_provider=step_fn.runtime_params_provider,
      geometry_provider=step_fn.geometry_provider,
      edge_outputs=edge_outputs,
      core_profiles=state.core_profiles,
  )
  names = runtime_params_t.numerics.evolving_names
  x_old = convertors.core_profiles_to_solver_x_tuple(state.core_profiles, names)
  cp1 = updaters.provide_core_profiles_t_plus_dt(
      dt=dt,
      runtime_params_t=runtime_params_t,
      runtime_params_t_plus_dt=rp1,
      geo_t_plus_dt=geo1,
      core_profiles_t=state.core_profiles,
  )
  flatten(f'{variant}/pts', pts, out)
  # A perturbed state so nothing is trivially symmetric.
  x0 = fvm_conversions.cell_variable_tuple_to_vec(x_old)
  pert = 1.0 + 0.02 * jnp.sin(jnp.arange(x0.size) * 0.7)
  x_pert = fvm_conversions.vec_to_cell_variable_tuple(x0 * pert, cp1, names)

  @jax.jit
  def perturbed_profiles(x_tuple, rp, geo, cp, cp_prev, dt):
    return updaters.update_core_profiles_during_step(
        x_tuple, rp, geo, cp, prev_core_profiles=cp_prev, dt=dt,
        evolving_names=names,
    )

  cpp = perturbed_profiles(x_pert, rp1, geo1, cp1, state.core_profiles, dt)

  for tag, cp in (('cp1', cp1), ('cpp', cpp)):
    @jax.jit
    def sources(rp, geo, cp, esp):
      cond = models.neoclassical_models.conductivity.calculate_conductivity(
          geo, cp
      )
      return source_profile_builders.build_source_profiles(
          runtime_params=rp,
          geo=geo,
          core_profiles=cp,
          source_models=models.source_models,
          neoclassical_models=models.neoclassical_models,
          explicit=False,
          explicit_source_profiles=esp,
          conductivity=cond,
      )

    flatten(f'{variant}/{tag}/sources', sources(rp1, geo1, cp, esp), out)
    # Explicit profiles too.
    @jax.jit
    def explicit_sources(rp, geo, cp):
      return source_profile_builders.build_source_profiles(
          runtime_params=rp,
          geo=geo,
          core_profiles=cp,
          source_models=models.source_models,
          neoclassical_models=models.neoclassical_models,
          explicit=True,
      )

    flatten(f'{variant}/{tag}/explicit_sources', explicit_sources(rp1, geo1, cp),
            out)
    for use_perev in (False, True):
      ct = transport_coefficients_builder.calculate_all_transport_coeffs(
          models.transport_model,
          models.neoclassical_models,
          models.internal_boundary_condition_model,
          rp1,
          geo1,
          cp,
          pts,
          use_pereverzev=use_perev,
      )
      flatten(f'{variant}/{tag}/transport_perev{int(use_perev)}', ct, out)

    @jax.jit
    def ibcs(rp, geo, cp, pts):
      return ibc_builder.build_internal_boundary_conditions(
          runtime_params=rp,
          geo=geo,
          core_profiles=cp,
          pedestal_transition_state=pts,
          internal_boundary_condition_model=models.internal_boundary_condition_model,
      )

    flatten(f'{variant}/{tag}/ibc', ibcs(rp1, geo1, cp, pts), out)
    # Eager evaluation of the same (op by op).
    flatten(
        f'{variant}/{tag}/ibc_eager',
        ibc_builder.build_internal_boundary_conditions(
            runtime_params=rp1,
            geo=geo1,
            core_profiles=cp,
            pedestal_transition_state=pts,
            internal_boundary_condition_model=models.internal_boundary_condition_model,
        ),
        out,
    )
    for use_perev in (False, True):
      coeffs = calc_coeffs.calc_coeffs(
          runtime_params=rp1,
          geo=geo1,
          core_profiles=cp,
          explicit_source_profiles=esp,
          models=models,
          evolving_names=names,
          use_pereverzev=use_perev,
          pedestal_transition_state=pts,
      )
      flatten(f'{variant}/{tag}/coeffs_perev{int(use_perev)}', coeffs, out)

  coeffs_callback = calc_coeffs.CoeffsCallback(
      models=models, evolving_names=names
  )
  coeffs_old = coeffs_callback(
      runtime_params_t,
      geo_t,
      state.core_profiles,
      prev_core_profiles=None,
      dt=None,
      x=x_old,
      explicit_source_profiles=esp,
      explicit_call=True,
      pedestal_transition_state=pts,
  )
  residual_fun = functools.partial(
      residual_and_loss.theta_method_block_residual,
      coeffs_old=coeffs_old,
      dt=dt,
      runtime_params_t_plus_dt=rp1,
      geo_t_plus_dt=geo1,
      x_old=x_old,
      core_profiles_t=state.core_profiles,
      core_profiles_t_plus_dt=cp1,
      explicit_source_profiles=esp,
      models=models,
      evolving_names=names,
      pedestal_transition_state=pts,
  )
  res = jax.jit(residual_fun)
  jac = jax.jit(jax.jacfwd(residual_fun))
  out[f'{variant}/residual_x0'] = np.asarray(res(x0))
  out[f'{variant}/residual_xp'] = np.asarray(res(x0 * pert))
  out[f'{variant}/jac_x0'] = np.asarray(jac(x0))
  out[f'{variant}/jac_xp'] = np.asarray(jac(x0 * pert))

  x_new, sno = newton_raphson_solve_block.newton_raphson_solve_block(
      dt=dt,
      runtime_params_t=runtime_params_t,
      runtime_params_t_plus_dt=rp1,
      geo_t=geo_t,
      geo_t_plus_dt=geo1,
      x_old=x_old,
      core_profiles_t=state.core_profiles,
      core_profiles_t_plus_dt=cp1,
      explicit_source_profiles=esp,
      models=models,
      coeffs_callback=coeffs_callback,
      evolving_names=names,
      initial_guess_mode=enums.InitialGuessMode.LINEAR,
      maxiter=30,
      tol=1e-10,
      coarse_tol=1e-2,
      delta_reduction_factor=0.5,
      tau_min=0.01,
      pedestal_transition_state=pts,
      max_linesearch_steps=100,
  )
  flatten(f'{variant}/nr_x_new', x_new, out)
  flatten(f'{variant}/nr_numeric', sno, out)
  mult = getattr(pts.pedestal_model_output, 'transport_multipliers', None)
  print(variant, 'confinement', int(pts.confinement_mode), 'multipliers',
        None if mult is None else jax.tree.map(float, mult), flush=True)


def main():
  out_path, variants = sys.argv[1], sys.argv[2:]
  out = {}
  for v in variants:
    run_variant(v, out)
  np.savez(out_path, **out)
  print('saved', len(out), 'arrays to', out_path, 'with torax', torax.__file__)


if __name__ == '__main__':
  main()
