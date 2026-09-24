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

"""Tests for the structured Jacobian of the Newton-Raphson solver."""

import copy
import dataclasses
import functools
import json
import os
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import immutabledict
import jax
import jax.numpy as jnp
import numpy as np
from torax._src import jax_utils
from torax._src.config import build_runtime_params
from torax._src.core_profiles import convertors
from torax._src.core_profiles import updaters
from torax._src.fvm import calc_coeffs
from torax._src.fvm import enums
from torax._src.fvm import fvm_conversions
from torax._src.fvm import newton_raphson_solve_block
from torax._src.fvm import residual_and_loss
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import run_simulation
from torax._src.orchestration import step_function_processing
from torax._src.solver import structured_jacobian
from torax._src.sources.ion_cyclotron_source import toric_nn
from torax._src.torax_pydantic import model_config
from torax.examples import iterhybrid_predictor_corrector


def _write_dummy_toric_nn(path: str) -> None:
  # pylint: disable=protected-access
  network = toric_nn._ToricNN(
      hidden_sizes=[3],
      pca_coeffs=4,
      input_dim=10,
      radial_nodes=toric_nn._TORIC_GRID_SIZE,
  )
  _, params = network.init_with_output(jax.random.PRNGKey(0), jnp.ones(10))
  config = dataclasses.asdict(network)
  weights = jax.tree_util.tree_map(lambda x: x.tolist(), params['params'])
  for name in (
      toric_nn._HELIUM3_ID,
      toric_nn._TRITIUM_SECOND_HARMONIC_ID,
      toric_nn._ELECTRON_ID,
  ):
    config[name] = weights
  # pylint: enable=protected-access
  with open(path, 'w') as f:
    json.dump(config, f)


class StructuredJacobianTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update('jax_enable_x64', True)
    self.toric_nn_path = os.path.join(tempfile.mkdtemp(), 'toric_nn.json')
    _write_dummy_toric_nn(self.toric_nn_path)

  def _config(self, variant: str, n_rho: int = 25) -> dict:
    """The example config with the Newton solver and the physics of variant."""
    config = copy.deepcopy(iterhybrid_predictor_corrector.CONFIG)
    config['geometry']['n_rho'] = n_rho
    config['solver'] = dict(
        solver_type='newton_raphson',
        use_predictor_corrector=True,
        n_corrector_steps=2,
        use_pereverzev=True,
    )
    match variant:
      case 'local':
        pass
      case 'global_sources':
        config['sources'].update({
            'cyclotron_radiation': {},
            'impurity_radiation': {
                'model_name': 'P_in_scaled_flat_profile',
                'fraction_P_heating': 0.1,
            },
            'icrh': {'model_path': self.toric_nn_path, 'P_total': 10e6},
        })
        # The ToricNN model needs the height of the magnetic axis.
        config['geometry'] = {'geometry_type': 'circular', 'n_rho': n_rho}
        qlknn = config['transport']['core_transport_models']['qlknn']
        qlknn['rotation_mode'] = 'half_radius'
      case 'implicit_pedestal':
        config['pedestal'] = {
            'model_name': 'set_P_ped_n_ped',
            'set_pedestal': True,
            'P_ped': 9e4,
            'n_e_ped': 0.62e20,
            'rho_norm_ped_top': 0.9,
            'explicit_pedestal': False,
            'pedestal_profile_form': 'MTANH',
        }
        faces = np.linspace(0.0, 1.0, n_rho + 1)
        config['geometry'].pop('n_rho')
        config['geometry']['face_centers'] = faces + 0.1 * faces * (1 - faces)
        config['numerics']['min_rho_norm'] = 0.1
      case 'adaptive_pedestal':
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
      case 'beta_poloidal_prime':
        config['pedestal']['set_pedestal'] = False
        config['profile_conditions']['internal_boundary_conditions'] = {
            'model_name': 'beta_poloidal_prime',
            'rho_norm_edge': 0.85,
            'n_e_edge': 0.4e20,
            'beta_poloidal_prime': 0.3,
            'Ti_Te_ratio': 1.0,
        }
    return config

  def _solve_block_kwargs(self, config: dict) -> dict:
    """The arguments of one Newton solve block at the initial state."""
    step_fn = run_simulation.make_step_fn(
        model_config.ToraxConfig.from_dict(config)
    )
    state, _ = initial_state_lib.get_initial_state_and_post_processed_outputs(
        step_fn=step_fn
    )
    models = step_fn.solver.models
    runtime_params_t, geo_t, explicit_source_profiles, edge_outputs, pts = (
        step_function_processing.pre_step(
            input_state=state,
            runtime_params_provider=step_fn.runtime_params_provider,
            geometry_provider=step_fn.geometry_provider,
            models=models,
        )
    )
    dt = jnp.asarray(0.05)
    runtime_params_t_plus_dt, geo_t_plus_dt = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=state.t + dt,
            runtime_params_provider=step_fn.runtime_params_provider,
            geometry_provider=step_fn.geometry_provider,
            edge_outputs=edge_outputs,
            core_profiles=state.core_profiles,
        )
    )
    evolving_names = runtime_params_t.numerics.evolving_names
    return dict(
        dt=dt,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t=geo_t,
        geo_t_plus_dt=geo_t_plus_dt,
        x_old=convertors.core_profiles_to_solver_x_tuple(
            state.core_profiles, evolving_names
        ),
        core_profiles_t=state.core_profiles,
        core_profiles_t_plus_dt=updaters.provide_core_profiles_t_plus_dt(
            dt=dt,
            runtime_params_t=runtime_params_t,
            runtime_params_t_plus_dt=runtime_params_t_plus_dt,
            geo_t_plus_dt=geo_t_plus_dt,
            core_profiles_t=state.core_profiles,
        ),
        explicit_source_profiles=explicit_source_profiles,
        models=models,
        coeffs_callback=calc_coeffs.CoeffsCallback(
            models=models, evolving_names=evolving_names
        ),
        evolving_names=evolving_names,
        initial_guess_mode=enums.InitialGuessMode.LINEAR,
        maxiter=30,
        tol=1e-8,
        coarse_tol=1e-2,
        delta_reduction_factor=0.5,
        tau_min=0.01,
        pedestal_transition_state=pts,
        max_linesearch_steps=100,
    )

  def _residual_fun(self, kwargs: dict) -> functools.partial:
    """The residual of the Newton solve block, as it builds it."""
    return functools.partial(
        residual_and_loss.theta_method_block_residual,
        coeffs_old=kwargs['coeffs_callback'](
            kwargs['runtime_params_t'],
            kwargs['geo_t'],
            kwargs['core_profiles_t'],
            prev_core_profiles=None,
            dt=None,
            x=kwargs['x_old'],
            explicit_source_profiles=kwargs['explicit_source_profiles'],
            explicit_call=True,
            pedestal_transition_state=kwargs['pedestal_transition_state'],
        ),
        **{
            k: kwargs[k]
            for k in (
                'dt',
                'runtime_params_t_plus_dt',
                'geo_t_plus_dt',
                'x_old',
                'core_profiles_t',
                'core_profiles_t_plus_dt',
                'explicit_source_profiles',
                'models',
                'evolving_names',
                'pedestal_transition_state',
            )
        },
    )

  @parameterized.parameters(
      'local',
      'global_sources',
      'implicit_pedestal',
      'adaptive_pedestal',
      'beta_poloidal_prime',
  )
  def test_jacobian_matches_jacfwd(self, variant: str):
    kwargs = self._solve_block_kwargs(self._config(variant))
    residual_fun = self._residual_fun(kwargs)
    x = fvm_conversions.cell_variable_tuple_to_vec(kwargs['x_old'])
    x = x * (1.0 + 0.05 * np.random.RandomState(0).standard_normal(x.shape))
    dense = np.asarray(jax.jit(jax.jacfwd(residual_fun))(x))
    structured = np.asarray(
        jax.jit(structured_jacobian.jacobian_fn(residual_fun))(x)
    )
    np.testing.assert_array_less(
        np.abs(structured - dense).max(axis=1),
        1e-10 * np.linalg.norm(dense, axis=1),
    )

  # Through the jitted solve block, as in a simulation: the grid and
  # min_rho_norm are traced there.
  @parameterized.parameters('global_sources', 'implicit_pedestal')
  def test_newton_solve_matches_dense(self, variant: str):
    kwargs = self._solve_block_kwargs(self._config(variant))
    x_dense, out_dense = newton_raphson_solve_block.newton_raphson_solve_block(
        jacobian_mode='dense', **kwargs
    )
    # With the per-iteration check of the assembly.
    with jax_utils.enable_errors(True):
      x_struct, out_struct = (
          newton_raphson_solve_block.newton_raphson_solve_block(
              jacobian_mode='structured', **kwargs
          )
      )
    self.assertEqual(int(out_dense.solver_error_state), 0)
    self.assertEqual(
        int(out_dense.inner_solver_iterations),
        int(out_struct.inner_solver_iterations),
    )
    for a, b in zip(x_dense, x_struct):
      np.testing.assert_allclose(a.value, b.value, rtol=1e-8)

  def test_user_defined_models_are_listed(self):
    torax_config = model_config.ToraxConfig.from_dict(
        self._config('local', n_rho=10)
    )
    models = torax_config.build_models()
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=0.0)
    self.assertEmpty(
        structured_jacobian._user_defined_models(models, runtime_params)
    )
    sources = dict(models.source_models.standard_sources)
    sources['generic_heat'] = dataclasses.replace(
        sources['generic_heat'], model_func=mock.Mock()
    )
    models = dataclasses.replace(
        models,
        source_models=dataclasses.replace(
            models.source_models,
            standard_sources=immutabledict.immutabledict(sources),
        ),
    )
    self.assertEqual(
        structured_jacobian._user_defined_models(models, runtime_params),
        ['model function of source generic_heat'],
    )

  def test_error_check_catches_missing_coupling(self):
    kwargs = self._solve_block_kwargs(self._config('local', n_rho=10))
    x = fvm_conversions.cell_variable_tuple_to_vec(kwargs['x_old'])
    with (
        mock.patch.object(
            structured_jacobian, '_transport_reach', return_value=(1, 0)
        ),
        jax_utils.enable_errors(True),
    ):
      jac_fn = jax.jit(
          structured_jacobian.jacobian_fn(self._residual_fun(kwargs))
      )
      with self.assertRaisesRegex(Exception, 'misses a coupling'):
        jax.block_until_ready(jac_fn(x))


if __name__ == '__main__':
  absltest.main()
