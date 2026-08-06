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
"""Solves constrained timesteps through the bordered Newton system."""

import functools

from absl.testing import absltest
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import math_utils
from torax._src.config import build_runtime_params
from torax._src.core_profiles import convertors
from torax._src.core_profiles import initialization
from torax._src.fvm import calc_coeffs
from torax._src.fvm import fvm_conversions
from torax._src.fvm import residual_and_loss
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.solver import constraints
from torax._src.solver import jacobian_pattern
from torax._src.solver import jax_root_finding
from torax._src.sources import source_profile_builders
from torax._src.test_utils import default_sources
from torax._src.torax_pydantic import model_config

_NUM_CELLS = 16
_DT = 0.2


def _build_case():
  """A small case with an implicit gas puff source."""
  torax_config = model_config.ToraxConfig.from_dict(
      dict(
          numerics=dict(
              evolve_ion_heat=True,
              evolve_electron_heat=True,
              evolve_density=True,
              evolve_current=True,
          ),
          plasma_composition=dict(),
          profile_conditions=dict(),
          geometry=dict(geometry_type='circular', n_rho=_NUM_CELLS),
          pedestal=dict(),
          sources=default_sources.get_default_source_config(),
          solver=dict(use_predictor_corrector=False, theta_implicit=1.0),
          transport=dict(
              model_name='combined',
              transport_models=[
                  dict(model_name='constant', chi_i=1.0),
              ],
              chi_min=0,
          ),
          time_step_calculator=dict(),
      )
  )
  models = torax_config.build_models()
  runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
      torax_config
  )(t=torax_config.numerics.t_initial)
  geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
  core_profiles = initialization.initial_core_profiles(
      runtime_params,
      geo,
      source_models=models.source_models,
      neoclassical_models=models.neoclassical_models,
  )
  evolving_names = ('T_i', 'T_e', 'n_e', 'psi')
  explicit_source_profiles = source_profile_builders.build_source_profiles(
      source_models=models.source_models,
      neoclassical_models=models.neoclassical_models,
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      explicit=True,
  )
  pedestal_transition_state = (
      pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode()
  )
  coeffs_old = calc_coeffs.calc_coeffs(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      models=models,
      explicit_source_profiles=explicit_source_profiles,
      evolving_names=evolving_names,
      use_pereverzev=False,
      pedestal_transition_state=pedestal_transition_state,
  )
  x_old = convertors.core_profiles_to_solver_x_tuple(
      core_profiles, evolving_names
  )

  def pde_residual(x_vec, runtime_params_t_plus_dt):
    return residual_and_loss.theta_method_block_residual(
        x_new_guess_vec=x_vec,
        dt=jnp.array(_DT),
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t_plus_dt=geo,
        x_old=x_old,
        core_profiles_t=core_profiles,
        core_profiles_t_plus_dt=core_profiles,
        explicit_source_profiles=explicit_source_profiles,
        models=models,
        coeffs_old=coeffs_old,
        evolving_names=evolving_names,
        pedestal_transition_state=pedestal_transition_state,
    )

  x_vec = fvm_conversions.cell_variable_tuple_to_vec(x_old)
  return pde_residual, runtime_params, geo, x_vec, evolving_names


class ConstraintsTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    jax.config.update('jax_enable_x64', True)
    (
        pde_residual,
        cls.runtime_params,
        cls.geo,
        cls.x_vec,
        cls.evolving_names,
    ) = _build_case()
    # staticmethod keeps the plain function from becoming a bound method
    # when accessed through the class.
    cls.pde_residual = staticmethod(pde_residual)
    n_e_index = cls.evolving_names.index('n_e')
    n_e = (
        cls.x_vec[n_e_index * _NUM_CELLS : (n_e_index + 1) * _NUM_CELLS]
        * convertors.SCALING_FACTORS['n_e']
    )
    cls.nbar_initial = float(math_utils.line_average(n_e, cls.geo))
    cls.s_total_reference = float(
        cls.runtime_params.sources['gas_puff'].S_total
    )

  def _solve(self, mode, target_factor, tau=0.5, u_min=None):
    config = constraints.ConstraintConfig.from_dict(
        dict(
            constraint='n_e_line_avg',
            target=self.nbar_initial * target_factor,
            actuator='sources.gas_puff.S_total',
            mode=mode,
            tau=tau,
            u_min=u_min,
        )
    )
    constraint = config.build_runtime_params(
        t=0.0, actuator_reference=self.s_total_reference
    )
    augmented = constraints.build_augmented_residual(
        pde_residual_fun=self.pde_residual,
        runtime_params_t_plus_dt=self.runtime_params,
        geo=self.geo,
        constraints=(constraint,),
        evolving_names=self.evolving_names,
        num_cells=_NUM_CELLS,
        dt=jnp.array(_DT),
    )
    z0 = jnp.concatenate([self.x_vec, jnp.array([constraint.u_hat_old])])
    z_root, metadata = jax_root_finding.root_newton_raphson(
        augmented, z0, tol=1e-8, use_jax_custom_root=False
    )
    self.assertEqual(int(metadata.error), 0)
    x_root = z_root[: self.x_vec.shape[0]]
    u_hat_root = float(z_root[-1])
    n_e_index = self.evolving_names.index('n_e')
    n_e = (
        x_root[n_e_index * _NUM_CELLS : (n_e_index + 1) * _NUM_CELLS]
        * convertors.SCALING_FACTORS['n_e']
    )
    nbar = float(math_utils.line_average(n_e, self.geo))
    return nbar, u_hat_root, constraint

  def test_hard_constraint_hits_target(self):
    # A huge tau freezes the actuator, giving the step's natural evolution.
    nbar_natural, _, _ = self._solve('relaxed', 1.0, tau=1e9)
    # Target 2% above the natural evolution: needs extra fuelling.
    target = nbar_natural * 1.02
    nbar, u_hat, _ = self._solve('hard', target / self.nbar_initial)
    self.assertLess(abs(nbar - target) / target, 1e-6)
    self.assertGreater(u_hat, 1.0)

  def test_bounded_hard_saturates_at_unreachable_target(self):
    """A target needing negative puff saturates the actuator at the bound."""
    # Unbounded hard mode reaches this target only with negative fuelling.
    _, u_unbounded, _ = self._solve('hard', 1.02)
    self.assertLess(u_unbounded, 0.0)
    # Bounded: the actuator sits at zero and the target is honestly missed.
    nbar, u_hat, constraint = self._solve('hard', 1.02, u_min=0.0)
    self.assertLess(abs(u_hat), 1e-6)
    g_hat = (nbar - float(constraint.target)) / float(constraint.target)
    self.assertGreater(g_hat, 0.0)
    # Complementarity: the active branch's partner is (numerically) zero.
    self.assertLess(min(abs(u_hat), abs(g_hat)), 1e-6)

  def test_bounded_hard_matches_unbounded_when_feasible(self):
    """With a reachable target the bound is inactive and changes nothing."""
    # Target above the natural evolution: met with positive fuelling.
    nbar_natural, _, _ = self._solve('relaxed', 1.0, tau=1e9)
    target_factor = nbar_natural * 1.02 / self.nbar_initial
    nbar_free, u_free, _ = self._solve('hard', target_factor)
    nbar_bounded, u_bounded, constraint = self._solve(
        'hard', target_factor, u_min=0.0
    )
    self.assertGreater(u_bounded, 0.0)
    self.assertLess(abs(u_bounded - u_free) / abs(u_free), 1e-6)
    self.assertLess(
        abs(nbar_bounded - float(constraint.target))
        / float(constraint.target),
        1e-6,
    )
    self.assertLess(abs(nbar_bounded - nbar_free) / nbar_free, 1e-8)

  def test_relaxed_constraint_satisfies_discrete_relaxation(self):
    tau = 0.5
    target_factor = 1.02
    nbar, u_hat, constraint = self._solve('relaxed', target_factor, tau=tau)
    g_hat = (nbar - float(constraint.target)) / float(constraint.target)
    # The converged step satisfies tau * (u - u_old) / dt + g_hat = 0.
    self.assertLess(abs(tau * (u_hat - 1.0) / _DT + g_hat), 1e-7)
    # The actuator moved against the residual violation: below target pushes
    # fuelling up, above target pushes it down.
    self.assertGreater((u_hat - 1.0) * (-g_hat), 0.0)

  def test_relaxation_time_dial(self):
    """Small tau approaches the hard constraint; large tau barely moves."""
    target_factor = 1.02
    target = self.nbar_initial * target_factor
    # One-step tracking requires tau / dt small against the actuator's
    # authority s = d(g_hat)/d(u_hat) (~4e-3 here): the converged violation
    # is g_natural * (tau/dt) / (s + tau/dt).
    nbar_fast, u_fast, _ = self._solve('relaxed', target_factor, tau=1e-5)
    nbar_slow, u_slow, _ = self._solve('relaxed', target_factor, tau=50.0)
    # Fast relaxation tracks the target closely within one step.
    self.assertLess(abs(nbar_fast - target) / target, 5e-3)
    # Slow relaxation leaves the actuator near its initial value.
    self.assertLess(abs(u_slow - 1.0), abs(u_fast - 1.0) / 10.0)
    self.assertLess(abs(nbar_slow - target) / target, 1.0)

  def test_augmented_pattern_covers_bordered_jacobian(self):
    config = constraints.ConstraintConfig.from_dict(
        dict(target=self.nbar_initial * 1.02, mode='relaxed')
    )
    constraint = config.build_runtime_params(
        t=0.0, actuator_reference=self.s_total_reference
    )
    augmented = constraints.build_augmented_residual(
        pde_residual_fun=self.pde_residual,
        runtime_params_t_plus_dt=self.runtime_params,
        geo=self.geo,
        constraints=(constraint,),
        evolving_names=self.evolving_names,
        num_cells=_NUM_CELLS,
        dt=jnp.array(_DT),
    )
    z0 = jnp.concatenate([self.x_vec, jnp.array([constraint.u_hat_old])])
    jacobian = np.asarray(jax.jacfwd(augmented)(z0))

    pde_pattern = jacobian_pattern.build_pattern(
        _NUM_CELLS, self.evolving_names, smoothing_matrix=None
    )
    declared = constraints.augment_pattern(
        pde_pattern, self.evolving_names, _NUM_CELLS, (constraint,)
    )
    abs_jacobian = np.abs(jacobian)
    row_max = np.maximum(abs_jacobian.max(axis=1, keepdims=True), 1e-300)
    measured = abs_jacobian > 1e-12 * row_max
    missed = measured & ~declared
    self.assertEqual(int(missed.sum()), 0)

    # The colored probe of the augmented pattern reconstructs the bordered
    # Jacobian and its Newton direction exactly.
    colors = jacobian_pattern.color_columns(declared)
    seeds, scatter = jacobian_pattern.build_seeds_and_scatter(
        declared, colors
    )
    _, jvp_fun = jax.linearize(augmented, z0)
    products = jax.vmap(jvp_fun)(jnp.asarray(seeds))
    reconstructed = np.asarray(
        jacobian_pattern.reconstruct_dense(products, scatter)
    )
    residual = np.asarray(augmented(z0))
    exact = np.linalg.solve(jacobian, -residual)
    probed = np.linalg.solve(reconstructed, -residual)
    np.testing.assert_allclose(probed, exact, rtol=1e-8)


if __name__ == '__main__':
  absltest.main()
