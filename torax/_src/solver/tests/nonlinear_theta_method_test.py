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

from unittest import mock

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from torax._src.core_profiles import convertors
from torax._src.fvm import calc_coeffs
from torax._src.fvm import fvm_conversions
from torax._src.fvm import residual_and_loss
from torax._src.orchestration import explicit_model_outputs as explicit_model_outputs_lib
from torax._src.orchestration import initial_state
from torax._src.orchestration import run_simulation
from torax._src.torax_pydantic import config

# Scaled residual of the unchanged state, between residual_tol (1e-5) and
# residual_coarse_tol (1e-2).
_RESIDUAL_SCALE = 1e-3


def _stalling_residual(x_new_guess_vec, *, x_old, evolving_names, **unused):
  """Residual whose Newton direction cannot reduce it, as at a coefficient jump.

  R = scale * C * (1 + u + 5 * [u != 0]) with u = (x - x_old) / scale. The
  Jacobian only sees the smooth part, so the Newton step looks like a descent
  direction, but any move off x_old hits the jump and raises the residual, as
  happens when a transport coefficient switches discontinuously. The line search
  therefore backtracks until the step is negligible and returns ~x_old.

  Args:
    x_new_guess_vec: Flattened trial state.
    x_old: State at the start of the step.
    evolving_names: Names of the evolving channels.
    **unused: Remaining residual arguments.

  Returns:
    The residual vector.
  """
  del unused
  x_old_vec = fvm_conversions.cell_variable_tuple_to_vec(x_old)
  scale = convertors.compute_residual_scaling_vector(evolving_names, x_old)
  u = (x_new_guess_vec - x_old_vec) / scale
  jump = jnp.where(jnp.any(u != 0.0), 5.0, 0.0)
  return scale * _RESIDUAL_SCALE * (1.0 + u + jump)


class NewtonRaphsonThetaMethodTest(absltest.TestCase):

  def _solve_stalled_step(self, solver_config):
    """Runs one Newton step on the stalling residual.

    Args:
      solver_config: Extra Newton-Raphson solver config fields.

    Returns:
      The new state, the old state (flattened) and the solver error state.
    """
    torax_config = config.ToraxConfig.from_dict(
        dict(
            numerics=dict(
                evolve_electron_heat=False,
                evolve_current=False,
                evolve_density=False,
            ),
            plasma_composition=dict(),
            profile_conditions=dict(
                I_p=15e6, I_p_source='profile_conditions', psi_source='j'
            ),
            geometry=dict(model_name='circular', n_rho=4),
            pedestal=dict(),
            sources=dict(),
            solver=dict(
                solver_type='newton_raphson',
                initial_guess_mode='x_old',
                use_pereverzev=False,
                **solver_config,
            ),
            transport=dict(
                core_transport_models=dict(
                    prescribed=dict(model_name='prescribed', chi_i=1)
                ),
            ),
            time_step_calculator=dict(),
        )
    )
    step_fn = run_simulation.make_step_fn(torax_config)
    runtime_params = step_fn.runtime_params_provider(
        t=torax_config.time_step_calculator.t_initial
    )
    geo, _ = step_fn.solver.models.geometry_model.initialize(runtime_params)
    sim_state = initial_state._get_initial_state(
        runtime_params=runtime_params, geo=geo, step_fn=step_fn
    )
    evolving_names = runtime_params.numerics.evolving_names
    solver_state = sim_state.core_profiles.to_solver_state()
    explicit_outputs = explicit_model_outputs_lib.ExplicitModelOutputs(
        sources={}, pedestal=sim_state.pedestal_output.structure
    )
    with mock.patch.object(
        residual_and_loss, 'theta_method_block_residual', _stalling_residual
    ):
      x_new, outputs = step_fn.solver._x_new_helper(
          dt=jnp.array(1e-3),
          runtime_params_t=runtime_params,
          runtime_params_t_plus_dt=runtime_params,
          geo_t=sim_state.geometry,
          geo_t_plus_dt=sim_state.geometry,
          solver_state_t=solver_state,
          solver_state_t_plus_dt=solver_state,
          explicit_model_outputs=explicit_outputs,
          coeffs_callback=calc_coeffs.CoeffsCallback(
              models=step_fn.solver.models, evolving_names=evolving_names
          ),
          evolving_names=evolving_names,
          model_state=sim_state.model_state,
      )
    x_old = convertors.solver_state_to_solver_x_tuple(
        solver_state, evolving_names
    )
    return (
        fvm_conversions.cell_variable_tuple_to_vec(x_new),
        fvm_conversions.cell_variable_tuple_to_vec(x_old),
        int(outputs.solver_error_state),
    )

  def test_stalled_step_is_accepted_at_coarse_tol_by_default(self):
    # Documents the profile freeze: the solve returns the unchanged state but
    # its residual is below residual_coarse_tol, so it is accepted (error 2).
    x_new, x_old, error = self._solve_stalled_step({})
    np.testing.assert_allclose(x_new, x_old, rtol=1e-6)
    self.assertEqual(error, 2)

  def test_residual_coarse_reduction_rejects_stalled_step(self):
    # With the check, a step that did not reduce the residual is not converged
    # (error 1), so the time step is retried with a smaller dt.
    _, _, error = self._solve_stalled_step({'residual_coarse_reduction': 0.5})
    self.assertEqual(error, 1)


if __name__ == '__main__':
  absltest.main()
