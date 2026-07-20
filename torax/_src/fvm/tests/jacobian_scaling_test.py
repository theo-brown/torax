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
"""Tests for jacobian_scaling and the use_jacobian_scaling solver option."""

from absl.testing import absltest
import numpy as np
from torax._src import state
from torax._src.orchestration import run_simulation
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


def _run(use_jacobian_scaling: bool):
  cfg = default_configs.get_default_config_dict()
  # A nonlinear transport model so the Newton solve is exercised for real.
  cfg['transport'] = {'model_name': 'CGM'}
  cfg['solver'] = {
      'solver_type': 'newton_raphson',
      'use_pereverzev': True,
      'residual_tol': 1e-8,
      'use_jacobian_scaling': use_jacobian_scaling,
  }
  cfg['numerics'] = {'t_final': 4.0, 'fixed_dt': 2.0}
  cfg['time_step_calculator'] = {'calculator_type': 'fixed'}
  return run_simulation.run_simulation(
      model_config.ToraxConfig.from_dict(cfg), progress_bar=False
  )


class JacobianScalingTest(absltest.TestCase):

  def test_scaled_solve_matches_unscaled(self):
    """The equilibrated solve must converge to the same solution."""
    data_tree_ref, history_ref = _run(use_jacobian_scaling=False)
    data_tree_scaled, history_scaled = _run(use_jacobian_scaling=True)
    self.assertEqual(history_ref.sim_error, state.SimError.NO_ERROR)
    self.assertEqual(history_scaled.sim_error, state.SimError.NO_ERROR)
    self.assertEqual(len(data_tree_ref.time), len(data_tree_scaled.time))
    for var in ['T_i', 'T_e', 'psi', 'n_e']:
      np.testing.assert_allclose(
          data_tree_scaled.profiles[var].to_numpy(),
          data_tree_ref.profiles[var].to_numpy(),
          rtol=1e-5,
          err_msg=f'{var} mismatch between scaled and unscaled solves',
      )
    # The nonlinear solve actually iterated.
    inner = data_tree_scaled.numerics.inner_solver_iterations.to_numpy()
    self.assertGreater(inner.sum(), 0)


if __name__ == '__main__':
  absltest.main()
