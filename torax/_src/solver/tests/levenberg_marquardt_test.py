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
"""Integration tests for the optimistix Levenberg-Marquardt solver."""

from absl.testing import absltest
import numpy as np
from torax._src import state
from torax._src.orchestration import run_simulation
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


def _run(solver_config: dict):
  cfg = default_configs.get_default_config_dict()
  # A nonlinear (critical-gradient) transport model so the nonlinear solve is
  # exercised for real: the constant-chi default converges in zero iterations.
  cfg['transport'] = {'model_name': 'CGM'}
  cfg['solver'] = solver_config | {'use_pereverzev': True}
  cfg['numerics'] = {'t_final': 4.0, 'fixed_dt': 2.0}
  cfg['time_step_calculator'] = {'calculator_type': 'fixed'}
  return run_simulation.run_simulation(
      model_config.ToraxConfig.from_dict(cfg), progress_bar=False
  )


class LevenbergMarquardtTest(absltest.TestCase):

  def test_levenberg_marquardt_matches_newton_raphson(self):
    """LM and NR must converge to the same theta-method solution."""
    data_tree_nr, history_nr = _run(
        {'solver_type': 'newton_raphson', 'residual_tol': 1e-7}
    )
    data_tree_lm, history_lm = _run(
        {'solver_type': 'levenberg_marquardt', 'residual_tol': 1e-7}
    )
    self.assertEqual(history_nr.sim_error, state.SimError.NO_ERROR)
    self.assertEqual(history_lm.sim_error, state.SimError.NO_ERROR)
    self.assertEqual(len(data_tree_nr.time), len(data_tree_lm.time))
    for var in ['T_i', 'T_e']:
      np.testing.assert_allclose(
          data_tree_lm.profiles[var].to_numpy(),
          data_tree_nr.profiles[var].to_numpy(),
          rtol=1e-5,
          err_msg=f'{var} mismatch between LM and NR solutions',
      )
    # LM actually iterated (the problem is genuinely nonlinear).
    inner = data_tree_lm.numerics.inner_solver_iterations.to_numpy()
    self.assertGreater(inner.sum(), 0)


if __name__ == '__main__':
  absltest.main()
