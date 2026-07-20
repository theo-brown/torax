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
"""Combined test for Jacobian scaling + smooth transport clipping."""

from absl.testing import absltest
import numpy as np
from torax._src import state
from torax._src.orchestration import run_simulation
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


class ScalingSmoothClipCombinedTest(absltest.TestCase):

  def test_combined_options_match_baseline(self):
    """Scaling + smooth clip together must reproduce the baseline solution.

    The scaling changes only the solver's internal representation and the
    smooth clip (at small width) perturbs transport coefficients only near
    the clip bounds, so the combination must stay close to the plain solve.
    """

    def run(enabled: bool):
      cfg = default_configs.get_default_config_dict()
      cfg['transport'] = {
          'model_name': 'CGM',
          'clip_smoothing_width': 0.01 if enabled else 0.0,
      }
      cfg['solver'] = {
          'solver_type': 'newton_raphson',
          'use_pereverzev': True,
          'residual_tol': 1e-8,
          'use_jacobian_scaling': enabled,
      }
      cfg['numerics'] = {'t_final': 4.0, 'fixed_dt': 2.0}
      cfg['time_step_calculator'] = {'calculator_type': 'fixed'}
      return run_simulation.run_simulation(
          model_config.ToraxConfig.from_dict(cfg), progress_bar=False
      )

    data_tree_base, history_base = run(False)
    data_tree_both, history_both = run(True)
    self.assertEqual(history_base.sim_error, state.SimError.NO_ERROR)
    self.assertEqual(history_both.sim_error, state.SimError.NO_ERROR)
    self.assertEqual(len(data_tree_base.time), len(data_tree_both.time))
    for var in ['T_i', 'T_e']:
      np.testing.assert_allclose(
          data_tree_both.profiles[var].to_numpy(),
          data_tree_base.profiles[var].to_numpy(),
          rtol=5e-2,
          err_msg=f'{var} mismatch between combined-options and baseline',
      )
    inner = data_tree_both.numerics.inner_solver_iterations.to_numpy()
    self.assertGreater(inner.sum(), 0)


if __name__ == '__main__':
  absltest.main()
