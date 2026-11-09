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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from torax._src.fvm import cell_variable
from torax._src.fvm import newton_raphson_solve_block as nrsb
from torax._src.orchestration import run_simulation
from torax._src.solver import jax_root_finding
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config as torax_config_lib


def _cell_variable(value):
  return cell_variable.CellVariable(
      value=jnp.asarray(value),
      face_centers=jnp.linspace(0.0, 1.0, len(value) + 1),
      right_face_constraint=jnp.asarray(1.0),
      right_face_grad_constraint=None,
  )


class LogVariablesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update('jax_enable_x64', True)

  def test_log_variable_mask(self):
    names = ('T_i', 'T_e', 'psi', 'n_e')
    x = tuple(_cell_variable(np.ones(3)) for _ in names)
    np.testing.assert_array_equal(
        nrsb.log_variable_mask(names, x),
        [True] * 6 + [False] * 3 + [True] * 3,
    )

  def test_round_trip(self):
    mask = jnp.array([True, True, False, False])
    x = jnp.array([2.0, 0.5, -3.0, 0.0])
    y = nrsb.to_log_variables(x, mask)
    np.testing.assert_allclose(y, [np.log(2.0), np.log(0.5), -3.0, 0.0])
    np.testing.assert_allclose(nrsb.from_log_variables(y, mask), x)

  def test_same_root_and_positive_path(self):
    # R(x) = log(x / a): a full Newton step from x0 = 10 lands at x < 0, where
    # R is NaN, so the plain iteration has to backtrack. In log variables the
    # residual is linear and Newton converges in a single step.
    a = jnp.array([0.1, 0.2])
    mask = jnp.array([True, True])
    residual = lambda x: jnp.log(x / a)
    x0 = jnp.array([10.0, 10.0])

    x_plain, meta_plain = jax_root_finding.root_newton_raphson(residual, x0)
    y_root, meta_log = jax_root_finding.root_newton_raphson(
        lambda y: residual(nrsb.from_log_variables(y, mask)),
        nrsb.to_log_variables(x0, mask),
    )
    x_log = nrsb.from_log_variables(y_root, mask)

    np.testing.assert_allclose(x_log, a, rtol=1e-8)
    np.testing.assert_allclose(x_plain, a, rtol=1e-4)
    self.assertEqual(int(meta_log.error), 0)
    self.assertEqual(int(meta_log.iterations), 1)
    self.assertGreater(int(meta_plain.iterations), int(meta_log.iterations))


class LogVariablesSimulationTest(absltest.TestCase):

  def test_default_config_matches_linear_variables(self):
    # A short simulation converges to the same profiles whether or not the
    # Newton iteration runs in log variables.
    outputs = {}
    for log_positive_variables in (False, True):
      config_dict = default_configs.get_default_config_dict()
      config_dict['numerics'] = {'evolve_density': True}
      config_dict['time_step_calculator'] = {
          'calculator_type': 'fixed',
          'dt': 0.1,
          't_final': 0.5,
      }
      config_dict['solver'] = {
          'solver_type': 'newton_raphson',
          'log_positive_variables': log_positive_variables,
      }
      torax_config = torax_config_lib.ToraxConfig.from_dict(config_dict)
      history = run_simulation.run_simulation(torax_config, progress_bar=False)
      outputs[log_positive_variables] = (
          history.simulation_output_to_xr().profiles
      )
    for name in ('T_i', 'T_e', 'n_e', 'psi'):
      np.testing.assert_allclose(
          outputs[True][name].values[-1],
          outputs[False][name].values[-1],
          rtol=1e-4,
          err_msg=name,
      )


if __name__ == '__main__':
  absltest.main()
