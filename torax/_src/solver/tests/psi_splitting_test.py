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
"""Tests for operator splitting of the psi equation."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.orchestration import run_simulation
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name

_EVOLVING = ('T_i', 'T_e', 'n_e', 'psi')


def _step(
    solver_overrides: dict[str, Any],
    numerics_overrides: dict[str, Any] | None = None,
    n_steps: int = 2,
) -> dict[str, np.ndarray]:
  """Runs a few steps of a small simulation and returns the final profiles."""
  config = default_configs.get_default_config_dict()
  config['numerics'] = {
      'evolve_ion_heat': True,
      'evolve_electron_heat': True,
      'evolve_density': True,
      'evolve_current': True,
      'fixed_dt': 0.05,
      't_final': 1.0,
      **(numerics_overrides or {}),
  }
  config['solver'] = {'solver_type': 'linear', **solver_overrides}
  torax_config = model_config.ToraxConfig.from_dict(config)
  state, post_processed, step_fn = run_simulation.prepare_simulation(
      torax_config
  )
  for _ in range(n_steps):
    state, post_processed = step_fn(state, post_processed)
  return {
      name: np.asarray(getattr(state.core_profiles, name).value)
      for name in _EVOLVING
  }


class PsiSplittingTest(parameterized.TestCase):

  @parameterized.parameters('lie', 'strang')
  def test_split_psi_stays_close_to_unsplit(self, order):
    """Splitting evolves every channel and only perturbs the solution weakly."""
    unsplit = _step({})
    split = _step({'split_psi': True, 'split_psi_order': order})
    for name in _EVOLVING:
      self.assertTrue(np.all(np.isfinite(split[name])), name)
      # Compare against the profile scale rather than element-wise: psi passes
      # through small values near the axis where a relative error is
      # meaningless.
      scale = np.abs(unsplit[name]).max()
      self.assertLess(
          np.abs(split[name] - unsplit[name]).max() / scale, 5e-2, name
      )

  def test_split_psi_is_noop_without_current_evolution(self):
    """With evolve_current=False there is nothing to split off."""
    numerics = {'evolve_current': False}
    unsplit = _step({}, numerics)
    split = _step({'split_psi': True}, numerics)
    for name in _EVOLVING:
      np.testing.assert_array_equal(split[name], unsplit[name], err_msg=name)

  def test_split_psi_is_noop_when_only_psi_evolves(self):
    """With psi as the only channel there is no coupled block to shrink."""
    numerics = {
        'evolve_ion_heat': False,
        'evolve_electron_heat': False,
        'evolve_density': False,
    }
    unsplit = _step({}, numerics)
    split = _step({'split_psi': True}, numerics)
    for name in _EVOLVING:
      np.testing.assert_array_equal(split[name], unsplit[name], err_msg=name)

  def test_strang_splitting_error_is_smaller_than_lie(self):
    """Strang is second order in the splitting error, Lie only first order.

    The unsplit solve at the same dt isolates the splitting error from the
    time-discretisation error, which is common to all three runs.
    """
    unsplit = _step({})
    lie = _step({'split_psi': True})
    strang = _step({'split_psi': True, 'split_psi_order': 'strang'})
    lie_error = np.abs(lie['psi'] - unsplit['psi']).max()
    strang_error = np.abs(strang['psi'] - unsplit['psi']).max()
    self.assertLess(strang_error, lie_error)


if __name__ == '__main__':
  absltest.main()
