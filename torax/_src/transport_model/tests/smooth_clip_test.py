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
"""Tests for the smooth transport-coefficient clipping."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import pydantic
from torax._src import state
from torax._src.orchestration import run_simulation
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import transport_model


class SmoothClipFunctionTest(absltest.TestCase):

  def test_converges_to_hard_clip_as_width_shrinks(self):
    x = jnp.linspace(-50.0, 150.0, 201)
    lo, hi = 0.15, 100.0
    hard = jnp.clip(x, lo, hi)
    for width in [1e-2, 1e-3, 1e-4]:
      smooth = transport_model.smooth_clip(x, lo, hi, width)
      # Max deviation from the hard clip is w_b * log(2) at each bound b,
      # with w_b = width * |b|.
      self.assertLessEqual(
          float(jnp.max(jnp.abs(smooth - hard))),
          1.1 * width * max(abs(lo), abs(hi)) * np.log(2.0),
      )

  def test_bound_relative_perturbation(self):
    # The perturbation of the lower bound must be relative to |lo|, not to
    # the (much larger) clip interval: with lo=0.05 and hi=100, values pinned
    # at the lower clip must stay within a few percent of lo.
    lo, hi = 0.05, 100.0
    width = 0.02
    y = transport_model.smooth_clip(jnp.array(0.0), lo, hi, width)
    self.assertLess(float(jnp.abs(y - lo)), 2 * width * lo)

  def test_asymptotes_and_monotonicity(self):
    lo, hi = -50.0, 50.0
    width = 0.02
    x = jnp.linspace(-5000.0, 5000.0, 2001)
    y = transport_model.smooth_clip(x, lo, hi, width)
    # Monotone non-decreasing.
    self.assertGreaterEqual(float(jnp.min(jnp.diff(y))), 0.0)
    # Asymptotes to the bounds far outside.
    np.testing.assert_allclose(float(y[0]), lo, atol=1e-8)
    np.testing.assert_allclose(float(y[-1]), hi, atol=1e-8)
    # Identity well inside the bounds. The deviation decays as
    # w_b * exp(-d / w_b) at distance d from bound b (w_b = 1 here), so a
    # margin of 15 w_b makes it negligible.
    w = width * abs(lo)
    inner = (x > lo + 15 * w) & (x < hi - 15 * w)
    np.testing.assert_allclose(y[inner], x[inner], atol=1e-3)

  def test_derivative_continuous_and_bounded(self):
    lo, hi = 0.15, 100.0
    width = 0.02
    grad = jax.vmap(
        jax.grad(lambda v: transport_model.smooth_clip(v, lo, hi, width))
    )
    x = jnp.linspace(-20.0, 120.0, 400001)
    g = grad(x)
    # Derivative in [0, 1]: no amplification, no negative slopes.
    self.assertGreaterEqual(float(jnp.min(g)), 0.0)
    self.assertLessEqual(float(jnp.max(g)), 1.0 + 1e-9)
    # The hard clip has a jump in the derivative at the bounds; the smooth
    # clip's derivative changes by at most O(dx / w_min) between neighbours,
    # where w_min is the smallest transition width (at the lower bound).
    dx = float(x[1] - x[0])
    w_min = width * abs(lo)
    self.assertLessEqual(float(jnp.max(jnp.abs(jnp.diff(g)))), dx / w_min)


class SmoothClipIntegrationTest(absltest.TestCase):

  def test_simulation_with_smooth_clip_close_to_hard_clip(self):
    def run(clip_smoothing_width):
      cfg = default_configs.get_default_config_dict()
      # CGM produces chi values that hit the chi_min clip near the edge, so
      # the clipping path is genuinely exercised.
      cfg['transport'] = {
          'model_name': 'CGM',
          'clip_smoothing_width': clip_smoothing_width,
      }
      cfg['solver'] = {'solver_type': 'newton_raphson', 'use_pereverzev': True}
      cfg['numerics'] = {'t_final': 4.0, 'fixed_dt': 2.0}
      cfg['time_step_calculator'] = {'calculator_type': 'fixed'}
      return run_simulation.run_simulation(
          model_config.ToraxConfig.from_dict(cfg), progress_bar=False
      )

    data_tree_hard, history_hard = run(0.0)
    data_tree_smooth, history_smooth = run(0.01)
    self.assertEqual(history_hard.sim_error, state.SimError.NO_ERROR)
    self.assertEqual(history_smooth.sim_error, state.SimError.NO_ERROR)
    for var in ['T_i', 'T_e']:
      a = data_tree_hard.profiles[var].to_numpy()
      b = data_tree_smooth.profiles[var].to_numpy()
      # A 1% transition width perturbs the transport coefficients only near
      # the clip bounds; profiles should agree to a few percent.
      np.testing.assert_allclose(b, a, rtol=5e-2)

  def test_width_validation(self):
    cfg = default_configs.get_default_config_dict()
    cfg['transport'] = {'model_name': 'constant', 'clip_smoothing_width': 0.6}
    with self.assertRaisesRegex(pydantic.ValidationError, 'clip_smoothing'):
      model_config.ToraxConfig.from_dict(cfg)


if __name__ == '__main__':
  absltest.main()
