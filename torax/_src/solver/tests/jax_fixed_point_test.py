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

import re
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize
from torax._src.solver import jax_fixed_point


def _func_np(x, c1, c2):
  return np.sqrt(c1 / (x + c2))


def _func_jnp(x, c1, c2):
  return jnp.sqrt(c1 / (x + c2))


class FixedPointTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update('jax_enable_x64', True)

  @parameterized.named_parameters(
      dict(
          testcase_name='maxiter',
          maxiter=2,
          atol=0.0,
          rtol=0.0,
      ),
      dict(
          testcase_name='atol',
          maxiter=500,
          atol=1e-8,
          rtol=0.0,
      ),
      dict(
          testcase_name='rtol',
          maxiter=500,
          atol=0.0,
          rtol=1e-5,
      ),
  )
  def test_fixed_point_convergence(self, maxiter, atol, rtol):
    c1 = np.array([10, 12.0])
    c2 = np.array([3, 5.0])
    x = np.array([1.2, 1.3])

    out_jnp = jax_fixed_point.fixed_point(
        _func_jnp,
        x,
        args=(c1, c2),
        maxiter=maxiter,
        atol=atol,
        rtol=rtol,
    )

    # Scipy's fixed_point raises a RuntimeError if the maximum number of
    # iterations is reached. As a fallback, we can extract the expected output
    # from the error message.
    try:
      out_expected = optimize.fixed_point(
          _func_np,
          x,
          args=(c1, c2),
          maxiter=maxiter,
          method='iteration',
      )
    except RuntimeError as e:
      out_expected = np.array(
          [float(f) for f in re.split(r'\[|]|\s+', e.args[0])[-3:-1]]
      )

    chex.assert_trees_all_close(out_expected, out_jnp)

  def test_fixed_point_backtracking(self):
    c1 = np.array([10, 12.0])
    c2 = np.array([3, 5.0])
    x = np.array([1.2, 1.3])

    out_with_backtracking = jax_fixed_point.fixed_point(
        _func_jnp,
        x,
        args=(c1, c2),
        maxiter=500,
        use_backtracking=True,
        step_size_reduction_factor=0.5,
        max_backtrack_steps=5,
        atol=1e-5,
    )

    out_without_backtracking = jax_fixed_point.fixed_point(
        _func_jnp,
        x,
        args=(c1, c2),
        maxiter=500,
        use_backtracking=False,
        atol=1e-5,
    )
    chex.assert_trees_all_close(
        out_with_backtracking, out_without_backtracking, atol=1e-5
    )

  @parameterized.named_parameters(
      dict(testcase_name='max_iterations', criterion='max_iterations'),
      dict(testcase_name='tolerance', criterion='tolerance'),
  )
  def test_anderson_matches_picard_fixed_point(self, criterion):
    """Anderson must converge to the same fixed point as plain Picard."""
    c1 = np.array([10, 12.0])
    c2 = np.array([3, 5.0])
    x = np.array([1.2, 1.3])

    out_anderson = jax_fixed_point.fixed_point(
        _func_jnp,
        x,
        args=(c1, c2),
        maxiter=100,
        atol=1e-12,
        rtol=0.0,
        termination_criterion=criterion,
        acceleration='anderson',
    )
    chex.assert_trees_all_close(
        out_anderson, _func_jnp(out_anderson, c1, c2), atol=1e-10
    )

  def test_anderson_converges_faster_than_picard(self):
    """Anderson should beat Picard on a slowly converging linear iteration."""
    rng = np.random.RandomState(0)
    n = 12
    a = rng.randn(n, n)
    # Spectral radius just below 1, so plain Picard converges very slowly.
    a = a / np.abs(np.linalg.eigvals(a)).max() * 0.95
    b = rng.randn(n)
    func = lambda x, a, b: a @ x + b
    x0 = np.zeros(n)

    def residual_norm(**kwargs):
      out = jax_fixed_point.fixed_point(
          func,
          x0,
          args=(a, b),
          maxiter=10,
          termination_criterion='max_iterations',
          **kwargs,
      )
      return float(jnp.linalg.norm(func(out, a, b) - out))

    self.assertLess(
        residual_norm(acceleration='anderson', anderson_depth=5),
        0.1 * residual_norm(),
    )

  def test_anderson_full_depth_terminates_finitely(self):
    """With full depth, Anderson on a linear problem is a Krylov method.

    It must therefore reach the exact solution in at most n iterations.
    """
    rng = np.random.RandomState(1)
    n = 8
    a = rng.randn(n, n)
    a = a / np.abs(np.linalg.eigvals(a)).max() * 0.9
    b = rng.randn(n)
    out = jax_fixed_point.fixed_point(
        lambda x, a, b: a @ x + b,
        np.zeros(n),
        args=(a, b),
        maxiter=n + 1,
        termination_criterion='max_iterations',
        acceleration='anderson',
        anderson_depth=n,
    )
    expected = np.linalg.solve(np.eye(n) - a, b)
    chex.assert_trees_all_close(np.asarray(out), expected, atol=1e-8)

  def test_anderson_preserves_constant_pytree_leaves(self):
    """Leaves that do not change across iterations must be passed through.

    The iterate pytree in TORAX is a tuple of `CellVariable`s, which carries
    the mesh and the boundary conditions alongside the evolving values.
    """
    constant = jnp.array([1.0, 2.0, 3.0])

    def func(x):
      value, const = x
      return (0.5 * value + 0.3 * jnp.sin(value) + const, const)

    out = jax_fixed_point.fixed_point(
        func,
        (jnp.ones(3), constant),
        maxiter=20,
        termination_criterion='max_iterations',
        acceleration='anderson',
    )
    chex.assert_trees_all_equal(out[1], constant)

  def test_anderson_is_differentiable(self):
    """The `max_iterations` path must stay reverse-differentiable."""
    rng = np.random.RandomState(2)
    a = rng.randn(5, 5) * 0.1

    def loss(b):
      out = jax_fixed_point.fixed_point(
          lambda x, a, b: jnp.tanh(a @ x) + b,
          jnp.zeros(5),
          args=(a, b),
          maxiter=8,
          termination_criterion='max_iterations',
          acceleration='anderson',
      )
      return jnp.sum(out**2)

    grad = jax.grad(loss)(jnp.asarray(rng.randn(5)))
    self.assertTrue(np.all(np.isfinite(np.asarray(grad))))

  def test_anderson_invalid_arguments(self):
    with self.assertRaises(ValueError):
      jax_fixed_point.fixed_point(
          _func_jnp, np.array([1.0]), acceleration='not_a_scheme'
      )
    with self.assertRaises(ValueError):
      jax_fixed_point.fixed_point(
          _func_jnp,
          np.array([1.0]),
          acceleration='anderson',
          anderson_depth=0,
      )


if __name__ == '__main__':
  absltest.main()
