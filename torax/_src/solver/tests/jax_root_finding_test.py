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
import functools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize
from torax._src.solver import jax_root_finding


# Adapted from the example in:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html
def function_to_find_root(x, a, b):
  array_construct = jnp.array if isinstance(x, jax.Array) else np.array
  return array_construct(
      [
          x[0] + 0.5 * (x[0] - b * x[1]) ** 3.0 - 1.0,
          a * (x[1] - b * x[0]) ** 3.0 + x[1],
      ],
      dtype=x.dtype,
  )


class NewtonRaphsonSolveBlockTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update('jax_enable_x64', True)

  @parameterized.named_parameters(
      # All search directions are positive.
      ('positive_search', 0.5, 0.1, (0.0, 0.0)),
      # Assume find the root even with a negative search direction.
      ('negative_search', 1.0, 1.0, (2.0, 1.0)),
  )
  def test_root_newton_raphson_basic(
      self, a: float, b: float, x0: tuple[float, float]
  ):
    dtype = np.float64
    tol = 1e-9
    f_closed = functools.partial(function_to_find_root, a=a, b=b)

    x_init = np.array(x0, dtype=dtype)
    sol_np = optimize.root(f_closed, x0, tol=tol)

    @jax.jit(static_argnames=['tol', 'maxiter'])
    def root_jax(x, tol, maxiter):
      return jax_root_finding.root_newton_raphson(
          f_closed, x, tol=tol, maxiter=maxiter
      )

    sol_jax, metadata = root_jax(x_init, tol=tol, maxiter=100)

    with self.subTest('solver_correctness_against_scipy'):
      chex.assert_trees_all_close(sol_np.x, sol_jax, atol=tol)
      self.assertFalse(bool(metadata.error))

    with self.subTest('auxiliary_data'):
      self.assertGreater(int(metadata.iterations), 0)
      self.assertEqual(int(metadata.error), 0)

    with self.subTest('maxiter'):
      _, metadata = root_jax(x_init, tol=tol, maxiter=1)
      self.assertEqual(int(metadata.iterations), 1)
      self.assertEqual(int(metadata.error), 1)
      self.assertTrue(jnp.isdtype(metadata.iterations.dtype, 'integral'))

    def loss(x, a, b):
      root = jax_root_finding.root_newton_raphson(
          functools.partial(function_to_find_root, a=a, b=b), x, tol=tol
      )[0]
      return jnp.sum(root**2)

    eps = 1e-4
    a_grad_diff = (loss(x_init, a + eps, b) - loss(sol_np.x, a - eps, b)) / (
        2 * eps
    )
    b_grad_diff = (loss(x_init, a, b + eps) - loss(sol_np.x, a, b - eps)) / (
        2 * eps
    )
    x_grad, a_grad, b_grad = jax.grad(loss, argnums=(0, 1, 2))(x_init, a, b)

    with self.subTest('gradient_correctness'):
      chex.assert_trees_all_equal(x_grad, jnp.array([0.0, 0.0], dtype=dtype))
      chex.assert_trees_all_close(a_grad, a_grad_diff, atol=1e-4)
      chex.assert_trees_all_close(b_grad, b_grad_diff, atol=1e-4)

  def test_root_newton_raphson_separate_norms(self):
    f_closed = functools.partial(function_to_find_root, a=0.5, b=0.1)
    x_init = np.array((0.0, 0.0), dtype=np.float64)
    sol_np = optimize.root(f_closed, x_init, tol=1e-9)

    # Use RMS norm for linesearch and Linf norm for convergence.
    linesearch_norm = lambda x: jnp.sqrt(jnp.sum(x**2))
    convergence_norm = lambda x: jnp.max(jnp.abs(x))

    sol_jax, metadata = jax_root_finding.root_newton_raphson(
        f_closed,
        x_init,
        tol=1e-9,
        maxiter=100,
        linesearch_norm=linesearch_norm,
        convergence_norm=convergence_norm,
    )
    chex.assert_trees_all_close(sol_np.x, sol_jax, atol=1e-9)
    self.assertEqual(int(metadata.error), 0)

  def test_make_scaled_norm_math(self):
    res = jnp.array([2.0, -6.0])
    scales = jnp.array([1.0, 2.0])

    rms_norm_fn = lambda x: jax_root_finding.rms_norm(x, scales)
    linf_norm_fn = lambda x: jax_root_finding.max_abs_norm(x, scales)

    # scaled = [2.0, -3.0]
    # RMS: sqrt(mean([4.0, 9.0])) = sqrt(6.5)
    np.testing.assert_allclose(rms_norm_fn(res), np.sqrt(6.5))
    # Linf: max(|2.0|, |-3.0|) = 3.0
    np.testing.assert_allclose(linf_norm_fn(res), 3.0)

  def test_root_newton_raphson_scaled_norms(self):
    f_closed = functools.partial(function_to_find_root, a=0.5, b=0.1)
    x_init = np.array((0.0, 0.0), dtype=np.float64)
    sol_np = optimize.root(f_closed, x_init, tol=1e-9)

    scales = jnp.array([2.0, 5.0], dtype=np.float64)
    rms_norm_fn = lambda x: jax_root_finding.rms_norm(x, scales)
    linf_norm_fn = lambda x: jax_root_finding.max_abs_norm(x, scales)

    sol_jax, metadata = jax_root_finding.root_newton_raphson(
        f_closed,
        x_init,
        tol=1e-9,
        maxiter=100,
        linesearch_norm=rms_norm_fn,
        convergence_norm=linf_norm_fn,
    )
    chex.assert_trees_all_close(sol_np.x, sol_jax, atol=1e-9)
    self.assertEqual(int(metadata.error), 0)

  @parameterized.parameters(
      dict(norm=1e-6, coarse_max_norm=None, expected=0),
      dict(norm=1e-6, coarse_max_norm=0.0, expected=0),
      dict(norm=5e-3, coarse_max_norm=None, expected=2),
      dict(norm=5e-3, coarse_max_norm=1e-2, expected=2),
      dict(norm=5e-3, coarse_max_norm=1e-3, expected=1),
      dict(norm=5e-2, coarse_max_norm=None, expected=1),
  )
  def test_get_error_flag_coarse_max_norm(
      self, norm, coarse_max_norm, expected
  ):
    error = jax_root_finding._get_error_flag(
        norm=jnp.array(norm),
        coarse_tol=1e-2,
        tol=1e-5,
        coarse_max_norm=(
            None if coarse_max_norm is None else jnp.array(coarse_max_norm)
        ),
    )
    self.assertEqual(int(error), expected)

  def test_coarse_max_norm_rejects_stalled_solve(self):
    """A stalled solve below coarse_tol is accepted unless it made progress."""
    # No root: |f| has a minimum of 3e-3 at x = 0, below coarse_tol = 1e-2.
    f = lambda x: 3e-3 * jnp.cosh(x)
    x0 = jnp.array([0.5])
    kwargs = dict(tol=1e-5, coarse_tol=1e-2, use_jax_custom_root=False)

    _, metadata = jax_root_finding.root_newton_raphson(f, x0, **kwargs)
    self.assertEqual(int(metadata.error), 2)

    ref_norm = jax_root_finding.max_abs_norm(f(x0))
    _, metadata = jax_root_finding.root_newton_raphson(
        f, x0, coarse_max_norm=0.5 * ref_norm, **kwargs
    )
    self.assertEqual(int(metadata.error), 1)


if __name__ == '__main__':
  absltest.main()
