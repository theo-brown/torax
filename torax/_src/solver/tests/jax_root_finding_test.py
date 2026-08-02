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
      self, a: float, b: float, x0: tuple[float, float]):
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


class GmresTest(parameterized.TestCase):
  """Tests for the right-preconditioned restarted GMRES."""

  def setUp(self):
    super().setUp()
    jax.config.update('jax_enable_x64', True)

  def _system(self, size=40, seed=0):
    rng = np.random.default_rng(seed)
    matrix = np.eye(size) + 0.3 * rng.normal(size=(size, size)) / np.sqrt(size)
    # Badly scaled rows, so that preconditioning has something to fix.
    scale = np.exp(rng.normal(size=size) * 2.0)
    matrix = matrix * scale[:, None]
    b = jnp.asarray(rng.normal(size=size))
    return jnp.asarray(matrix), b, jnp.asarray(scale)

  @parameterized.named_parameters(
      # Restarted GMRES is only tested with the preconditioner: with a badly
      # scaled matrix and a subspace this small it stagnates without one, which
      # is a known property of restarting rather than a defect here.
      ('full_space', 40, 40, False),
      ('restarted', 8, 80, True),
  )
  def test_converges_to_the_exact_solution(
      self, restart, max_krylov, precondition
  ):
    matrix, b, scale = self._system()
    preconditioner = (lambda v: v / scale) if precondition else (lambda v: v)
    x, iterations = jax.jit(
        functools.partial(
            jax_root_finding.gmres_right_preconditioned,
            lambda v: matrix @ v,
            preconditioner=preconditioner,
            restart=restart,
            max_krylov=max_krylov,
            rtol=1e-12,
        )
    )(b)
    np.testing.assert_allclose(matrix @ x, b, atol=1e-8)
    self.assertGreater(int(iterations), 0)
    self.assertLessEqual(int(iterations), max_krylov)

  def test_preconditioning_reduces_iterations(self):
    matrix, b, scale = self._system()
    run = lambda pre: jax.jit(
        functools.partial(
            jax_root_finding.gmres_right_preconditioned,
            lambda v: matrix @ v,
            preconditioner=pre,
            restart=40,
            max_krylov=40,
            rtol=1e-10,
        )
    )(b)
    _, plain_iterations = run(lambda v: v)
    x, preconditioned_iterations = run(lambda v: v / scale)
    np.testing.assert_allclose(matrix @ x, b, atol=1e-6)
    self.assertLess(int(preconditioned_iterations), int(plain_iterations))

  def test_loose_tolerance_stops_early(self):
    """The inexact-Newton forcing term must actually buy fewer iterations."""
    matrix, b, scale = self._system()
    run = lambda rtol: jax.jit(
        functools.partial(
            jax_root_finding.gmres_right_preconditioned,
            lambda v: matrix @ v,
            preconditioner=lambda v: v / scale,
            restart=40,
            max_krylov=40,
            rtol=rtol,
        )
    )(b)
    x_loose, loose_iterations = run(1e-1)
    _, tight_iterations = run(1e-10)
    self.assertLess(int(loose_iterations), int(tight_iterations))
    # The loose solve must still respect the tolerance it was given.
    relative = jnp.linalg.norm(b - matrix @ x_loose) / jnp.linalg.norm(b)
    self.assertLess(float(relative), 1e-1)


class JfnkRootFindingTest(parameterized.TestCase):
  """Tests the JFNK path of root_newton_raphson against the direct path."""

  def setUp(self):
    super().setUp()
    jax.config.update('jax_enable_x64', True)

  def _jfnk_kwargs(self, f_closed):
    """Uses the identity as a (deliberately trivial) preconditioner."""

    def residual_with_operator(x):
      return f_closed(x), None

    return dict(
        linear_solver='jfnk',
        residual_with_operator_fun=residual_with_operator,
        preconditioner_apply=lambda _, v: v,
        jfnk_max_krylov=8,
        jfnk_restart=8,
        jfnk_rtol=1e-8,
    )

  @parameterized.named_parameters(
      ('positive_search', 0.5, 0.1, (0.0, 0.0)),
      ('negative_search', 1.0, 1.0, (2.0, 1.0)),
  )
  def test_jfnk_matches_direct_solver(self, a, b, x0):
    tol = 1e-9
    f_closed = functools.partial(function_to_find_root, a=a, b=b)
    x_init = np.array(x0, dtype=np.float64)

    direct, direct_metadata = jax.jit(
        functools.partial(
            jax_root_finding.root_newton_raphson, f_closed, tol=tol
        )
    )(x_init)
    jfnk, jfnk_metadata = jax.jit(
        functools.partial(
            jax_root_finding.root_newton_raphson,
            f_closed,
            tol=tol,
            **self._jfnk_kwargs(f_closed),
        )
    )(x_init)

    with self.subTest('same_root'):
      chex.assert_trees_all_close(direct, jfnk, atol=1e-8)
    with self.subTest('converged'):
      self.assertEqual(int(jfnk_metadata.error), 0)
    with self.subTest('same_newton_iteration_count'):
      # A tight Krylov tolerance reproduces the exact Newton step, so the
      # nonlinear iteration counts must agree.
      self.assertEqual(
          int(direct_metadata.iterations), int(jfnk_metadata.iterations)
      )
    with self.subTest('krylov_iterations_reported'):
      self.assertIsNone(direct_metadata.krylov_iterations)
      self.assertGreater(int(jfnk_metadata.krylov_iterations), 0)
      self.assertTrue(
          jnp.isdtype(jfnk_metadata.krylov_iterations.dtype, 'integral')
      )

  def test_jfnk_works_inside_custom_root(self):
    """JFNK must remain differentiable via jax.lax.custom_root."""
    tol = 1e-9

    def loss(x, a, b):
      f_closed = functools.partial(function_to_find_root, a=a, b=b)
      root = jax_root_finding.root_newton_raphson(
          f_closed,
          x,
          tol=tol,
          use_jax_custom_root=True,
          **self._jfnk_kwargs(f_closed),
      )[0]
      return jnp.sum(root**2)

    def direct_loss(x, a, b):
      root = jax_root_finding.root_newton_raphson(
          functools.partial(function_to_find_root, a=a, b=b), x, tol=tol
      )[0]
      return jnp.sum(root**2)

    x_init = np.array([0.0, 0.0], dtype=np.float64)
    a, b = 0.5, 0.1
    chex.assert_trees_all_close(
        loss(x_init, a, b), direct_loss(x_init, a, b), atol=1e-8
    )
    jfnk_grads = jax.grad(loss, argnums=(1, 2))(x_init, a, b)
    direct_grads = jax.grad(direct_loss, argnums=(1, 2))(x_init, a, b)
    chex.assert_trees_all_close(jfnk_grads, direct_grads, atol=1e-6)

  def test_jfnk_requires_preconditioner(self):
    f_closed = functools.partial(function_to_find_root, a=0.5, b=0.1)
    with self.assertRaisesRegex(ValueError, 'requires'):
      jax_root_finding.root_newton_raphson(
          f_closed, np.zeros(2), linear_solver='jfnk'
      )

  def test_unknown_linear_solver_raises(self):
    f_closed = functools.partial(function_to_find_root, a=0.5, b=0.1)
    with self.assertRaisesRegex(ValueError, 'Unknown linear_solver'):
      jax_root_finding.root_newton_raphson(
          f_closed, np.zeros(2), linear_solver='nope'
      )


if __name__ == '__main__':
  absltest.main()
