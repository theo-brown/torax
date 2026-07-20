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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from torax._src.orchestration import whilei_loop


class WhileiLoopTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('no_loops', -1),
      ('one_loop', 1),
      ('four_loops', 4),
  )
  def test_whilei_loop(
      self,
      terminating_step,
  ):

    @jax.jit
    def f(alpha: jax.Array,) -> jax.Array:
      x = jnp.cos(alpha)

      def cond_fun(state, unused_alpha):
        _, counter = state
        return counter < terminating_step

      def compute_state(counter, unused_prev_state, loop_statistics, alpha):
        loop_statistics += 20
        return (jnp.sin(alpha * counter), counter), loop_statistics

      result = whilei_loop.whilei_loop(
          cond_fun,
          compute_state,
          ((x, jnp.array(0),), jnp.array(0)),
          x,
      )
      x = result.state[0]
      x = jnp.cos(x)
      return x, result.loop_statistics

    @jax.jit
    def g(alpha: jax.Array) -> jax.Array:
      x = jnp.cos(alpha)
      if terminating_step > 0:
        x = jnp.sin(x * terminating_step)
      x = jnp.cos(x)
      return x

    with self.subTest('value_test'):
      self.assertEqual(f(0.4)[0], g(0.4))
    with self.subTest('grad_test'):
      self.assertEqual(jax.grad(f, has_aux=True)(0.4)[0], jax.grad(g)(0.4))
    with self.subTest('loop_statistics_test'):
      expected_loop_statistics = (
          20 * (terminating_step+1) if terminating_step > 0 else 0.0
      )
      self.assertEqual(f(0.4)[1], expected_loop_statistics)

  def test_grad_raises_when_taking_grad_of_loop_statistics(self):
    terminating_step = 4
    aux_output = 0
    x = jnp.cos(0.4)

    def cond_fun(state, unused_alpha):
      _, counter = state
      return counter < terminating_step

    def compute_state(counter, unused_prev_state, loop_statistics, alpha):
      loop_statistics += 20
      return (jnp.sin(alpha * counter), counter), loop_statistics

    def f(aux_output):
      whilei_loop.whilei_loop(
          cond_fun,
          compute_state,
          ((x, jnp.array(0),), jnp.array(aux_output)),
          x,
      )

    with self.assertRaises(TypeError):
      jax.grad(f)(aux_output)

  def test_whilei_loop_compute_state_multiple_args(self):
    terminating_step = 4

    @jax.jit
    def f(alpha: jax.Array, beta: jax.Array) -> jax.Array:
      x = jnp.cos(alpha)

      def cond_fun(state, unused_alpha, unused_beta):
        _, counter = state
        return counter < terminating_step

      def compute_state(
          counter, unused_prev_state, loop_statistics, alpha, beta
      ):
        loop_statistics += 20
        return (jnp.sin(alpha * counter) * beta, counter), loop_statistics

      result = whilei_loop.whilei_loop(
          cond_fun,
          compute_state,
          ((x, jnp.array(0),), jnp.array(0)),
          x, beta,
      )
      x = result.state[0]
      x = jnp.cos(x)
      return x

    @jax.jit
    def g(alpha: jax.Array, beta: jax.Array) -> jax.Array:
      x = jnp.cos(alpha)
      x = jnp.sin(x * terminating_step) * beta
      x = jnp.cos(x)
      return x

    with self.subTest('value_test'):
      self.assertEqual(f(0.4, 0.5), g(0.4, 0.5))
    with self.subTest('grad_test'):
      self.assertEqual(jax.grad(f)(0.4, 0.5), jax.grad(g)(0.4, 0.5))

  def test_whilei_loop_passes_previous_state(self):
    """prev_state passed to compute_state is the previously computed state."""
    terminating_step = 4

    def cond_fun(state, unused_alpha):
      _, counter = state
      return counter < terminating_step

    def compute_state(counter, prev_state, loop_statistics, alpha):
      # Record the previous state's value in the new state so we can check
      # the chaining outside the loop. The final *value* only depends on
      # counter/alpha (whilei_loop contract); prev_val is just carried along.
      prev_val, _ = prev_state
      return (jnp.sin(alpha * counter) + 0.0 * prev_val, counter), (
          loop_statistics + 1
      )

    x = jnp.cos(0.4)
    result = whilei_loop.whilei_loop(
        cond_fun,
        compute_state,
        ((x, jnp.array(0)), jnp.array(0)),
        x,
    )
    # After the loop, prev_state is the state that was passed into the final
    # compute_state call, i.e. the state computed at counter - 2.
    self.assertEqual(result.counter, terminating_step + 1)
    self.assertEqual(result.state[1], terminating_step)
    self.assertEqual(result.prev_state[1], terminating_step - 1)
    self.assertEqual(
        result.prev_state[0], jnp.sin(x * (terminating_step - 1))
    )

  def test_whilei_loop_no_gradient_through_prev_state(self):
    """Gradients treat prev_state as a constant (zero tangent)."""
    terminating_step = 3

    @jax.jit
    def f(alpha: jax.Array) -> jax.Array:
      def cond_fun(state, unused_alpha):
        _, counter = state
        return counter < terminating_step

      def compute_state(counter, prev_state, loop_statistics, alpha):
        prev_val, _ = prev_state
        # Use prev_state in a way that does not change the primal value:
        # (prev_val - stop_gradient(prev_val)) == 0, but a naive gradient
        # through prev_state would produce a nonzero contribution. The
        # whilei_loop contract zeroes the prev_state tangent, so the gradient
        # must equal the gradient of sin(alpha * counter) alone.
        zero = prev_val - jax.lax.stop_gradient(prev_val)
        return (jnp.sin(alpha * counter) + zero, counter), loop_statistics

      x = jnp.cos(alpha)
      result = whilei_loop.whilei_loop(
          cond_fun,
          compute_state,
          ((x, jnp.array(0)), jnp.array(0)),
          alpha,
      )
      return result.state[0]

    @jax.jit
    def g(alpha: jax.Array) -> jax.Array:
      return jnp.sin(alpha * terminating_step)

    self.assertEqual(f(0.4), g(0.4))
    self.assertEqual(jax.grad(f)(0.4), jax.grad(g)(0.4))


if __name__ == '__main__':
  absltest.main()
