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

"""JAX fixed point functions."""

from typing import Any, Callable, NamedTuple, TypeAlias

import jax
from jax import flatten_util
import jax.numpy as jnp
import jaxtyping as jt
from torax._src import array_typing
from torax._src import jax_utils
from torax._src.solver import linesearch

PyTree: TypeAlias = Any


class AndersonState(NamedTuple):
  """History buffers carried by the Anderson acceleration.

  The buffers have a fixed leading dimension `m` (the depth) so that the state
  is a valid `lax` loop carry; new columns are pushed in at the end and the
  oldest column falls off the front. Everything is stored in the flattened
  (ravelled) representation of the iterate pytree, since the least-squares
  problem is over flat vectors.

  Attributes:
    x_prev: Flattened previous iterate `x_{k-1}`.
    g_prev: Flattened previous residual `g_{k-1} = f(x_{k-1}) - x_{k-1}`.
    dx: Buffer of iterate differences `x_{j+1} - x_j`, oldest first.
    dg: Buffer of residual differences `g_{j+1} - g_j`, oldest first.
  """

  x_prev: jt.Float[array_typing.Array, 'n']
  g_prev: jt.Float[array_typing.Array, 'n']
  dx: jt.Float[array_typing.Array, 'm n']
  dg: jt.Float[array_typing.Array, 'm n']


def _init_anderson_state(
    x0_flat: jt.Float[array_typing.Array, 'n'],
    g0_flat: jt.Float[array_typing.Array, 'n'],
    depth: int,
) -> AndersonState:
  """Returns an empty Anderson history seeded with the initial iterate."""
  empty = jnp.zeros((depth, x0_flat.size), dtype=x0_flat.dtype)
  return AndersonState(x_prev=x0_flat, g_prev=g0_flat, dx=empty, dg=empty)


def _anderson_accelerate(
    x: PyTree,
    residual: PyTree,
    state: AndersonState,
    is_first_iteration: array_typing.BoolScalar,
    beta: float,
    regularization: float,
) -> tuple[PyTree, AndersonState]:
  """Returns the Anderson-accelerated iterate and the updated history.

  This is the difference (unconstrained) form of Anderson mixing, see Walker &
  Ni, SIAM J. Numer. Anal. 49 (2011). With `g_k = f(x_k) - x_k` and the history
  of differences `dx_j = x_{j+1} - x_j`, `dg_j = g_{j+1} - g_j`,

    gamma = argmin_gamma || g_k - dg^T gamma ||
    x_{k+1} = x_k + beta * g_k - (dx + beta * dg)^T gamma

  The difference form is preferred over the constrained form
  (`min ||G alpha||` subject to `sum(alpha) = 1`) because it needs no
  constraint, and because leaves that are constant across the iteration (e.g.
  the mesh and boundary conditions carried along inside a `CellVariable`) have
  identically zero residual and differences, so they are reproduced bit-exactly
  rather than being rebuilt from a near-unit sum of weights. With an empty
  history and `beta = 1` the update degenerates to plain Picard,
  `x_{k+1} = f(x_k)`.

  Args:
    x: The current iterate `x_k`.
    residual: The current residual `g_k = f(x_k) - x_k`.
    state: The Anderson history from the previous iteration.
    is_first_iteration: Whether this is the first iteration, in which case there
      is no meaningful previous iterate to difference against.
    beta: Damping (mixing) parameter. 1.0 is undamped.
    regularization: Tikhonov regularization of the least-squares normal
      equations, relative to the mean squared column norm of `dg`.

  Returns:
    A tuple of the next iterate (same pytree structure as `x`) and the updated
    history.
  """
  x_flat, unravel = flatten_util.ravel_pytree(x)
  g_flat, _ = flatten_util.ravel_pytree(residual)

  # On the very first iteration the "previous" iterate is the initial guess
  # itself, so differencing against it would push a spurious column with
  # dx = 0 and dg != 0. The least-squares would then cancel the residual
  # without moving x at all, stalling the iteration. Push zeros instead.
  keep = jnp.where(is_first_iteration, 0.0, 1.0)
  dx = jnp.roll(state.dx, -1, axis=0).at[-1].set((x_flat - state.x_prev) * keep)
  dg = jnp.roll(state.dg, -1, axis=0).at[-1].set((g_flat - state.g_prev) * keep)

  # Normal equations for min ||g_k - dg^T gamma||. Slots of the history that
  # have not been filled yet are exactly zero and are annihilated by the
  # regularization, so they need no separate masking. The regularization is
  # essential rather than cosmetic: as the iteration converges the columns of
  # dg collapse onto each other and the Gram matrix becomes numerically
  # singular, at which point an unregularized solve produces a huge gamma and
  # throws the iterate away.
  gram = dg @ dg.T
  rhs = dg @ g_flat
  mean_sq_col_norm = jnp.trace(gram) / dg.shape[0]
  # An entirely empty history gives gram = 0 and rhs = 0; fall back to the
  # identity so that gamma = 0 exactly rather than 0/0.
  damping = jnp.where(
      mean_sq_col_norm > 0.0, regularization * mean_sq_col_norm, 1.0
  )
  gamma = jnp.linalg.solve(gram + damping * jnp.eye(dg.shape[0]), rhs)

  x_next_flat = x_flat + beta * g_flat - gamma @ (dx + beta * dg)
  next_state = AndersonState(x_prev=x_flat, g_prev=g_flat, dx=dx, dg=dg)
  return unravel(x_next_flat), next_state


def fixed_point(
    func: Callable[..., PyTree],
    x0: PyTree,
    args: tuple[PyTree, ...] = (),
    maxiter: int = 500,
    atol: float = 1e-8,
    rtol: float = 1e-6,
    termination_criterion: str = 'tolerance',
    use_backtracking: bool = False,
    sufficient_decrease: float = 0.5,
    step_size_reduction_factor: float = 0.5,
    max_backtrack_steps: int = 50,
    acceleration: str = 'none',
    anderson_depth: int = 5,
    anderson_beta: float = 1.0,
    anderson_regularization: float = 1e-10,
) -> PyTree:
  """Solves `func(x, *args) = x` for `x` with backtracking linesearch.

  Iterates x_new = func(x_old, *args) until either the requested tolerance is
  satisfied or the maximum number of iterations is reached.
  If `use_backtracking` is True, the iteration is of the form
  x_new = x_old + alpha * (f(x_old) - x_old), where alpha is chosen via
  backtracking linesearch.
  If `acceleration` is 'anderson', the plain (or backtracked) update is replaced
  by an Anderson mixing step over a history of the last `anderson_depth`
  iterates, which can turn the linear convergence of the Picard iteration into
  superlinear convergence at the cost of one small least-squares solve per
  iteration and no extra evaluations of `func`.

  Args:
    func: The function to solve, of the form `f(x, *args)` returning a `PyTree`
      of the same structure as `x`.
    x0: The initial guess.
    args: Additional arguments to pass to the function.
    maxiter: The maximum number of iterations to perform.
    atol: Absolute tolerance on the residual norm.
    rtol: Relative tolerance on the residual norm.
    termination_criterion: The criterion to use for terminating the iteration.
      If 'max_iterations', the iteration will terminate after `maxiter`
      iterations. If 'tolerance', the iteration will terminate when the residual
      norm is below the tolerance specified by `atol` and `rtol`.
    use_backtracking: If true, use backtracking linesearch.
    sufficient_decrease: Control parameter for Armijo condition in backtracking
      linesearch. Residual norm must decrease by at least this factor for the
      step to be accepted.
    step_size_reduction_factor: Factor by which step_size is reduced during
      backtracking linesearch.
    max_backtrack_steps: Maximum number of backtracking steps.
    acceleration: Acceleration scheme applied to the iteration. Either 'none'
      for plain (damped) Picard, or 'anderson' for Anderson mixing.
    anderson_depth: Number of previous iterates retained by the Anderson
      history. Ignored unless `acceleration` is 'anderson'.
    anderson_beta: Anderson damping (mixing) parameter. 1.0 is undamped, which
      reduces to plain Picard when the history is empty.
    anderson_regularization: Relative Tikhonov regularization of the Anderson
      least-squares problem.

  Returns:
    The fixed point `PyTree`.
  """
  if maxiter <= 0:
    raise ValueError(f'Invalid maxiter: {maxiter} must be positive.')
  if termination_criterion not in ['max_iterations', 'tolerance']:
    raise ValueError(
        f'Invalid termination criterion: {termination_criterion} must be'
        ' "max_iterations" or "tolerance".'
    )
  if acceleration not in ['none', 'anderson']:
    raise ValueError(
        f'Invalid acceleration: {acceleration} must be "none" or "anderson".'
    )
  use_anderson = acceleration == 'anderson'
  if use_anderson and anderson_depth <= 0:
    raise ValueError(
        f'Invalid anderson_depth: {anderson_depth} must be positive.'
    )

  def residual_fn(f_x, x):
    """Computes the residual R(x) = f(x) - x."""
    return jax.tree.map(lambda a, b: a - b, f_x, x)

  def sq_norm_fn(x):
    """Computes the squared L2 norm of a PyTree."""
    return sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(x))

  def body(carry):
    x, _, count, anderson_state = carry
    f_x = func(x, *args)
    residual = residual_fn(f_x, x)
    residual_sq_norm = sq_norm_fn(residual)

    if use_backtracking:
      residual_norm = jnp.sqrt(residual_sq_norm)

      def armijo_condition(step_size, trial_norm):
        return (
            trial_norm <= (1 - sufficient_decrease * step_size) * residual_norm
        )

      # Damped Picard update is x_k+1 = x_k + alpha * (f(x_k) - x_k), where
      # alpha is the step size. We do linesearch to find an acceptable alpha.
      # Note: this will *fail* if f'(x_k) > 1.
      ls_state = linesearch.backtracking_linesearch(
          residual_fn=lambda x: residual_fn(func(x, *args), x),
          x_init=x,
          direction=residual,
          accept_fn=armijo_condition,
          norm_fn=lambda residual: jnp.sqrt(sq_norm_fn(residual)),
          initial_residual=residual,
          initial_residual_norm=residual_norm,
          delta_reduction_factor=step_size_reduction_factor,
          max_steps=max_backtrack_steps,
      )
      x = ls_state.x
      f_x = func(x, *args)
      residual = residual_fn(f_x, x)
      residual_sq_norm = sq_norm_fn(residual)

    if use_anderson:
      # Anderson composes on top of whatever step the code above settled on:
      # it extrapolates from the accepted iterate x and its residual, so it
      # costs no additional evaluation of `func`.
      x_next, anderson_state = _anderson_accelerate(
          x,
          residual,
          anderson_state,
          is_first_iteration=count == 0,
          beta=anderson_beta,
          regularization=anderson_regularization,
      )
    else:
      x_next = f_x

    count += 1
    return x_next, residual_sq_norm, count, anderson_state

  # TODO(b/515250945): Ensure that automatic differentiation is supported.
  # Currently, the branch using fori_loop supports autodiff, but differentiates
  # through the entire loop. The branch using while_loop does not allow for
  # automatic differentiation. Consider switching to whilei_loop.
  if termination_criterion == 'max_iterations':
    count = jnp.array(0, dtype=jax_utils.get_int_dtype())
    initial_sq_norm = jnp.array(jnp.inf, dtype=jax_utils.get_dtype())
    if use_anderson:
      x0_flat, _ = flatten_util.ravel_pytree(x0)
      # No residual has been evaluated yet; the first body iteration discards
      # the difference against this placeholder (see `is_first_iteration`).
      initial_anderson_state = _init_anderson_state(
          x0_flat, jnp.zeros_like(x0_flat), anderson_depth
      )
    else:
      initial_anderson_state = ()
    initial_carry = (x0, initial_sq_norm, count, initial_anderson_state)
    x_final, _, _, _ = jax.lax.fori_loop(
        0, maxiter, lambda i, val: body(val), initial_carry
    )
    return x_final

  else:
    # Precompute the tolerance for convergence.
    # TODO(b/515255142): pass in the initial residual as an argument, and use it
    # as the basis for the tolerance instead of calculating it here.
    f_x0 = func(x0, *args)
    initial_residual = residual_fn(f_x0, x0)
    initial_sq_norm = sq_norm_fn(initial_residual)
    initial_residual_norm = jnp.sqrt(initial_sq_norm)
    tol = atol + rtol * initial_residual_norm
    sq_tol = tol**2

    def cond(carry):
      _, sq_norm, count, _ = carry
      is_converged = sq_norm <= sq_tol
      return (count < maxiter) & jnp.logical_not(is_converged)

    if use_anderson:
      # This path already has x0 and its residual in hand, so the history can
      # be seeded exactly and the first body iteration produces a usable
      # difference column immediately.
      initial_anderson_state = _init_anderson_state(
          flatten_util.ravel_pytree(x0)[0],
          flatten_util.ravel_pytree(initial_residual)[0],
          anderson_depth,
      )
    else:
      initial_anderson_state = ()
    # Initial count starts at 1 since we do one evaluation of `func` in the
    # initialization above.
    initial_count = jnp.array(1, dtype=jax_utils.get_int_dtype())
    initial_carry = (
        f_x0,
        initial_sq_norm,
        initial_count,
        initial_anderson_state,
    )
    x_final, _, _, _ = jax.lax.while_loop(cond, body, initial_carry)
    return x_final
