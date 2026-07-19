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

"""JAX root finding functions."""

import dataclasses
import functools
from typing import Callable, Final

import jax
import jax.numpy as jnp
import numpy as np
from torax._src import jax_utils
from torax._src.solver import linesearch

# Delta is a vector. If no entry of delta is above this magnitude, we terminate
# the delta loop. This is to avoid getting stuck in an infinite loop in edge
# cases with bad numerics.
MIN_DELTA: Final[float] = 1e-7


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RootMetadata:
  iterations: jax.Array
  residual: jax.Array
  last_tau: jax.Array
  error: jax.Array


def root_newton_raphson(
    fun: Callable[[jax.Array], jax.Array],
    x0: jax.Array | np.ndarray,
    *,
    maxiter: int = 30,
    tol: float = 1e-5,
    coarse_tol: float = 1e-2,
    delta_reduction_factor: float = 0.5,
    tau_min: float = 0.01,
    sufficient_decrease: float = 1e-4,
    log_iterations: bool = False,
    use_jax_custom_root: bool = True,
    custom_jac: Callable[[jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, RootMetadata]:
  """A differentiable Newton-Raphson root finder.

  A similar API to scipy.optimize.root.

  Args:
    fun: The function to find the root of.
    x0: The initial guess of the location of the root.
    maxiter: Quit iterating after this many iterations reached.
    tol: Quit iterating after the average absolute value of the residual is <=
      tol.
    coarse_tol: Coarser allowed tolerance for cases when solver develops small
      steps in the vicinity of the solution.
    delta_reduction_factor: Multiply by delta_reduction_factor after each failed
      line search step.
    tau_min: Minimum delta/delta_original allowed before the newton raphson
      routine resets at a lower timestep.
    sufficient_decrease: Acceptance threshold for sufficient decrease in the
      line search.
    log_iterations: If true, output diagnostic information from within iteration
      loop.
    use_jax_custom_root: If true, use jax.lax.custom_root to allow for
      differentiable solving. This can increase compile times even when no
      derivatives are requested.
    custom_jac: If provided, use this function to compute the Jacobian of `fun`
      instead of jax.jacfwd.

  Returns:
    A tuple `(x_root, RootMetadata(...))`.
  """

  def _newton_raphson(f, x, jacobian_fun=None):
    init_x_new_vec = x
    f = jax.jit(f)

    residual_fun = jax_utils.xla_metadata_call(
        f, compilation_unit='residual_fun_block'
    )

    if jacobian_fun is None:
      jacobian_fun = jax.jacfwd(f)
      jacobian_fun = jax_utils.xla_metadata_call(
          jax.jit(jacobian_fun), compilation_unit='jacobian_fun_block'
      )

    # initialize state dict being passed around Newton-Raphson iterations
    residual_vec_init_x_new = residual_fun(init_x_new_vec)
    initial_state = {
        'x': init_x_new_vec,
        # jax.lax.custom_root is broken with aux outputs of integer type. Use
        # float for the iterations https://github.com/jax-ml/jax/issues/24295.
        'iterations': jnp.array(0, dtype=jax_utils.get_dtype()),
        'residual': residual_vec_init_x_new,
        'last_tau': jnp.array(1.0, dtype=jax_utils.get_dtype()),
    }

    # carry out iterations.
    cond_fun = functools.partial(
        _cond, tol=tol, tau_min=tau_min, maxiter=maxiter
    )
    body_fun = functools.partial(
        _body,
        jacobian_fun=jacobian_fun,
        residual_fun=residual_fun,
        log_iterations=log_iterations,
        delta_reduction_factor=delta_reduction_factor,
        sufficient_decrease=sufficient_decrease,
    )
    output_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    x_out = output_state.pop('x')
    return x_out, output_state

  # jax.lax.custom_root allows for differentiating through the solver,
  # efficiently. As the solver has a jax.lax.while_loop, it cannot be
  # reverse-mode differentiated. But even if we could, this would be highly
  # inefficient. This uses the implicit function theorem to differentiate
  # through the solver with only needing the result of the solver,
  # rather than the entire solver computational graph.
  # See also this discussion:
  # https://docs.jax.dev/en/latest/advanced-autodiff.html#example-implicit-function-differentiation-of-iterative-implementations

  def back(g, y):
    return jnp.linalg.solve(jax.jacfwd(g)(y), y)

  if use_jax_custom_root:
    if custom_jac is not None:
      raise ValueError('custom_jac is not compatible with use_jax_custom_root.')
    x_out, metadata = jax.lax.custom_root(
        f=fun,
        initial_guess=x0,
        solve=_newton_raphson,
        tangent_solve=back,
        has_aux=True,
    )
  else:
    x_out, metadata = _newton_raphson(fun, x0, jacobian_fun=custom_jac)

  # Tell the caller whether or not x_new successfully reduces the residual below
  # the tolerance by providing an extra output, error.
  # error = 0: residual converged within fine tolerance (tol)
  # error = 1: not converged. Possibly backtrack to smaller dt and retry
  # error = 2: residual not strictly converged but is still within reasonable
  # tolerance (coarse_tol). Can occur when solver exits early due to small steps
  # in solution vicinity. Proceed but provide a warning to user.
  error = _error_cond(
      residual=metadata['residual'], coarse_tol=coarse_tol, tol=tol
  )
  # Workaround for https://github.com/google/jax/issues/24295: cast iterations
  # to the correct int dtype.
  metadata['iterations'] = metadata['iterations'].astype(
      jax_utils.get_int_dtype()
  )
  return x_out, RootMetadata(**metadata, error=error)  # pytype: disable=bad-return-type


def _error_cond(residual: jax.Array, coarse_tol: float, tol: float):
  return jax.lax.cond(
      _residual_scalar(residual) < tol,
      lambda: 0,  # Called when True
      lambda: jax.lax.cond(  # Called when False
          _residual_scalar(residual) < coarse_tol,
          lambda: 2,  # Called when True
          lambda: 1,  # Called when False
      ),
  )


def _residual_scalar(x):
  return jnp.mean(jnp.abs(x))


def _cond(
    state: dict[str, jax.Array],
    tau_min: float,
    maxiter: int,
    tol: float,
) -> bool:
  """Check if exit condition reached for Newton-Raphson iterations."""
  iteration = state['iterations'][...]
  return jnp.bool_(
      jnp.logical_and(
          jnp.logical_and(
              _residual_scalar(state['residual']) > tol, iteration < maxiter
          ),
          state['last_tau'] > tau_min,
      )
  )


def _linesearch_update(
    input_state: dict[str, jax.Array],
    direction: jax.Array,
    residual_fun: Callable[[jax.Array], jax.Array],
    log_iterations: bool,
    delta_reduction_factor: float,
    sufficient_decrease: float,
) -> dict[str, jax.Array]:
  """Backtracking line search along `direction` and iteration state update."""
  dtype = input_state['x'].dtype

  def norm_fn(res):
    return jnp.mean(jnp.abs(res))

  init_norm = norm_fn(input_state['residual'])

  def accept_fn(step_size, trial_norm):
    return (
        trial_norm <= (1.0 - sufficient_decrease * step_size) * init_norm
    ) & (~jnp.isnan(trial_norm))

  ls_state = linesearch.backtracking_linesearch(
      residual_fn=residual_fun,
      x_init=input_state['x'],
      direction=direction,
      accept_fn=accept_fn,
      norm_fn=norm_fn,
      initial_residual=input_state['residual'],
      initial_residual_norm=init_norm,
      delta_reduction_factor=delta_reduction_factor,
      max_steps=100,
      min_step_norm=MIN_DELTA,
  )

  output_state = {
      'x': ls_state.x,
      'residual': ls_state.residual,
      'iterations': jnp.array(input_state['iterations'][...], dtype=dtype) + 1,
      'last_tau': ls_state.step_size,
  }

  if log_iterations:
    jax.debug.print(
        'Iteration: {iteration:d}. Residual: {residual:.16f}. tau = {tau:.6f}',
        iteration=output_state['iterations'].astype(jax_utils.get_int_dtype()),
        residual=_residual_scalar(output_state['residual']),
        tau=ls_state.step_size,
    )

  return output_state


def _body(
    input_state: dict[str, jax.Array],
    jacobian_fun: Callable[[jax.Array], jax.Array],
    residual_fun: Callable[[jax.Array], jax.Array],
    log_iterations: bool,
    delta_reduction_factor: float,
    sufficient_decrease: float,
) -> dict[str, jax.Array]:
  """Calculates next guess in Newton-Raphson iteration."""
  a_mat = jacobian_fun(input_state['x'])
  rhs = -input_state['residual']

  direction = jnp.linalg.solve(a_mat, rhs)

  return _linesearch_update(
      input_state=input_state,
      direction=direction,
      residual_fun=residual_fun,
      log_iterations=log_iterations,
      delta_reduction_factor=delta_reduction_factor,
      sufficient_decrease=sufficient_decrease,
  )


def root_newton_krylov(
    fun: Callable[[jax.Array], jax.Array],
    x0: jax.Array | np.ndarray,
    *,
    maxiter: int = 30,
    tol: float = 1e-5,
    coarse_tol: float = 1e-2,
    delta_reduction_factor: float = 0.5,
    tau_min: float = 0.01,
    sufficient_decrease: float = 1e-4,
    gmres_rtol: float = 1e-2,
    gmres_restart: int = 20,
    gmres_maxiter: int = 1,
    precond_apply: Callable[[jax.Array], jax.Array] | None = None,
    log_iterations: bool = False,
) -> tuple[jax.Array, RootMetadata]:
  """A Jacobian-free Newton-Krylov (JFNK) root finder.

  Prototype alternative to `root_newton_raphson`. Instead of materializing the
  dense Jacobian with `jax.jacfwd` (cost ~= N residual evaluations) and solving
  with `jnp.linalg.solve`, each Newton direction is obtained by solving
  J(x) d = -R(x) with matrix-free GMRES, where J v products are computed via
  `jax.linearize` (cost ~= 1 residual evaluation per Krylov iteration).

  The outer iteration (backtracking line search, convergence and tau_min exit
  conditions, error classification) is identical to `root_newton_raphson`.

  Note: this prototype does not support `jax.lax.custom_root`; use
  `root_newton_raphson` if implicit differentiation through the solve is
  required.

  Args:
    fun: The function to find the root of.
    x0: The initial guess of the location of the root.
    maxiter: Quit iterating after this many Newton iterations.
    tol: Quit iterating after the average absolute value of the residual is <=
      tol.
    coarse_tol: Coarser allowed tolerance for cases when solver develops small
      steps in the vicinity of the solution.
    delta_reduction_factor: Multiply by delta_reduction_factor after each failed
      line search step.
    tau_min: Minimum step_size allowed before the Newton routine gives up.
    sufficient_decrease: Acceptance threshold for sufficient decrease in the
      line search.
    gmres_rtol: Relative tolerance for the inner GMRES solve (an inexact-Newton
      forcing term). The Krylov solve stops when the linear residual norm is
      below gmres_rtol * ||R(x)||.
    gmres_restart: Size of the Krylov subspace per GMRES cycle. Each Krylov
      iteration costs one J v product (~1 residual evaluation).
    gmres_maxiter: Number of GMRES restart cycles. Total J v products are at
      most gmres_restart * gmres_maxiter per Newton iteration.
    precond_apply: Optional preconditioner. A callable approximating v ->
      J^{-1} v, applied within GMRES. For the theta-method residual, an
      effective and cheap choice is a Thomas solve against the block-
      tridiagonal LHS matrix of the discretized PDE (which is the exact
      Jacobian minus the transport-coefficient sensitivity terms).
    log_iterations: If true, output diagnostic information from within
      iteration loop.

  Returns:
    A tuple `(x_root, RootMetadata(...))`.
  """
  residual_fun = jax_utils.xla_metadata_call(
      jax.jit(fun), compilation_unit='residual_fun_block'
  )

  residual_vec_init = residual_fun(x0)
  initial_state = {
      'x': x0,
      'iterations': jnp.array(0, dtype=jax_utils.get_dtype()),
      'residual': residual_vec_init,
      'last_tau': jnp.array(1.0, dtype=jax_utils.get_dtype()),
  }

  def krylov_body(input_state: dict[str, jax.Array]) -> dict[str, jax.Array]:
    x = input_state['x']
    residual_now = input_state['residual']
    # One primal evaluation; jvp_fun then computes J v at ~1 residual eval per
    # call without rematerializing the primal.
    _, jvp_fun = jax.linearize(residual_fun, x)
    direction, _ = jax.scipy.sparse.linalg.gmres(
        jvp_fun,
        -residual_now,
        tol=gmres_rtol,
        atol=0.0,
        restart=gmres_restart,
        maxiter=gmres_maxiter,
        M=precond_apply,
        solve_method='batched',
    )
    return _linesearch_update(
        input_state=input_state,
        direction=direction,
        residual_fun=residual_fun,
        log_iterations=log_iterations,
        delta_reduction_factor=delta_reduction_factor,
        sufficient_decrease=sufficient_decrease,
    )

  cond_fun = functools.partial(
      _cond, tol=tol, tau_min=tau_min, maxiter=maxiter
  )
  output_state = jax.lax.while_loop(cond_fun, krylov_body, initial_state)
  x_out = output_state.pop('x')

  error = _error_cond(
      residual=output_state['residual'], coarse_tol=coarse_tol, tol=tol
  )
  output_state['iterations'] = output_state['iterations'].astype(
      jax_utils.get_int_dtype()
  )
  return x_out, RootMetadata(**output_state, error=error)
