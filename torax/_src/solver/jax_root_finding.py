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

# Guard against dividing by a vanishing norm during a lucky GMRES breakdown.
_GMRES_EPS: Final[float] = 1e-30


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RootMetadata:
  iterations: jax.Array
  residual: jax.Array
  last_tau: jax.Array
  error: jax.Array


def _gmres_cycle(
    matvec: Callable[[jax.Array], jax.Array],
    preconditioner: Callable[[jax.Array], jax.Array],
    target_norm: jax.Array,
    x: jax.Array,
    residual: jax.Array,
    residual_norm: jax.Array,
    restart: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Runs one restart cycle of right-preconditioned GMRES.

  Builds an Arnoldi basis for the Krylov space of `matvec . preconditioner`,
  keeps the least-squares problem in triangular form with Givens rotations, and
  stops as soon as the (exactly tracked) residual norm drops below
  `target_norm`. Right preconditioning is used rather than left so that the
  quantity being driven below the tolerance is the true residual `|b - A x|`
  and not a preconditioned surrogate; that is what makes the inexact-Newton
  forcing term meaningful.

  Args:
    matvec: Applies the (linear) system matrix A.
    preconditioner: Applies the approximate inverse M^-1.
    target_norm: Stop once the residual norm is at or below this.
    x: Current iterate.
    residual: b - A x at the current iterate.
    residual_norm: Norm of `residual`.
    restart: Maximum Krylov dimension of this cycle. Static; it sizes the
      Arnoldi buffers.

  Returns:
    A tuple of (x, residual_norm, num_iterations) after the cycle.
  """
  size = x.shape[0]
  dtype = x.dtype
  # Arnoldi basis, one row per vector. `hessenberg` stores the triangularised
  # least-squares factor column by column (row k is column k), so the upper
  # triangular factor is its transpose.
  basis = jnp.zeros((restart + 1, size), dtype=dtype).at[0].set(
      residual / jnp.where(residual_norm > _GMRES_EPS, residual_norm, 1.0)
  )
  hessenberg = jnp.zeros((restart, restart), dtype=dtype)
  cos = jnp.zeros((restart,), dtype=dtype)
  sin = jnp.zeros((restart,), dtype=dtype)
  # Right-hand side of the triangular least-squares problem. Its trailing entry
  # is the exact residual norm of the current best iterate.
  givens_rhs = jnp.zeros((restart + 1,), dtype=dtype).at[0].set(residual_norm)
  basis_index = jnp.arange(restart + 1)

  def cond_fun(carry):
    iteration, _, _, _, _, _, norm = carry
    return (iteration < restart) & (norm > target_norm)

  def body_fun(carry):
    iteration, basis, hessenberg, cos, sin, givens_rhs, _ = carry
    w = matvec(preconditioner(basis[iteration]))

    # Only the first `iteration + 1` basis vectors are populated; the rest of
    # the buffer is stale, so mask their projections out.
    active = basis_index <= iteration

    def orthogonalise(w):
      coeffs = jnp.where(active, basis @ w, 0.0)
      return w - coeffs @ basis, coeffs

    # Classical Gram-Schmidt applied twice. One pass vectorises into a single
    # (restart+1, size) matmul, which is negligible next to a JVP through the
    # physics, but loses orthogonality; a second pass restores it to the same
    # accuracy as modified Gram-Schmidt without serialising over the basis.
    w, coeffs_first = orthogonalise(w)
    w, coeffs_second = orthogonalise(w)
    column = (coeffs_first + coeffs_second).at[iteration + 1].set(
        jnp.linalg.norm(w)
    )
    next_norm = column[iteration + 1]
    basis = basis.at[iteration + 1].set(
        w / jnp.where(next_norm > _GMRES_EPS, next_norm, 1.0)
    )

    # Apply the rotations accumulated by previous iterations, then eliminate
    # the new subdiagonal entry with a fresh rotation.
    def rotate(i, column):
      rotated = cos[i] * column[i] + sin[i] * column[i + 1]
      column = column.at[i + 1].set(
          -sin[i] * column[i] + cos[i] * column[i + 1]
      )
      return column.at[i].set(rotated)

    column = jax.lax.fori_loop(0, iteration, rotate, column)
    hypot = jnp.sqrt(column[iteration] ** 2 + column[iteration + 1] ** 2)
    hypot = jnp.where(hypot > _GMRES_EPS, hypot, 1.0)
    cos_k = column[iteration] / hypot
    sin_k = column[iteration + 1] / hypot
    column = column.at[iteration].set(
        cos_k * column[iteration] + sin_k * column[iteration + 1]
    ).at[iteration + 1].set(0.0)

    rhs_k = givens_rhs[iteration]
    givens_rhs = givens_rhs.at[iteration].set(cos_k * rhs_k).at[
        iteration + 1
    ].set(-sin_k * rhs_k)

    return (
        iteration + 1,
        basis,
        hessenberg.at[iteration].set(column[:restart]),
        cos.at[iteration].set(cos_k),
        sin.at[iteration].set(sin_k),
        givens_rhs,
        jnp.abs(givens_rhs[iteration + 1]),
    )

  iteration, basis, hessenberg, _, _, givens_rhs, norm = jax.lax.while_loop(
      cond_fun,
      body_fun,
      (0, basis, hessenberg, cos, sin, givens_rhs, residual_norm),
  )

  # Solve the triangular system over the columns that were actually built.
  # Unused columns are replaced by the identity and their right-hand side by
  # zero so the fixed-size solve returns zero for them.
  active = jnp.arange(restart) < iteration
  upper = jnp.where(
      active[None, :], hessenberg.T, jnp.eye(restart, dtype=dtype)
  )
  y = jax.scipy.linalg.solve_triangular(
      upper, jnp.where(active, givens_rhs[:restart], 0.0), lower=False
  )
  # Right preconditioning solves A M^-1 u = b, so the correction has to be
  # pulled back through M^-1 once at the end of the cycle.
  return x + preconditioner(y @ basis[:restart]), norm, iteration


def gmres_right_preconditioned(
    matvec: Callable[[jax.Array], jax.Array],
    b: jax.Array,
    preconditioner: Callable[[jax.Array], jax.Array],
    restart: int,
    max_krylov: int,
    rtol: float,
) -> tuple[jax.Array, jax.Array]:
  """Solves A x = b with restarted, right-preconditioned GMRES.

  Args:
    matvec: Applies the system matrix A.
    b: Right-hand side.
    preconditioner: Applies the approximate inverse M^-1.
    restart: Krylov dimension per cycle. Static.
    max_krylov: Maximum total Arnoldi iterations across all cycles.
    rtol: Relative tolerance; the solve stops once |b - A x| <= rtol * |b|.

  Returns:
    A tuple of (x, total_krylov_iterations).
  """
  b_norm = jnp.linalg.norm(b)
  target_norm = rtol * b_norm

  if restart >= max_krylov:
    # A cycle stops either because it converged, or because it used its full
    # Krylov dimension. When `restart >= max_krylov` the second case exhausts
    # the total budget too, so a second cycle is unreachable. Emitting the
    # restart loop anyway would put a second instantiation of `matvec` -- a JVP
    # through the whole physics model -- into the graph for a branch that never
    # runs, which is a large fraction of the compiled program. Since `restart`
    # and `max_krylov` are static, resolve it here instead.
    x, norm, iterations = _gmres_cycle(
        matvec, preconditioner, target_norm, jnp.zeros_like(b), b, b_norm,
        restart,
    )
    return x, iterations

  def cond_fun(carry):
    _, _, norm, total = carry
    return (norm > target_norm) & (total < max_krylov)

  def body_fun(carry):
    x, residual, _, total = carry
    x, norm, iterations = _gmres_cycle(
        matvec,
        preconditioner,
        target_norm,
        x,
        residual,
        jnp.linalg.norm(residual),
        restart,
    )
    total += iterations
    # Recomputing the residual costs a matvec, so only pay for it when another
    # cycle will actually consume it. In the common single-cycle case this
    # saves one JVP through the whole physics model per Newton iteration.
    residual = jax.lax.cond(
        (norm > target_norm) & (total < max_krylov),
        lambda: b - matvec(x),
        lambda: residual,
    )
    return x, residual, norm, total

  x, _, _, total = jax.lax.while_loop(
      cond_fun, body_fun, (jnp.zeros_like(b), b, b_norm, 0)
  )
  return x, total



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


def _body(
    input_state: dict[str, jax.Array],
    jacobian_fun: Callable[[jax.Array], jax.Array],
    residual_fun: Callable[[jax.Array], jax.Array],
    log_iterations: bool,
    delta_reduction_factor: float,
    sufficient_decrease: float,
) -> dict[str, jax.Array]:
  """Calculates next guess in Newton-Raphson iteration."""
  dtype = input_state['x'].dtype
  a_mat = jacobian_fun(input_state['x'])
  rhs = -input_state['residual']

  direction = jnp.linalg.solve(a_mat, rhs)

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
