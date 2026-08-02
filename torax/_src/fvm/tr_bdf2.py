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
"""TR-BDF2 (ESDIRK2) time integration for the FVM core profile solver.

TR-BDF2 splits one step ``t_n -> t_n + dt`` into a trapezoidal sub-step to
``t_n + gamma*dt`` followed by a BDF2 sub-step to ``t_n + dt``. It is second
order, L-stable and stiffly accurate, which the theta method is not: backward
Euler (``theta_implicit=1``) is L-stable but only first order, and
Crank-Nicolson (``theta_implicit=0.5``) is second order but only A-stable, so
it rings on the stiff, quasi-steady transport problem TORAX solves.

Both stages have the same shape as the backward-Euler solve that already
exists, so each is handed to the existing Newton-Raphson solver unchanged:

  * Stage 1 is *exactly* the theta method with ``theta_implicit=0.5`` over a
    step of length ``gamma*dt``, so it needs no new residual at all.
  * Stage 2 needs a new residual, built here, which differs from the theta
    residual only in its right-hand side.

See `tr_bdf2_stage2_block_residual` for the derivation of the stage-2 residual
in the conservative form TORAX uses.
"""

import dataclasses
from typing import Final

import jax
from jax import numpy as jnp
import numpy as np
from torax._src import models as models_lib
from torax._src import state
from torax._src import tridiagonal
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import updaters
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import fvm_conversions
from torax._src.fvm import residual_and_loss
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.sources import source_profiles

# pylint: disable=invalid-name

# The stage-1 (trapezoidal) end point, as a fraction of the full step.
# gamma = 2 - sqrt(2) is the unique value for which the two stages share the
# same implicit coefficient, gamma/2 * dt: stage 1 contributes
# theta * (gamma*dt) = gamma/2 * dt and stage 2 contributes
# B_IMPLICIT * dt = gamma/2 * dt. That shared diagonal is what makes the method
# L-stable and stiffly accurate, and it means both stages present the Newton
# solver with an identically conditioned Jacobian.
GAMMA: Final[float] = 2.0 - np.sqrt(2.0)

# Stage 1 is the trapezoidal rule, i.e. the theta method with theta = 1/2.
TRAPEZOIDAL_THETA: Final[float] = 0.5

# Stage-2 (BDF2) weights on the three points t_n, t_n + gamma*dt, t_n + dt.
# In closed form for gamma = 2 - sqrt(2):
#   A_STAGE1 = (1 + sqrt(2)) / 2, A_START = (1 - sqrt(2)) / 2, B = gamma / 2.
A_STAGE1: Final[float] = 1.0 / (GAMMA * (2.0 - GAMMA))
A_START: Final[float] = -((1.0 - GAMMA) ** 2) / (GAMMA * (2.0 - GAMMA))
B_IMPLICIT: Final[float] = (1.0 - GAMMA) / (2.0 - GAMMA)

# Butcher weights of the second-order method, b = [w, w, d], and of the
# embedded third-order method, b_hat. The step controller signal is driven by
# their difference, which sums to zero.
_W: Final[float] = np.sqrt(2.0) / 4.0
_D: Final[float] = GAMMA / 2.0
# b_hat = [(1 - w)/3, (3w + 1)/3, d/3] satisfies all four third-order
# conditions for the TR-BDF2 tableau, so b_hat - b annihilates any solution the
# second-order method integrates exactly and leaves the leading O(dt^3) term.
EMBEDDED_WEIGHT_START: Final[float] = (1.0 - 4.0 * _W) / 3.0
EMBEDDED_WEIGHT_STAGE1: Final[float] = 1.0 / 3.0
EMBEDDED_WEIGHT_NEW: Final[float] = -2.0 * _D / 3.0


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StageInputs:
  """Runtime inputs evaluated at the TR-BDF2 stage-1 time ``t_n + gamma*dt``.

  The solver cannot build these itself: interpolating the runtime params and
  the geometry to an intermediate time needs the providers, which live in the
  orchestration layer. They are therefore built there (see
  `step_function_processing.build_tr_bdf2_stage_inputs`) and passed down.

  Attributes:
    runtime_params: Runtime parameters at ``t_n + gamma*dt``.
    geo: Geometry at ``t_n + gamma*dt``.
    core_profiles: Core profiles at ``t_n + gamma*dt``, carrying the updated
      boundary conditions and prescribed profiles for that time.
  """

  runtime_params: runtime_params_lib.RuntimeParams
  geo: geometry.Geometry
  core_profiles: state.CoreProfiles


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Stage2Data:
  """The ``t_n`` information the BDF2 stage needs on top of the stage-1 state.

  Attributes:
    x_start: Evolving profiles at ``t_n``.
    tc_in_start: ``transient_in_cell`` at ``t_n``, stacked to shape
      (num_cells, num_channels). Carried explicitly because the BDF2 weights
      act on the conserved quantity ``tc_in * x``, not on ``x``, and ``tc_in``
      is state dependent (it contains ``n_i``, ``n_e`` and ``vpr``).
  """

  x_start: tuple[cell_variable.CellVariable, ...]
  tc_in_start: jax.Array


@jax.jit(
    static_argnames=[
        'evolving_names',
        'models',
    ],
)
def tr_bdf2_stage2_block_residual(
    x_new_guess_vec: jax.Array,
    dt: jax.Array,
    runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t: state.CoreProfiles,
    core_profiles_t_plus_dt: state.CoreProfiles,
    explicit_source_profiles: source_profiles.SourceProfiles,
    models: models_lib.Models,
    coeffs_old: block_1d_coeffs.Block1DCoeffs,
    evolving_names: tuple[str, ...],
    pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
    stage2: Stage2Data,
) -> jax.Array:
  # pyformat: disable  # pyformat removes line breaks needed for readability
  """Residual of the TR-BDF2 BDF2 stage for core profiles at ``t_n + dt``.

  Derivation
  ----------
  As in `residual_and_loss.theta_method_matrix_equation`, the system solved is

    tc_out partial (tc_in x) / partial t = F

  with `tc_in` inside the time derivative and `tc_out` outside it. Writing the
  conserved quantity as `u = tc_in x` and `G(x) = F(x) / tc_out`, this is

    partial u / partial t = G.

  This distinction is the crux of applying a multi-stage method here: TR-BDF2
  is a Runge-Kutta method, so its stage combination is a linear combination of
  `u` at the stage points, *not* of `x`. `tc_in` is itself state dependent
  (it contains `n_i`, `n_e` and `vpr`), so each stage carries its own `tc_in`,
  evaluated at that stage's state, time and geometry.

  Label the three points of the step `n` (at `t_n`), `g` (the stage-1 point at
  `t_n + gamma*dt`) and `new` (at `t_n + dt`). Stage 1 is the trapezoidal rule
  over `gamma*dt`, i.e. exactly the theta method with theta = 1/2, and needs no
  code of its own:

    | (tc_in_g x_g - tc_in_n x_n) / (gamma dt) =
    | 1/2 F_g / tc_out_g + 1/2 F_n / tc_out_n

  Stage 2 is the BDF2 formula on `u_n`, `u_g`, `u_new`:

    | u_new = A_STAGE1 u_g + A_START u_n + B_IMPLICIT dt G_new

  i.e., in terms of `x` and the transient coefficients,

    | tc_in_new x_new =
    | A_STAGE1 tc_in_g x_g + A_START tc_in_n x_n
    | + B_IMPLICIT dt F_new / tc_out_new

  Note that `A_STAGE1 + A_START = 1`, so the stage is consistent, and that the
  only place the step size enters is through the implicit coefficient
  `dt_implicit = B_IMPLICIT dt`. This function takes that product directly as
  its `dt` argument, which makes the stage structurally identical to a backward
  Euler solve of length `dt_implicit` started from the extrapolated state
  `A_STAGE1 u_g + A_START u_n`.

  As in the theta method, the equation lives on the cell grid where `tc` is
  never zero, so we divide through by `tc_in_new` to scale the residual to `x`:

    | x_new - dt_implicit F_new / (tc_out_new tc_in_new) =
    | A_STAGE1 (tc_in_g / tc_in_new) x_g + A_START (tc_in_n / tc_in_new) x_n

  Substituting `F = C x + c` gives the assembled system:

    | (I - dt_implicit diag(1/(tc_out_new tc_in_new)) C_new) x_new
    | - dt_implicit diag(1/(tc_out_new tc_in_new)) c_new
    | =
    | diag(A_STAGE1 tc_in_g / tc_in_new) x_g
    | + A_START (tc_in_n / tc_in_new) x_n

  The left-hand side is identical to the theta-method left-hand side with
  `theta_implicit = 1` and step `dt_implicit`, so it is assembled by calling
  `theta_method_matrix_equation` directly. Only the right-hand side is new: the
  stage-1 term is the theta method's `diag(tc_in_old / tc_in_new)` rescaled by
  `A_STAGE1`, and the `t_n` term is a pure forcing vector since `x_n` is known.

  Args:
    x_new_guess_vec: Flattened array of the current guess of x at `t_n + dt`.
    dt: The *implicit* step `B_IMPLICIT * dt_full`, see derivation above.
    runtime_params_t_plus_dt: Runtime parameters for time `t_n + dt`.
    geo_t_plus_dt: The geometry at time `t_n + dt`.
    x_old: The stage-1 solution `x_g`, at `t_n + gamma*dt`.
    core_profiles_t: Core profiles at the stage-1 point, carrying the prescribed
      quantities there.
    core_profiles_t_plus_dt: Core profiles at `t_n + dt` carrying the evolved
      boundary conditions and prescribed profiles.
    explicit_source_profiles: Pre-calculated sources implemented as explicit
      sources in the PDE.
    models: Models used for the calculations.
    coeffs_old: The coefficients calculated at the stage-1 point. Only
      `transient_in_cell` is read, so the reduced coefficients suffice.
    evolving_names: The names of variables within the core profiles that should
      evolve.
    pedestal_transition_state: State for tracking pedestal L-H and H-L
      transitions.
    stage2: The `t_n` state and transient coefficients.

  Returns:
    residual: Vector residual between LHS and RHS of the BDF2 stage equation.
  """
  # pyformat: enable
  x_new_guess = fvm_conversions.vec_to_cell_variable_tuple(
      x_new_guess_vec, core_profiles_t_plus_dt, evolving_names
  )
  core_profiles_t_plus_dt = updaters.update_core_profiles_during_step(
      x_new_guess,
      runtime_params_t_plus_dt,
      geo_t_plus_dt,
      core_profiles_t_plus_dt,
      prev_core_profiles=core_profiles_t,
      dt=dt,
      evolving_names=evolving_names,
  )
  coeffs_new = calc_coeffs.calc_coeffs(
      runtime_params=runtime_params_t_plus_dt,
      geo=geo_t_plus_dt,
      core_profiles=core_profiles_t_plus_dt,
      explicit_source_profiles=explicit_source_profiles,
      models=models,
      evolving_names=evolving_names,
      use_pereverzev=False,
      pedestal_transition_state=pedestal_transition_state,
  )

  solver_params = runtime_params_t_plus_dt.solver
  # theta_implicit=1 makes this assemble exactly the BDF2 stage's left-hand
  # side, and makes the returned right-hand side the bare transient term
  # diag(tc_in_g / tc_in_new) with no off-diagonal or coupling bands.
  lhs, lhs_vec, rhs, _ = residual_and_loss.theta_method_matrix_equation(
      dt=dt,
      x_old=x_old,
      x_new_guess=x_new_guess,
      coeffs_old=coeffs_old,
      coeffs_new=coeffs_new,
      theta_implicit=1.0,
      convection_dirichlet_mode=solver_params.convection_dirichlet_mode,
      convection_neumann_mode=solver_params.convection_neumann_mode,
  )

  # Apply the BDF2 weight to the stage-1 term, and put the (known) t_n term in
  # the forcing vector. Both are weighted by their own tc_in, because the BDF2
  # combination acts on u = tc_in * x.
  tc_in_new = jnp.stack(coeffs_new.transient_in_cell, axis=-1)
  x_start_array = fvm_conversions.cell_variable_tuple_to_array(
      stage2.x_start, axis=1
  )
  rhs = tridiagonal.ChannelTriDiagonal(diagonal=A_STAGE1 * rhs.diagonal)
  rhs_vec = A_START * stage2.tc_in_start * x_start_array / tc_in_new

  # Apply direct IBC enforcement via matrix row replacement.
  if coeffs_new.has_internal_boundary_conditions:
    lhs, lhs_vec, rhs, rhs_vec = (
        residual_and_loss.apply_internal_boundary_conditions(
            lhs,
            lhs_vec,
            rhs,
            rhs_vec,
            coeffs_new.internal_boundary_condition_mask,
            coeffs_new.internal_boundary_condition_target_vec,
        )
    )

  x_old_array = fvm_conversions.cell_variable_tuple_to_array(x_old, axis=1)
  num_cells, num_channels = x_old_array.shape
  x_new_array = x_new_guess_vec.reshape(num_channels, num_cells).T

  lhs_result = lhs.matvec(x_new_array) + lhs_vec
  rhs_result = rhs.matvec(x_old_array) + rhs_vec

  return (lhs_result - rhs_result).T.reshape(-1)


def embedded_error_estimate(
    dt: jax.Array,
    u_start: jax.Array,
    u_stage1: jax.Array,
    u_new: jax.Array,
    stage_derivative_start: jax.Array,
    tc_in_new: jax.Array,
) -> jax.Array:
  # pyformat: disable  # pyformat removes line breaks needed for readability
  """Returns the embedded TR-BDF2 error estimate, scaled to `x`.

  The embedded third-order method differs from the second-order one only in its
  weights, so the estimate is

    err = dt sum_i (b_hat_i - b_i) G_i

  over the three stage derivatives `G_i = F_i / tc_out_i` of the conserved
  quantity `u = tc_in x`. Rather than evaluate `F` three more times, we recover
  two of the three derivatives from the stage equations themselves, which hold
  exactly at the converged solution:

    * Stage 1 (trapezoidal) gives
        G_n + G_g = 2 (u_g - u_n) / (gamma dt)
    * Stage 2 (BDF2) gives
        B_IMPLICIT dt G_new = u_new - A_STAGE1 u_g - A_START u_n

  so only `G_n` has to be evaluated directly, and it is the cheapest of the
  three because the coefficients at `t_n` are already assembled for stage 1.

  The result is divided by `tc_in_new` for the same reason the residual is: it
  converts an error in the conserved quantity to an error in `x`, which is the
  O(1) quantity the step controller should compare against `atol + rtol |x|`.

  Args:
    dt: The full step size.
    u_start: Conserved quantity `tc_in x` at `t_n`.
    u_stage1: Conserved quantity at the stage-1 point.
    u_new: Conserved quantity at `t_n + dt`.
    stage_derivative_start: `G_n = F_n / tc_out_n`, the derivative of `u` at
      `t_n`.
    tc_in_new: `transient_in_cell` at `t_n + dt`.

  Returns:
    The per-(cell, channel) error estimate in the units of `x`.
  """
  # pyformat: enable
  derivative_sum = 2.0 * (u_stage1 - u_start) / (GAMMA * dt)
  stage_derivative_stage1 = derivative_sum - stage_derivative_start
  stage_derivative_new = (
      u_new - A_STAGE1 * u_stage1 - A_START * u_start
  ) / (B_IMPLICIT * dt)
  error_in_u = dt * (
      EMBEDDED_WEIGHT_START * stage_derivative_start
      + EMBEDDED_WEIGHT_STAGE1 * stage_derivative_stage1
      + EMBEDDED_WEIGHT_NEW * stage_derivative_new
  )
  return error_in_u / tc_in_new
