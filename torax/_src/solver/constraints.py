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
"""Constraint/actuator pairs as a bordered extension of the Newton system.

A constraint/actuator pair imposes a physics target (e.g. line-averaged
density = Y) and frees a scalar actuator (e.g. the gas puff rate) as an
extra unknown that the nonlinear solver finds. Each pair appends one border
column (how the actuator enters the PDE) and one border row (the constraint
equation) to the Newton system; see docs/constraints_and_actuators.rst for
the design rationale and `torax._src.config.constraints` for the config
model.

Two constraint modes are supported:

* ``'hard'``: the constraint is an algebraic equation, enforced exactly at
  every timestep (index-1 DAE semantics). The initial state should satisfy
  the constraint, and the actuator must be able to reach the target.
* ``'relaxed'``: the actuator integrates the constraint violation with time
  constant tau (an implicit integral controller), which regularises the
  system against actuator saturation and inconsistent initial conditions.
  As tau -> 0 the hard constraint is recovered.

The actuator state u_hat travels through the step as follows: the previous
step's converged values are injected into
``runtime_params_t_plus_dt.constraints[j].u_hat_old`` by the orchestration
(`inject_actuator_state`), the Newton solve finds the new values jointly
with the profiles, and they are returned in
``SolverNumericOutputs.actuators`` which persists in the ``SimState``.
"""

import dataclasses
from typing import Callable, Final

import jax
from jax import numpy as jnp
import numpy as np
from torax._src import math_utils
from torax._src.config import constraints as constraints_config
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.geometry import geometry as geometry_lib

# Re-exported for convenience: the config-side models.
ConstraintConfig = constraints_config.ConstraintConfig
ConstraintRuntimeParams = constraints_config.ConstraintRuntimeParams

# Constraint quantities: the evolving channel each constraint reads. Used for
# both the constraint evaluation and the declared border-row pattern.
_CONSTRAINT_CHANNELS: dict[str, str] = {
    'n_e_line_avg': 'n_e',
}

# Actuators: dotted runtime-params path -> the evolving channels its source
# deposits into. Used for the declared border-column pattern.
_ACTUATOR_CHANNELS: dict[str, tuple[str, ...]] = {
    'sources.gas_puff.S_total': ('n_e',),
}

# Smoothing of the Fischer-Burmeister function at its origin. The
# complementarity solution is perturbed by O(epsilon^2 / max(a, b)), so at
# 1e-8 the enforcement error is at machine level while the corner of
# sqrt(a^2 + b^2) stays differentiable for the Newton solve.
_FB_EPSILON: Final[float] = 1e-8


def initial_actuators(
    constraints: tuple[ConstraintRuntimeParams, ...],
) -> jax.Array | None:
  """Returns the initial nondimensional actuator vector, or None if empty."""
  if not constraints:
    return None
  return jnp.ones(len(constraints))


def inject_actuator_state(
    runtime_params: runtime_params_lib.RuntimeParams,
    actuators: jax.Array | None,
) -> runtime_params_lib.RuntimeParams:
  """Writes the previous step's actuator values into the runtime params.

  Args:
    runtime_params: Runtime params at t + dt, whose constraints carry the
      provider-default u_hat_old.
    actuators: Nondimensional actuator values from the previous step's
      SolverNumericOutputs, or None to keep the provider defaults (used at
      the first step, where the actuator starts at its reference magnitude).

  Returns:
    Runtime params with u_hat_old replaced per constraint.
  """
  if actuators is None or not runtime_params.constraints:
    return runtime_params
  new_constraints = tuple(
      dataclasses.replace(constraint, u_hat_old=actuators[j])
      for j, constraint in enumerate(runtime_params.constraints)
  )
  return dataclasses.replace(runtime_params, constraints=new_constraints)


def _replace_path(obj, parts: list[str], value):
  """Returns a copy of a runtime-params pytree with one leaf replaced."""
  head, *rest = parts
  if isinstance(obj, dict):
    child = obj[head]
    new_child = value if not rest else _replace_path(child, rest, value)
    new_obj = dict(obj)
    new_obj[head] = new_child
    return new_obj
  child = getattr(obj, head)
  new_child = value if not rest else _replace_path(child, rest, value)
  return dataclasses.replace(obj, **{head: new_child})


def substitute_actuators(
    runtime_params: runtime_params_lib.RuntimeParams,
    constraints: tuple[ConstraintRuntimeParams, ...],
    u_hat: jax.Array,
) -> runtime_params_lib.RuntimeParams:
  """Substitutes the border unknowns into the runtime params.

  Args:
    runtime_params: Runtime params at t + dt.
    constraints: The constraint runtime params, in border order.
    u_hat: Nondimensional actuator vector of shape (num_constraints,).

  Returns:
    Runtime params with each actuated parameter replaced by its unknown in
    physical units. The actuated source must be implicit, so that the
    substitution is seen by the residual and its Jacobian.
  """
  for j, constraint in enumerate(constraints):
    value = u_hat[j] * constraint.actuator_reference
    runtime_params = _replace_path(
        runtime_params, constraint.actuator.split('.'), value
    )
  return runtime_params


def _constraint_violation(
    constraint: ConstraintRuntimeParams,
    x_vec: jax.Array,
    geo: geometry_lib.Geometry,
    evolving_names: tuple[str, ...],
    num_cells: int,
) -> jax.Array:
  """Returns g_hat = (value - target) / target for one constraint."""
  channel = _CONSTRAINT_CHANNELS[constraint.constraint]
  index = evolving_names.index(channel)
  solver_values = x_vec[index * num_cells : (index + 1) * num_cells]
  physical_values = solver_values * convertors.SCALING_FACTORS[channel]
  # Line average = cell integration over rho_norm in [0, 1], written with an
  # explicit quadrature rather than math_utils.line_average: the latter goes
  # through geo.drho_norm, a cached property whose first evaluation inside
  # the Newton trace would cache (and leak) a tracer on the grid object.
  value = jnp.sum(physical_values * jnp.diff(geo.rho_face_norm))
  return (value - constraint.target) / constraint.target


def build_augmented_residual(
    pde_residual_fun: Callable[
        [jax.Array, runtime_params_lib.RuntimeParams], jax.Array
    ],
    runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
    geo: geometry_lib.Geometry,
    constraints: tuple[ConstraintRuntimeParams, ...],
    evolving_names: tuple[str, ...],
    num_cells: int,
    dt: jax.Array,
) -> Callable[[jax.Array], jax.Array]:
  """Builds the bordered residual over the augmented vector [x, u_hat].

  The augmented residual stacks the PDE residual (evaluated with the
  actuators substituted into the runtime params) with one constraint row per
  pair:

  * 'hard':    g_hat(x) = 0
  * 'hard' with a lower bound u_min: the Fischer-Burmeister complementarity
    row phi(u_hat - u_hat_min, g_hat) = 0 with
    phi(a, b) = a + b - sqrt(a^2 + b^2 + eps^2), which enforces
    a >= 0, b >= 0, a * b ~= 0: either the target is met (g_hat = 0) with a
    feasible actuator, or the actuator sits at the bound and the constraint
    is violated in the only direction it can be (g_hat > 0). At saturation
    the row degenerates toward the well-conditioned u_hat = u_hat_min rather
    than toward a vanishing Schur complement.
  * 'relaxed': tau * (u_hat - u_hat_old) / dt + g_hat(x) = 0,

  the fully implicit discretisation of tau * du_hat/dt = -g_hat, so the
  actuator rises while the quantity is below target (assuming the actuator
  increases the quantity). Constraint rows are dimensionless and O(1),
  matching the scaled PDE residual. The start-of-step actuator values are
  read from each constraint's u_hat_old field.

  Args:
    pde_residual_fun: The theta-method residual as a function of the profile
      vector and the runtime params at t + dt.
    runtime_params_t_plus_dt: Runtime params at t + dt.
    geo: Geometry at t + dt.
    constraints: Constraint runtime params, defining the border order.
    evolving_names: Evolving channel names, in solver order.
    num_cells: Number of radial cells.
    dt: Timestep duration.

  Returns:
    A function mapping the augmented vector of size
    (num_cells * num_channels + num_constraints) to the augmented residual.
  """
  size = num_cells * len(evolving_names)

  def augmented_residual(z: jax.Array) -> jax.Array:
    x_vec = z[:size]
    u_hat = z[size:]
    substituted = substitute_actuators(
        runtime_params_t_plus_dt, constraints, u_hat
    )
    pde_residual = pde_residual_fun(x_vec, substituted)
    rows = []
    for j, constraint in enumerate(constraints):
      g_hat = _constraint_violation(
          constraint, x_vec, geo, evolving_names, num_cells
      )
      if constraint.mode == 'hard':
        if constraint.u_min is None:
          rows.append(g_hat)
        else:
          # Fischer-Burmeister complementarity between the actuator's
          # distance to its bound and the constraint violation. The pairing
          # relies on the actuator increasing the constrained quantity, so
          # saturation at the lower bound can only leave g_hat positive.
          a = u_hat[j] - constraint.u_min / constraint.actuator_reference
          rows.append(
              a + g_hat - jnp.sqrt(a**2 + g_hat**2 + _FB_EPSILON**2)
          )
      else:
        rows.append(
            constraint.tau * (u_hat[j] - constraint.u_hat_old) / dt + g_hat
        )
    return jnp.concatenate([pde_residual, jnp.stack(rows)])

  return augmented_residual


def augment_pattern(
    pattern: np.ndarray,
    evolving_names: tuple[str, ...],
    num_cells: int,
    constraints: tuple[ConstraintRuntimeParams, ...],
) -> np.ndarray:
  """Extends a declared Jacobian pattern with the constraint border.

  Args:
    pattern: Declared (size, size) PDE pattern from
      `jacobian_pattern.build_pattern`.
    evolving_names: Evolving channel names, in solver order.
    num_cells: Number of radial cells.
    constraints: Constraint runtime params, in border order.

  Returns:
    Boolean array of shape (size + m, size + m). Border rows carry the
    constraint quantity's support; border columns carry the actuated
    source's deposition support; the border diagonal is declared for the
    relaxation term (and conservatively in hard mode).
  """
  size = pattern.shape[0]
  m = len(constraints)
  augmented = np.zeros((size + m, size + m), dtype=bool)
  augmented[:size, :size] = pattern
  for j, constraint in enumerate(constraints):
    row_channel = _CONSTRAINT_CHANNELS[constraint.constraint]
    row_index = evolving_names.index(row_channel)
    augmented[
        size + j, row_index * num_cells : (row_index + 1) * num_cells
    ] = True
    for column_channel in _ACTUATOR_CHANNELS[constraint.actuator]:
      column_index = evolving_names.index(column_channel)
      augmented[
          column_index * num_cells : (column_index + 1) * num_cells,
          size + j,
      ] = True
    augmented[size + j, size + j] = True
  return augmented
