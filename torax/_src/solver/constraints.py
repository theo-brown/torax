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
the design rationale.

Two constraint modes are supported:

* ``'hard'``: the constraint is an algebraic equation, enforced exactly at
  every timestep (index-1 DAE semantics). The initial state should satisfy
  the constraint, and the actuator must be able to reach the target.
* ``'relaxed'``: the actuator integrates the constraint violation with time
  constant tau (an implicit integral controller), which regularises the
  system against actuator saturation and inconsistent initial conditions.
  As tau -> 0 the hard constraint is recovered.

This module provides the solver-level machinery: the config model, the
augmented residual (actuator substitution + constraint rows), and the
declared-pattern extension for the probe coloring. Wiring the actuator state
through the orchestration loop and outputs is follow-up work; the tests
exercise the augmented system directly through the Newton root finder.
"""

import dataclasses
from typing import Annotated, Callable, Literal

import jax
from jax import numpy as jnp
import numpy as np
import pydantic
from torax._src import array_typing
from torax._src import math_utils
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.geometry import geometry as geometry_lib
from torax._src.torax_pydantic import torax_pydantic

# Constraint quantities: how the scalar g reads the solver state vector. Each
# entry maps a name to (channel read, reduction). The reduction acts on the
# channel's physical-units cell values.
_CONSTRAINT_CHANNELS: dict[str, str] = {
    'n_e_line_avg': 'n_e',
}

# Actuators: dotted runtime-params path -> the evolving channels its source
# deposits into. Used for the declared border-column pattern.
_ACTUATOR_CHANNELS: dict[str, tuple[str, ...]] = {
    'sources.gas_puff.S_total': ('n_e',),
}


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConstraintRuntimeParams:
  """Runtime parameters for one constraint/actuator pair at a time slice.

  Attributes:
    target: The constraint target Y(t), in the constraint quantity's physical
      units.
    tau: Relaxation time constant [s]. Unused in 'hard' mode.
    actuator_reference: Reference magnitude of the actuator, used to
      nondimensionalise the border unknown so it is O(1) alongside the
      scaled profile unknowns.
    constraint: Name of the constraint quantity.
    actuator: Dotted runtime-params path of the actuated parameter.
    mode: 'hard' or 'relaxed'.
  """

  target: array_typing.FloatScalar
  tau: array_typing.FloatScalar
  actuator_reference: array_typing.FloatScalar
  constraint: str = dataclasses.field(metadata={'static': True})
  actuator: str = dataclasses.field(metadata={'static': True})
  mode: str = dataclasses.field(metadata={'static': True})


class ConstraintConfig(torax_pydantic.BaseModelFrozen):
  """Configuration for one constraint/actuator pair.

  Attributes:
    constraint: The physics quantity to hold at the target.
    target: The target value Y(t), in the quantity's physical units.
    actuator: Dotted config path of the parameter freed as an unknown. The
      configured value of that parameter becomes the actuator's reference
      magnitude and initial condition.
    mode: 'relaxed' integrates the constraint violation with time constant
      tau (implicit integral controller); 'hard' enforces the constraint as
      an algebraic equation at every step.
    tau: Relaxation time constant [s]; only used in 'relaxed' mode.
  """

  constraint: Annotated[Literal['n_e_line_avg'], torax_pydantic.JAX_STATIC] = (
      'n_e_line_avg'
  )
  target: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      1e20
  )
  actuator: Annotated[
      Literal['sources.gas_puff.S_total'], torax_pydantic.JAX_STATIC
  ] = 'sources.gas_puff.S_total'
  mode: Annotated[Literal['relaxed', 'hard'], torax_pydantic.JAX_STATIC] = (
      'relaxed'
  )
  tau: pydantic.PositiveFloat = 0.5

  def build_runtime_params(
      self, t: float, actuator_reference: float
  ) -> ConstraintRuntimeParams:
    return ConstraintRuntimeParams(
        target=self.target.get_value(t),
        tau=self.tau,
        actuator_reference=actuator_reference,
        constraint=self.constraint,
        actuator=self.actuator,
        mode=self.mode,
    )


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
  value = math_utils.line_average(physical_values, geo)
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
    u_hat_old: jax.Array,
) -> Callable[[jax.Array], jax.Array]:
  """Builds the bordered residual over the augmented vector [x, u_hat].

  The augmented residual stacks the PDE residual (evaluated with the
  actuators substituted into the runtime params) with one constraint row per
  pair:

  * 'hard':    g_hat(x) = 0
  * 'relaxed': tau * (u_hat - u_hat_old) / dt + g_hat(x) = 0,

  the fully implicit discretisation of tau * du_hat/dt = -g_hat, so the
  actuator rises while the quantity is below target (assuming the actuator
  increases the quantity). Constraint rows are dimensionless and O(1),
  matching the scaled PDE residual.

  Args:
    pde_residual_fun: The theta-method residual as a function of the profile
      vector and the runtime params at t + dt.
    runtime_params_t_plus_dt: Runtime params at t + dt.
    geo: Geometry at t + dt.
    constraints: Constraint runtime params, defining the border order.
    evolving_names: Evolving channel names, in solver order.
    num_cells: Number of radial cells.
    dt: Timestep duration.
    u_hat_old: Nondimensional actuator values at the start of the step.

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
        rows.append(g_hat)
      else:
        rows.append(
            constraint.tau * (u_hat[j] - u_hat_old[j]) / dt + g_hat
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
