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
"""Configuration for constraint/actuator pairs.

A constraint/actuator pair imposes a physics target (e.g. line-averaged
density = Y) and frees a scalar actuator (e.g. the gas puff rate) as an
extra unknown found by the nonlinear solver. See
docs/constraints_and_actuators.rst for the design, and
torax/_src/solver/constraints.py for the solver-level mechanics.

This module is a leaf of the config dependency graph so that
`config.runtime_params` can carry the runtime dataclass without import
cycles.
"""

import dataclasses
from typing import Annotated, Literal

import jax
import pydantic
from torax._src import array_typing
from torax._src.torax_pydantic import torax_pydantic


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
    u_hat_old: The nondimensional actuator value at the start of the step.
      The provider default of 1.0 (the reference magnitude) is overwritten by
      the orchestration with the previous step's converged value.
    constraint: Name of the constraint quantity.
    actuator: Dotted runtime-params path of the actuated parameter.
    mode: 'hard' or 'relaxed'.
  """

  target: array_typing.FloatScalar
  tau: array_typing.FloatScalar
  actuator_reference: array_typing.FloatScalar
  u_hat_old: array_typing.FloatScalar
  constraint: str = dataclasses.field(metadata={'static': True})
  actuator: str = dataclasses.field(metadata={'static': True})
  mode: str = dataclasses.field(metadata={'static': True})
  # Bounds of the actuator in physical units, or None when unbounded. With a
  # bound, the hard constraint becomes a complementarity condition: either
  # the target is met with a feasible actuator, or the actuator sits at a
  # bound and the constraint is violated in the only direction that bound
  # allows. See `solver.constraints.build_augmented_residual`.
  u_min: array_typing.FloatScalar | None = None
  u_max: array_typing.FloatScalar | None = None
  # Radial location of point-valued constraints (e.g. the pedestal top).
  # Unused by integral constraints.
  rho_norm: array_typing.FloatScalar = 0.0


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
    u_min: Optional lower bound of the actuator, in the actuated parameter's
      physical units (e.g. 0.0 to forbid a negative gas puff rate). Only
      supported in 'hard' mode, where the constraint row becomes a
      Fischer-Burmeister complementarity condition: the target is met when
      the actuator is feasible, and the actuator saturates at the bound
      (with the constraint honestly violated) when it is not. Bounded
      'relaxed' mode needs anti-windup and is not yet implemented.
    u_max: Optional upper bound of the actuator, in the same units. Combining
      both bounds gives a box complementarity condition.
    rho_norm: Radial location of point-valued constraints such as 'T_e_ped'
      (the pedestal-top temperature). Ignored by integral constraints.
  """

  constraint: Annotated[
      Literal['n_e_line_avg', 'T_e_ped'], torax_pydantic.JAX_STATIC
  ] = 'n_e_line_avg'
  target: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      1e20
  )
  actuator: Annotated[
      Literal[
          'sources.gas_puff.S_total',
          'transport.pedestal_suppression',
      ],
      torax_pydantic.JAX_STATIC,
  ] = 'sources.gas_puff.S_total'
  mode: Annotated[Literal['relaxed', 'hard'], torax_pydantic.JAX_STATIC] = (
      'relaxed'
  )
  tau: pydantic.PositiveFloat = 0.5
  u_min: float | None = None
  u_max: float | None = None
  rho_norm: torax_pydantic.UnitInterval = 0.9

  @pydantic.model_validator(mode='after')
  def _validate_bounds(self):
    bounded = self.u_min is not None or self.u_max is not None
    if bounded and self.mode != 'hard':
      raise ValueError(
          'actuator bounds are only supported with mode="hard"; bounded'
          ' relaxed mode (anti-windup) is not yet implemented.'
      )
    if (
        self.u_min is not None
        and self.u_max is not None
        and self.u_max <= self.u_min
    ):
      raise ValueError(
          f'u_max ({self.u_max}) must exceed u_min ({self.u_min}).'
      )
    return self

  def build_runtime_params(
      self, t: float, actuator_reference: float
  ) -> ConstraintRuntimeParams:
    return ConstraintRuntimeParams(
        target=self.target.get_value(t),
        tau=self.tau,
        actuator_reference=actuator_reference,
        u_hat_old=1.0,
        constraint=self.constraint,
        actuator=self.actuator,
        mode=self.mode,
        u_min=self.u_min,
        u_max=self.u_max,
        rho_norm=self.rho_norm,
    )
