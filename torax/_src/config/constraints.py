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
        u_hat_old=1.0,
        constraint=self.constraint,
        actuator=self.actuator,
        mode=self.mode,
    )
