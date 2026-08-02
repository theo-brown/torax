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

"""The Solver class.

Abstract base class defining updates to State.
"""

import abc
import dataclasses

import jax
import jax.numpy as jnp
from torax._src import jax_utils
from torax._src import models as models_lib
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import updaters
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.solver import psi_splitting
from torax._src.sources import source_profiles


@dataclasses.dataclass(frozen=True, eq=False)
class Solver(static_dataclass.StaticDataclass, abc.ABC):
  """Solves for a single time step's update to State.

  Attributes:
    models: Physics models.
  """

  models: models_lib.Models

  @jax.jit(
      static_argnames=[
          'self',
      ],
  )
  def __call__(
      self,
      t: jax.Array,
      dt: jax.Array,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """Applies a time step update.

    Args:
      t: Time.
      dt: Time step duration.
      runtime_params_t: Runtime parameters for time t (the start time of the
        step). These runtime params can change from step to step without
        triggering a recompilation.
      runtime_params_t_plus_dt: Runtime parameters for time t + dt, used for
        implicit calculations in the solver.
      geo_t: Geometry of the torus at time t.
      geo_t_plus_dt: Geometry of the torus at time t + dt.
      core_profiles_t: Core plasma profiles at the beginning of the time step.
      core_profiles_t_plus_dt: Core plasma profiles which contain all available
        prescribed quantities at the end of the time step. This includes
        evolving boundary conditions and prescribed time-dependent profiles that
        are not being evolved by the PDE system.
      explicit_source_profiles: Source profiles of all explicit sources (as
        configured by the input params). All implicit source's profiles will be
        set to 0 in this object. These explicit source profiles were calculated
        either based on the original core profiles at the start of the time step
        or were independent of the core profiles. Because they were calculated
        outside the possibly-JAX-jitted solver logic, they can be calculated in
        non-JAX-friendly ways.
      pedestal_transition_state: State for tracking pedestal L-H and H-L
        transitions.

    Returns:
      x_new: Tuple containing new cell-grid values of the evolving variables.
      solver_numeric_output: Error and solver iteration info.
    """

    # This base class method can be completely overridden by a subclass, but
    # most can make use of the boilerplate here and just implement `_x_new`.

    evolving_names = runtime_params_t.numerics.evolving_names
    # Splitting psi out of the block only means anything if psi is being
    # evolved together with at least one kinetic channel, so the flag is a
    # no-op otherwise (in particular when evolve_current=False).
    split_psi = (
        runtime_params_t.solver.split_psi
        and psi_splitting.PSI_NAME in evolving_names
        and len(evolving_names) > 1
    )

    # Don't call solver functions on an empty list
    if split_psi:
      (
          x_new,
          solver_numeric_output,
      ) = self._x_new_split_psi(
          dt=dt,
          runtime_params_t=runtime_params_t,
          runtime_params_t_plus_dt=runtime_params_t_plus_dt,
          geo_t=geo_t,
          geo_t_plus_dt=geo_t_plus_dt,
          core_profiles_t=core_profiles_t,
          core_profiles_t_plus_dt=core_profiles_t_plus_dt,
          explicit_source_profiles=explicit_source_profiles,
          evolving_names=evolving_names,
          pedestal_transition_state=pedestal_transition_state,
      )
    elif evolving_names:
      (
          x_new,
          solver_numeric_output,
      ) = self._x_new(
          dt=dt,
          runtime_params_t=runtime_params_t,
          runtime_params_t_plus_dt=runtime_params_t_plus_dt,
          geo_t=geo_t,
          geo_t_plus_dt=geo_t_plus_dt,
          core_profiles_t=core_profiles_t,
          core_profiles_t_plus_dt=core_profiles_t_plus_dt,
          explicit_source_profiles=explicit_source_profiles,
          evolving_names=evolving_names,
          pedestal_transition_state=pedestal_transition_state,
      )
    else:
      x_new = tuple()
      solver_numeric_output = state.SolverNumericOutputs(
          sawtooth_crash=False,
          solver_error_state=jnp.array(0, jax_utils.get_int_dtype()),
          inner_solver_iterations=jnp.array(0, jax_utils.get_int_dtype()),
          outer_solver_iterations=jnp.array(0, jax_utils.get_int_dtype()),
      )

    return (
        x_new,
        solver_numeric_output,
    )

  def _x_new_split_psi(
      self,
      dt: jax.Array,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      evolving_names: tuple[str, ...],
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """Advances the step with psi split out of the coupled block.

    The kinetic channels are advanced by the solver's own `_x_new` with psi
    frozen, and psi is advanced by a single linear implicit solve driven by the
    conductivity and bootstrap current of the updated kinetic profiles. See
    `psi_splitting` for why the psi equation does not need to be inside the
    nonlinear block.

    Args:
      dt: Time step duration.
      runtime_params_t: Runtime parameters at time t.
      runtime_params_t_plus_dt: Runtime parameters at time t + dt.
      geo_t: Geometry at time t.
      geo_t_plus_dt: Geometry at time t + dt.
      core_profiles_t: Core plasma profiles at the beginning of the time step.
      core_profiles_t_plus_dt: Core plasma profiles carrying the prescribed
        quantities and boundary conditions at the end of the time step.
      explicit_source_profiles: See the docstring of __call__.
      evolving_names: The names of the evolving core_profiles variables, which
        must contain 'psi' and at least one kinetic channel.
      pedestal_transition_state: State for tracking pedestal transitions.

    Returns:
      x_new: The values of the evolving variables at time t + dt, ordered to
        match `evolving_names`.
      solver_numeric_output: Error and iteration info from the kinetic solve.
        The psi sub-solve is direct and contributes no iterations.
    """
    kinetic_names = tuple(
        name for name in evolving_names if name != psi_splitting.PSI_NAME
    )
    strang = runtime_params_t.solver.split_psi_order == 'strang'

    if strang:
      # Strang: half a psi step, the full kinetic step, then the second half
      # psi step. The kinetic block then sees psi (and hence q, s) at the
      # midpoint of the interval rather than at its left edge, which cancels
      # the leading O(dt) splitting error.
      psi_half = psi_splitting.psi_sub_step(
          models=self.models,
          dt=dt / 2.0,
          runtime_params_old=runtime_params_t,
          runtime_params_new=runtime_params_t,
          geo_old=geo_t,
          geo_new=geo_t,
          core_profiles_old=core_profiles_t,
          core_profiles_new=dataclasses.replace(
              core_profiles_t,
              psi=psi_splitting.interpolate_boundary_conditions(
                  core_profiles_t.psi, core_profiles_t_plus_dt.psi, 0.5
              ),
          ),
          explicit_source_profiles=explicit_source_profiles,
          pedestal_transition_state=pedestal_transition_state,
      )
      # The kinetic solve reads psi from both endpoints, so freeze both at the
      # midpoint value.
      core_profiles_t_kinetic = dataclasses.replace(
          core_profiles_t, psi=psi_half
      )
      core_profiles_t_plus_dt_kinetic = dataclasses.replace(
          core_profiles_t_plus_dt, psi=psi_half
      )
      psi_old = psi_half
      # The second half step starts from psi_half but must land on the t + dt
      # boundary conditions.
      psi_guess = dataclasses.replace(
          core_profiles_t_plus_dt.psi, value=psi_half.value
      )
      psi_sub_step_dt = dt / 2.0
    else:
      core_profiles_t_kinetic = core_profiles_t
      core_profiles_t_plus_dt_kinetic = core_profiles_t_plus_dt
      psi_old = core_profiles_t.psi
      # core_profiles_t_plus_dt.psi already holds the time-t values with the
      # t + dt boundary conditions, which is exactly the guess we want.
      psi_guess = core_profiles_t_plus_dt.psi
      psi_sub_step_dt = dt

    x_kinetic, solver_numeric_output = self._x_new(
        dt=dt,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t=geo_t,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=core_profiles_t_kinetic,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt_kinetic,
        explicit_source_profiles=explicit_source_profiles,
        evolving_names=kinetic_names,
        pedestal_transition_state=pedestal_transition_state,
    )

    # Rebuild the derived quantities (n_i, Z_eff, ...) that the conductivity
    # and the bootstrap current depend on, so that the psi sub-step sees the
    # freshly advanced kinetic profiles.
    core_profiles_kinetic_new = updaters.update_core_profiles_during_step(
        x_kinetic,
        runtime_params_t_plus_dt,
        geo_t_plus_dt,
        core_profiles=core_profiles_t_plus_dt,
        prev_core_profiles=None,
        dt=None,
        evolving_names=kinetic_names,
    )
    psi_new = psi_splitting.psi_sub_step(
        models=self.models,
        dt=psi_sub_step_dt,
        runtime_params_old=runtime_params_t,
        runtime_params_new=runtime_params_t_plus_dt,
        geo_old=geo_t,
        geo_new=geo_t_plus_dt,
        core_profiles_old=dataclasses.replace(core_profiles_t, psi=psi_old),
        core_profiles_new=dataclasses.replace(
            core_profiles_kinetic_new, psi=psi_guess
        ),
        explicit_source_profiles=explicit_source_profiles,
        pedestal_transition_state=pedestal_transition_state,
    )

    # Reassemble in the caller's channel order; downstream code indexes x_new
    # by position in `evolving_names`.
    x_by_name = dict(zip(kinetic_names, x_kinetic))
    x_by_name[psi_splitting.PSI_NAME] = psi_new
    x_new = tuple(x_by_name[name] for name in evolving_names)
    return x_new, solver_numeric_output

  def _x_new(
      self,
      dt: jax.Array,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      evolving_names: tuple[str, ...],
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """Calculates new values of the changing variables.

    Subclasses must either implement `_x_new` so that `Solver.__call__`
    will work, or implement a different `__call__`.

    Args:
      dt: Time step duration.
      runtime_params_t: Runtime parameters for time t (the start time of the
        step). These runtime params can change from step to step without
        triggering a recompilation.
      runtime_params_t_plus_dt: Runtime parameters for time t + dt, used for
        implicit calculations in the solver.
      geo_t: Geometry of the torus for time t.
      geo_t_plus_dt: Geometry of the torus for time t + dt.
      core_profiles_t: Core plasma profiles at the beginning of the time step.
      core_profiles_t_plus_dt: Core plasma profiles which contain all available
        prescribed quantities at the end of the time step. This includes
        evolving boundary conditions and prescribed time-dependent profiles that
        are not being evolved by the PDE system.
      explicit_source_profiles: see the docstring of __call__
      evolving_names: The names of core_profiles variables that should evolve.
      pedestal_transition_state: State for tracking pedestal L-H and H-L
        transitions.

    Returns:
      x_new: The values of the evolving variables at time t + dt.
      solver_numeric_output: Error and solver iteration info.
    """

    raise NotImplementedError(
        f'{type(self)} must implement `_x_new` or '
        'implement a different `__call__` that does not'
        ' need `_x_new`.'
    )
