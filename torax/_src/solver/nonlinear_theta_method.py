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

"""The NonLinearThetaMethod class."""
import abc
import dataclasses

import jax
from jax import numpy as jnp
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.core_profiles import updaters
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import enums
from torax._src.fvm import newton_raphson_solve_block
from torax._src.fvm import optimizer_solve_block
from torax._src.fvm import tr_bdf2
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.solver import runtime_params as solver_runtime_params_lib
from torax._src.solver import solver
from torax._src.sources import source_profiles


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class OptimizerRuntimeParams(solver_runtime_params_lib.RuntimeParams):
  n_max_iterations: int
  loss_tol: float
  initial_guess_mode: int = dataclasses.field(metadata={'static': True})


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NewtonRaphsonRuntimeParams(solver_runtime_params_lib.RuntimeParams):
  maxiter: int
  residual_tol: float
  residual_coarse_tol: float
  tau_min: float
  initial_guess_mode: int = dataclasses.field(metadata={'static': True})
  log_iterations: bool = dataclasses.field(metadata={'static': True})


def _combine_stage_error_states(
    error_state_a: jax.Array,
    error_state_b: jax.Array,
) -> jax.Array:
  """Combines the error states of the two TR-BDF2 stages.

  The codes are not ordered by severity (0 = converged, 1 = did not converge,
  2 = converged to the coarse tolerance only), so a plain maximum would report
  a non-convergence as a coarse convergence and let the step through.

  Args:
    error_state_a: Error state of the first stage.
    error_state_b: Error state of the second stage.

  Returns:
    The error state of the combined step.
  """
  did_not_converge = (error_state_a == 1) | (error_state_b == 1)
  coarse = (error_state_a == 2) | (error_state_b == 2)
  return jnp.where(did_not_converge, 1, jnp.where(coarse, 2, 0)).astype(
      jax_utils.get_int_dtype()
  )


class NonlinearThetaMethod(solver.Solver):
  """Time step update using nonlinear solvers and the theta method."""

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
      tr_bdf2_stage_inputs: tr_bdf2.StageInputs | None = None,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """See Solver._x_new docstring."""

    coeffs_callback = calc_coeffs.CoeffsCallback(
        models=self.models,
        evolving_names=evolving_names,
    )
    (
        x_new,
        solver_numeric_outputs,
    ) = self._x_new_helper(
        dt=dt,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t=geo_t,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=core_profiles_t,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        explicit_source_profiles=explicit_source_profiles,
        coeffs_callback=coeffs_callback,
        evolving_names=evolving_names,
        pedestal_transition_state=pedestal_transition_state,
        tr_bdf2_stage_inputs=tr_bdf2_stage_inputs,
    )

    return (
        x_new,
        solver_numeric_outputs,
    )

  @abc.abstractmethod
  def _x_new_helper(
      self,
      dt: jax.Array,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      coeffs_callback: calc_coeffs.CoeffsCallback,
      evolving_names: tuple[str, ...],
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
      tr_bdf2_stage_inputs: tr_bdf2.StageInputs | None = None,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """Abstract method for subclasses to implement the specific nonlinear solve.

    This helper method is called by `_x_new` after it has constructed the
    `coeffs_callback`. Subclasses should implement this method to call their
    respective nonlinear solver implementation (e.g., Newton-Raphson or an
    optimizer-based approach).

    Args:
      dt: Time step duration.
      runtime_params_t: Runtime parameters for time t (the start time of the
        step).
      runtime_params_t_plus_dt: Runtime parameters for time t + dt, used for
        implicit calculations in the solver.
      geo_t: Magnetic geometry at time t.
      geo_t_plus_dt: Magnetic geometry at time t + dt.
      core_profiles_t: Core plasma profiles at the beginning of the time step.
      core_profiles_t_plus_dt: Core plasma profiles which contain all available
        prescribed quantities at the end of the time step. This includes
        evolving boundary conditions and prescribed time-dependent profiles that
        are not being evolved by the PDE system.
      explicit_source_profiles: Pre-calculated sources implemented as explicit
        sources in the PDE.
      coeffs_callback: Calculates diffusion, convection etc. coefficients given
        a core_profiles, geometry, runtime_params. Repeatedly called by the
        iterative solvers.
      evolving_names: The names of variables within the core profiles that
        should evolve.
      pedestal_transition_state: State for tracking pedestal L-H and H-L
        transitions.
      tr_bdf2_stage_inputs: See the `Solver.__call__` docstring.

    Returns:
      A tuple containing:
        - The new values of the evolving variables at time t + dt.
        - Solver iteration and error info.
    """
    ...


class OptimizerThetaMethod(NonlinearThetaMethod):
  """Minimize the squared norm of the residual of the theta method equation."""

  def _x_new_helper(
      self,
      dt: jax.Array,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      coeffs_callback: calc_coeffs.CoeffsCallback,
      evolving_names: tuple[str, ...],
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
      tr_bdf2_stage_inputs: tr_bdf2.StageInputs | None = None,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """See abstract method docstring in NonlinearThetaMethod."""
    # The optimizer solver only implements the theta method.
    del tr_bdf2_stage_inputs
    solver_params = runtime_params_t.solver
    assert isinstance(solver_params, OptimizerRuntimeParams)
    (
        x_new,
        solver_numeric_outputs,
    ) = optimizer_solve_block.optimizer_solve_block(
        dt=dt,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t=geo_t,
        geo_t_plus_dt=geo_t_plus_dt,
        x_old=convertors.core_profiles_to_solver_x_tuple(
            core_profiles_t, evolving_names
        ),
        core_profiles_t=core_profiles_t,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        models=self.models,
        explicit_source_profiles=explicit_source_profiles,
        coeffs_callback=coeffs_callback,
        evolving_names=evolving_names,
        initial_guess_mode=enums.InitialGuessMode(
            solver_params.initial_guess_mode,
        ),
        maxiter=solver_params.n_max_iterations,
        tol=solver_params.loss_tol,
        pedestal_transition_state=pedestal_transition_state,
    )
    return (
        x_new,
        solver_numeric_outputs,
    )


class NewtonRaphsonThetaMethod(NonlinearThetaMethod):
  """Nonlinear theta method using Newton Raphson."""

  def _x_new_helper(
      self,
      dt: jax.Array,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      coeffs_callback: calc_coeffs.CoeffsCallback,
      evolving_names: tuple[str, ...],
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
      tr_bdf2_stage_inputs: tr_bdf2.StageInputs | None = None,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """See abstract method docstring in NonlinearThetaMethod."""
    solver_params = runtime_params_t.solver
    assert isinstance(solver_params, NewtonRaphsonRuntimeParams)

    x_old = convertors.core_profiles_to_solver_x_tuple(
        core_profiles_t, evolving_names
    )
    # `time_integrator` is a static field, so this branch is resolved at trace
    # time and the theta path below is exactly what it was before TR-BDF2.
    if solver_params.time_integrator == 'tr_bdf2':
      if tr_bdf2_stage_inputs is None:
        raise ValueError(
            'TR-BDF2 requires tr_bdf2_stage_inputs, which must be built by the'
            ' caller because the stage-1 time needs the runtime params and'
            ' geometry providers.'
        )
      return self._tr_bdf2_x_new(
          dt=dt,
          solver_params=solver_params,
          runtime_params_t=runtime_params_t,
          runtime_params_t_plus_dt=runtime_params_t_plus_dt,
          geo_t=geo_t,
          geo_t_plus_dt=geo_t_plus_dt,
          x_old=x_old,
          core_profiles_t=core_profiles_t,
          core_profiles_t_plus_dt=core_profiles_t_plus_dt,
          explicit_source_profiles=explicit_source_profiles,
          coeffs_callback=coeffs_callback,
          evolving_names=evolving_names,
          pedestal_transition_state=pedestal_transition_state,
          stage_inputs=tr_bdf2_stage_inputs,
      )

    (
        x_new,
        solver_numeric_outputs,
    ) = newton_raphson_solve_block.newton_raphson_solve_block(
        dt=dt,
        runtime_params_t=runtime_params_t,
        runtime_params_t_plus_dt=runtime_params_t_plus_dt,
        geo_t=geo_t,
        geo_t_plus_dt=geo_t_plus_dt,
        x_old=x_old,
        core_profiles_t=core_profiles_t,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        explicit_source_profiles=explicit_source_profiles,
        models=self.models,
        coeffs_callback=coeffs_callback,
        evolving_names=evolving_names,
        log_iterations=solver_params.log_iterations,
        initial_guess_mode=enums.InitialGuessMode(
            solver_params.initial_guess_mode
        ),
        maxiter=solver_params.maxiter,
        tol=solver_params.residual_tol,
        coarse_tol=solver_params.residual_coarse_tol,
        delta_reduction_factor=solver_params.delta_reduction_factor,
        tau_min=solver_params.tau_min,
        pedestal_transition_state=pedestal_transition_state,
    )
    return (
        x_new,
        solver_numeric_outputs,
    )

  def _tr_bdf2_x_new(
      self,
      dt: jax.Array,
      solver_params: NewtonRaphsonRuntimeParams,
      runtime_params_t: runtime_params_lib.RuntimeParams,
      runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      x_old: tuple[cell_variable.CellVariable, ...],
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
      coeffs_callback: calc_coeffs.CoeffsCallback,
      evolving_names: tuple[str, ...],
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
      stage_inputs: tr_bdf2.StageInputs,
  ) -> tuple[
      tuple[cell_variable.CellVariable, ...],
      state.SolverNumericOutputs,
  ]:
    """Advances one step with TR-BDF2, as two Newton-Raphson solves.

    Both stages are backward-Euler-shaped implicit solves and are handed to the
    unmodified `newton_raphson_solve_block`. Keeping both stages inside the
    solver means one `SimulationStepFn` step is still exactly one `dt`, so the
    step function, the adaptive dt backtracking and the output cadence are all
    unaffected.

    Args:
      dt: The full step duration.
      solver_params: The Newton-Raphson runtime params.
      runtime_params_t: Runtime parameters at `t`.
      runtime_params_t_plus_dt: Runtime parameters at `t + dt`.
      geo_t: Geometry at `t`.
      geo_t_plus_dt: Geometry at `t + dt`.
      x_old: Evolving profiles at `t`.
      core_profiles_t: Core profiles at `t`.
      core_profiles_t_plus_dt: Core profiles carrying the prescribed quantities
        at `t + dt`.
      explicit_source_profiles: Pre-calculated explicit sources.
      coeffs_callback: Coefficient callback.
      evolving_names: The names of the evolving variables.
      pedestal_transition_state: State for tracking pedestal transitions.
      stage_inputs: Runtime params, geometry and core profiles at the stage-1
        time `t + GAMMA*dt`.

    Returns:
      A tuple of the evolving variables at `t + dt` and the combined solver
      iteration and error info of both stages.
    """
    solve_kwargs = dict(
        explicit_source_profiles=explicit_source_profiles,
        models=self.models,
        coeffs_callback=coeffs_callback,
        evolving_names=evolving_names,
        log_iterations=solver_params.log_iterations,
        initial_guess_mode=enums.InitialGuessMode(
            solver_params.initial_guess_mode
        ),
        maxiter=solver_params.maxiter,
        tol=solver_params.residual_tol,
        coarse_tol=solver_params.residual_coarse_tol,
        delta_reduction_factor=solver_params.delta_reduction_factor,
        tau_min=solver_params.tau_min,
        pedestal_transition_state=pedestal_transition_state,
    )

    # --- Stage 1: trapezoidal rule over GAMMA*dt ---
    # This is exactly the theta method with theta = 1/2 over a shorter step, so
    # it reuses the existing residual with no changes. Note that the theta
    # override also switches `calc_coeffs` to the full (rather than reduced)
    # explicit coefficients, which the trapezoidal rule needs for F(x_n).
    x_stage1, stage1_outputs = (
        newton_raphson_solve_block.newton_raphson_solve_block(
            dt=tr_bdf2.GAMMA * dt,
            runtime_params_t=runtime_params_lib.with_theta_implicit(
                runtime_params_t, tr_bdf2.TRAPEZOIDAL_THETA
            ),
            runtime_params_t_plus_dt=runtime_params_lib.with_theta_implicit(
                stage_inputs.runtime_params, tr_bdf2.TRAPEZOIDAL_THETA
            ),
            # The Pereverzev-stabilised linear guess must stay fully implicit;
            # see `newton_raphson_solve_block`.
            initial_guess_theta_implicit=1.0,
            geo_t=geo_t,
            geo_t_plus_dt=stage_inputs.geo,
            x_old=x_old,
            core_profiles_t=core_profiles_t,
            core_profiles_t_plus_dt=stage_inputs.core_profiles,
            **solve_kwargs,
        )
    )

    # --- Stage 2: BDF2 over the three points t, t + GAMMA*dt, t + dt ---
    # The BDF2 weights act on the conserved quantity tc_in*x, so tc_in at t is
    # needed explicitly. Only `transient_in_cell` is required, which is far
    # cheaper than a full coefficient evaluation.
    tc_in_start = jnp.stack(
        calc_coeffs.calc_transient_in_coeffs(
            geo_t, core_profiles_t, evolving_names
        ).transient_in_cell,
        axis=-1,
    )
    # The stage-1 core profiles: the prescribed quantities at t + GAMMA*dt with
    # the evolving subset replaced by the stage-1 solution.
    core_profiles_stage1 = updaters.update_core_profiles_during_step(
        x_stage1,
        stage_inputs.runtime_params,
        stage_inputs.geo,
        stage_inputs.core_profiles,
        prev_core_profiles=core_profiles_t,
        dt=tr_bdf2.GAMMA * dt,
        evolving_names=evolving_names,
    )
    x_new, stage2_outputs = (
        newton_raphson_solve_block.newton_raphson_solve_block(
            # The BDF2 stage is a backward Euler solve of length
            # B_IMPLICIT*dt started from an extrapolation of the two known
            # points, so passing that product as `dt` makes both the residual
            # and the linear initial guess consistent with the stage.
            dt=tr_bdf2.B_IMPLICIT * dt,
            runtime_params_t=stage_inputs.runtime_params,
            runtime_params_t_plus_dt=runtime_params_t_plus_dt,
            geo_t=stage_inputs.geo,
            geo_t_plus_dt=geo_t_plus_dt,
            x_old=x_stage1,
            core_profiles_t=core_profiles_stage1,
            core_profiles_t_plus_dt=core_profiles_t_plus_dt,
            tr_bdf2_stage2=tr_bdf2.Stage2Data(
                x_start=x_old, tc_in_start=tc_in_start
            ),
            **solve_kwargs,
        )
    )

    solver_numeric_outputs = state.SolverNumericOutputs(
        inner_solver_iterations=(
            stage1_outputs.inner_solver_iterations
            + stage2_outputs.inner_solver_iterations
        ),
        solver_error_state=_combine_stage_error_states(
            stage1_outputs.solver_error_state,
            stage2_outputs.solver_error_state,
        ),
        # One TR-BDF2 step is still one step of the outer loop.
        outer_solver_iterations=jnp.array(1, jax_utils.get_int_dtype()),
        sawtooth_crash=False,
    )
    return x_new, solver_numeric_outputs
