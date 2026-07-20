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
"""The `levenberg_marquardt_solve_block` function.

See function docstring for details.
"""

import functools

import jax
import jax.numpy as jnp
import optimistix as optx
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import models as models_lib
from torax._src import state as state_module
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import enums
from torax._src.fvm import fvm_conversions
from torax._src.fvm import residual_and_loss
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.solver import predictor_corrector_method
from torax._src.sources import source_profiles


@jax.jit(
    static_argnames=[
        'evolving_names',
        'models',
        'coeffs_callback',
        'initial_guess_mode',
        'maxiter',
        'step_rtol',
        'step_atol',
    ],
)
def levenberg_marquardt_solve_block(
    dt: array_typing.FloatScalar,
    runtime_params_t: runtime_params_lib.RuntimeParams,
    runtime_params_t_plus_dt: runtime_params_lib.RuntimeParams,
    geo_t: geometry.Geometry,
    geo_t_plus_dt: geometry.Geometry,
    x_old: tuple[cell_variable.CellVariable, ...],
    core_profiles_t: state_module.CoreProfiles,
    core_profiles_t_plus_dt: state_module.CoreProfiles,
    explicit_source_profiles: source_profiles.SourceProfiles,
    models: models_lib.Models,
    coeffs_callback: calc_coeffs.CoeffsCallback,
    evolving_names: tuple[str, ...],
    initial_guess_mode: enums.InitialGuessMode,
    maxiter: int,
    tol: array_typing.FloatScalar,
    coarse_tol: array_typing.FloatScalar,
    step_rtol: float,
    step_atol: float,
    pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
) -> tuple[
    tuple[cell_variable.CellVariable, ...],
    state_module.SolverNumericOutputs,
]:
  """Runs one theta-method time step using optimistix's Levenberg-Marquardt.

  Solves the nonlinear theta-method equation R(x_new) = 0 by minimizing
  ||R(x_new)||^2 with a damped least-squares (Levenberg-Marquardt) trust-region
  method. Compared to the line-searched Newton-Raphson solver, LM interpolates
  between Gauss-Newton and gradient-descent steps and is typically more robust
  far from the solution (e.g. with stiff transport models), at a similar
  per-iteration cost (one Jacobian evaluation).

  Convergence/error classification is identical to the Newton-Raphson solver:
  the returned error state is 0 if mean|R| < tol at exit, 2 if it is within
  coarse_tol, and 1 otherwise (triggering dt backoff if adaptive_dt is
  enabled).

  Args:
    dt: Discrete time step.
    runtime_params_t: Runtime parameters for time t.
    runtime_params_t_plus_dt: Runtime parameters for time t + dt.
    geo_t: Geometry at time t.
    geo_t_plus_dt: Geometry at time t + dt.
    x_old: Tuple containing CellVariables for each channel with their values at
      the start of the time step.
    core_profiles_t: Core plasma profiles at the start of the time step.
    core_profiles_t_plus_dt: Core plasma profiles containing all available
      prescribed quantities at the end of the time step.
    explicit_source_profiles: Pre-calculated sources implemented as explicit
      sources in the PDE.
    models: Models used for the calculations.
    coeffs_callback: Calculates diffusion, convection etc. coefficients given a
      core_profiles.
    evolving_names: The names of variables within the core profiles that should
      evolve.
    initial_guess_mode: Chooses the initial guess for the iterative method,
      either x_old or a linear step (see newton_raphson_solve_block docstring).
    maxiter: Maximum number of LM iterations.
    tol: Residual tolerance (mean absolute residual) for successful exit.
    coarse_tol: Coarser acceptable residual tolerance.
    step_rtol: Relative step-size tolerance for LM termination: the solver
      terminates when the step is smaller than step_atol + step_rtol * |x|.
    step_atol: Absolute step-size tolerance for LM termination.
    pedestal_transition_state: State for tracking pedestal L-H and H-L
      transitions.

  Returns:
    x_new: Tuple, with x_new[i] giving channel i of x at the next time step.
    solver_numeric_outputs: Iteration and error info. For the error, 0
      signifies mean|R| < tol at exit, 2 signifies mean|R| < coarse_tol, and 1
      signifies mean|R| > coarse_tol.
  """
  coeffs_old = coeffs_callback(
      runtime_params_t,
      geo_t,
      core_profiles_t,
      prev_core_profiles=None,
      dt=None,
      x=x_old,
      explicit_source_profiles=explicit_source_profiles,
      explicit_call=True,
      pedestal_transition_state=pedestal_transition_state,
  )

  match initial_guess_mode:
    # LINEAR initial guess will provide the initial guess using the predictor-
    # corrector method if predictor_corrector=True in the solver config
    case enums.InitialGuessMode.LINEAR:
      # returns transport coefficients with additional pereverzev terms
      # if set by runtime_params, needed if stiff transport models
      # (e.g. qlknn) are used.
      coeffs_exp_linear = coeffs_callback(
          runtime_params_t,
          geo_t,
          core_profiles=core_profiles_t,
          prev_core_profiles=None,
          dt=None,
          x=x_old,
          explicit_source_profiles=explicit_source_profiles,
          allow_pereverzev=True,
          explicit_call=True,
          pedestal_transition_state=pedestal_transition_state,
      )

      # See linear_theta_method.py for comments on the predictor_corrector API
      x_new_guess = convertors.core_profiles_to_solver_x_tuple(
          core_profiles_t_plus_dt, evolving_names
      )
      init_x_new = predictor_corrector_method.predictor_corrector_method(
          dt=dt,
          runtime_params_t_plus_dt=runtime_params_t_plus_dt,
          geo_t_plus_dt=geo_t_plus_dt,
          x_old=x_old,
          x_new_guess=x_new_guess,
          core_profiles_t=core_profiles_t,
          core_profiles_t_plus_dt=core_profiles_t_plus_dt,
          coeffs_exp=coeffs_exp_linear,
          coeffs_callback=coeffs_callback,
          explicit_source_profiles=explicit_source_profiles,
          pedestal_transition_state=pedestal_transition_state,
      )
      init_x_new_vec = fvm_conversions.cell_variable_tuple_to_vec(init_x_new)
    case enums.InitialGuessMode.X_OLD:
      init_x_new_vec = fvm_conversions.cell_variable_tuple_to_vec(x_old)
    case _:
      raise ValueError(
          f'Unknown option for first guess in iterations: {initial_guess_mode}'
      )

  # Create a residual() function with only one argument: x_new.
  # The other arguments (dt, x_old, etc.) are fixed.
  residual_fun = functools.partial(
      residual_and_loss.theta_method_block_residual,
      dt=dt,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      core_profiles_t=core_profiles_t,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      models=models,
      explicit_source_profiles=explicit_source_profiles,
      coeffs_old=coeffs_old,
      evolving_names=evolving_names,
      pedestal_transition_state=pedestal_transition_state,
  )

  solver = optx.LevenbergMarquardt(
      rtol=step_rtol,
      atol=step_atol,
  )
  solution = optx.least_squares(
      lambda y, args: residual_fun(y),
      solver,
      init_x_new_vec,
      max_steps=maxiter,
      # Do not raise on non-convergence: the error state below reports it, and
      # the orchestration layer may retry at reduced dt (adaptive_dt).
      throw=False,
  )
  x_new_vec = solution.value

  # Create updated CellVariable instances based on core_profiles_t_plus_dt
  # which has updated boundary conditions and prescribed profiles.
  x_new = fvm_conversions.vec_to_cell_variable_tuple(
      x_new_vec, core_profiles_t_plus_dt, evolving_names
  )

  # Classify convergence identically to the Newton-Raphson solver, based on
  # the mean absolute residual at the LM solution. Note that LM's own
  # termination criterion is step-size based (step_rtol/step_atol); the
  # residual test below is what the orchestration layer acts on.
  final_residual = residual_fun(x_new_vec)
  mean_abs_residual = jnp.mean(jnp.abs(final_residual))
  solver_error_state = jax.lax.cond(
      mean_abs_residual < tol,
      lambda: 0,
      lambda: jax.lax.cond(
          mean_abs_residual < coarse_tol,
          lambda: 2,
          lambda: 1,
      ),
  )

  solver_numeric_outputs = state_module.SolverNumericOutputs(
      inner_solver_iterations=jnp.array(
          solution.stats['num_steps'], jax_utils.get_int_dtype()
      ),
      solver_error_state=jnp.array(
          solver_error_state, jax_utils.get_int_dtype()
      ),
      outer_solver_iterations=jnp.array(1, jax_utils.get_int_dtype()),
      sawtooth_crash=False,
  )

  return x_new, solver_numeric_outputs
