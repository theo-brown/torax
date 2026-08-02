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

"""Operator splitting of the psi (current diffusion) equation.

The psi equation is the odd one out among the evolving channels:

*   It is linear in psi given the plasma conductivity. Its diffusion
    coefficient ``geo.g2g3_over_rhon_face`` is purely geometric, it has no
    convection term, it has no implicit source matrix, and its transient
    coefficient depends on ``sigma(T_e, n_e, Z_eff)`` but not on psi. The only
    residual psi dependence is inside the *source* vector, through the
    bootstrap current (which sees psi via q) and through the moving-boundary
    ``Phi_b_dot * psi.grad()`` term. Both are lagged in a single implicit solve.
*   It evolves on the resistive timescale, orders of magnitude slower than the
    heat and particle transport timescales, so the error made by advancing it
    separately over one kinetic time step is small.

Together these mean psi does not need to sit inside the monolithic Newton
block, where it costs a full extra channel of dense Jacobian tangents. This
module implements the psi half of a Lie or Strang splitting: a single
theta-method block-tridiagonal solve for psi alone, with no root finding.
"""

import dataclasses
from typing import Final

import jax
from torax._src import array_typing
from torax._src import models as models_lib
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import convertors
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import implicit_solve_block
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.sources import source_profiles as source_profiles_lib

# The psi sub-solve always evolves exactly this one channel.
PSI_NAME: Final[str] = 'psi'
PSI_NAMES: Final[tuple[str, ...]] = (PSI_NAME,)


@jax.jit(static_argnames=['models'])
def psi_sub_step(
    models: models_lib.Models,
    dt: array_typing.FloatScalar,
    runtime_params_old: runtime_params_lib.RuntimeParams,
    runtime_params_new: runtime_params_lib.RuntimeParams,
    geo_old: geometry.Geometry,
    geo_new: geometry.Geometry,
    core_profiles_old: state.CoreProfiles,
    core_profiles_new: state.CoreProfiles,
    explicit_source_profiles: source_profiles_lib.SourceProfiles,
    pedestal_transition_state: (
        pedestal_transition_state_lib.PedestalTransitionState
    ),
) -> cell_variable.CellVariable:
  """Advances psi by `dt` with a single linear implicit solve.

  Args:
    models: Physics models.
    dt: Duration of the psi sub-step. This is the full step for Lie splitting
      and half the step for Strang splitting.
    runtime_params_old: Runtime parameters at the start of the sub-step.
    runtime_params_new: Runtime parameters at the end of the sub-step, used for
      the implicit coefficients.
    geo_old: Geometry at the start of the sub-step.
    geo_new: Geometry at the end of the sub-step.
    core_profiles_old: Core profiles at the start of the sub-step. Only
      `psi` is read as the initial condition; the other profiles set the
      explicit (theta < 1) coefficients.
    core_profiles_new: Core profiles carrying the kinetic profiles at which the
      implicit coefficients (conductivity, bootstrap current) are evaluated,
      and the psi boundary conditions targeted at the end of the sub-step. Its
      `psi` values are the initial guess at which the psi-dependent source
      terms are lagged.
    explicit_source_profiles: Precomputed explicit source profiles.
    pedestal_transition_state: State for tracking pedestal transitions.

  Returns:
    The psi CellVariable at the end of the sub-step.
  """
  coeffs_callback = calc_coeffs.CoeffsCallback(
      models=models,
      evolving_names=PSI_NAMES,
  )
  x_old = convertors.core_profiles_to_solver_x_tuple(
      core_profiles_old, PSI_NAMES
  )
  x_new_guess = convertors.core_profiles_to_solver_x_tuple(
      core_profiles_new, PSI_NAMES
  )
  solver_params = runtime_params_new.solver

  # Pereverzev-Corrigan terms only stabilise the stiff turbulent heat and
  # particle fluxes; the psi diffusion coefficient is purely geometric, so they
  # are never needed here.
  coeffs_old = coeffs_callback(
      runtime_params_old,
      geo_old,
      core_profiles_old,
      prev_core_profiles=None,
      dt=None,
      x=x_old,
      explicit_source_profiles=explicit_source_profiles,
      pedestal_transition_state=pedestal_transition_state,
      explicit_call=True,
  )
  coeffs_new = coeffs_callback(
      runtime_params_new,
      geo_new,
      core_profiles_new,
      prev_core_profiles=core_profiles_old,
      dt=dt,
      x=x_new_guess,
      explicit_source_profiles=explicit_source_profiles,
      pedestal_transition_state=pedestal_transition_state,
  )

  # A single Thomas solve is exact for this equation: the psi block of the
  # theta-method matrix is independent of psi, so there is nothing for a Newton
  # iteration to converge to beyond the lagged source terms.
  x_new = implicit_solve_block.implicit_solve_block(
      dt=dt,
      x_old=x_old,
      x_new_guess=x_new_guess,
      coeffs_old=coeffs_old,
      coeffs_new=coeffs_new,
      theta_implicit=solver_params.theta_implicit,
      convection_dirichlet_mode=solver_params.convection_dirichlet_mode,
      convection_neumann_mode=solver_params.convection_neumann_mode,
      implicit_solver_type=solver_params.implicit_solver_type,
  )
  return x_new[0]


def interpolate_boundary_conditions(
    cv_old: cell_variable.CellVariable,
    cv_new: cell_variable.CellVariable,
    fraction: float,
) -> cell_variable.CellVariable:
  """Returns `cv_old` with its boundary constraints moved towards `cv_new`.

  Strang splitting needs psi boundary conditions at the midpoint of the step,
  but the step function only evaluates runtime parameters at t and t + dt.
  Linear interpolation of the constraints is exact whenever the driving
  parameters are piecewise linear in time (as `Ip`, which sets the psi edge
  gradient, typically is) and second-order accurate otherwise, so it does not
  degrade the order of the splitting.

  Args:
    cv_old: CellVariable whose values are kept.
    cv_new: CellVariable supplying the target boundary constraints.
    fraction: Interpolation weight in [0, 1]; 0 returns `cv_old` constraints.

  Returns:
    A CellVariable with `cv_old` values and interpolated constraints.
  """

  def _lerp(old, new):
    if old is None or new is None:
      return old
    return old + fraction * (new - old)

  return dataclasses.replace(
      cv_old,
      left_face_constraint=_lerp(
          cv_old.left_face_constraint, cv_new.left_face_constraint
      ),
      left_face_grad_constraint=_lerp(
          cv_old.left_face_grad_constraint, cv_new.left_face_grad_constraint
      ),
      right_face_constraint=_lerp(
          cv_old.right_face_constraint, cv_new.right_face_constraint
      ),
      right_face_grad_constraint=_lerp(
          cv_old.right_face_grad_constraint, cv_new.right_face_grad_constraint
      ),
  )
