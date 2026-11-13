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

"""Bohm-gyroBohm transport with a PID-controlled multiplier tracking H98.

All Bohm-gyroBohm coefficients are scaled by a single multiplier `m`. Before
each step, `m` is updated by a PID controller acting on the H98 confinement
factor of the previous step (one-step lag), in log space since `m` is
multiplicative:

  e = H98 - target_H98                (H98 too high -> more transport)
  I <- clip(I + Ki * e * dt, min_integral, max_integral)
  m = clip(m_ff * exp(Kp * e + I + Kd * de/dt), m_min, m_max)

`m` is constant within a step, so the controller adds no coupling to the
nonlinear solve. Since `D_e` and `V_e` are proportional to `chi` in the BgB
model, scaling the four chi multipliers scales all four channels by `m`.
"""

import dataclasses
from typing import Any

import jax
from jax import numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import jax_utils
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import core_profiles as core_profiles_lib
from torax._src.geometry import geometry
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.physics import formulas as physics_formulas
from torax._src.transport_model import bohm_gyrobohm
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import transport_state as transport_state_lib
from typing_extensions import override
import xarray as xr

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(bohm_gyrobohm.RuntimeParams):
  """Runtime params for the H98-feedback BgB transport model.

  Attributes:
    target_H98: Reference H98 [dimensionless].
    multiplier_feedforward: Multiplier `m_ff` at zero controller output.
    proportional_gain: `Kp` [per unit H98 error].
    integral_gain: `Ki` [per unit H98 error per second].
    derivative_gain: `Kd` [s per unit H98 error].
    derivative_smoothing_time_scale: Exponential smoothing time [s] of de/dt.
    min_integral: Lower anti-windup bound on the integral term `I`.
    max_integral: Upper anti-windup bound on the integral term `I`.
    multiplier_min: Lower bound on `m`.
    multiplier_max: Upper bound on `m`.
  """

  target_H98: array_typing.FloatScalar
  multiplier_feedforward: array_typing.FloatScalar
  proportional_gain: array_typing.FloatScalar
  integral_gain: array_typing.FloatScalar
  derivative_gain: array_typing.FloatScalar
  derivative_smoothing_time_scale: array_typing.FloatScalar
  min_integral: array_typing.FloatScalar
  max_integral: array_typing.FloatScalar
  multiplier_min: array_typing.FloatScalar
  multiplier_max: array_typing.FloatScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class H98FeedbackState(transport_state_lib.ComponentTransportState):
  """Cross-step PID controller state.

  Attributes:
    multiplier: Transport multiplier `m` applied during the step.
    H98_input: Lagged H98 used for the latest controller update.
    integral_term: Integral term `I` [dimensionless].
    previous_error: H98 error at time `t`.
    smoothed_derivative: Smoothed error derivative [1/s] at time `t`.
    t: Time [s] of the latest controller update.
    initialized: Whether the controller has received a valid H98.
  """

  multiplier: array_typing.FloatScalar
  H98_input: array_typing.FloatScalar
  integral_term: array_typing.FloatScalar
  previous_error: array_typing.FloatScalar
  smoothed_derivative: array_typing.FloatScalar
  t: array_typing.FloatScalar
  initialized: array_typing.BoolScalar

  @override
  def to_xr_datatree(
      self,
      context: output_grid_context.OutputGridContext,
  ) -> xr.DataTree | None:
    pack = lambda key, x: context.pack(key, np.atleast_1d(np.asarray(x)))
    data_vars = {
        output_keys.TRANSPORT_MULTIPLIER: pack(
            output_keys.TRANSPORT_MULTIPLIER, self.multiplier
        ),
        output_keys.H98_FEEDBACK_INPUT: pack(
            output_keys.H98_FEEDBACK_INPUT, self.H98_input
        ),
        output_keys.FEEDBACK_INTEGRAL_TERM: pack(
            output_keys.FEEDBACK_INTEGRAL_TERM, self.integral_term
        ),
        output_keys.FEEDBACK_PREVIOUS_ERROR: pack(
            output_keys.FEEDBACK_PREVIOUS_ERROR, self.previous_error
        ),
        output_keys.FEEDBACK_SMOOTHED_ERROR_DERIVATIVE: pack(
            output_keys.FEEDBACK_SMOOTHED_ERROR_DERIVATIVE,
            self.smoothed_derivative,
        ),
        output_keys.FEEDBACK_TIME: pack(output_keys.FEEDBACK_TIME, self.t),
    }
    return xr.DataTree(dataset=context.build_dataset(data_vars))


def _initial_state(t: array_typing.FloatScalar) -> H98FeedbackState:
  dtype = jax_utils.get_dtype()
  zero = jnp.array(0.0, dtype=dtype)
  return H98FeedbackState(
      multiplier=jnp.array(1.0, dtype=dtype),
      H98_input=zero,
      integral_term=zero,
      previous_error=zero,
      smoothed_derivative=zero,
      t=jnp.asarray(t, dtype=dtype),
      initialized=jnp.array(False),
  )


def update_controller(
    state: H98FeedbackState,
    H98: array_typing.FloatScalar,
    t: array_typing.FloatScalar,
    params: RuntimeParams,
) -> H98FeedbackState:
  """Advances the PID controller with the H98 measured at time `t`.

  The output is held if H98 is not finite and positive (only the controller
  clock advances), and the whole state is held if `t` is not later than the
  previous update. The first valid H98 initializes
  the controller without a derivative spike or integral update.

  Args:
    state: Controller state from the previous update.
    H98: H98 measured at time `t` (previous step's post-processed output).
    t: Time [s] of the H98 measurement.
    params: Runtime params of this transport model.

  Returns:
    The updated controller state.
  """
  dtype = jax_utils.get_dtype()
  H98 = jnp.asarray(H98, dtype=dtype)
  t = jnp.asarray(t, dtype=dtype)
  valid = jnp.isfinite(H98) & (H98 > 0.0)
  error = H98 - params.target_H98
  dt = t - state.t

  def _with_output(integral, smoothed_derivative):
    u = (
        params.proportional_gain * error
        + integral
        + params.derivative_gain * smoothed_derivative
    )
    multiplier = jnp.clip(
        params.multiplier_feedforward * jnp.exp(u),
        params.multiplier_min,
        params.multiplier_max,
    )
    return H98FeedbackState(
        multiplier=multiplier,
        H98_input=H98,
        integral_term=integral,
        previous_error=error,
        smoothed_derivative=smoothed_derivative,
        t=t,
        initialized=jnp.array(True),
    )

  def _initialize():
    integral = jnp.clip(
        state.integral_term, params.min_integral, params.max_integral
    )
    return _with_output(integral, jnp.array(0.0, dtype=dtype))

  def _advance():
    safe_dt = jnp.where(dt > 0.0, dt, 1.0)
    _, smoothed_derivative = (
        physics_formulas.calculate_time_derivative_with_exponential_smoothing(
            current_value=error,
            previous_value=state.previous_error,
            previous_smoothed_derivative=state.smoothed_derivative,
            dt=safe_dt,
            smoothing_time_scale=params.derivative_smoothing_time_scale,
        )
    )
    integral = jnp.clip(
        state.integral_term + params.integral_gain * error * dt,
        params.min_integral,
        params.max_integral,
    )
    return _with_output(integral, smoothed_derivative)

  def _update():
    return jax.lax.cond(
        state.initialized,
        lambda: jax.lax.cond(dt > 0.0, _advance, lambda: state),
        _initialize,
    )

  def _skip():
    # Hold the output, but advance the clock so the next valid sample is not
    # integrated over the whole invalid period (e.g. P_loss <= 0 transients).
    return dataclasses.replace(state, t=jnp.maximum(state.t, t))

  return jax.lax.cond(valid, _update, _skip)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class BohmGyroBohmH98FeedbackTransportModel(
    bohm_gyrobohm.BohmGyroBohmTransportModel
):
  """Bohm-gyroBohm transport scaled by a PID multiplier tracking H98."""

  @override
  def initial_state(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
  ) -> H98FeedbackState:
    return _initial_state(runtime_params.t)

  @override
  def initial_state_from_datatree(
      self,
      datatree: xr.DataTree | None,
      runtime_params: runtime_params_lib.RuntimeParams,
  ) -> H98FeedbackState:
    """Restores the controller state on restart, or starts fresh."""
    if datatree is None or datatree.dataset is None:
      return _initial_state(runtime_params.t)
    ds = datatree.dataset
    if output_keys.FEEDBACK_INTEGRAL_TERM not in ds.data_vars:
      return _initial_state(runtime_params.t)
    dtype = jax_utils.get_dtype()
    get = lambda key: jnp.asarray(np.squeeze(ds[key].values), dtype=dtype)
    return H98FeedbackState(
        multiplier=get(output_keys.TRANSPORT_MULTIPLIER),
        H98_input=get(output_keys.H98_FEEDBACK_INPUT),
        integral_term=get(output_keys.FEEDBACK_INTEGRAL_TERM),
        previous_error=get(output_keys.FEEDBACK_PREVIOUS_ERROR),
        smoothed_derivative=get(
            output_keys.FEEDBACK_SMOOTHED_ERROR_DERIVATIVE
        ),
        t=get(output_keys.FEEDBACK_TIME),
        initialized=jnp.array(True),
    )

  @override
  def ingest_post_processed_outputs(
      self,
      component_state: transport_state_lib.ComponentTransportState,
      post_processed_outputs: Any,
      transport_runtime_params: (
          transport_runtime_params_lib.ComponentRuntimeParams
      ),
      runtime_params: runtime_params_lib.RuntimeParams,
  ) -> H98FeedbackState:
    assert isinstance(component_state, H98FeedbackState)
    assert isinstance(transport_runtime_params, RuntimeParams)
    return update_controller(
        component_state,
        post_processed_outputs.H98,
        runtime_params.t,
        transport_runtime_params,
    )

  @override
  def call_implementation(
      self,
      transport_runtime_params: (
          transport_runtime_params_lib.ComponentRuntimeParams
      ),
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: core_profiles_lib.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
      component_state: transport_state_lib.ComponentTransportState,
  ) -> tuple[
      bohm_gyrobohm.BohmGyroBohmTransportOutput,
      transport_state_lib.ComponentTransportState,
  ]:
    """BgB coefficients scaled by the controller multiplier of this step."""
    assert isinstance(transport_runtime_params, RuntimeParams)
    assert isinstance(component_state, H98FeedbackState)
    p = transport_runtime_params
    m = jnp.where(
        component_state.initialized,
        component_state.multiplier,
        p.multiplier_feedforward,
    )
    scaled_params = dataclasses.replace(
        p,
        chi_e_bohm_multiplier=p.chi_e_bohm_multiplier * m,
        chi_e_gyrobohm_multiplier=p.chi_e_gyrobohm_multiplier * m,
        chi_i_bohm_multiplier=p.chi_i_bohm_multiplier * m,
        chi_i_gyrobohm_multiplier=p.chi_i_gyrobohm_multiplier * m,
    )
    return super().call_implementation(
        scaled_params,
        runtime_params,
        geo,
        core_profiles,
        two_point_mask=two_point_mask,
        component_state=component_state,
    )
