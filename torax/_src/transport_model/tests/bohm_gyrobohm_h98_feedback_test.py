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

import copy
import types

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from torax._src.config import config_loader
from torax._src.orchestration import run_simulation
from torax._src.output_tools import output_grid_context
from torax._src.torax_pydantic import config as torax_config_lib
from torax._src.transport_model import bohm_gyrobohm_h98_feedback as h98fb
from torax._src.transport_model import config as transport_config
from torax._src.transport_model import transport_state as transport_state_lib

# pylint: disable=invalid-name


def _params(**kwargs) -> h98fb.RuntimeParams:
  return transport_config.BohmGyroBohmH98FeedbackTransportModelConfig(
      **kwargs
  ).build_runtime_params(t=0.0)


def _state(**kwargs) -> h98fb.H98FeedbackState:
  state = h98fb.BohmGyroBohmH98FeedbackTransportModel().initial_state(
      types.SimpleNamespace(t=0.0)
  )
  return h98fb.H98FeedbackState(**{**vars(state), **kwargs})


def _base_config() -> dict[str, object]:
  return copy.deepcopy(
      config_loader.import_module('tests/test_data/test_bohmgyrobohm_all.py')[
          'CONFIG'
      ]
  )


def _run(transport_model: dict[str, object], t_final: float):
  config = _base_config()
  config['transport']['core_transport_models'] = {'bgb': transport_model}
  config['time_step_calculator'].update(t_final=t_final, max_steps=1000)
  output = run_simulation.run_simulation(
      torax_config_lib.ToraxConfig.from_dict(config), progress_bar=False
  )
  return output.simulation_output_to_xr()


class ControllerTest(absltest.TestCase):

  def test_first_valid_sample_initializes_with_proportional_term(self):
    params = _params(target_H98=1.0, proportional_gain=2.0, integral_gain=5.0)
    state = h98fb.update_controller(_state(), 0.8, 0.0, params)
    self.assertTrue(state.initialized)
    self.assertEqual(float(state.integral_term), 0.0)
    np.testing.assert_allclose(state.multiplier, np.exp(2.0 * -0.2))

  def test_integral_accumulates_and_is_clamped(self):
    params = _params(
        target_H98=1.0, proportional_gain=0.0, integral_gain=1.0,
        max_integral=0.25,
    )
    state = h98fb.update_controller(_state(), 1.2, 0.0, params)
    state = h98fb.update_controller(state, 1.2, 0.5, params)
    np.testing.assert_allclose(state.integral_term, 0.1)
    state = h98fb.update_controller(state, 1.2, 5.0, params)
    np.testing.assert_allclose(state.integral_term, 0.25)
    np.testing.assert_allclose(state.multiplier, np.exp(0.25))

  def test_hold_when_time_does_not_advance(self):
    params = _params(proportional_gain=1.0, integral_gain=1.0)
    state = h98fb.update_controller(_state(), 1.3, 1.0, params)
    held = h98fb.update_controller(state, 0.7, 1.0, params)
    for field in ('multiplier', 'integral_term', 'previous_error', 't'):
      np.testing.assert_array_equal(
          getattr(held, field), getattr(state, field)
      )

  def test_invalid_H98_holds_output_and_advances_clock(self):
    params = _params(proportional_gain=1.0, integral_gain=1.0)
    state = h98fb.update_controller(_state(), 1.3, 1.0, params)
    for bad in (0.0, -1.0, np.nan, np.inf):
      held = h98fb.update_controller(state, bad, 2.0, params)
      np.testing.assert_array_equal(held.multiplier, state.multiplier)
      np.testing.assert_array_equal(held.integral_term, state.integral_term)
      np.testing.assert_allclose(held.t, 2.0)
    # Uninitialized controller stays uninitialized on invalid input.
    self.assertFalse(
        h98fb.update_controller(_state(), 0.0, 1.0, params).initialized
    )

  def test_multiplier_bounds_and_direction(self):
    params = _params(
        target_H98=1.0, proportional_gain=10.0, multiplier_min=0.5,
        multiplier_max=3.0,
    )
    high = h98fb.update_controller(_state(), 2.0, 0.0, params)
    low = h98fb.update_controller(_state(), 0.1, 0.0, params)
    # H98 above target -> more transport, clipped at the bounds.
    np.testing.assert_allclose(high.multiplier, 3.0)
    np.testing.assert_allclose(low.multiplier, 0.5)

  def test_config_rejects_inverted_bounds(self):
    with self.assertRaises(ValueError):
      transport_config.BohmGyroBohmH98FeedbackTransportModelConfig(
          multiplier_min=2.0, multiplier_max=1.0
      )
    with self.assertRaises(ValueError):
      transport_config.BohmGyroBohmH98FeedbackTransportModelConfig(
          min_integral=1.0, max_integral=0.0
      )

  def test_state_datatree_round_trip(self):
    params = _params(proportional_gain=1.0, integral_gain=1.0)
    state = h98fb.update_controller(_state(), 1.3, 1.0, params)
    state = h98fb.update_controller(state, 1.1, 2.0, params)
    context = output_grid_context.OutputGridContext(
        times=np.array([2.0]),
        rho_face_norm=np.linspace(0, 1, 3),
        rho_cell_norm=np.array([0.25, 0.75]),
        rho_cell_plus_boundaries_norm=np.array([0.0, 0.25, 0.75, 1.0]),
    )
    tree = state.to_xr_datatree(context)
    model = h98fb.BohmGyroBohmH98FeedbackTransportModel()
    restored = model.initial_state_from_datatree(
        tree, types.SimpleNamespace(t=2.0)
    )
    for field in (
        'multiplier', 'H98_input', 'integral_term', 'previous_error',
        'smoothed_derivative', 't',
    ):
      np.testing.assert_allclose(
          getattr(restored, field), getattr(state, field)
      )
    self.assertTrue(restored.initialized)


class HookTest(absltest.TestCase):

  def test_stateless_models_pass_through(self):
    config = _base_config()
    config['transport']['core_transport_models'] = {
        'bgb': {'model_name': 'bohm_gyrobohm'},
        'fb': {'model_name': 'bohm_gyrobohm_h98_feedback'},
    }
    torax_config = torax_config_lib.ToraxConfig.from_dict(config)
    transport_model = torax_config.transport.build_transport_model()
    runtime_params = types.SimpleNamespace(
        t=jnp.array(0.0),
        transport=torax_config.transport.build_runtime_params(t=0.0),
    )
    state = transport_model.initial_state(runtime_params)
    new = transport_model.update_state_from_post_processed_outputs(
        state, types.SimpleNamespace(H98=jnp.array(1.5)), runtime_params
    )
    self.assertIs(
        new.core_transport_states['bgb'], state.core_transport_states['bgb']
    )
    self.assertIsInstance(
        new.core_transport_states['bgb'],
        transport_state_lib.ComponentTransportState,
    )
    self.assertTrue(new.core_transport_states['fb'].initialized)


class SimulationTest(absltest.TestCase):

  def test_zero_gains_match_bohm_gyrobohm(self):
    plain = _run({'model_name': 'bohm_gyrobohm'}, t_final=1.0)
    feedback = _run(
        {
            'model_name': 'bohm_gyrobohm_h98_feedback',
            'proportional_gain': 0.0,
            'integral_gain': 0.0,
        },
        t_final=1.0,
    )
    for name in ('T_e', 'T_i', 'n_e', 'psi'):
      np.testing.assert_array_equal(
          feedback.profiles[name].values, plain.profiles[name].values
      )

  def test_feedback_moves_H98_toward_target(self):
    target = 1.2
    plain = _run({'model_name': 'bohm_gyrobohm'}, t_final=4.0)
    feedback = _run(
        {
            'model_name': 'bohm_gyrobohm_h98_feedback',
            'target_H98': target,
            'proportional_gain': 1.0,
            'integral_gain': 1.0,
        },
        t_final=4.0,
    )
    multiplier = feedback['/model_state/transport/core/bgb'].dataset[
        'transport_multiplier'
    ].values
    # Plain BgB sits below target here, so transport must be reduced.
    H98_plain = plain.scalars.H98.values[-1]
    H98_feedback = feedback.scalars.H98.values[-1]
    self.assertLess(H98_plain, target)
    self.assertLess(multiplier[-1], 1.0)
    self.assertLess(
        abs(H98_feedback - target), 0.6 * abs(H98_plain - target)
    )


if __name__ == '__main__':
  absltest.main()
