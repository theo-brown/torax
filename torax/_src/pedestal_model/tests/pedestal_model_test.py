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

"""Tests for the PedestalModel base class transport multiplier logic."""

import dataclasses
from unittest import mock

from absl.testing import absltest
import jax
from jax import numpy as jnp
import numpy as np
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.pedestal_model.formation import base as formation_base
from torax._src.pedestal_model.saturation import base as saturation_base

# pylint: disable=invalid-name


def _make_multipliers(value):
  return pedestal_model_output.TransportMultipliers(
      chi_e_multiplier=jnp.array(value),
      chi_i_multiplier=jnp.array(value),
      D_e_multiplier=jnp.array(value),
      v_e_multiplier=jnp.array(value),
  )


@dataclasses.dataclass(frozen=True, eq=False)
class _FixedFormationModel(formation_base.FormationModel):
  value: float = 0.01

  def __call__(self, *args, **kwargs):
    return _make_multipliers(self.value)


@dataclasses.dataclass(frozen=True, eq=False)
class _FixedSaturationModel(saturation_base.SaturationModel):
  value: float = 2.0

  def __call__(self, *args, **kwargs):
    return _make_multipliers(self.value)


@dataclasses.dataclass(frozen=True, eq=False)
class _DummyPedestalModel(pedestal_model_lib.PedestalModel):

  def _call_implementation(self, *args, **kwargs):
    raise NotImplementedError('Not needed for these tests.')


class ComputeTransportMultipliersTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.formation_value = 0.01
    self.saturation_value = 2.0
    self.model = _DummyPedestalModel(
        formation_model=_FixedFormationModel(value=self.formation_value),
        saturation_model=_FixedSaturationModel(value=self.saturation_value),
    )
    self.instantaneous = self.formation_value * self.saturation_value
    self.prev_value = 1.0
    self.transition_state = dataclasses.replace(
        pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode(),
        transport_multipliers_prev=_make_multipliers(self.prev_value),
    )
    self.pedestal_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=jnp.array(0.9),
        T_i_ped=jnp.array(4.0),
        T_e_ped=jnp.array(4.0),
        n_e_ped=jnp.array(0.7e20),
    )
    self.geo = mock.MagicMock()
    self.core_profiles = mock.MagicMock()
    self.source_profiles = mock.MagicMock()

  def _make_runtime_params(self, relaxation_time):
    runtime_params = mock.MagicMock()
    runtime_params.pedestal.transport_multiplier_relaxation_time = jnp.array(
        relaxation_time
    )
    return runtime_params

  def _compute(self, relaxation_time, dt):
    return self.model.compute_transport_multipliers(
        self._make_runtime_params(relaxation_time),
        self.geo,
        self.core_profiles,
        self.source_profiles,
        self.transition_state,
        self.pedestal_output,
        dt,
    )

  def test_instantaneous_when_dt_is_none(self):
    result = self._compute(relaxation_time=1.0, dt=None)
    for leaf in jax.tree.leaves(result):
      np.testing.assert_allclose(leaf, self.instantaneous, rtol=1e-12)

  def test_instantaneous_when_relaxation_time_is_zero(self):
    result = self._compute(relaxation_time=0.0, dt=jnp.array(0.1))
    for leaf in jax.tree.leaves(result):
      np.testing.assert_allclose(leaf, self.instantaneous, rtol=1e-12)

  def test_log_space_relaxation(self):
    tau = 0.2
    dt = 0.1
    result = self._compute(relaxation_time=tau, dt=jnp.array(dt))
    w = -np.expm1(-dt / tau)
    expected = np.exp(
        (1.0 - w) * np.log(self.prev_value) + w * np.log(self.instantaneous)
    )
    for leaf in jax.tree.leaves(result):
      np.testing.assert_allclose(leaf, expected, rtol=1e-12)
    # The relaxed multiplier lies between the previous and instantaneous
    # values.
    self.assertLess(float(result.chi_e_multiplier), self.prev_value)
    self.assertGreater(float(result.chi_e_multiplier), self.instantaneous)

  def test_long_relaxation_time_pins_to_previous_multiplier(self):
    result = self._compute(relaxation_time=1e6, dt=jnp.array(1e-3))
    for leaf in jax.tree.leaves(result):
      np.testing.assert_allclose(leaf, self.prev_value, atol=1e-4)

  def test_relaxation_reduces_sensitivity_to_instantaneous_value(self):
    """The in-step gain is scaled by w = 1 - exp(-dt/tau)."""
    tau = 0.2
    dt = 0.1
    w = -np.expm1(-dt / tau)

    def relaxed_log_multiplier(saturation_value):
      model = _DummyPedestalModel(
          formation_model=_FixedFormationModel(value=self.formation_value),
          saturation_model=_FixedSaturationModel(value=saturation_value),
      )
      result = model.compute_transport_multipliers(
          self._make_runtime_params(tau),
          self.geo,
          self.core_profiles,
          self.source_profiles,
          self.transition_state,
          self.pedestal_output,
          jnp.array(dt),
      )
      return float(jnp.log(result.chi_e_multiplier))

    d_saturation = 1e-3
    gain = (
        relaxed_log_multiplier(self.saturation_value + d_saturation)
        - relaxed_log_multiplier(self.saturation_value)
    ) / d_saturation
    instantaneous_gain = 1.0 / self.saturation_value  # d log(S) / dS
    np.testing.assert_allclose(gain, w * instantaneous_gain, rtol=1e-3)


if __name__ == '__main__':
  absltest.main()
