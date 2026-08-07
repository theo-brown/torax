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

"""Tests for min/max clipping of transport coefficients.

The transport coefficients are bounded for PDE stability. These tests pin down
what those bounds do to the *derivatives* of the transport model output, which
matters for gradient-based workflows (optimisation, sensitivity analysis):

  * With `clip_mode='hard'` (the default, `jnp.clip`), a saturated coefficient
    is exactly constant, so its derivative with respect to any upstream
    quantity is exactly zero and the gradient signal is destroyed.
  * With `clip_mode='soft'`, the bound is imposed smoothly, so saturated
    coefficients keep a small but non-zero derivative while the bound itself is
    still respected.
"""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.orchestration import jit_run_loop
from torax._src.orchestration import run_simulation
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.sources import source_profile_builders
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import enums
from torax._src.transport_model import transport_model as transport_model_lib

# pylint: disable=invalid-name

jax.config.update('jax_enable_x64', True)

_CHI_MIN = 0.05
_CHI_MAX = 10.0
_D_E_MIN = 0.05
_D_E_MAX = 10.0
_V_E_MIN = -5.0
_V_E_MAX = 5.0


def _build_model_inputs(**transport_overrides):
  """Builds a combined+constant transport model and everything it needs."""
  config = default_configs.get_default_config_dict()
  config['geometry'] = {'geometry_type': 'circular', 'n_rho': 10}
  config['transport'] = {
      'model_name': 'combined',
      'chi_min': _CHI_MIN,
      'chi_max': _CHI_MAX,
      'D_e_min': _D_E_MIN,
      'D_e_max': _D_E_MAX,
      'V_e_min': _V_E_MIN,
      'V_e_max': _V_E_MAX,
      # Smoothing mixes neighbouring grid points, which would blur the
      # per-point saturation behaviour we want to isolate here.
      'smoothing_width': 0.0,
      'transport_models': [{'model_name': 'constant'}],
      **transport_overrides,
  }
  torax_config = model_config.ToraxConfig.from_dict(config)

  runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
      torax_config
  )(t=0.0)
  geo = torax_config.geometry.build_provider(t=0.0)
  source_models = torax_config.sources.build_models()
  neoclassical_models = torax_config.neoclassical.build_models()
  core_profiles = initialization.initial_core_profiles(
      runtime_params, geo, source_models, neoclassical_models
  )
  source_profiles = source_profile_builders.build_source_profiles(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      neoclassical_models=neoclassical_models,
      explicit=True,
  )
  pedestal_model = torax_config.pedestal.build_pedestal_model()
  pedestal_output = pedestal_model(
      runtime_params,
      geo,
      core_profiles,
      source_profiles,
      pedestal_transition_state=pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode(),
  )
  return (
      torax_config.transport.build_transport_model(),
      runtime_params,
      geo,
      core_profiles,
      pedestal_output,
  )


def _make_channel_fn(channel: str, **transport_overrides):
  """Returns f(x) -> transport output for `channel`, with input set to x.

  The returned function drives the underlying constant transport model with a
  uniform value `x` in the requested channel and runs the full transport model
  `__call__`, so the returned profile has been through clipping.

  Args:
    channel: One of 'chi_i', 'chi_e', 'D_e', 'V_e'.
    **transport_overrides: Overrides for the transport config dict.
  """
  output_names = {
      'chi_i': 'chi_face_ion',
      'chi_e': 'chi_face_el',
      'D_e': 'd_face_el',
      'V_e': 'v_face_el',
  }
  model, runtime_params, geo, core_profiles, pedestal_output = (
      _build_model_inputs(**transport_overrides)
  )
  transport_rp = runtime_params.transport
  constant_rp = transport_rp.transport_model_params[0]

  def f(x):
    new_constant = dataclasses.replace(
        constant_rp, **{channel: jnp.full_like(constant_rp.chi_i, x)}
    )
    new_transport = dataclasses.replace(
        transport_rp, transport_model_params=(new_constant,)
    )
    coeffs = model(
        dataclasses.replace(runtime_params, transport=new_transport),
        geo,
        core_profiles,
        pedestal_output,
    )
    return getattr(coeffs, output_names[channel])

  return f


def _grad_of_mean(f):
  """d/dx of the mean of the profile returned by f."""
  return jax.grad(lambda x: jnp.mean(f(x)))


_CHANNELS = (
    # channel, lower bound, upper bound
    ('chi_i', _CHI_MIN, _CHI_MAX),
    ('chi_e', _CHI_MIN, _CHI_MAX),
    ('D_e', _D_E_MIN, _D_E_MAX),
    ('V_e', _V_E_MIN, _V_E_MAX),
)


class HardClipGradientTest(parameterized.TestCase):
  """Characterises the gradient behaviour of the default hard clip."""

  @parameterized.named_parameters(*[(c[0], *c) for c in _CHANNELS])
  def test_gradient_vanishes_above_upper_bound(self, channel, _, upper):
    """The gradient is exactly zero once the model saturates at its max."""
    f = _make_channel_fn(channel)
    grad = _grad_of_mean(f)

    # Sanity check: below the bound the gradient passes through unattenuated.
    self.assertAlmostEqual(float(grad(0.5 * upper)), 1.0)

    for x in [1.01 * upper, 2 * upper, 10 * upper]:
      with self.subTest(x=x):
        np.testing.assert_allclose(f(x), upper)
        # Exactly zero, not just small: the gradient signal is destroyed.
        self.assertEqual(float(grad(x)), 0.0)

  @parameterized.named_parameters(*[(c[0], *c) for c in _CHANNELS])
  def test_gradient_vanishes_below_lower_bound(self, channel, lower, _):
    """The gradient is exactly zero once the model saturates at its min."""
    f = _make_channel_fn(channel)
    grad = _grad_of_mean(f)

    for x in [lower - abs(lower) - 1.0, lower - 100.0]:
      with self.subTest(x=x):
        np.testing.assert_allclose(f(x), lower)
        self.assertEqual(float(grad(x)), 0.0)


class SoftClipGradientTest(parameterized.TestCase):
  """The soft clip keeps a non-zero gradient in the saturated region."""

  @parameterized.named_parameters(*[(c[0], *c) for c in _CHANNELS])
  def test_gradient_nonzero_above_upper_bound(self, channel, _, upper):
    f = _make_channel_fn(channel, clip_mode='soft')
    grad = _grad_of_mean(f)

    for x in [1.01 * upper, 1.2 * upper]:
      with self.subTest(x=x):
        self.assertGreater(float(grad(x)), 0.0)

  @parameterized.named_parameters(*[(c[0], *c) for c in _CHANNELS])
  def test_gradient_nonzero_below_lower_bound(self, channel, lower, _):
    f = _make_channel_fn(channel, clip_mode='soft')
    grad = _grad_of_mean(f)

    for x in [lower - 0.1 * abs(lower) - 0.01, lower - 0.2 * abs(lower) - 0.02]:
      with self.subTest(x=x):
        self.assertGreater(float(grad(x)), 0.0)

  @parameterized.named_parameters(*[(c[0], *c) for c in _CHANNELS])
  def test_bounds_are_still_respected(self, channel, lower, upper):
    """The soft clip must not let the coefficients escape their bounds."""
    f = _make_channel_fn(channel, clip_mode='soft', clip_softness=0.2)
    for x in [-1e3, -1.0, 0.0, 1.0, 1e3, 1e8]:
      with self.subTest(x=x):
        out = np.asarray(f(x))
        # Allow the sub-permille slack introduced at the lower bound by
        # composing the two smooth bounds (see soft_clip_widths).
        slack = 2e-3 * (upper - lower)
        self.assertGreaterEqual(out.min(), lower - slack)
        self.assertLessEqual(out.max(), upper + 1e-12)

  @parameterized.named_parameters(*[(c[0], *c) for c in _CHANNELS])
  def test_agrees_with_hard_clip_away_from_bounds(self, channel, lower, upper):
    """Well inside the bounds, soft clipping is a no-op to high accuracy."""
    hard = _make_channel_fn(channel)
    soft = _make_channel_fn(channel, clip_mode='soft')
    # Midway between the bounds is many transition widths from either.
    x = 0.5 * (lower + upper)
    np.testing.assert_allclose(soft(x), hard(x), rtol=1e-4)
    np.testing.assert_allclose(
        _grad_of_mean(soft)(x), _grad_of_mean(hard)(x), rtol=1e-4
    )

  def test_larger_softness_gives_larger_saturated_gradient(self):
    """`clip_softness` controls how deep into saturation gradients survive."""
    x = 1.5 * _CHI_MAX
    grads = [
        float(
            _grad_of_mean(
                _make_channel_fn('chi_i', clip_mode='soft', clip_softness=s)
            )(x)
        )
        for s in [0.02, 0.05, 0.1, 0.2]
    ]
    self.assertSequenceEqual(sorted(grads), grads)
    self.assertGreater(grads[-1], grads[0])


class EndToEndSimulationGradientTest(parameterized.TestCase):
  """The same effect, measured through a full (short) TORAX simulation.

  The scalar being differentiated is the volume-averaged ion temperature at the
  end of the run, and the input is the prescribed `chi_i` of a constant
  transport model. `chi_i` reaches the solver *only* through the transport
  clip, so a hard clip severs the only path from input to output and the
  gradient is identically zero.
  """

  CHI_MAX = 2.0

  def _make_loss_fn(self, **transport_overrides):
    config = default_configs.get_default_config_dict()
    config['geometry'] = {'geometry_type': 'circular', 'n_rho': 25}
    config['numerics'] = {'t_final': 0.1, 'fixed_dt': 0.05}
    config['solver'] = {'solver_type': 'linear'}
    config['transport'] = {
        'model_name': 'combined',
        'chi_min': _CHI_MIN,
        'chi_max': self.CHI_MAX,
        'smoothing_width': 0.0,
        'transport_models': [{'model_name': 'constant', 'chi_i': 1.0}],
        **transport_overrides,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    step_fn = run_simulation.make_step_fn(torax_config)
    provider = step_fn.runtime_params_provider
    rho_norm = provider.transport_model.transport_models[0].chi_i.grid.cell_centers

    @jax.jit
    def loss(chi_i_value):
      update = interpolated_param_2d.TimeVaryingArrayUpdate(
          value=jnp.full((1, rho_norm.shape[0]), chi_i_value),
          rho_norm=rho_norm,
      )
      overrides = provider.update_provider(
          lambda p: (p.transport_model.transport_models[0].chi_i,),
          (update,),
      )
      states, _, final_i = jit_run_loop.run_loop_jit(
          step_fn=step_fn,
          max_steps=4,
          runtime_params_overrides=overrides,
      )
      return jnp.mean(states.core_profiles.T_i.value[final_i])

    return loss

  def test_hard_clip_kills_the_simulation_gradient(self):
    grad = jax.grad(self._make_loss_fn())
    # Unsaturated: increasing chi_i cools the plasma, so the gradient is
    # non-zero and negative.
    self.assertLess(float(grad(0.5 * self.CHI_MAX)), 0.0)
    # Saturated: the gradient is exactly zero.
    self.assertEqual(float(grad(1.5 * self.CHI_MAX)), 0.0)
    self.assertEqual(float(grad(4.0 * self.CHI_MAX)), 0.0)

  def test_soft_clip_preserves_the_simulation_gradient(self):
    loss = self._make_loss_fn(clip_mode='soft', clip_softness=0.2)
    grad = jax.grad(loss)
    self.assertLess(float(grad(0.5 * self.CHI_MAX)), 0.0)
    self.assertLess(float(grad(1.5 * self.CHI_MAX)), 0.0)

    # The recovered gradient is a genuine derivative of the simulation, not
    # just a non-zero number: check it against a central difference.
    x = 1.5 * self.CHI_MAX
    eps = 1e-4
    fd = (loss(x + eps) - loss(x - eps)) / (2 * eps)
    np.testing.assert_allclose(grad(x), fd, rtol=1e-4)

  @parameterized.named_parameters(
      ('hard', 'hard', False),
      ('soft', 'soft', True),
  )
  def test_gradient_descent_escapes_saturation(self, clip_mode, expect_escape):
    """The practical consequence: can an optimiser leave the saturated region?

    Recovers `chi_i` by gradient descent on a target volume-averaged `T_i`,
    starting from an initial guess that is saturated at `chi_max`. Under a hard
    clip the gradient is identically zero and the optimiser never takes a
    single step. Under a soft clip it descends out of saturation and converges.
    """
    chi_true = 1.0
    chi_start = 1.5 * self.CHI_MAX
    loss_fn = self._make_loss_fn(clip_mode=clip_mode, clip_softness=0.2)
    target = loss_fn(chi_true)
    objective = jax.jit(
        jax.grad(lambda chi: (loss_fn(chi) - target) ** 2)
    )

    chi = chi_start
    for _ in range(30):
      chi = float(chi - 0.8 * objective(chi))

    if expect_escape:
      self.assertAlmostEqual(chi, chi_true, delta=0.05)
    else:
      self.assertEqual(chi, chi_start)


class ClipModeConfigTest(parameterized.TestCase):

  def test_default_clip_mode_is_hard(self):
    """Default behaviour is unchanged, so reference results are unaffected."""
    _, runtime_params, _, _, _ = _build_model_inputs()
    self.assertEqual(runtime_params.transport.clip_mode, enums.ClipMode.HARD)
    self.assertEqual(
        runtime_params.transport.transport_model_params[0].clip_mode,
        enums.ClipMode.HARD,
    )

  def test_hard_and_soft_agree_when_nothing_saturates(self):
    hard = _make_channel_fn('chi_i')
    soft = _make_channel_fn('chi_i', clip_mode='soft')
    x = 1.0
    np.testing.assert_allclose(soft(x), hard(x), rtol=1e-6)

  @parameterized.parameters(0.0, -0.1, 0.5, 1.0)
  def test_invalid_softness_rejected(self, softness):
    with self.assertRaises(ValueError):
      _build_model_inputs(clip_mode='soft', clip_softness=softness)


class SoftClipFunctionTest(parameterized.TestCase):
  """Unit tests for the `soft_clip` primitive itself."""

  @parameterized.parameters(
      (0.05, 10.0),
      (-5.0, 5.0),
      (0.0, 100.0),  # Degenerate lower bound at exactly zero.
  )
  def test_converges_to_hard_clip_as_softness_goes_to_zero(self, lower, upper):
    x = jnp.linspace(lower - (upper - lower), upper + (upper - lower), 101)
    hard = jnp.clip(x, lower, upper)
    previous_error = np.inf
    for softness in [0.1, 0.01, 1e-3, 1e-4]:
      soft = transport_model_lib.apply_bounds(
          x, lower, upper, enums.ClipMode.SOFT, softness
      )
      error = float(jnp.max(jnp.abs(soft - hard)))
      self.assertLess(error, previous_error)
      previous_error = error
    np.testing.assert_allclose(soft, hard, atol=1e-3 * (upper - lower))

  @parameterized.parameters(0.01, 0.05, 0.2)
  def test_monotonic_non_decreasing(self, softness):
    x = jnp.linspace(-50.0, 150.0, 401)
    y = transport_model_lib.apply_bounds(
        x, 0.05, 10.0, enums.ClipMode.SOFT, softness
    )
    self.assertTrue(bool(jnp.all(jnp.diff(y) >= 0.0)))

  @parameterized.parameters(0.01, 0.05, 0.2)
  def test_strictly_increasing_within_reach_of_the_bounds(self, softness):
    """Strict monotonicity holds out to the float underflow depth.

    Far enough into saturation the sigmoid underflows and the soft clip
    degenerates to the hard clip exactly; see
    `test_gradient_underflows_far_into_saturation`.
    """
    lower, upper = 0.05, 10.0
    lower_width, upper_width = transport_model_lib.soft_clip_widths(
        lower, upper, softness
    )
    x = jnp.linspace(lower - 30 * lower_width, upper + 30 * upper_width, 601)
    y = transport_model_lib.apply_bounds(
        x, lower, upper, enums.ClipMode.SOFT, softness
    )
    self.assertTrue(bool(jnp.all(jnp.diff(y) > 0.0)))

  @parameterized.parameters(0.01, 0.05, 0.2)
  def test_derivative_positive_within_reach_of_the_bounds(self, softness):
    lower, upper = 0.05, 10.0
    lower_width, upper_width = transport_model_lib.soft_clip_widths(
        lower, upper, softness
    )
    dy_dx = jax.vmap(
        jax.grad(
            lambda x: transport_model_lib.apply_bounds(
                x, lower, upper, enums.ClipMode.SOFT, softness
            )
        )
    )(jnp.linspace(lower - 30 * lower_width, upper + 30 * upper_width, 201))
    self.assertTrue(bool(jnp.all(dy_dx > 0.0)))
    # And bounded above by the identity slope.
    self.assertLessEqual(float(jnp.max(dy_dx)), 1.0 + 1e-12)

  def test_gradient_underflows_far_into_saturation(self):
    """The soft clip is not a cure for arbitrarily deep saturation.

    The residual gradient decays like `exp(-depth / width)`, so it underflows
    to exactly zero once the saturation depth exceeds ~745 transition widths
    in float64 (~88 in float32). Beyond that the soft clip behaves exactly
    like the hard clip. This bounds how far a gradient-based optimiser can be
    expected to see.
    """
    lower, upper, softness = 0.05, 10.0, 0.05
    _, upper_width = transport_model_lib.soft_clip_widths(
        lower, upper, softness
    )
    grad = jax.grad(
        lambda x: transport_model_lib.apply_bounds(
            x, lower, upper, enums.ClipMode.SOFT, softness
        )
    )
    self.assertGreater(float(grad(upper + 700 * upper_width)), 0.0)
    self.assertEqual(float(grad(upper + 800 * upper_width)), 0.0)

  def test_zero_width_bound_degenerates_to_hard_bound(self):
    """A bound at exactly zero must not produce NaNs."""
    lower, upper = 0.0, 100.0
    x = jnp.linspace(-10.0, 110.0, 121)
    y = transport_model_lib.apply_bounds(
        x, lower, upper, enums.ClipMode.SOFT, 0.05
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(y))))
    # The smooth upper bound undershoots `lower` by
    # upper_width * exp(-(upper - lower) / upper_width), which is ~1e-8 here.
    self.assertGreaterEqual(float(jnp.min(y)), lower - 1e-7 * (upper - lower))
    dy_dx = jax.vmap(
        jax.grad(
            lambda x: transport_model_lib.apply_bounds(
                x, lower, upper, enums.ClipMode.SOFT, 0.05
            )
        )
    )(x)
    self.assertTrue(bool(jnp.all(jnp.isfinite(dy_dx))))

  def test_gradient_matches_finite_differences(self):
    lower, upper, softness = 0.05, 10.0, 0.05

    def g(x):
      return transport_model_lib.apply_bounds(
          x, lower, upper, enums.ClipMode.SOFT, softness
      )

    eps = 1e-6
    for x in [-1.0, 0.5, 5.0, 9.9, 10.0, 10.5, 12.0]:
      with self.subTest(x=x):
        fd = (g(x + eps) - g(x - eps)) / (2 * eps)
        np.testing.assert_allclose(jax.grad(g)(x), fd, rtol=1e-5, atol=1e-9)

  def test_saturated_gradient_decays_like_a_sigmoid(self):
    """Documents the residual gradient available at a given saturation depth.

    The soft clip does not give a *constant* gradient in the saturated region:
    it decays like `sigmoid(-(x - upper) / width)`. This test pins that down so
    that the practical reach of the soft clip is explicit.
    """
    lower, upper, softness = 0.05, 10.0, 0.05
    lower_width, upper_width = transport_model_lib.soft_clip_widths(
        lower, upper, softness
    )
    del lower_width

    def g(x):
      return transport_model_lib.apply_bounds(
          x, lower, upper, enums.ClipMode.SOFT, softness
      )

    for n_widths in [1.0, 3.0, 5.0, 10.0]:
      with self.subTest(n_widths=n_widths):
        x = upper + n_widths * upper_width
        expected = jax.nn.sigmoid(-n_widths)
        np.testing.assert_allclose(jax.grad(g)(x), expected, rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
