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
"""Tests for the TR-BDF2 coefficients and embedded error estimate.

These tests pin down the method's algebra independently of the TORAX PDE: if
the Butcher tableau or the embedded weights are wrong, the full simulation
would still run and merely lose an order of accuracy, which is exactly the sort
of failure that is easy to miss.
"""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.fvm import tr_bdf2


# The TR-BDF2 tableau, written out here from the method definition rather than
# imported, so the test is an independent statement of what the constants mean.
#   c = [0, gamma, 1]
#   row 2 = [gamma/2, gamma/2, 0]      (trapezoidal rule to t + gamma*dt)
#   row 3 = [w, w, d]                  (BDF2 to t + dt)
_GAMMA = 2.0 - np.sqrt(2.0)
_D = _GAMMA / 2.0
_W = np.sqrt(2.0) / 4.0
_C = np.array([0.0, _GAMMA, 1.0])
_A = np.array([
    [0.0, 0.0, 0.0],
    [_GAMMA / 2.0, _GAMMA / 2.0, 0.0],
    [_W, _W, _D],
])
_B = np.array([_W, _W, _D])
_B_HAT = np.array([(1.0 - _W) / 3.0, (3.0 * _W + 1.0) / 3.0, _D / 3.0])


class TrBdf2CoefficientsTest(parameterized.TestCase):

  def test_gamma_gives_both_stages_the_same_implicit_coefficient(self):
    # This is the defining property of gamma = 2 - sqrt(2): stage 1 contributes
    # theta * gamma = gamma / 2 and stage 2 contributes B_IMPLICIT.
    np.testing.assert_allclose(
        tr_bdf2.TRAPEZOIDAL_THETA * tr_bdf2.GAMMA, _GAMMA / 2.0
    )
    np.testing.assert_allclose(tr_bdf2.B_IMPLICIT, _GAMMA / 2.0)

  def test_bdf2_stage_weights_are_consistent(self):
    # The two explicit weights must sum to one, otherwise a constant solution
    # would not be preserved.
    np.testing.assert_allclose(tr_bdf2.A_STAGE1 + tr_bdf2.A_START, 1.0)

  def test_bdf2_stage_weights_match_the_butcher_tableau(self):
    # Eliminating dt*(f_n + f_g) between the two stage equations turns
    # row 3 of the tableau into the three-point BDF2 form used by the residual.
    np.testing.assert_allclose(tr_bdf2.A_STAGE1, 2.0 * _W / _GAMMA)
    np.testing.assert_allclose(tr_bdf2.A_START, 1.0 - 2.0 * _W / _GAMMA)

  @parameterized.named_parameters(
      ('second_order', _B, 2),
      ('embedded_third_order', _B_HAT, 3),
  )
  def test_order_conditions(self, b, order):
    """Checks the Runge-Kutta order conditions up to `order`."""
    np.testing.assert_allclose(np.sum(b), 1.0, atol=1e-14)
    if order >= 2:
      np.testing.assert_allclose(b @ _C, 1.0 / 2.0, atol=1e-14)
    if order >= 3:
      np.testing.assert_allclose(b @ (_C**2), 1.0 / 3.0, atol=1e-14)
      np.testing.assert_allclose(b @ (_A @ _C), 1.0 / 6.0, atol=1e-14)

  def test_second_order_method_is_not_third_order(self):
    # Otherwise the embedded estimate would be identically zero.
    self.assertNotAlmostEqual(_B @ (_C**2), 1.0 / 3.0)

  def test_embedded_weights_are_b_hat_minus_b(self):
    embedded = np.array([
        tr_bdf2.EMBEDDED_WEIGHT_START,
        tr_bdf2.EMBEDDED_WEIGHT_STAGE1,
        tr_bdf2.EMBEDDED_WEIGHT_NEW,
    ])
    np.testing.assert_allclose(embedded, _B_HAT - _B, atol=1e-14)
    # The difference of two consistent methods annihilates constants.
    np.testing.assert_allclose(np.sum(embedded), 0.0, atol=1e-14)

  def test_method_is_l_stable(self):
    """The stability function must vanish as the stiff limit z -> -infinity."""
    # R(z) = 1 + z b^T (I - z A)^-1 e
    def stability_function(z):
      e = np.ones(3)
      return 1.0 + z * _B @ np.linalg.solve(np.eye(3) - z * _A, e)

    # A-stability on the negative real axis, and L-stability in the limit:
    # |R(z)| decays like 1/|z|, so the stiffest modes are killed in one step.
    # (The limit is checked at z = -1e6 rather than further out because the
    # explicit linear solve loses accuracy to cancellation beyond that.)
    for z in (-1.0, -1e1, -1e3, -1e6):
      self.assertLessEqual(abs(stability_function(z)), 1.0)
    self.assertLess(abs(stability_function(-1e3)), 1e-2)
    self.assertLess(abs(stability_function(-1e6)), 1e-5)

  def test_crank_nicolson_is_not_l_stable(self):
    """Sanity check on the stability test: theta=0.5 rings instead of damping."""
    # The theta=0.5 amplification factor tends to -1, which is why the theta
    # method cannot be used at second order on this stiff problem.
    self.assertAlmostEqual((1.0 + 0.5 * -1e10) / (1.0 - 0.5 * -1e10), -1.0)


class TrBdf2ConvergenceTest(parameterized.TestCase):
  """Integrates a scalar problem with the tableau to confirm the order."""

  def _integrate(self, lam, y0, t_final, n_steps):
    """Solves y' = lam*y exactly stage by stage, using the tableau."""
    dt = t_final / n_steps
    y = y0
    for _ in range(n_steps):
      # Stage 1: trapezoidal over gamma*dt.
      y_g = y * (1.0 + _D * lam * dt) / (1.0 - _D * lam * dt)
      # Stage 2: BDF2, in the same three-point form the residual assembles.
      y = (tr_bdf2.A_STAGE1 * y_g + tr_bdf2.A_START * y) / (
          1.0 - tr_bdf2.B_IMPLICIT * lam * dt
      )
    return y

  def test_observed_order_is_two(self):
    lam, y0, t_final = -1.5, 1.0, 1.0
    exact = y0 * np.exp(lam * t_final)
    errors = np.array([
        abs(self._integrate(lam, y0, t_final, n) - exact)
        for n in (10, 20, 40, 80)
    ])
    orders = np.log2(errors[:-1] / errors[1:])
    np.testing.assert_allclose(orders, 2.0, atol=0.05)

  def test_embedded_estimate_scales_as_dt_cubed(self):
    """The estimate of a second-order method's error is O(dt^3) per step."""
    lam, y0 = -1.5, 1.0
    estimates = []
    for dt in (0.1, 0.05, 0.025):
      # One step, with unit transient coefficients so u == x.
      y_g = y0 * (1.0 + _D * lam * dt) / (1.0 - _D * lam * dt)
      y_new = (tr_bdf2.A_STAGE1 * y_g + tr_bdf2.A_START * y0) / (
          1.0 - tr_bdf2.B_IMPLICIT * lam * dt
      )
      estimates.append(
          abs(
              tr_bdf2.embedded_error_estimate(
                  dt=np.array(dt),
                  u_start=np.array(y0),
                  u_stage1=np.array(y_g),
                  u_new=np.array(y_new),
                  stage_derivative_start=np.array(lam * y0),
                  tc_in_new=np.array(1.0),
              )
          )
      )
    estimates = np.array(estimates)
    orders = np.log2(estimates[:-1] / estimates[1:])
    np.testing.assert_allclose(orders, 3.0, atol=0.1)

  def test_embedded_estimate_tracks_the_true_local_error(self):
    """On one step the estimate should be the right size, not just the right order."""
    lam, y0, dt = -1.5, 1.0, 0.05
    y_g = y0 * (1.0 + _D * lam * dt) / (1.0 - _D * lam * dt)
    y_new = (tr_bdf2.A_STAGE1 * y_g + tr_bdf2.A_START * y0) / (
        1.0 - tr_bdf2.B_IMPLICIT * lam * dt
    )
    # The estimate is b_hat - b, i.e. (embedded solution) - (2nd order
    # solution). Since the embedded solution is the more accurate one, this is
    # minus the local error of the step that was actually taken.
    true_local_error = y0 * np.exp(lam * dt) - y_new
    estimate = tr_bdf2.embedded_error_estimate(
        dt=np.array(dt),
        u_start=np.array(y0),
        u_stage1=np.array(y_g),
        u_new=np.array(y_new),
        stage_derivative_start=np.array(lam * y0),
        tc_in_new=np.array(1.0),
    )
    # The embedded method is third order, so the estimate should recover the
    # local error to within its own O(dt^4) truncation.
    np.testing.assert_allclose(estimate, true_local_error, rtol=0.05)


if __name__ == '__main__':
  absltest.main()
