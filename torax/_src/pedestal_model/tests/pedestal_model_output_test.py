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

import dataclasses
from unittest import mock
from absl.testing import absltest
from jax import numpy as jnp
import numpy as np
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib

# pylint: disable=invalid-name


class PedestalModelOutputTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = mock.create_autospec(geometry.Geometry, instance=True)
    self.geo.rho_face_norm = jnp.linspace(0, 1, 10)
    self.geo.rho = jnp.linspace(0, 1, 9)
    self.geo.rho_norm = jnp.linspace(0, 1, 9)
    self.pedestal_model_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=0.99,
        T_i_ped=1.0,
        T_e_ped=1.1,
        n_e_ped=1.2e19,
        barrier_state=pedestal_model_output.BarrierState(
            barrier_fraction=jnp.array(1.0),
            chi_i_openness=jnp.array(0.25),
            chi_e_openness=jnp.array(0.5),
            D_e_openness=jnp.array(0.75),
        ),
    )

  def test_to_internal_boundary_conditions(self):
    ibc = self.pedestal_model_output.to_internal_boundary_conditions(self.geo)
    idx = jnp.argmin(
        jnp.abs(self.geo.rho_norm - self.pedestal_model_output.rho_norm_ped_top)
    )
    with self.subTest('T_i'):
      np.testing.assert_allclose(
          ibc.T_i[idx], self.pedestal_model_output.T_i_ped
      )
      np.testing.assert_allclose(
          jnp.sum(ibc.T_i), self.pedestal_model_output.T_i_ped
      )
    with self.subTest('T_e'):
      np.testing.assert_allclose(
          ibc.T_e[idx], self.pedestal_model_output.T_e_ped
      )
      np.testing.assert_allclose(
          jnp.sum(ibc.T_e), self.pedestal_model_output.T_e_ped
      )
    with self.subTest('n_e'):
      np.testing.assert_allclose(
          ibc.n_e[idx], self.pedestal_model_output.n_e_ped
      )
      np.testing.assert_allclose(
          jnp.sum(ibc.n_e), self.pedestal_model_output.n_e_ped
      )

  def _make_uniform_core_transport(
      self, value=1.0, chi_neo=None, D_neo=None
  ):
    """Builds a uniform CoreTransport.

    Args:
      value: Uniform value for the raw (turbulent) fields, including the
        diagnostic components and Pereverzev pairs.
      chi_neo: Uniform value for chi_neo_i / chi_neo_e (the barrier residual
        floor for the heat channels). Defaults to `value` when unspecified.
      D_neo: Uniform value for D_neo_e (the barrier residual floor for the
        particle channel). Defaults to `value` when unspecified.
    """
    if chi_neo is None:
      chi_neo = value
    if D_neo is None:
      D_neo = value
    n_face = self.geo.rho_face_norm.shape[0]
    return state.CoreTransport(
        chi_face_ion=value * jnp.ones(n_face),
        chi_face_el=value * jnp.ones(n_face),
        d_face_el=value * jnp.ones(n_face),
        v_face_el=value * jnp.ones(n_face),
        chi_face_el_bohm=value * jnp.ones(n_face),
        chi_face_el_gyrobohm=value * jnp.ones(n_face),
        chi_face_ion_bohm=value * jnp.ones(n_face),
        chi_face_ion_gyrobohm=value * jnp.ones(n_face),
        chi_face_el_itg=value * jnp.ones(n_face),
        chi_face_el_tem=value * jnp.ones(n_face),
        chi_face_el_etg=value * jnp.ones(n_face),
        chi_face_ion_itg=value * jnp.ones(n_face),
        chi_face_ion_tem=value * jnp.ones(n_face),
        d_face_el_itg=value * jnp.ones(n_face),
        d_face_el_tem=value * jnp.ones(n_face),
        v_face_el_itg=value * jnp.ones(n_face),
        v_face_el_tem=value * jnp.ones(n_face),
        chi_neo_i=chi_neo * jnp.ones(n_face),
        chi_neo_e=chi_neo * jnp.ones(n_face),
        D_neo_e=D_neo * jnp.ones(n_face),
        V_neo_e=value * jnp.ones(n_face),
        V_neo_ware_e=value * jnp.ones(n_face),
        chi_face_ion_pereverzev=value * jnp.ones(n_face),
        chi_face_el_pereverzev=value * jnp.ones(n_face),
        full_v_heat_face_ion_pereverzev=value * jnp.ones(n_face),
        full_v_heat_face_el_pereverzev=value * jnp.ones(n_face),
        d_face_el_pereverzev=value * jnp.ones(n_face),
        v_face_el_pereverzev=value * jnp.ones(n_face),
    )

  def _make_pedestal_runtime_params(self):
    pedestal_runtime_params = mock.create_autospec(
        pedestal_runtime_params_lib.RuntimeParams, instance=True
    )
    pedestal_runtime_params.pedestal_top_smoothing_width = jnp.array(0.0)
    return pedestal_runtime_params

  def _make_output(self, barrier_fraction, openness, rho_norm_ped_top=0.5):
    return pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=rho_norm_ped_top,
        T_i_ped=1.0,
        T_e_ped=1.0,
        n_e_ped=1e19,
        barrier_state=pedestal_model_output.BarrierState(
            barrier_fraction=jnp.array(barrier_fraction),
            chi_i_openness=jnp.array(openness),
            chi_e_openness=jnp.array(openness),
            D_e_openness=jnp.array(openness),
        ),
    )

  def test_modify_core_transport_applies_barrier_blend(self):
    # Raw (turbulent) fields at 1.0; neoclassical residual floors at 0.05
    # (heat channels) / 0.02 (particle channel), so cap = max(1.0, floor) =
    # 1.0 for all three channels.
    core_transport = self._make_uniform_core_transport(
        value=1.0, chi_neo=0.05, D_neo=0.02
    )
    pedestal_runtime_params = self._make_pedestal_runtime_params()

    modified_core_transport = self.pedestal_model_output.modify_core_transport(
        core_transport, self.geo, pedestal_runtime_params
    )
    pedestal_mask = (
        self.geo.rho_face_norm > self.pedestal_model_output.rho_norm_ped_top
    )

    # The barrier fraction is 1, so in the pedestal region the total
    # coefficients equal the barrier branch residual + r * (cap - residual)
    # with per-channel openness r (chi_i: 0.25, chi_e: 0.5, D_e: 0.75), cap
    # = max(raw, residual) and residual = the local neoclassical level.
    def expected_barrier(openness, cap, residual):
      return residual + openness * (cap - residual)

    with self.subTest('total_fields_get_barrier_branch'):
      np.testing.assert_allclose(
          modified_core_transport.chi_face_ion,
          jnp.where(pedestal_mask, expected_barrier(0.25, 1.0, 0.05), 1.0),
          rtol=1e-6,
      )
      np.testing.assert_allclose(
          modified_core_transport.chi_face_el,
          jnp.where(pedestal_mask, expected_barrier(0.5, 1.0, 0.05), 1.0),
          rtol=1e-6,
      )
      np.testing.assert_allclose(
          modified_core_transport.d_face_el,
          jnp.where(pedestal_mask, expected_barrier(0.75, 1.0, 0.02), 1.0),
          rtol=1e-6,
      )

    with self.subTest('pinch_suppressed_to_zero_in_barrier'):
      np.testing.assert_allclose(
          modified_core_transport.v_face_el,
          jnp.where(pedestal_mask, 0.0, 1.0),
          atol=1e-12,
      )

    with self.subTest('components_get_relative_suppression'):
      # Diagnostic components scale by (1 - g) + g * r = r at g = 1, without
      # the residual floor.
      for field_name in ['chi_face_ion_bohm', 'chi_face_ion_gyrobohm']:
        np.testing.assert_allclose(
            getattr(modified_core_transport, field_name),
            jnp.where(pedestal_mask, 0.25, 1.0),
            rtol=1e-6,
        )
      for field_name in ['chi_face_el_bohm', 'chi_face_el_etg']:
        np.testing.assert_allclose(
            getattr(modified_core_transport, field_name),
            jnp.where(pedestal_mask, 0.5, 1.0),
            rtol=1e-6,
        )
      np.testing.assert_allclose(
          modified_core_transport.d_face_el_itg,
          jnp.where(pedestal_mask, 0.75, 1.0),
          rtol=1e-6,
      )
      np.testing.assert_allclose(
          modified_core_transport.v_face_el_itg,
          jnp.where(pedestal_mask, 0.0, 1.0),
          atol=1e-12,
      )

    with self.subTest('pereverzev_pairs_share_channel_factor'):
      for field_name in [
          'chi_face_ion_pereverzev',
          'full_v_heat_face_ion_pereverzev',
      ]:
        np.testing.assert_allclose(
            getattr(modified_core_transport, field_name),
            jnp.where(pedestal_mask, 0.25, 1.0),
            rtol=1e-6,
        )
      for field_name in [
          'chi_face_el_pereverzev',
          'full_v_heat_face_el_pereverzev',
      ]:
        np.testing.assert_allclose(
            getattr(modified_core_transport, field_name),
            jnp.where(pedestal_mask, 0.5, 1.0),
            rtol=1e-6,
        )
      for field_name in ['d_face_el_pereverzev', 'v_face_el_pereverzev']:
        np.testing.assert_allclose(
            getattr(modified_core_transport, field_name),
            jnp.where(pedestal_mask, 0.75, 1.0),
            rtol=1e-6,
        )

    with self.subTest('neoclassical_untouched'):
      for field_name in [
          'chi_neo_i',
          'chi_neo_e',
          'D_neo_e',
          'V_neo_e',
          'V_neo_ware_e',
      ]:
        np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
            getattr(modified_core_transport, field_name),
            getattr(core_transport, field_name),
        )

  def test_modify_core_transport_identity_at_zero_barrier_fraction(self):
    """With barrier fraction 0 (L-mode), coefficients are unmodified."""
    core_transport = self._make_uniform_core_transport(value=3.0)
    pedestal_runtime_params = self._make_pedestal_runtime_params()
    output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=0.5,
        T_i_ped=1.0,
        T_e_ped=1.0,
        n_e_ped=1e19,
        barrier_state=pedestal_model_output.BarrierState.default(),
    )
    modified = output.modify_core_transport(
        core_transport, self.geo, pedestal_runtime_params
    )
    # No cap/residual blend is applied at all when the barrier fraction is 0
    # (the L-mode branch is unmodified).
    np.testing.assert_allclose(modified.chi_face_ion, 3.0)
    np.testing.assert_allclose(modified.chi_face_el, 3.0)
    np.testing.assert_allclose(modified.d_face_el, 3.0)
    np.testing.assert_allclose(modified.v_face_el, 3.0)

  def test_modify_core_transport_is_continuous_in_barrier_fraction(self):
    """chi is a continuous function of the barrier fraction.

    The blend is linear in g, so sweeping g from 0 (L-mode) to 1 (barrier)
    must change chi continuously, with no jumps, even though the barrier
    branch spans the full range from the neoclassical residual floor up to
    the raw (turbulent) coefficient.
    """
    core_transport = self._make_uniform_core_transport(value=5.0, chi_neo=0.05)
    pedestal_runtime_params = self._make_pedestal_runtime_params()

    fractions = np.linspace(0.0, 1.0, 401)
    ped_top_idx = -1  # Last face is inside the pedestal (rho > 0.5).
    chi_values = []
    for g in fractions:
      output = self._make_output(barrier_fraction=g, openness=0.5)
      modified = output.modify_core_transport(
          core_transport, self.geo, pedestal_runtime_params
      )
      chi_values.append(float(modified.chi_face_ion[ped_top_idx]))
    chi_values = np.array(chi_values)

    # The full swing over the sweep is large (raw 5.0 vs barrier ~2.5), but
    # each step must be a small fraction of the swing: no jumps.
    total_swing = np.max(chi_values) - np.min(chi_values)
    max_step = np.max(np.abs(np.diff(chi_values)))
    self.assertGreater(total_swing, 1.0)
    self.assertLess(max_step, 0.05 * total_swing)

    # At g = 0 the coefficient is untouched; at g = 1 it is the barrier
    # branch value: residual + r * (cap - residual), with cap = max(raw,
    # residual) = 5.0 and residual = 0.05.
    np.testing.assert_allclose(chi_values[0], 5.0, rtol=1e-6)
    np.testing.assert_allclose(
        chi_values[-1], 0.05 + 0.5 * (5.0 - 0.05), rtol=1e-6
    )

  def test_modify_core_transport_preserves_pereverzev_cancellation(self):
    """Pereverzev diffusion/convection pairs must scale by the same factor.

    The Pereverzev-Corrigan scheme relies on exact cancellation between its
    diffusion and compensating convection terms at the current profile. Both
    members of each pair must therefore be scaled identically, with no
    clipping, even when their magnitudes exceed the barrier caps.
    """
    n_face = self.geo.rho_face_norm.shape[0]
    core_transport = self._make_uniform_core_transport(value=1.0)
    # Pereverzev terms are typically much larger than the barrier caps.
    core_transport = dataclasses.replace(
        core_transport,
        chi_face_ion_pereverzev=30.0 * jnp.ones(n_face),
        chi_face_el_pereverzev=30.0 * jnp.ones(n_face),
        full_v_heat_face_ion_pereverzev=-25.0 * jnp.ones(n_face),
        full_v_heat_face_el_pereverzev=-25.0 * jnp.ones(n_face),
        d_face_el_pereverzev=30.0 * jnp.ones(n_face),
        v_face_el_pereverzev=-20.0 * jnp.ones(n_face),
    )
    pedestal_runtime_params = self._make_pedestal_runtime_params()
    # Fully formed barrier, slightly open.
    barrier_fraction = 0.99
    openness = 0.1
    output = self._make_output(barrier_fraction, openness)
    modified = output.modify_core_transport(
        core_transport, self.geo, pedestal_runtime_params
    )
    pedestal_mask = self.geo.rho_face_norm > 0.5
    expected_factor = (1.0 - barrier_fraction) + barrier_fraction * openness

    for chi_name, v_heat_name in [
        ('chi_face_ion_pereverzev', 'full_v_heat_face_ion_pereverzev'),
        ('chi_face_el_pereverzev', 'full_v_heat_face_el_pereverzev'),
        ('d_face_el_pereverzev', 'v_face_el_pereverzev'),
    ]:
      chi_ratio = getattr(modified, chi_name) / getattr(
          core_transport, chi_name
      )
      v_ratio = getattr(modified, v_heat_name) / getattr(
          core_transport, v_heat_name
      )
      with self.subTest(pair=chi_name):
        # Same scaling factor for both pair members everywhere.
        np.testing.assert_allclose(chi_ratio, v_ratio, rtol=1e-12)
        # In the pedestal region, the factor is (1-g) + g*r, unclipped.
        np.testing.assert_allclose(
            chi_ratio[pedestal_mask], expected_factor, rtol=1e-12
        )

  def test_residual_floors_apply_under_full_suppression(self):
    """At g=1, r=0 the barrier transport sits at the local neoclassical floor.

    At g=1, r=1 it reverts to the raw (turbulent) coefficient itself: the cap
    is not an externally chosen constant.
    """
    core_transport = self._make_uniform_core_transport(
        value=1.0, chi_neo=0.07, D_neo=0.5
    )
    pedestal_runtime_params = self._make_pedestal_runtime_params()
    pedestal_mask = self.geo.rho_face_norm > 0.5

    with self.subTest('closed_barrier_sits_at_residuals'):
      modified = self._make_output(
          barrier_fraction=1.0, openness=0.0
      ).modify_core_transport(core_transport, self.geo, pedestal_runtime_params)
      np.testing.assert_allclose(
          modified.d_face_el[pedestal_mask], 0.5, rtol=1e-6
      )
      np.testing.assert_allclose(
          modified.chi_face_ion[pedestal_mask], 0.07, rtol=1e-6
      )
      np.testing.assert_allclose(
          modified.v_face_el[pedestal_mask], 0.0, atol=1e-12
      )
      # Outside the region: untouched.
      np.testing.assert_allclose(modified.d_face_el[~pedestal_mask], 1.0)

    with self.subTest('open_barrier_reverts_to_raw_coefficient'):
      # At r=1 the barrier transport equals the cap, which is max(raw,
      # residual) = the raw coefficient itself (1.0) here, independent of how
      # large the residual floors are.
      modified = self._make_output(
          barrier_fraction=1.0, openness=1.0
      ).modify_core_transport(core_transport, self.geo, pedestal_runtime_params)
      np.testing.assert_allclose(
          modified.chi_face_ion[pedestal_mask], 1.0, rtol=1e-6
      )
      np.testing.assert_allclose(
          modified.d_face_el[pedestal_mask], 1.0, rtol=1e-6
      )

    with self.subTest('zero_barrier_fraction_ignores_residuals'):
      modified = self._make_output(
          barrier_fraction=0.0, openness=0.0
      ).modify_core_transport(core_transport, self.geo, pedestal_runtime_params)
      np.testing.assert_allclose(modified.d_face_el, 1.0)

  def test_cap_clamps_to_residual_when_raw_is_below_it(self):
    """The cap never falls below the residual, even if raw transport does.

    If the raw (turbulent) coefficient dips below the local neoclassical
    floor, cap = max(raw, residual) clamps to the residual, so the barrier
    branch collapses to a flat residual value (no throttling authority)
    rather than inverting the sign of (cap - residual).
    """
    core_transport = self._make_uniform_core_transport(
        value=0.01, chi_neo=0.07, D_neo=0.02
    )
    pedestal_runtime_params = self._make_pedestal_runtime_params()
    pedestal_mask = self.geo.rho_face_norm > 0.5

    for openness in (0.0, 0.5, 1.0):
      modified = self._make_output(
          barrier_fraction=1.0, openness=openness
      ).modify_core_transport(core_transport, self.geo, pedestal_runtime_params)
      with self.subTest(openness=openness):
        np.testing.assert_allclose(
            modified.chi_face_ion[pedestal_mask], 0.07, rtol=1e-6
        )
        np.testing.assert_allclose(
            modified.d_face_el[pedestal_mask], 0.02, rtol=1e-6
        )

  def test_to_internal_boundary_conditions_tanh_profiles(self):
    """Tests mtanh-shaped IBC when pedestal_profile_form=MTANH."""
    geo = circular_geometry.CircularConfig(n_rho=25).build_geometry()
    n_cell = geo.torax_mesh.nx

    # Use psi ~ rho^2 so the mapping is analytically known.
    psi_cell = geo.rho_norm**2

    rho_ped_top = 0.9
    T_i_ped = 5.0
    T_e_ped = 4.0
    n_e_ped = 0.7e20

    # Separatrix values (rightmost face).
    T_i_sep = 0.1
    T_e_sep = 0.08
    n_e_sep = 0.1e20

    face_centers = geo.rho_face_norm

    def _make_standard_cell_var(cell_vals, right_face_val):
      return cell_variable.CellVariable(
          value=cell_vals,
          face_centers=face_centers,
          right_face_constraint=jnp.array(right_face_val),
          right_face_grad_constraint=None,
      )

    core_profiles = mock.MagicMock(spec=state.CoreProfiles)
    core_profiles.psi = cell_variable.CellVariable(
        value=psi_cell,
        face_centers=face_centers,
        right_face_constraint=None,
        right_face_grad_constraint=jnp.array(2.0),
    )
    core_profiles.T_i = _make_standard_cell_var(
        jnp.linspace(10.0, T_i_sep, n_cell), T_i_sep
    )
    core_profiles.T_e = _make_standard_cell_var(
        jnp.linspace(8.0, T_e_sep, n_cell), T_e_sep
    )
    core_profiles.n_e = _make_standard_cell_var(
        jnp.linspace(1.0e20, n_e_sep, n_cell), n_e_sep
    )

    ped_output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=rho_ped_top,
        T_i_ped=T_i_ped,
        T_e_ped=T_e_ped,
        n_e_ped=n_e_ped,
    )

    ibc_tanh = ped_output.to_internal_boundary_conditions(
        geo,
        core_profiles=core_profiles,
        pedestal_profile_form=pedestal_runtime_params_lib.PedestalProfileForm.MTANH,
    )
    ibc_point = ped_output.to_internal_boundary_conditions(geo)

    ped_mask = geo.rho_norm >= rho_ped_top

    with self.subTest('tanh_has_multiple_nonzero_cells'):
      # mtanh should have non-zero values at multiple cells.
      n_nonzero_tanh = jnp.sum(jnp.abs(ibc_tanh.T_i) > 0.0)
      n_nonzero_point = jnp.sum(jnp.abs(ibc_point.T_i) > 0.0)
      self.assertGreater(int(n_nonzero_tanh), int(n_nonzero_point))

    with self.subTest('core_region_is_zero'):
      # Core cells (rho < rho_ped_top) should be exactly zero.
      core_mask = ~ped_mask
      np.testing.assert_allclose(ibc_tanh.T_i[core_mask], 0.0)
      np.testing.assert_allclose(ibc_tanh.T_e[core_mask], 0.0)
      np.testing.assert_allclose(ibc_tanh.n_e[core_mask], 0.0)

    with self.subTest('pedestal_values_are_positive'):
      # All values in the pedestal region should be positive.
      self.assertTrue(jnp.all(ibc_tanh.T_i[ped_mask] > 0.0))
      self.assertTrue(jnp.all(ibc_tanh.T_e[ped_mask] > 0.0))
      self.assertTrue(jnp.all(ibc_tanh.n_e[ped_mask] > 0.0))

    with self.subTest('profile_decreases_from_top_to_separatrix'):
      # T_i in the pedestal region should be monotonically decreasing.
      ped_T_i = ibc_tanh.T_i[ped_mask]
      if len(ped_T_i) > 1:
        self.assertTrue(jnp.all(jnp.diff(ped_T_i) <= 0.0))

    with self.subTest('exact_profile_comparison'):
      # Independently compute expected mtanh profiles from the formula:
      #   f(psi) = f_sep + a0 * [tanh(1) - tanh(2*(psi - psi_mid)/delta)]
      # with psi_top = interp(rho_ped_top), delta = (1 - psi_top) / 1.5,
      # psi_mid = 1 - delta / 2.

      # Compute psi_norm at cell centers using face_value() to match impl.
      psi_fv = core_profiles.psi.face_value()
      psi_norm_cell = (psi_cell - psi_fv[0]) / (psi_fv[-1] - psi_fv[0])  # pyrefly: ignore[bad-index]

      # Derive delta from nearest cell to rho_ped_top (matching impl).
      ped_top_idx = jnp.argmin(jnp.abs(geo.rho_norm - rho_ped_top))
      psi_top = psi_norm_cell[ped_top_idx]
      delta = (1.0 - psi_top) / 1.5
      psi_mid = 1.0 - delta / 2.0

      tanh1 = jnp.tanh(1.0)
      tanh2 = jnp.tanh(2.0)

      def _expected_mtanh(val_top, val_sep):
        a0 = (val_top - val_sep) / (tanh1 + tanh2)
        profile = val_sep + a0 * (
            tanh1 - jnp.tanh(2.0 * (psi_norm_cell - psi_mid) / delta)
        )
        return jnp.where(ped_mask, profile, 0.0)

      expected_T_i = _expected_mtanh(T_i_ped, T_i_sep)
      expected_T_e = _expected_mtanh(T_e_ped, T_e_sep)
      expected_n_e = _expected_mtanh(n_e_ped, n_e_sep)

      np.testing.assert_allclose(ibc_tanh.T_i, expected_T_i, rtol=1e-5)
      np.testing.assert_allclose(ibc_tanh.T_e, expected_T_e, rtol=1e-5)
      np.testing.assert_allclose(ibc_tanh.n_e, expected_n_e, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
