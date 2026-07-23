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
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.pedestal_model.saturation import base as saturation_base

# pylint: disable=invalid-name


class PedestalModelOutputTest(parameterized.TestCase):

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
        H_mode_fraction=jnp.array(1.0),
        saturation_fraction=saturation_base.SaturationFraction(
            chi_i_saturation_fraction=jnp.array(0.25),
            chi_e_saturation_fraction=jnp.array(0.5),
            D_e_saturation_fraction=jnp.array(0.75),
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

  def _make_uniform_core_transport(self, value=1.0):
    """CoreTransport with every coefficient set to a uniform face array."""
    n_face = self.geo.rho_face_norm.shape[0]
    return state.CoreTransport(**{
        field.name: value * jnp.ones(n_face)
        for field in dataclasses.fields(state.CoreTransport)
    })

  def _make_pedestal_runtime_params(self):
    pedestal_runtime_params = mock.create_autospec(
        pedestal_runtime_params_lib.RuntimeParams, instance=True
    )
    pedestal_runtime_params.chi_H_mode_max = jnp.array(1.0)
    pedestal_runtime_params.D_e_H_mode_max = jnp.array(1.0)
    pedestal_runtime_params.chi_H_mode_min = jnp.array(0.05)
    pedestal_runtime_params.D_e_H_mode_min = jnp.array(0.02)
    pedestal_runtime_params.pedestal_top_smoothing_width = jnp.array(0.0)
    return pedestal_runtime_params

  def _make_output(self, H_mode_fraction, saturation_fraction, rho_norm_ped_top=0.5):
    return pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=rho_norm_ped_top,
        T_i_ped=1.0,
        T_e_ped=1.0,
        n_e_ped=1e19,
        H_mode_fraction=jnp.array(H_mode_fraction),
        saturation_fraction=saturation_base.SaturationFraction(
            chi_i_saturation_fraction=jnp.array(saturation_fraction),
            chi_e_saturation_fraction=jnp.array(saturation_fraction),
            D_e_saturation_fraction=jnp.array(saturation_fraction),
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='chi_i',
          total_field='chi_face_ion',
          component_fields=(
              'chi_face_ion_bohm',
              'chi_face_ion_gyrobohm',
              'chi_face_ion_itg',
              'chi_face_ion_tem',
          ),
          pereverzev_fields=(
              'chi_face_ion_pereverzev',
              'full_v_heat_face_ion_pereverzev',
          ),
          saturation_fraction=0.25,
          residual=0.05,
          cap=1.0,
      ),
      dict(
          testcase_name='chi_e',
          total_field='chi_face_el',
          component_fields=(
              'chi_face_el_bohm',
              'chi_face_el_gyrobohm',
              'chi_face_el_itg',
              'chi_face_el_tem',
              'chi_face_el_etg',
          ),
          pereverzev_fields=(
              'chi_face_el_pereverzev',
              'full_v_heat_face_el_pereverzev',
          ),
          saturation_fraction=0.5,
          residual=0.05,
          cap=1.0,
      ),
      dict(
          testcase_name='D_e',
          total_field='d_face_el',
          component_fields=('d_face_el_itg', 'd_face_el_tem'),
          pereverzev_fields=('d_face_el_pereverzev', 'v_face_el_pereverzev'),
          saturation_fraction=0.75,
          residual=0.02,
          cap=1.0,
      ),
      dict(
          testcase_name='v_e_pinch',
          total_field='v_face_el',
          component_fields=('v_face_el_itg', 'v_face_el_tem'),
          pereverzev_fields=(),
          # The pinch has no saturation fraction of its own: residual = cap =
          # 0 makes the H-mode branch (and thus the suppression factor) 0
          # regardless of the fixture's per-channel saturation fractions.
          saturation_fraction=0.0,
          residual=0.0,
          cap=0.0,
      ),
  )
  def test_modify_core_transport_applies_blend(
      self,
      total_field,
      component_fields,
      pereverzev_fields,
      saturation_fraction,
      residual,
      cap,
  ):
    core_transport = self._make_uniform_core_transport(value=1.0)
    pedestal_runtime_params = self._make_pedestal_runtime_params()

    modified_core_transport = self.pedestal_model_output.modify_core_transport(
        core_transport, self.geo, pedestal_runtime_params
    )
    pedestal_mask = (
        self.geo.rho_face_norm > self.pedestal_model_output.rho_norm_ped_top
    )

    # The fixture's H-mode fraction is 1, so in the pedestal region the total
    # coefficient equals the H-mode branch residual + r * (cap - residual),
    # and diagnostic components/Pereverzev pairs scale by the suppression
    # factor (1 - g) + g * r = r, without the residual floor.
    expected_total = residual + saturation_fraction * (cap - residual)
    with self.subTest('total_field_gets_H_mode_branch'):
      np.testing.assert_allclose(
          getattr(modified_core_transport, total_field),
          jnp.where(pedestal_mask, expected_total, 1.0),
          rtol=1e-6,
          atol=1e-12,
      )
    with self.subTest('components_get_relative_suppression'):
      for field_name in component_fields + pereverzev_fields:
        np.testing.assert_allclose(
            getattr(modified_core_transport, field_name),
            jnp.where(pedestal_mask, saturation_fraction, 1.0),
            rtol=1e-6,
            atol=1e-12,
        )

  def test_modify_core_transport_neoclassical_untouched(self):
    core_transport = self._make_uniform_core_transport(value=1.0)
    pedestal_runtime_params = self._make_pedestal_runtime_params()
    modified_core_transport = self.pedestal_model_output.modify_core_transport(
        core_transport, self.geo, pedestal_runtime_params
    )
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

  def test_modify_core_transport_identity_at_zero_H_mode_fraction(self):
    """With H-mode fraction 0 (L-mode), coefficients are unmodified."""
    core_transport = self._make_uniform_core_transport(value=3.0)
    pedestal_runtime_params = self._make_pedestal_runtime_params()
    output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=0.5,
        T_i_ped=1.0,
        T_e_ped=1.0,
        n_e_ped=1e19,
    )
    modified = output.modify_core_transport(
        core_transport, self.geo, pedestal_runtime_params
    )
    # Even though the coefficient (3.0) exceeds chi_H_mode_max (1.0), no cap is
    # applied when the H-mode fraction is 0 (L-mode branch is unclipped).
    np.testing.assert_allclose(modified.chi_face_ion, 3.0)
    np.testing.assert_allclose(modified.chi_face_el, 3.0)
    np.testing.assert_allclose(modified.d_face_el, 3.0)
    np.testing.assert_allclose(modified.v_face_el, 3.0)

  def test_modify_core_transport_is_continuous_in_H_mode_fraction(self):
    """chi is a continuous function of the H-mode fraction.

    The blend is linear in g, so sweeping g from 0 (L-mode) to 1 (H-mode)
    with a raw coefficient well above the H-mode cap must change chi
    continuously, with no jumps.
    """
    core_transport = self._make_uniform_core_transport(value=5.0)
    pedestal_runtime_params = self._make_pedestal_runtime_params()

    fractions = np.linspace(0.0, 1.0, 401)
    ped_top_idx = -1  # Last face is inside the pedestal (rho > 0.5).
    chi_values = []
    for g in fractions:
      output = self._make_output(H_mode_fraction=g, saturation_fraction=0.5)
      modified = output.modify_core_transport(
          core_transport, self.geo, pedestal_runtime_params
      )
      chi_values.append(float(modified.chi_face_ion[ped_top_idx]))
    chi_values = np.array(chi_values)

    # The full swing over the sweep is large (raw 5 vs H-mode ~0.5), but
    # each step must be a small fraction of the swing: no jumps.
    total_swing = np.max(chi_values) - np.min(chi_values)
    max_step = np.max(np.abs(np.diff(chi_values)))
    self.assertGreater(total_swing, 1.0)
    self.assertLess(max_step, 0.05 * total_swing)

    # At g = 0 the coefficient is untouched; at g = 1 it is the H-mode
    # branch value, independent of the raw coefficient.
    np.testing.assert_allclose(chi_values[0], 5.0, rtol=1e-6)
    np.testing.assert_allclose(
        chi_values[-1], 0.05 + 0.5 * (1.0 - 0.05), rtol=1e-6
    )

  def test_modify_core_transport_preserves_pereverzev_cancellation(self):
    """Pereverzev diffusion/convection pairs must scale by the same factor.

    The Pereverzev-Corrigan scheme relies on exact cancellation between its
    diffusion and compensating convection terms at the current profile. Both
    members of each pair must therefore be scaled identically, with no
    clipping, even when their magnitudes exceed the H-mode caps.
    """
    n_face = self.geo.rho_face_norm.shape[0]
    core_transport = self._make_uniform_core_transport(value=1.0)
    # Pereverzev terms are typically much larger than the H-mode caps.
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
    # Fully formed H-mode edge, slightly open.
    H_mode_fraction = 0.99
    saturation_fraction = 0.1
    output = self._make_output(H_mode_fraction, saturation_fraction)
    modified = output.modify_core_transport(
        core_transport, self.geo, pedestal_runtime_params
    )
    pedestal_mask = self.geo.rho_face_norm > 0.5
    expected_factor = (1.0 - H_mode_fraction) + H_mode_fraction * saturation_fraction

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
    """At g=1, r=0 the H-mode transport equals the floors."""
    core_transport = self._make_uniform_core_transport(value=1.0)
    pedestal_runtime_params = self._make_pedestal_runtime_params()
    pedestal_runtime_params.chi_H_mode_min = jnp.array(0.07)
    pedestal_runtime_params.D_e_H_mode_min = jnp.array(0.5)
    pedestal_mask = self.geo.rho_face_norm > 0.5

    with self.subTest('fully_suppressed_sits_at_floors'):
      modified = self._make_output(
          H_mode_fraction=1.0, saturation_fraction=0.0
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

    with self.subTest('full_saturation_fraction_reaches_caps'):
      # At r=1 the H-mode transport equals the caps, independent of the raw
      # coefficient: the saturation authority is bounded by chi_H_mode_max/D_e_H_mode_max.
      modified = self._make_output(
          H_mode_fraction=1.0, saturation_fraction=1.0
      ).modify_core_transport(core_transport, self.geo, pedestal_runtime_params)
      np.testing.assert_allclose(
          modified.chi_face_ion[pedestal_mask],
          pedestal_runtime_params.chi_H_mode_max,
          rtol=1e-6,
      )
      np.testing.assert_allclose(
          modified.d_face_el[pedestal_mask],
          pedestal_runtime_params.D_e_H_mode_max,
          rtol=1e-6,
      )

    with self.subTest('zero_H_mode_fraction_ignores_residuals'):
      modified = self._make_output(
          H_mode_fraction=0.0, saturation_fraction=0.0
      ).modify_core_transport(core_transport, self.geo, pedestal_runtime_params)
      np.testing.assert_allclose(modified.d_face_el, 1.0)

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
