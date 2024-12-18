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

"""Unit tests for torax.physics."""

from typing import Callable
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax import core_profile_setters
from torax import geometry
from torax import physics
from torax import state
from torax.fvm import cell_variable
from torax.sources import runtime_params as source_runtime_params
from torax.sources import source_models as source_models_lib
from torax.tests.test_lib import torax_refs

_trapz = jax.scipy.integrate.trapezoid


class PhysicsTest(torax_refs.ReferenceValueTest):
  """Unit tests for the `torax.physics` module."""

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_calc_q_from_psi(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    """Compare `calc_q_from_psi` function to a reference implementation."""
    references = references_getter()

    runtime_params = references.runtime_params
    dynamic_runtime_params_slice, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            runtime_params,
            references.geometry_provider,
        )
    )

    q_face_jax, q_cell_jax = physics.calc_q_from_psi(
        geo,
        references.psi,
        dynamic_runtime_params_slice.numerics.q_correction_factor,
    )

    # Make ground truth
    def calc_q_from_psi(runtime_params, geo):
      """Reference implementation from PINT."""
      iota = np.zeros(geo.torax_mesh.nx + 1)  # on face grid
      # We use the reference value of psi here because the original code
      # for calculating psi depends on FiPy, and we don't want to install that
      iota[1:] = np.abs(
          references.psi_face_grad[1:] / (2 * geo.Phib * geo.rho_face_norm[1:])
      )
      iota[0] = np.abs(
          references.psi_face_grad[1] / (2 * geo.Phib * geo.drho_norm)
      )
      q = 1 / iota
      q *= runtime_params.numerics.q_correction_factor

      def face_to_cell(face):
        cell = np.zeros(geo.torax_mesh.nx)
        cell[:] = 0.5 * (face[1:] + face[:-1])
        return cell

      q_cell = face_to_cell(q)
      return q, q_cell

    q_face_np, q_cell_np = calc_q_from_psi(runtime_params, geo)

    np.testing.assert_allclose(q_face_jax, q_face_np)
    np.testing.assert_allclose(q_cell_jax, q_cell_np)

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_update_psi_from_j(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    """Compare `update_psi_from_j` function to a reference implementation."""
    references = references_getter()

    runtime_params = references.runtime_params
    source_models_builder = source_models_lib.SourceModelsBuilder()
    # Turn on the external current source.
    source_models_builder.runtime_params['generic_current_source'].mode = (
        source_runtime_params.Mode.FORMULA_BASED
    )
    source_models = source_models_builder()
    dynamic_runtime_params_slice, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            runtime_params,
            references.geometry_provider,
            sources=source_models_builder.runtime_params,
        )
    )
    initial_core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice,
        geo,
        source_models=source_models,
    )

    # pylint: disable=protected-access
    if isinstance(geo, geometry.CircularAnalyticalGeometry):
      currents = core_profile_setters._prescribe_currents_no_bootstrap(
          dynamic_runtime_params_slice,
          geo,
          source_models=source_models,
          core_profiles=initial_core_profiles,
      )
      psi = core_profile_setters._update_psi_from_j(
          dynamic_runtime_params_slice, geo, currents.jtot_hires
      ).value
    elif isinstance(geo, geometry.StandardGeometry):
      psi = geo.psi_from_Ip
    else:
      raise ValueError(f'Unknown geometry type: {geo.geometry_type}')
    # pylint: enable=protected-access

    np.testing.assert_allclose(psi, references.psi.value)

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_calc_jtot_from_psi(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    """Compare `calc_jtot_from_psi` to a reference value."""
    references = references_getter()
    geo = references.geometry_provider(
        references.runtime_params.numerics.t_initial
    )
    # pylint: disable=invalid-name
    j, _, Ip_profile_face = physics.calc_jtot_from_psi(
        geo,
        references.psi,
    )
    # pylint: enable=invalid-name
    np.testing.assert_allclose(j, references.jtot)

    if references.Ip_from_parameters:
      np.testing.assert_allclose(
          Ip_profile_face[-1],
          references.runtime_params.profile_conditions.Ip_tot * 1e6,
      )
    else:
      assert(isinstance(geo, geometry.StandardGeometry))
      np.testing.assert_allclose(
          Ip_profile_face[-1],
          geo.Ip_profile_face[-1],
      )

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_calc_s_from_psi(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    """Compare `calc_s_from_psi` to a reference value."""
    references = references_getter()
    geo = references.geometry_provider(
        references.runtime_params.numerics.t_initial
    )

    s = physics.calc_s_from_psi(
        geo,
        references.psi,
    )

    np.testing.assert_allclose(s, references.s)

  def test_fast_ion_fractional_heating_formula(self):
    """Compare `ion_heat_fraction` to a reference value."""
    # Inertial energy small compared to critical energy, all energy to ions.
    birth_energy = 1e-3
    temp_el = jnp.array(0.1, dtype=jnp.float32)
    fast_ion_mass = 1
    frac_i = physics.fast_ion_fractional_heating_formula(
        birth_energy, temp_el, fast_ion_mass
    )
    np.testing.assert_allclose(frac_i, 1.0, atol=1e-3)

    # Inertial energy large compared to critical energy, all energy to e-.
    birth_energy = 1e10
    frac_i = physics.fast_ion_fractional_heating_formula(
        birth_energy, temp_el, fast_ion_mass
    )
    np.testing.assert_allclose(frac_i, 0.0, atol=1e-9)

  # TODO(b/377225415): generalize to arbitrary number of ions.
  @parameterized.parameters([
      dict(Aimp=20.0, Zimp=10.0, Zi=1.0, Ai=1.0, ni=1.0, expected=1.0),
      dict(Aimp=20.0, Zimp=10.0, Zi=1.0, Ai=2.0, ni=1.0, expected=0.5),
      dict(Aimp=20.0, Zimp=10.0, Zi=2.0, Ai=4.0, ni=0.5, expected=0.5),
      dict(Aimp=20.0, Zimp=10.0, Zi=1.0, Ai=2.0, ni=0.9, expected=0.5),
      dict(Aimp=40.0, Zimp=20.0, Zi=1.0, Ai=2.0, ni=0.92, expected=0.5),
  ])
  # pylint: disable=invalid-name
  def test_calculate_weighted_Zeff(self, Aimp, Zimp, Zi, Ai, ni, expected):
    """Compare `_calculate_weighted_Zeff` to a reference value."""
    references = torax_refs.circular_references()
    geo = references.geometry_provider(
        references.runtime_params.numerics.t_initial
    )
    ne = 1.0
    nimp = (ne - ni * Zi) / Zimp
    core_profiles = state.CoreProfiles(
        ne=cell_variable.CellVariable(
            value=jnp.array(ne),
            dr=jnp.array(1.0),
        ),
        ni=cell_variable.CellVariable(
            value=jnp.array(ni),
            dr=jnp.array(1.0),
        ),
        nimp=cell_variable.CellVariable(
            value=jnp.array(nimp),
            dr=jnp.array(1.0),
        ),
        temp_ion=cell_variable.CellVariable(
            value=jnp.array(0.0),
            dr=jnp.array(1.0),
        ),
        temp_el=cell_variable.CellVariable(
            value=jnp.array(0.0),
            dr=jnp.array(1.0),
        ),
        psi=cell_variable.CellVariable(
            value=jnp.array(0.0),
            dr=jnp.array(1.0),
        ),
        psidot=cell_variable.CellVariable(
            value=jnp.array(0.0),
            dr=jnp.array(1.0),
        ),
        currents=state.Currents.zeros(geo),
        q_face=jnp.array(0.0),
        s_face=jnp.array(0.0),
        Zi=Zi,
        Ai=Ai,
        Zimp=Zimp,
        Aimp=Aimp,
        nref=1e20,
    )
    # pylint: enable=invalid-name
    # pylint: disable=protected-access
    np.testing.assert_allclose(
        physics._calculate_weighted_Zeff(core_profiles), expected
    )
    # pylint: enable=protected-access


if __name__ == '__main__':
  absltest.main()
