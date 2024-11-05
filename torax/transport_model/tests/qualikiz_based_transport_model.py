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

"""Unit tests for torax.transport_model.qualikiz_transport_model."""
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax.numpy as jnp
from torax import core_profile_setters
from torax import geometry
from torax import state
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import source_models as source_models_lib
from torax.transport_model import qualikiz_based_transport_model
from torax.transport_model import quasilinear_transport_model
from torax.transport_model import runtime_params as runtime_params_lib


def _get_model_inputs(transport: qualikiz_based_transport_model.RuntimeParams):
  """Returns the model inputs for testing."""
  runtime_params = general_runtime_params.GeneralRuntimeParams()
  geo = geometry.build_circular_geometry()
  source_models_builder = source_models_lib.SourceModelsBuilder()
  source_models = source_models_builder()
  dynamic_runtime_params_slice = (
      runtime_params_slice.DynamicRuntimeParamsSliceProvider(
          runtime_params=runtime_params,
          transport=transport,
          sources=source_models_builder.runtime_params,
          torax_mesh=geo.torax_mesh,
      )(
          t=runtime_params.numerics.t_initial,
      )
  )
  core_profiles = core_profile_setters.initial_core_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      geo=geo,
      source_models=source_models,
  )
  return dynamic_runtime_params_slice, geo, core_profiles


class QualikizTransportModelTest(parameterized.TestCase):
  """Unit tests for the `torax.transport_model.qualikiz_transport_model` module."""

  def test_qualikiz_based_transport_model_output_shapes(self):
    """Tests that the core transport output has the right shapes."""
    transport = qualikiz_based_transport_model.RuntimeParams(
        coll_mult=1.0,
        avoid_big_negative_s=True,
        q_sawtooth_proxy=True,
        **runtime_params_lib.RuntimeParams()
    )
    transport_model = FakeQualikizBasedTransportModel()
    dynamic_runtime_params_slice, geo, core_profiles = _get_model_inputs(
        transport
    )

    core_transport = transport_model(
        dynamic_runtime_params_slice, geo, core_profiles
    )
    expected_shape = geo.rho_face_norm.shape
    self.assertEqual(core_transport.chi_face_ion.shape, expected_shape)
    self.assertEqual(core_transport.chi_face_el.shape, expected_shape)
    self.assertEqual(core_transport.d_face_el.shape, expected_shape)
    self.assertEqual(core_transport.v_face_el.shape, expected_shape)

  def test_qualikiz_based_transport_model_prepare_qualikiz_inputs_shapes(self):
    """Tests that the qualikiz inputs have the expected shapes."""
    transport = qualikiz_based_transport_model.RuntimeParams(
        coll_mult=1.0,
        avoid_big_negative_s=True,
        q_sawtooth_proxy=True,
        smag_alpha_correction=True,
    )
    dynamic_runtime_params_slice, geo, core_profiles = _get_model_inputs(
        transport
    )
    transport_model = FakeQualikizBasedTransportModel()
    assert isinstance(
        dynamic_runtime_params_slice.transport,
        qualikiz_based_transport_model.DynamicRuntimeParams,
    )
    qualikiz_inputs = transport_model.prepare_qualikiz_inputs(
        Zeff_face=dynamic_runtime_params_slice.plasma_composition.Zeff_face,
        nref=dynamic_runtime_params_slice.numerics.nref,
        q_correction_factor=dynamic_runtime_params_slice.numerics.q_correction_factor,
        transport=dynamic_runtime_params_slice.transport,
        geo=geo,
        core_profiles=core_profiles,
    )

    # 1D array qualikiz_inputs
    vector_keys = [
        'Zeff_face',
        'Ati',
        'Ate',
        'Ane',
        'Ani0',
        'Ani1',
        'q',
        'smag',
        'x',
        'Ti_Te',
        'log_nu_star_face',
        'normni',
        'chiGB',
    ]
    scalar_keys = ['Rmaj', 'Rmin']
    expected_vector_length = geo.rho_face_norm.shape[0]
    for key in vector_keys:
      self.assertEqual(
          getattr(qualikiz_inputs, key).shape, (expected_vector_length,)
      )
    for key in scalar_keys:
      self.assertEqual(getattr(qualikiz_inputs, key).shape, ())


class FakeQualikizBasedTransportModel(
    qualikiz_based_transport_model.QualikizBasedTransportModel
):
  """Fake QualikizBasedTransportModel for testing purposes."""

  def __init__(self):
    super().__init__()
    self._frozen = True

  # pylint: disable=invalid-name
  def prepare_qualikiz_inputs(
      self,
      Zeff_face: chex.Array,
      nref: chex.Numeric,
      q_correction_factor: chex.Numeric,
      transport: qualikiz_based_transport_model.DynamicRuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> qualikiz_based_transport_model.QualikizInputs:
    """Exposing prepare_qualikiz_inputs for testing."""
    return self._prepare_qualikiz_inputs(
        Zeff_face, nref, q_correction_factor, transport, geo, core_profiles
    )
  # pylint: enable=invalid-name

  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> state.CoreTransport:
    transport = dynamic_runtime_params_slice.transport
    # Assert required for pytype.
    assert isinstance(
        transport,
        qualikiz_based_transport_model.DynamicRuntimeParams,
    )
    qualikiz_inputs = self._prepare_qualikiz_inputs(
        Zeff_face=dynamic_runtime_params_slice.plasma_composition.Zeff_face,
        nref=dynamic_runtime_params_slice.numerics.nref,
        q_correction_factor=dynamic_runtime_params_slice.numerics.q_correction_factor,
        transport=dynamic_runtime_params_slice.transport,
        geo=geo,
        core_profiles=core_profiles,
    )
    # Assert required for pytype.
    assert isinstance(
        transport,
        quasilinear_transport_model.DynamicRuntimeParams,
    )
    return self._make_core_transport(
        qi=jnp.ones(geo.rho_face_norm.shape) * 0.4,
        qe=jnp.ones(geo.rho_face_norm.shape) * 0.5,
        pfe=jnp.ones(geo.rho_face_norm.shape) * 1.6,
        quasilinear_inputs=qualikiz_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
    )


if __name__ == '__main__':
  absltest.main()