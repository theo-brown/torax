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
"""TGLFNN-ukaea transport model."""

from __future__ import annotations

import dataclasses
from typing import Literal

from fusion_surrogates.tglfnn_ukaea import tglfnn_ukaea_model
import jax
import jax.numpy as jnp
import tglfnn_ukaea as tglfnn_ukaea_lib
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import transport_model as transport_model_lib


# Inputs whose training-space bounds are recorded in log10 in the checkpoint
# but which are fed to the network in linear units.
_LOG10_SAMPLED_INPUTS = ('XNUE', 'BETAE')


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(tglf_based_transport_model.RuntimeParams):
  """Runtime parameters for the TGLFNN-ukaea transport model.

  Attributes:
    clip_inputs: Whether to clip the network inputs to the training-set
      bounds recorded in the model checkpoint before inference, avoiding
      uncontrolled extrapolation outside the training hypercube.
    clip_margin: Margin for the clipping, with the same semantics as the
      qlknn model's `clip_margin`: bounds are shrunk towards zero by
      `|bound| * (1 - clip_margin)`.
  """

  clip_inputs: bool
  clip_margin: float


def clip_inputs_to_bounds(
    feature_tensor: jax.Array,
    clip_margin: float,
    input_bounds: tuple[tuple[float, float], ...],
) -> jax.Array:
  """Clips network inputs to the training-set bounds + optional margin."""
  bounds = jnp.asarray(input_bounds)
  min_vals = bounds[:, 0]
  max_vals = bounds[:, 1]
  min_vals += jnp.abs(min_vals) * (1 - clip_margin)
  max_vals -= jnp.abs(max_vals) * (1 - clip_margin)
  return jnp.clip(feature_tensor, min_vals, max_vals)


@dataclasses.dataclass(frozen=True, eq=False)
class TGLFNNukaeaTransportModel(
    tglf_based_transport_model.TGLFBasedTransportModel
):
  """TGLFNN-ukaea transport model."""

  machine: Literal["step", "multimachine", "multimachine_student"]

  # The following fields are set by __post_init__
  model: tglfnn_ukaea_model.TGLFNNukaeaModel = dataclasses.field(init=False)
  # Training-set bounds per network input, in the network's input order and
  # in linear units, read from the checkpoint's recorded parameter space.
  input_bounds: tuple[tuple[float, float], ...] = dataclasses.field(init=False)

  def __post_init__(self):
    # Load weights in post-init, so that they are not reloaded on every call.
    # Use __setattr__ as this is a frozen dataclass, so we can't just do
    # self.model = ...
    object.__setattr__(
        self, "model", tglfnn_ukaea_model.TGLFNNukaeaModel(self.machine)
    )
    checkpoint = tglfnn_ukaea_lib.load(self.machine)
    param_space = checkpoint["config"]["param_space"]
    bounds = []
    for label in checkpoint["input_labels"]:
      low, high = (float(b) for b in param_space[label])
      if label in _LOG10_SAMPLED_INPUTS:
        low, high = 10.0**low, 10.0**high
      bounds.append((low, high))
    object.__setattr__(self, "input_bounds", tuple(bounds))
    super().__post_init__()

  def _make_input_tensor_step(
      self,
      tglf_inputs: tglf_based_transport_model.TGLFInputs,
  ) -> jax.Array:
    # Note: TGLFNN-ukaea uses a different definition of the magnetic shear
    # to TGLF. This is not the same as S_HAT_LOC in s-alpha geometry.
    s_hat = (
        tglf_inputs.RMIN_LOC / tglf_inputs.Q_LOC
    ) ** 2 * tglf_inputs.Q_PRIME_LOC
    return jnp.stack(
        [
            tglf_inputs.RLNS_1,
            tglf_inputs.RLTS_1,
            tglf_inputs.RLTS_2,
            tglf_inputs.TAUS_2,
            tglf_inputs.RMIN_LOC,
            tglf_inputs.DRMAJDX_LOC,
            tglf_inputs.Q_LOC,
            s_hat,
            tglf_inputs.XNUE,
            tglf_inputs.KAPPA_LOC,
            tglf_inputs.S_KAPPA_LOC,
            tglf_inputs.DELTA_LOC,
            tglf_inputs.S_DELTA_LOC,
            tglf_inputs.BETAE,
            tglf_inputs.ZEFF,
        ],
        axis=-1,
    )

  def _make_input_tensor_multimachine(
      self,
      tglf_inputs: tglf_based_transport_model.TGLFInputs,
  ) -> jax.Array:
    # Note: TGLFNN-ukaea uses a different definition of the magnetic shear
    # to TGLF. This is not the same as S_HAT_LOC in s-alpha geometry.
    s_hat = (
        tglf_inputs.RMIN_LOC / tglf_inputs.Q_LOC
    ) ** 2 * tglf_inputs.Q_PRIME_LOC

    return jnp.stack(
        [
            tglf_inputs.RLNS_1,
            tglf_inputs.RLTS_1,
            tglf_inputs.RLTS_2,
            tglf_inputs.TAUS_2,
            tglf_inputs.RMIN_LOC,
            tglf_inputs.DRMAJDX_LOC,
            tglf_inputs.Q_LOC,
            s_hat,
            tglf_inputs.XNUE,
            tglf_inputs.KAPPA_LOC,
            tglf_inputs.DELTA_LOC,
            tglf_inputs.ZEFF,
            tglf_inputs.VEXB_SHEAR,
        ],
        axis=-1,
    )

  def _prepare_tglfnn_inputs(
      self,
      tglf_inputs: tglf_based_transport_model.TGLFInputs,
  ) -> jax.Array:
    match self.machine:
      case "step":
        return self._make_input_tensor_step(tglf_inputs)
      case "multimachine" | "multimachine_student":
        # The student is a distillation of the multimachine ensemble and
        # shares its inputs, outputs and normalisation stats.
        return self._make_input_tensor_multimachine(tglf_inputs)
      case _:
        raise ValueError(f"Unsupported machine: {self.machine}")

  def call_implementation(
      self,
      transport: RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    del pedestal_model_output  # unused
    tglf_inputs = self._prepare_tglf_inputs(
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
        poloidal_velocity_multiplier=runtime_params.neoclassical.poloidal_velocity_multiplier,
    )
    tglfnn_inputs = self._prepare_tglfnn_inputs(tglf_inputs)
    tglfnn_inputs = jax.lax.cond(
        transport.clip_inputs,
        lambda: clip_inputs_to_bounds(
            tglfnn_inputs, transport.clip_margin, self.input_bounds
        ),
        lambda: tglfnn_inputs,
    )
    predictions = self.model.predict(tglfnn_inputs)

    # TODO(b/323504363): expose variance outputs
    return self._make_core_transport(
        ion_heat_flux_GB=predictions["efi_gb"][..., 0],
        electron_heat_flux_GB=predictions["efe_gb"][..., 0],
        # TODO(b/323504363): Convert pfi to pfe for multi-ion plasmas
        electron_particle_flux_GB=predictions["pfi_gb"][..., 0],
        tglf_inputs=tglf_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
    )
