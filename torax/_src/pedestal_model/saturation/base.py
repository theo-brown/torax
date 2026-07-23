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

"""Base class for pedestal saturation models."""

import abc
import dataclasses
import typing

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry

if typing.TYPE_CHECKING:
  from torax._src.pedestal_model import pedestal_model_output


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class SaturationFraction:
  """Per-channel saturation fraction from a saturation model.

  Each value is bounded in (0, 1), mirroring the formation model's H-mode
  fraction: 0 far below the limit the saturation model regulates towards, 1
  at or beyond it (see `SaturationModel`).

  Attributes:
    chi_i_saturation_fraction: Saturation fraction of the ion heat channel.
    chi_e_saturation_fraction: Saturation fraction of the electron heat
      channel.
    D_e_saturation_fraction: Saturation fraction of the particle diffusivity
      channel.
  """

  chi_i_saturation_fraction: array_typing.FloatScalar
  chi_e_saturation_fraction: array_typing.FloatScalar
  D_e_saturation_fraction: array_typing.FloatScalar

  @classmethod
  def default(cls):
    return cls(
        chi_i_saturation_fraction=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        chi_e_saturation_fraction=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        D_e_saturation_fraction=jnp.array(0.0, dtype=jax_utils.get_dtype()),
    )


@dataclasses.dataclass(frozen=True, eq=False)
class SaturationModel(static_dataclass.StaticDataclass, abc.ABC):
  """Base class for pedestal saturation models.

  A saturation model senses how close the pedestal is to the limit it
  regulates towards (a prescribed pedestal-top target, a stability
  boundary, ...) and returns a bounded (0, 1) saturation fraction per
  transport channel, mirroring how a formation model returns the H-mode
  fraction.
  """

  @abc.abstractmethod
  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_output: "pedestal_model_output.PedestalModelOutput",
  ) -> SaturationFraction:
    """Calculates the per-channel saturation fraction.

    Args:
      runtime_params: Runtime parameters.
      geo: Geometry.
      core_profiles: Core profiles.
      pedestal_output: Output from the pedestal model implementation.

    Returns:
      Per-channel saturation fraction values in (0, 1): 0 far below the
      limit being regulated towards, 1 at or beyond it.
    """
