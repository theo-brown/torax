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
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output


@dataclasses.dataclass(frozen=True, eq=False)
class SaturationModel(static_dataclass.StaticDataclass, abc.ABC):
  """Base class for pedestal saturation models.

  A saturation model senses how close the pedestal is to the limit it
  regulates towards (a prescribed pedestal-top target, a stability
  boundary, ...) and returns one dimensionless proximity signal per transport
  channel. The signals are mapped to barrier openness by the shared bounded
  response in `PedestalModel.compute_barrier_state`; saturation models choose
  the signal, not the response shape.
  """

  @abc.abstractmethod
  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_output: pedestal_model_output.PedestalModelOutput,
  ) -> pedestal_model_output.BarrierSignals:
    """Calculates the per-channel proximity-to-limit signals.

    Args:
      runtime_params: Runtime parameters.
      geo: Geometry.
      core_profiles: Core profiles.
      pedestal_output: Output from the pedestal model implementation.

    Returns:
      Per-channel dimensionless signals, normalized so that a signal is 0 at
      the limit, negative below it and positive beyond it.
    """
