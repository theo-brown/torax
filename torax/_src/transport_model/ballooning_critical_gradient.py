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

"""The BallooningCriticalGradientTransportModel class.

A pedestal/edge transport model in which transport is limited by the ideal
ballooning normalized pressure gradient. Intended as a physics-based
alternative to enforcing a prescribed pedestal height: the edge pressure
gradient is clamped near the s-alpha ballooning stability boundary, so the
pedestal height emerges from the stability limit, the barrier width, and the
edge boundary conditions, rather than from a user-specified target.
"""
import dataclasses

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.physics import formulas
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import transport_model


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(transport_runtime_params_lib.RuntimeParams):
  """Runtime params for the ballooning critical gradient transport model."""

  alpha_crit_multiplier: array_typing.FloatScalar
  s_min: array_typing.FloatScalar
  chi_ceiling: array_typing.FloatScalar
  alpha_width: array_typing.FloatScalar
  chi_e_i_ratio: array_typing.FloatScalar
  chi_D_ratio: array_typing.FloatScalar
  chi_floor: array_typing.FloatScalar
  D_e_floor: array_typing.FloatScalar


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class BallooningCriticalGradientTransportModel(transport_model.TransportModel):
  """Pressure-gradient-limited transport for the pedestal/edge region."""

  def call_implementation(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
  ) -> transport_model.TurbulentTransport:
    r"""Calculates transport from the ballooning normalized pressure gradient.

    The normalized pressure gradient is the standard s-alpha model parameter

    .. math::
        \alpha = -\frac{2 \mu_0 q^2 R_0}{B_0^2} \frac{dp}{dr}

    and the critical value is approximated from the s-alpha ideal ballooning
    stability boundary as :math:`\alpha_{crit} = c_\alpha \max(s, s_{min})`
    (Connor, Hastie & Taylor first-stability branch, with the shear floor
    avoiding a vanishing limit near :math:`s = 0`).

    The transport response is a bounded, smooth (C-infinity) sigmoid:

    .. math::
        \chi_i = \chi_{floor} + \chi_{ceiling}\,
        \sigma\!\left(\frac{\alpha - \alpha_{crit}}{w_\alpha}\right)

    which saturates at :math:`\chi_{ceiling}` above the boundary (turbulent
    transport is physically bounded) and decays to the residual barrier
    transport :math:`\chi_{floor}` below it. Because :math:`\alpha` involves
    the total pressure gradient, temperature and density channels are limited
    jointly, and the particle diffusivity shares the MHD-driven term via
    ``chi_D_ratio``. The finite width and ceiling keep the implicit solver
    residual smooth with physically-calibrated stiffness.

    Args:
      transport_runtime_params: Input runtime parameters for this transport
        model at the current time.
      runtime_params: Input runtime parameters at the current time.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      pedestal_model_output: Output of the pedestal model.

    Returns:
      coeffs: The transport coefficients.
    """
    del pedestal_model_output  # Unused: the limit is stability-based.

    # Required for pytype
    assert isinstance(transport_runtime_params, RuntimeParams)

    s = core_profiles.s_face
    alpha = formulas.calc_ballooning_alpha_face(geo, core_profiles)

    alpha_crit = transport_runtime_params.alpha_crit_multiplier * jnp.maximum(
        s, transport_runtime_params.s_min
    )

    # Bounded smooth response: floor below the boundary, ceiling above it.
    drive = transport_runtime_params.chi_ceiling * jax.nn.sigmoid(
        (alpha - alpha_crit) / transport_runtime_params.alpha_width
    )

    chi_face_ion = transport_runtime_params.chi_floor + drive
    chi_face_el = (
        transport_runtime_params.chi_floor
        + drive / transport_runtime_params.chi_e_i_ratio
    )
    d_face_el = (
        transport_runtime_params.D_e_floor
        + drive / transport_runtime_params.chi_D_ratio
    )
    v_face_el = jnp.zeros_like(d_face_el)

    return transport_model.TurbulentTransport(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )
