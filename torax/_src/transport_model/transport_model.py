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

"""The TransportModel abstract base class.

The transport model calculates heat and particle turbulent transport
coefficients.
"""

import abc
import dataclasses
import functools
from typing import Final, Mapping, Sequence

import immutabledict
import jax
from jax import numpy as jnp
from torax._src import constants
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.transport_model import enums
from torax._src.transport_model import runtime_params as transport_runtime_params_lib

# pylint: disable=invalid-name

# Map main channels to their sub-channels (if any) and disable flags
# TODO(b/434175938): Upgrade TransportModel to encapsulate this structure.
CHANNEL_CONFIG_STRUCT: Final[Mapping[str, dict[str, Sequence[str] | str]]] = (
    immutabledict.immutabledict({
        'chi_face_ion': {
            'sub_channels': [
                'chi_face_ion_bohm',
                'chi_face_ion_gyrobohm',
                'chi_face_ion_itg',
                'chi_face_ion_tem',
            ],
            'disable_flag': 'disable_chi_i',
        },
        'chi_face_el': {
            'sub_channels': [
                'chi_face_el_bohm',
                'chi_face_el_gyrobohm',
                'chi_face_el_itg',
                'chi_face_el_tem',
                'chi_face_el_etg',
            ],
            'disable_flag': 'disable_chi_e',
        },
        'd_face_el': {
            'sub_channels': ['d_face_el_itg', 'd_face_el_tem'],
            'disable_flag': 'disable_D_e',
        },
        'v_face_el': {
            'sub_channels': ['v_face_el_itg', 'v_face_el_tem'],
            'disable_flag': 'disable_V_e',
        },
    })
)


def _softplus(x: jax.Array, width: jax.Array | float) -> jax.Array:
  """Numerically stable `width * log(1 + exp(x / width))`.

  This is a smooth approximation to `max(x, 0)` which converges to it as
  `width -> 0`. It is strictly positive and strictly increasing everywhere, so
  its derivative (a sigmoid) is never exactly zero.

  Args:
    x: Input array.
    width: Scale over which the transition from 0 to `x` takes place. Must be
      strictly positive.

  Returns:
    The smoothed `max(x, 0)`.
  """
  return width * jax.nn.softplus(x / width)


def soft_clip(
    x: jax.Array,
    lower: jax.Array | float,
    upper: jax.Array | float,
    lower_width: jax.Array | float,
    upper_width: jax.Array | float,
) -> jax.Array:
  """Smooth, strictly monotonic version of `jnp.clip`.

  Unlike `jnp.clip`, the output is an infinitely differentiable and strictly
  increasing function of `x`, so `d(output)/dx` is never exactly zero. Instead
  of collapsing to zero at the bound, the derivative decays smoothly like a
  sigmoid over the width of the respective transition region. This keeps a
  usable (if small) gradient signal for saturated points, which matters for
  gradient-based optimisation and sensitivity analysis.

  The bounds are still respected: the output is strictly within
  `(lower, upper)` provided the widths are small compared to `upper - lower`.
  See `soft_clip_widths` for how the widths are chosen from a single
  dimensionless softness parameter.

  Args:
    x: Input array.
    lower: Lower bound.
    upper: Upper bound.
    lower_width: Width of the transition region around `lower`. Must be
      strictly positive.
    upper_width: Width of the transition region around `upper`. Must be
      strictly positive.

  Returns:
    The softly clipped array. Converges to `jnp.clip(x, lower, upper)` as both
    widths go to zero.
  """
  # Smooth `max(x, lower)`, then smooth `min(., upper)`. This matches the
  # ordering of `jnp.clip`, which applies the lower bound first.
  x = lower + _softplus(x - lower, lower_width)
  return upper - _softplus(upper - x, upper_width)


# Relative floor on the soft-clip transition widths. Only relevant for the
# degenerate case of a bound at exactly zero, where it makes that bound behave
# as a hard bound instead of producing a division by zero.
_MIN_CLIP_WIDTH_FRACTION: Final[float] = 1e-9


def soft_clip_widths(
    lower: jax.Array | float,
    upper: jax.Array | float,
    softness: float,
) -> tuple[jax.Array, jax.Array]:
  """Derives soft clip transition widths from a dimensionless softness.

  The width of each transition region is `softness * |bound|`, i.e. relative to
  the magnitude of the bound it applies to. This makes `softness` a single
  dimensionless knob which behaves sensibly across the chi, D_e and V_e
  channels, which have different units and very different magnitudes, and it
  keeps a soft bound near zero from leaking a large offset into the
  unsaturated region.

  The width is additionally capped at `softness * (upper - lower)` so that a
  bound with a large magnitude but a narrow permitted range (e.g.
  `chi_min=99, chi_max=100`) cannot produce a transition wider than the range
  itself, which would make the two bounds mutually inconsistent.

  Args:
    lower: Lower bound.
    upper: Upper bound.
    softness: Transition width as a fraction of the bound magnitude.

  Returns:
    A tuple of (lower_width, upper_width).
  """
  span = upper - lower

  def width(bound):
    return jnp.maximum(
        softness * jnp.minimum(jnp.abs(bound), span),
        _MIN_CLIP_WIDTH_FRACTION * span,
    )

  return width(lower), width(upper)


def apply_bounds(
    x: jax.Array,
    lower: jax.Array | float,
    upper: jax.Array | float,
    clip_mode: enums.ClipMode,
    softness: float,
) -> jax.Array:
  """Constrains `x` to `[lower, upper]` using the configured clipping mode.

  Args:
    x: Input array.
    lower: Lower bound.
    upper: Upper bound.
    clip_mode: Whether to use a hard (`jnp.clip`) or soft (smooth) bound.
    softness: Only used for `ClipMode.SOFT`. See `soft_clip_widths`.

  Returns:
    The bounded array.
  """
  match clip_mode:
    case enums.ClipMode.HARD:
      return jnp.clip(x, lower, upper)
    case enums.ClipMode.SOFT:
      lower_width, upper_width = soft_clip_widths(lower, upper, softness)
      return soft_clip(x, lower, upper, lower_width, upper_width)
    case _:
      raise ValueError(f'Unknown clip mode: {clip_mode}')


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TurbulentTransport:
  """Turbulent transport coefficients calculated by a transport model.

  Attributes:
    chi_face_ion: Ion heat conductivity, on the face grid.
    chi_face_el: Electron heat conductivity, on the face grid.
    d_face_el: Diffusivity of electron density, on the face grid.
    v_face_el: Convection strength of electron density, on the face grid.
    chi_face_el_bohm: (Optional) Bohm contribution for electron heat
      conductivity.
    chi_face_el_gyrobohm: (Optional) GyroBohm contribution for electron heat
      conductivity.
    chi_face_ion_bohm: (Optional) Bohm contribution for ion heat conductivity.
    chi_face_ion_gyrobohm: (Optional) GyroBohm contribution for ion heat
      conductivity.
    chi_face_ion_itg: (Optional) ITG contribution for ion heat conductivity.
    chi_face_ion_tem: (Optional) TEM contribution for ion heat conductivity.
    chi_face_el_itg: (Optional) ITG contribution for electron heat conductivity.
    chi_face_el_tem: (Optional) TEM contribution for electron heat conductivity.
    chi_face_el_etg: (Optional) ETG contribution for electron heat conductivity.
    d_face_el_itg: (Optional) ITG contribution for electron diffusivity.
    d_face_el_tem: (Optional) TEM contribution for electron diffusivity.
    v_face_el_itg: (Optional) ITG contribution for electron convection.
    v_face_el_tem: (Optional) TEM contribution for electron convection.
  """

  chi_face_ion: jax.Array
  chi_face_el: jax.Array
  d_face_el: jax.Array
  v_face_el: jax.Array
  chi_face_el_bohm: jax.Array | None = None
  chi_face_el_gyrobohm: jax.Array | None = None
  chi_face_ion_bohm: jax.Array | None = None
  chi_face_ion_gyrobohm: jax.Array | None = None
  chi_face_ion_itg: jax.Array | None = None
  chi_face_ion_tem: jax.Array | None = None
  chi_face_el_itg: jax.Array | None = None
  chi_face_el_tem: jax.Array | None = None
  chi_face_el_etg: jax.Array | None = None
  d_face_el_itg: jax.Array | None = None
  d_face_el_tem: jax.Array | None = None
  v_face_el_itg: jax.Array | None = None
  v_face_el_tem: jax.Array | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class TransportModel(static_dataclass.StaticDataclass, abc.ABC):
  """Calculates various coefficients related to heat and particle transport."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
  ) -> TurbulentTransport:
    transport_runtime_params = runtime_params.transport

    # Calculate the transport coefficients
    transport_coeffs = self.call_implementation(
        transport_runtime_params,
        runtime_params,
        geo,
        core_profiles,
        pedestal_model_output,
    )

    # Apply masking to selectively enable/disable specific channels
    transport_coeffs = self.zero_out_disabled_channels(
        transport_runtime_params, transport_coeffs
    )

    # Restrict the model to operating in its permissible rho domain
    transport_coeffs = self._apply_domain_restriction(
        transport_runtime_params,
        runtime_params,
        geo,
        transport_coeffs,
        pedestal_model_output,
    )

    # Apply min/max clipping
    transport_coeffs = self._apply_clipping(
        transport_runtime_params,
        transport_coeffs,
    )

    # Apply inner and outer transport patch
    transport_coeffs = self._apply_transport_patches(
        transport_runtime_params,
        runtime_params,
        geo,
        transport_coeffs,
    )

    transport_coeffs = self._smooth_coeffs(
        runtime_params,
        geo,
        transport_coeffs,
        pedestal_model_output,
    )

    return transport_coeffs

  @abc.abstractmethod
  def call_implementation(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
  ) -> TurbulentTransport:
    pass

  def zero_out_disabled_channels(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      transport_coeffs: TurbulentTransport,
  ) -> TurbulentTransport:
    """Sets coefficients to zero for channels that are disabled."""
    to_replace = {}

    for channel_name, config in CHANNEL_CONFIG_STRUCT.items():
      disable_flag = getattr(transport_runtime_params, config['disable_flag'])  # pyrefly: ignore[bad-argument-type]

      # Handle main channel
      val = getattr(transport_coeffs, channel_name)
      to_replace[channel_name] = jnp.where(disable_flag, 0.0, val)

      # Handle sub-channels
      for sub_channel in config['sub_channels']:
        sub_value = getattr(transport_coeffs, sub_channel)
        if sub_value is not None:
          sub_value = jnp.where(disable_flag, 0.0, sub_value)
        to_replace[sub_channel] = sub_value

    return dataclasses.replace(transport_coeffs, **to_replace)

  def _apply_domain_restriction(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      transport_coeffs: TurbulentTransport,
      pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
  ) -> TurbulentTransport:
    """Sets transport coefficients to zero outside the model's domain."""
    active_mask = compute_core_domain_mask(
        transport_runtime_params, runtime_params, geo, pedestal_model_output
    )

    coeffs_dict = dataclasses.asdict(transport_coeffs)
    to_replace = {}

    for channel_name, config in CHANNEL_CONFIG_STRUCT.items():
      # Mask main channel
      val = coeffs_dict[channel_name]
      to_replace[channel_name] = jnp.where(active_mask, val, 0.0)  # pyrefly: ignore[bad-argument-type]

      # Mask sub-channels
      for sub_channel in config['sub_channels']:
        sub_val = coeffs_dict[sub_channel]
        if sub_val is not None:
          to_replace[sub_channel] = jnp.where(active_mask, sub_val, 0.0)

    return dataclasses.replace(transport_coeffs, **to_replace)

  def _apply_clipping(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      transport_coeffs: TurbulentTransport,
  ) -> TurbulentTransport:
    """Applies min/max clipping to transport coefficients for PDE stability."""
    clip = functools.partial(
        apply_bounds,
        clip_mode=transport_runtime_params.clip_mode,
        softness=transport_runtime_params.clip_softness,
    )
    chi_face_ion = clip(
        transport_coeffs.chi_face_ion,
        transport_runtime_params.chi_min,
        transport_runtime_params.chi_max,
    )
    chi_face_el = clip(
        transport_coeffs.chi_face_el,
        transport_runtime_params.chi_min,
        transport_runtime_params.chi_max,
    )
    d_face_el = clip(
        transport_coeffs.d_face_el,
        transport_runtime_params.D_e_min,
        transport_runtime_params.D_e_max,
    )
    v_face_el = clip(
        transport_coeffs.v_face_el,
        transport_runtime_params.V_e_min,
        transport_runtime_params.V_e_max,
    )

    return dataclasses.replace(
        transport_coeffs,
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )

  def _apply_transport_patches(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      transport_coeffs: TurbulentTransport,
  ) -> TurbulentTransport:
    """Applies inner and outer transport patches to transport coefficients."""
    consts = constants.CONSTANTS

    # Apply inner and outer patch constant transport coefficients. rho_inner and
    # rho_outer are shifted by consts.eps (1e-7) to avoid ambiguities if their
    # values are close to and geo.rho_face_norm values.
    chi_face_ion = jnp.where(
        jnp.logical_and(
            transport_runtime_params.apply_inner_patch,
            geo.rho_face_norm < transport_runtime_params.rho_inner + consts.eps,
        ),
        transport_runtime_params.chi_i_inner,
        transport_coeffs.chi_face_ion,
    )
    chi_face_el = jnp.where(
        jnp.logical_and(
            transport_runtime_params.apply_inner_patch,
            geo.rho_face_norm < transport_runtime_params.rho_inner + consts.eps,
        ),
        transport_runtime_params.chi_e_inner,
        transport_coeffs.chi_face_el,
    )
    d_face_el = jnp.where(
        jnp.logical_and(
            transport_runtime_params.apply_inner_patch,
            geo.rho_face_norm < transport_runtime_params.rho_inner + consts.eps,
        ),
        transport_runtime_params.D_e_inner,
        transport_coeffs.d_face_el,
    )
    v_face_el = jnp.where(
        jnp.logical_and(
            transport_runtime_params.apply_inner_patch,
            geo.rho_face_norm < transport_runtime_params.rho_inner + consts.eps,
        ),
        transport_runtime_params.V_e_inner,
        transport_coeffs.v_face_el,
    )

    # Apply outer patch constant transport coefficients.
    # Due to Pereverzev-Corrigan convection, it is required
    # for the convection modes to be 'ghost' to avoid numerical instability
    chi_face_ion = jnp.where(
        jnp.logical_and(
            jnp.logical_and(
                transport_runtime_params.apply_outer_patch,
                jnp.logical_not(runtime_params.pedestal.set_pedestal),
            ),
            geo.rho_face_norm > transport_runtime_params.rho_outer - consts.eps,
        ),
        transport_runtime_params.chi_i_outer,
        chi_face_ion,
    )
    chi_face_el = jnp.where(
        jnp.logical_and(
            jnp.logical_and(
                transport_runtime_params.apply_outer_patch,
                jnp.logical_not(runtime_params.pedestal.set_pedestal),
            ),
            geo.rho_face_norm > transport_runtime_params.rho_outer - consts.eps,
        ),
        transport_runtime_params.chi_e_outer,
        chi_face_el,
    )
    d_face_el = jnp.where(
        jnp.logical_and(
            jnp.logical_and(
                transport_runtime_params.apply_outer_patch,
                jnp.logical_not(runtime_params.pedestal.set_pedestal),
            ),
            geo.rho_face_norm > transport_runtime_params.rho_outer - consts.eps,
        ),
        transport_runtime_params.D_e_outer,
        d_face_el,
    )
    v_face_el = jnp.where(
        jnp.logical_and(
            jnp.logical_and(
                transport_runtime_params.apply_outer_patch,
                jnp.logical_not(runtime_params.pedestal.set_pedestal),
            ),
            geo.rho_face_norm > transport_runtime_params.rho_outer - consts.eps,
        ),
        transport_runtime_params.V_e_outer,
        v_face_el,
    )

    return dataclasses.replace(
        transport_coeffs,
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )

  def _smooth_coeffs(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      transport_coeffs: TurbulentTransport,
      pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
  ) -> TurbulentTransport:
    """Gaussian smoothing of turbulent transport coefficients."""
    smoothing_matrix = _build_smoothing_matrix(
        runtime_params,
        geo,
        pedestal_model_output,
    )

    # Iterate over fields of the CoreTransport dataclass.
    # Ignore optional fields that are made all zero in post_init.
    def smooth_single_coeff(coeff):
      return jax.lax.cond(
          jnp.all(coeff == 0.0),
          lambda: coeff,
          lambda: jnp.dot(smoothing_matrix, coeff),
      )

    return jax.tree_util.tree_map(smooth_single_coeff, transport_coeffs)


def _build_smoothing_matrix(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
) -> jax.Array:
  """Builds a smoothing matrix for the turbulent transport model.

  Uses a Gaussian kernel of HWHM defined in the transport config.

  Args:
    runtime_params: Input runtime parameters of the simulation.
    geo: Geometry of the torus.
    pedestal_model_output: Output of the pedestal model.

  Returns:
    kernel: A smoothing matrix for convolution with the transport outputs.
  """

  # To reduce the range of the convolution, weights under lower_cutoff are
  # clipped to zero
  lower_cutoff = 0.01

  # used for eps, small number to avoid divisions by zero for sigma = 0
  consts = constants.CONSTANTS

  # 1. Kernel matrix
  kernel = jnp.exp(
      -jnp.log(2)
      * (geo.rho_face_norm[:, jnp.newaxis] - geo.rho_face_norm) ** 2
      / (runtime_params.transport.smoothing_width**2 + consts.eps)
  )

  # 2. Masking: we do not want transport coefficients calculated in pedestal
  # region or in inner and outer transport patch regions to impact
  # transport_model calculated coefficients
  if (
      runtime_params.pedestal.mode
      == pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION
  ):
    # If in INTERNAL_BOUNDARY_CONDITION mode: if set_pedestal is True, mask
    # according to the pedestal top. Otherwise, mask according to the outer
    # patch, if set.
    mask_outer_edge = jnp.where(
        runtime_params.pedestal.set_pedestal,
        pedestal_model_output.rho_norm_ped_top - consts.eps,
        jnp.where(
            runtime_params.transport.apply_outer_patch,
            runtime_params.transport.rho_outer - consts.eps,
            jnp.inf,
        ),
    )
  else:
    # If in ADAPTIVE_TRANSPORT mode, only mask according to the outer patch.
    mask_outer_edge = jnp.where(
        runtime_params.transport.apply_outer_patch,
        runtime_params.transport.rho_outer - consts.eps,
        jnp.inf,
    )

  mask_inner_edge = jax.lax.cond(
      runtime_params.transport.apply_inner_patch,
      lambda: runtime_params.transport.rho_inner + consts.eps,
      lambda: -consts.eps,
  )

  mask = jnp.where(
      jnp.logical_or(
          runtime_params.transport.smooth_everywhere,
          jnp.logical_and(
              geo.rho_face_norm > mask_inner_edge,
              geo.rho_face_norm < mask_outer_edge,
          ),
      ),
      1.0,
      0.0,
  )

  # remove impact of smoothing on inner and outer patch, or pedestal zone

  # first zero out all rows corresponding to grid points not to be impacted
  diag_mask = jnp.diag(mask)
  kernel = jnp.dot(diag_mask, kernel)
  # now zero out all columns corresponding to grid points not to be impacted,
  # such that they don't impact the smoothing of the other grid points
  num_rows = len(mask)
  mask_mat = jnp.tile(mask, (num_rows, 1))
  kernel *= mask_mat
  # now restore identity to the zero rows, such that smoothing is a no-op for
  # on the grid points where it shouldn't impact
  zero_row_mask = jnp.all(kernel == 0, axis=1)
  kernel = jnp.where(
      zero_row_mask[:, jnp.newaxis], jnp.eye(kernel.shape[0]), kernel
  )

  # 3. Normalization
  row_sums = jnp.sum(kernel, axis=1)
  kernel /= row_sums[:, jnp.newaxis]

  # 4. Remove small numbers
  kernel = jnp.where(kernel < lower_cutoff, 0.0, kernel)

  # 5. Final Normalization following removal of small numbers
  row_sums = jnp.sum(kernel, axis=1)
  kernel /= row_sums[:, jnp.newaxis]
  return kernel


def compute_core_domain_mask(
    transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
) -> jax.Array:
  """Calculates the active domain mask for core transport models.

  Args:
    transport_runtime_params: Runtime parameters for the transport model.
    runtime_params: Runtime parameters for the simulation.
    geo: Geometry of the torus.
    pedestal_model_output: Output of the pedestal model.

  Returns:
    active_mask: A boolean array indicating the active domain.
  """
  # Active range is rho_min < rho <= rho_max
  # (AND rho <= rho_norm_ped_top, if pedestal is in INTERNAL_BOUNDARY_CONDITION
  # mode)
  active_mask = (geo.rho_face_norm > transport_runtime_params.rho_min) & (
      geo.rho_face_norm <= transport_runtime_params.rho_max
  )
  if (
      runtime_params.pedestal.mode
      == pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION
  ):
    active_mask = active_mask & (
        jnp.logical_not(runtime_params.pedestal.set_pedestal)
        | (geo.rho_face_norm <= pedestal_model_output.rho_norm_ped_top)
    )

  # Special case: if rho_min is 0, lower bound of active range is the first
  # grid point.
  active_mask = (
      jnp.asarray(active_mask).at[0].set(transport_runtime_params.rho_min == 0)
  )
  return active_mask
