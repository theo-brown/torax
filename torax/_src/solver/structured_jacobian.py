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

"""Structured Jacobian of the theta-method residual.

The residual couples neighboring cells only, except through the
post-processing ``T`` of the raw turbulent transport coefficients ``h(x)``
(their smoothing and the scaling of an ADAPTIVE_TRANSPORT pedestal; see
``transport_coefficients_builder.postprocess_turbulent_transport``) and
through a few global quantities ``q(x)`` of the state (see
``calc_coeffs.StateGlobals``: the globals of some sources, a state-dependent
pedestal and the values of the state that whole internal boundary profiles
use). With ``G`` the residual evaluated on injected coefficients and globals,
``R(x) = G(x, T(h(x), q(x)), q(x))`` and

  ``J = dG/dx + dG/dc dT/dh dh/dx + (dG/dq + dG/dc dT/dq) dq/dx``.

``dG/dx`` and ``dh/dx`` are banded and are assembled from a grid-independent
number of colored forward-mode passes (Curtis-Powell-Reid coloring),
``dT/dh`` is a dense matrix per coefficient from one pass per face, and the
last term has rank ``len(q)`` and costs one JVP and one VJP per global
quantity. The bands hold on any radial grid and for any
``numerics.min_rho_norm``, which are only known at run time. The result equals
``jax.jacfwd`` to round-off.
"""

from collections.abc import Callable
import dataclasses
import functools

from absl import logging
import jax
from jax import flatten_util
import jax.numpy as jnp
import numpy as np
from torax._src import jax_utils
from torax._src import models as models_lib
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import updaters
from torax._src.fvm import calc_coeffs
from torax._src.fvm import fvm_conversions
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source as source_lib
from torax._src.transport_model import qualikiz_based_transport_model
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import transport_coefficients_builder
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib

# With T and q frozen, residual cell i depends on state cells |i - j| <= 2,
# and cells below numerics.min_rho_norm on psi in _AXIS_CELLS cells around the
# first cell beyond it, whose current density they take.
_BANDWIDTH = 2
_AXIS_CELLS = 4
_N_COEFFS = 4  # chi_i, chi_e, D_e, V_e on the face grid.
# Cells left and right of face f that its raw transport coefficients depend
# on. Face values reach one cell either side and the three-point face
# gradients one more to the right, and the magnetic shear uses q on the
# neighboring faces. The E x B shear of rotating models is a face gradient of a
# face-to-cell average of a face value built from these; the inputs of
# TGLF-based models reach one cell further on either side (measured). On a
# uniform grid the third weight of the face gradients vanishes and each reach
# is one cell less to the right, but the grid is only known at run time.
_TRANSPORT_REACH = (2, 2)
_TRANSPORT_REACH_WITH_ROTATION = (3, 4)
_TRANSPORT_REACH_WITH_TGLF_ROTATION = (4, 5)


def jacobian_fn(
    residual_fun: functools.partial,
) -> Callable[[jax.Array], jax.Array]:
  """Returns a jitted x -> dR/dx for `residual_fun`.

  Args:
    residual_fun: `residual_and_loss.theta_method_block_residual` with every
      argument but the state bound by keyword, as `newton_raphson_solve_block`
      builds it. The bound arguments define the physics of the step.

  Returns:
    The Jacobian function, equal to `jax.jacfwd(residual_fun)` to round-off.
  """
  kw = residual_fun.keywords
  models, geo = kw['models'], kw['geo_t_plus_dt']
  runtime_params, names = kw['runtime_params_t_plus_dt'], kw['evolving_names']
  n_cells = int(geo.torax_mesh.nx)
  user_models = _user_defined_models(models, runtime_params)
  if user_models and not jax_utils.errors_enabled():
    logging.warning(
        "jacobian_mode='structured' assumes that these user-defined models"
        ' couple neighboring cells only (a source can declare its global'
        ' quantities with a torax.sources.SplitModelFunction): %s. Set'
        ' TORAX_ERRORS_ENABLED=True to check the Jacobian at every Newton'
        ' iteration.',
        ', '.join(user_models),
    )

  def core_profiles(x):
    x_tuple = fvm_conversions.vec_to_cell_variable_tuple(
        x, kw['core_profiles_t_plus_dt'], names
    )
    return updaters.update_core_profiles_during_step(
        x_tuple,
        runtime_params,
        geo,
        kw['core_profiles_t_plus_dt'],
        prev_core_profiles=kw['core_profiles_t'],
        dt=kw['dt'],
        evolving_names=names,
    )

  def state_globals(x):
    """q(x)."""
    return calc_coeffs.calc_state_globals(
        runtime_params,
        geo,
        core_profiles(x),
        kw['explicit_source_profiles'],
        models,
        kw['pedestal_transition_state'],
    )

  def pedestal_transition_state(q):
    if q.pedestal is None:
      return kw['pedestal_transition_state']
    return dataclasses.replace(
        kw['pedestal_transition_state'], pedestal_model_output=q.pedestal
    )

  def raw_transport(x, q):
    """h(x): the turbulent transport coefficients before post-processing.

    The transport model sees the pedestal only through the location of its
    top, so h does not depend on q differentiably.
    """
    transport = transport_coefficients_builder.calculate_all_transport_coeffs(
        models.transport_model,
        models.neoclassical_models,
        models.internal_boundary_condition_model,
        runtime_params,
        geo,
        core_profiles(x),
        pedestal_transition_state(q),
        postprocess=False,
    )
    return _coeffs_to_vec(transport.turbulent.total)

  def postprocess(h, q):
    """T(h, q)."""
    return _coeffs_to_vec(
        transport_coefficients_builder.postprocess_turbulent_transport(
            models.transport_model,
            runtime_params,
            geo,
            pedestal_transition_state(q),
            _vec_to_coeffs(h),
        )
    )

  def residual(x, c, q):
    """G(x, c, q): the residual with c and q injected."""
    return residual_fun(
        x, turbulent_transport=_vec_to_coeffs(c), state_globals=q
    )

  assemble = functools.partial(
      _assemble,
      residual=residual,
      raw_transport=raw_transport,
      postprocess=postprocess,
      state_globals=state_globals,
      n_cells=n_cells,
      n_channels=len(names),
      h_reach=_transport_reach(runtime_params.transport),
      rho_norm=geo.rho_norm,
      min_rho_norm=runtime_params.numerics.min_rho_norm,
      psi_channel=names.index('psi') if 'psi' in names else None,
      full_residual=residual_fun,
  )
  # Unlike the dense Jacobian in jax_root_finding, not jax.Inline.XLA_LATE:
  # XLA's late inlining of this many instructions can take tens of minutes,
  # depending on the depth of the Python call stack.
  return jax.jit(assemble)


def _assemble(
    x: jax.Array,
    *,
    residual: Callable[
        [jax.Array, jax.Array, calc_coeffs.StateGlobals], jax.Array
    ],
    raw_transport: Callable[[jax.Array, calc_coeffs.StateGlobals], jax.Array],
    postprocess: Callable[[jax.Array, calc_coeffs.StateGlobals], jax.Array],
    state_globals: Callable[[jax.Array], calc_coeffs.StateGlobals],
    n_cells: int,
    n_channels: int,
    h_reach: tuple[int, int],
    rho_norm: jax.Array,
    min_rho_norm: jax.Array,
    psi_channel: int | None,
    full_residual: Callable[[jax.Array], jax.Array],
) -> jax.Array:
  """Assembles dR/dx at x from the linearized factors."""
  n_state, n_faces = n_cells * n_channels, n_cells + 1
  cells = np.arange(n_state) % n_cells
  faces = np.arange(_N_COEFFS * n_faces) % n_faces

  # Column colorings and sparsity patterns of dG/dx and dh/dx.
  colors_x, n_x = _coloring(n_cells, n_channels, 2 * _BANDWIDTH + 1)
  mask_x = np.abs(cells[:, None] - cells[None, :]) <= _BANDWIDTH
  if psi_channel is not None:
    # The current density of the cells below min_rho_norm is that of the first
    # cell at or beyond it, or of the first cell if there is none
    # (psi_calculations._extrapolate_cell_profile_to_axis), which depends on
    # psi from the cell before it to two cells after it (three-point face
    # gradients). These columns, known only at run time, get a color each.
    first = jnp.argmax(rho_norm >= min_rho_norm)
    start = jnp.clip(first - 1, 0, n_cells - _AXIS_CELLS)
    axis = psi_channel * n_cells + start + jnp.arange(_AXIS_CELLS)
    colors_x = jnp.asarray(colors_x).at[axis].set(n_x + jnp.arange(_AXIS_CELLS))
    mask_x = jnp.asarray(mask_x).at[:, axis].set(True)
    n_x += _AXIS_CELLS
  colors_h, n_h = _coloring(n_cells, n_channels, sum(h_reach) + 1)
  mask_h = (cells[None, :] >= faces[:, None] - h_reach[0]) & (
      cells[None, :] <= faces[:, None] + h_reach[1]
  )
  # dG/dc only needs the two parities of each coefficient: residual cell i
  # uses faces i and i + 1.
  colors_c = np.repeat(np.arange(_N_COEFFS), n_faces) * 2 + faces % 2
  n_c = 2 * _N_COEFFS

  def seeds(colors, n):
    return jax.nn.one_hot(colors, n, dtype=x.dtype).T

  def zeros(rows, cols):
    return jnp.zeros((rows, cols), x.dtype)

  # Each factor is linearized once and evaluated on all its seeds in one batch.
  q_tree, q_vjp = jax.vjp(state_globals, x)
  q, unravel = flatten_util.ravel_pytree(q_tree)
  q = q.astype(x.dtype)  # ravel_pytree gives float32 for an empty tree.
  k = q.size
  seeds_q = jnp.eye(k, dtype=x.dtype)
  h, h_lin = jax.linearize(lambda x: raw_transport(x, q_tree), x)
  c, t_lin = jax.linearize(lambda h, q: postprocess(h, unravel(q)), h, q)
  # T acts on each coefficient separately, so one seed per face gives that
  # column of dT/dh for all of them: dt_dh[i, f, g] = dT_i(f) / dh_i(g).
  dt_dh = jax.vmap(lambda s: t_lin(s, jnp.zeros_like(q)), out_axes=1)(
      jnp.tile(jnp.eye(n_faces, dtype=x.dtype), (1, _N_COEFFS))
  ).reshape(_N_COEFFS, n_faces, n_faces)
  dt_dq = jax.vmap(lambda e: t_lin(jnp.zeros_like(h), e))(seeds_q)

  # One batch of JVPs of G, by blocks of (dx, dc, dq) seeds: the compressed
  # dG/dx, the compressed dG/dc, and dG/dq + dG/dc dT/dq.
  _, g_lin = jax.linearize(lambda x, c, q: residual(x, c, unravel(q)), x, c, q)
  blocks = (
      (seeds(colors_x, n_x), zeros(n_x, c.size), zeros(n_x, k)),
      (zeros(n_c, n_state), seeds(colors_c, n_c), zeros(n_c, k)),
      (zeros(k, n_state), dt_dq, seeds_q),
  )
  dx, dc, dq = (jnp.concatenate(block) for block in zip(*blocks))
  compressed = jax.vmap(g_lin)(dx, dc, dq)
  jac = _decompress(compressed[:n_x], colors_x, mask_x)

  # dG/dc dT/dh dh/dx, applied row by row.
  dh_dx = _decompress(jax.vmap(h_lin)(seeds(colors_h, n_h)), colors_h, mask_h)
  dc_dx = dt_dh @ dh_dx.reshape(_N_COEFFS, n_faces, n_state)
  rows = np.arange(n_state)
  for i in range(_N_COEFFS):
    for face in (cells, cells + 1):  # Residual cell i uses faces i and i + 1.
      dg_dc = compressed[n_x + 2 * i + face % 2, rows]
      jac += dg_dc[:, None] * dc_dx[i, face]

  # (dG/dq + dG/dc dT/dq) dq/dx, of rank k.
  if k:
    dq_dx = jax.vmap(lambda v: q_vjp(unravel(v))[0])(seeds_q)
    dr_dq = compressed[n_x + n_c :]
    # A global that the residual does not use can have a non-finite derivative
    # (e.g. the multipliers of an adaptive pedestal that is switched off);
    # forward mode drops it, and so must this product.
    used = jnp.any(dr_dq != 0.0, axis=1, keepdims=True)
    jac += dr_dq.T @ jnp.where(used, dq_dx, 0.0)

  if jax_utils.errors_enabled():
    # A coupling missing from the bands shows up in its row, against a JVP of
    # the full residual along a random probe.
    probe = jnp.asarray(
        np.random.default_rng(0).uniform(1.0, 2.0, n_state), x.dtype
    )
    jv = jax.jvp(full_residual, (x,), (probe,))[1]
    jp = jac @ probe
    tol = jnp.sqrt(jnp.finfo(x.dtype).eps) * (jnp.abs(jac) @ probe)
    jac = jax_utils.error_if(
        jac,
        (jnp.abs(jp - jv) > tol) | (jnp.isfinite(jv) & ~jnp.isfinite(jp)),
        'The structured Jacobian does not match the residual, e.g. because a'
        " model couples distant cells; use jacobian_mode='dense'.",
    )
  return jac


def _user_defined_models(
    models: models_lib.Models,
    runtime_params: runtime_params_lib.RuntimeParams,
) -> list[str]:
  """The models in the residual that are not TORAX's, and so assumed local.

  These are registered transport models and the model functions of implicit
  model-based sources that are not `SplitModelFunction`s. Pedestal models are
  not listed: their output is a global quantity.
  """

  def outside_torax(obj) -> bool:
    if isinstance(obj, functools.partial):
      obj = obj.func
    return not (getattr(obj, '__module__', None) or '').startswith('torax.')

  transport = models.transport_model
  names = [
      f'transport model {name}'
      for name, model in (
          *transport.core_transport_models.items(),
          *transport.pedestal_transport_models.items(),
      )
      if outside_torax(model)
  ]
  for name, source in models.source_models.standard_sources.items():
    params = runtime_params.sources[name]
    if (
        params.mode == sources_runtime_params_lib.Mode.MODEL_BASED
        and not params.is_explicit
        and not isinstance(source.model_func, source_lib.SplitModelFunction)
        and (outside_torax(source) or outside_torax(source.model_func))
    ):
      names.append(f'source {name}')
  return names


def _transport_reach(
    transport_params: transport_runtime_params_lib.RuntimeParams,
) -> tuple[int, int]:
  """Cells left and right of face f that its raw coefficients depend on."""
  params = (
      *transport_params.core_transport_model_params.values(),
      *transport_params.pedestal_transport_model_params.values(),
  )
  if any(
      isinstance(p, tglf_based_transport_model.RuntimeParams) and p.use_rotation
      for p in params
  ):
    return _TRANSPORT_REACH_WITH_TGLF_ROTATION
  if any(
      isinstance(p, qualikiz_based_transport_model.RuntimeParams)
      and p.rotation_mode != qualikiz_based_transport_model.RotationMode.OFF
      for p in params
  ):
    return _TRANSPORT_REACH_WITH_ROTATION
  return _TRANSPORT_REACH


def _coloring(
    n_cells: int, n_channels: int, period: int
) -> tuple[np.ndarray, int]:
  """Colors the channel-major state cyclically with `period` per channel.

  Args:
    n_cells: Number of cells per channel.
    n_channels: Number of channels.
    period: Number of colors per channel: cells of a channel closer than this
      get different colors.

  Returns:
    The color of each unknown and the number of colors.
  """
  channel, cell = np.divmod(np.arange(n_cells * n_channels), n_cells)
  return channel * period + cell % period, n_channels * period


def _decompress(
    compressed: jax.Array,
    colors: jax.typing.ArrayLike,
    mask: jax.typing.ArrayLike,
) -> jax.Array:
  """The Jacobian from its compressed columns (one per color) and sparsity."""
  return jnp.where(mask, compressed[colors].T, 0.0)


def _coeffs_to_vec(coeffs: transport_coeffs_lib.TransportCoeffs) -> jax.Array:
  return jnp.concatenate([
      coeffs.chi_face_ion,
      coeffs.chi_face_el,
      coeffs.d_face_el,
      coeffs.v_face_el,
  ])


def _vec_to_coeffs(vec: jax.Array) -> transport_coeffs_lib.TransportCoeffs:
  c = vec.reshape(_N_COEFFS, -1)
  return transport_coeffs_lib.TransportCoeffs(
      chi_face_ion=c[0], chi_face_el=c[1], d_face_el=c[2], v_face_el=c[3]
  )
