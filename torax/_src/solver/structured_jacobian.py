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

The residual couples neighbouring cells only, except through the
post-processing ``T`` of the raw turbulent transport coefficients ``h(x)``
(their smoothing and, with an adaptive pedestal, its scaling; see
``transport_coefficients_builder.postprocess_turbulent_transport``) and
through a few global quantities ``q(x)`` of the state (see
``calc_coeffs.StateGlobals``: the globals of some sources, a state-dependent
pedestal and the values of the state that whole internal boundary profiles
use). With ``G`` the residual evaluated on injected coefficients and globals,
``R(x) = G(x, T(h(x), q(x)), q(x))`` and

  ``J = dG/dx + dG/dc dT/dh dh/dx + (dG/dq + dG/dc dT/dq) dq/dx``.

``dG/dx`` and ``dh/dx`` have fixed stencils and are assembled from a
grid-independent number of coloured forward-mode passes (Curtis-Powell-Reid
colouring), ``dT/dh`` is a dense matrix per coefficient from one pass per face,
and the last term has rank ``len(q)`` and costs one JVP and one VJP per global
quantity. The stencils hold on any radial grid and for any
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
from torax._src.core_profiles import updaters
from torax._src.fvm import calc_coeffs
from torax._src.fvm import fvm_conversions
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source as source_lib
from torax._src.transport_model import qualikiz_based_transport_model
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import transport_coefficients_builder
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib

# With T and q frozen, residual cell i depends on state cells |i - j| <= 2,
# and cells below numerics.min_rho_norm on psi in _AXIS_CELLS cells around the
# first cell beyond it, whose current density they take.
_BANDWIDTH = 2
_AXIS_CELLS = 4
_N_COEFFS = 4  # chi_i, chi_e, D_e, V_e on the face grid.


def jacobian_fn(
    residual_fun: functools.partial,
) -> Callable[[jax.Array], jax.Array]:
  """Returns x -> dR/dx for `residual_fun`.

  Args:
    residual_fun: `theta_method_block_residual` with everything but the state
      bound; its bound arguments define the physics of the step.
  """
  kw = residual_fun.keywords
  models, geo = kw['models'], kw['geo_t_plus_dt']
  runtime_params, names = kw['runtime_params_t_plus_dt'], kw['evolving_names']
  n_cells = int(geo.torax_mesh.nx)
  user_models = _user_defined_models(models, runtime_params)
  if user_models and not jax_utils.errors_enabled():
    logging.warning(
        "jacobian_mode='structured' assumes that these user-defined models"
        ' couple neighbouring cells only (a source can declare its global'
        ' quantities with a source.SplitModelFunction): %s. Set'
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
    """q(x), unflattened."""
    return calc_coeffs.calc_state_globals(
        runtime_params,
        geo,
        core_profiles(x),
        kw['explicit_source_profiles'],
        models,
        kw['pedestal_transition_state'],
    )

  x0 = fvm_conversions.cell_variable_tuple_to_vec(kw['x_old'])
  layout = jax.eval_shape(state_globals, x0)
  _, unravel = flatten_util.ravel_pytree(
      jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), layout)
  )

  def pedestal_transition_state(q):
    pedestal = unravel(q).pedestal
    if pedestal is None:
      return kw['pedestal_transition_state']
    return dataclasses.replace(
        kw['pedestal_transition_state'], pedestal_model_output=pedestal
    )

  def raw_transport(x, q):
    """h(x): the turbulent transport coefficients before post-processing.

    The transport model sees the pedestal only through the location of its
    top, so h does not depend on q differentiably.
    """
    transport = transport_coefficients_builder.calculate_all_transport_coeffs(
        transport_model=models.transport_model,
        neoclassical_models=models.neoclassical_models,
        internal_boundary_condition_model=models.internal_boundary_condition_model,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles(x),
        pedestal_transition_state=pedestal_transition_state(q),
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
            _vec_to_coeffs(h, n_cells + 1),
        )
    )

  def residual(x, c, q):
    """G(x, c, q): the residual with c and q injected."""
    return residual_fun(
        x,
        turbulent_transport=_vec_to_coeffs(c, n_cells + 1),
        state_globals=unravel(q),
    )

  assemble = functools.partial(
      _assemble,
      residual=residual,
      raw_transport=raw_transport,
      postprocess=postprocess,
      state_globals=lambda x: flatten_util.ravel_pytree(state_globals(x))[
          0
      ].astype(x.dtype),
      n_cells=n_cells,
      n_channels=len(names),
      h_reach=_transport_reach(runtime_params.transport),
      rho_norm=geo.rho_norm,
      min_rho_norm=runtime_params.numerics.min_rho_norm,
      psi_channel=names.index('psi') if 'psi' in names else None,
      check_against=residual_fun,
  )
  # Not jax.Inline.XLA_LATE: XLA's late inlining of this many instructions can
  # take tens of minutes, depending on the depth of the Python call stack.
  return jax.jit(assemble)


def _assemble(
    x,
    *,
    residual,
    raw_transport,
    postprocess,
    state_globals,
    n_cells,
    n_channels,
    h_reach,
    rho_norm,
    min_rho_norm,
    psi_channel,
    check_against,
):
  """Assembles dR/dx at x from the linearised factors."""
  n_state, n_faces = n_cells * n_channels, n_cells + 1
  cells = np.arange(n_state) % n_cells
  faces = np.arange(_N_COEFFS * n_faces) % n_faces

  # Column colourings and sparsity patterns of dG/dx and dh/dx.
  colours_x, n_x = _colouring(n_cells, n_channels, 2 * _BANDWIDTH + 1)
  mask_x = np.abs(cells[:, None] - cells[None, :]) <= _BANDWIDTH
  if psi_channel is not None:
    # The current density of the cells below min_rho_norm is that of the first
    # cell at or beyond it, or of the first cell if there is none
    # (psi_calculations._extrapolate_cell_profile_to_axis), which depends on
    # psi from the cell before it to two cells after it (three-point face
    # gradients). These columns, known only at run time, get a colour each.
    first = jnp.argmax(rho_norm >= min_rho_norm)
    start = jnp.clip(first - 1, 0, n_cells - _AXIS_CELLS)
    axis = psi_channel * n_cells + start + jnp.arange(_AXIS_CELLS)
    colours_x = (
        jnp.asarray(colours_x).at[axis].set(n_x + jnp.arange(_AXIS_CELLS))
    )
    mask_x = jnp.asarray(mask_x).at[:, axis].set(True)
    n_x += _AXIS_CELLS
  colours_h, n_h = _colouring(n_cells, n_channels, sum(h_reach) + 1)
  mask_h = (cells[None, :] >= faces[:, None] - h_reach[0]) & (
      cells[None, :] <= faces[:, None] + h_reach[1]
  )
  # dG/dc only needs the two parities of each coefficient: residual cell i
  # uses faces i and i + 1.
  colours_c = np.repeat(np.arange(_N_COEFFS), n_faces) * 2 + faces % 2
  n_c = 2 * _N_COEFFS

  # Each factor is linearised once and evaluated on all its seeds in one
  # batch; the dG/dx, dG/dc and dG/dq seeds are stacked into a single batch.
  q, q_lin = jax.linearize(state_globals, x)
  k = q.shape[0]
  seeds_q = jnp.eye(k, dtype=x.dtype)
  raw, h_lin = jax.linearize(lambda x: raw_transport(x, q), x)
  c, t_lin = jax.linearize(postprocess, raw, q)
  # T acts on each coefficient separately, so one seed per face gives the
  # corresponding column of dT/dh for all of them.
  dt_dh = jax.vmap(lambda s: t_lin(s, jnp.zeros_like(q)))(
      jnp.tile(jnp.eye(n_faces, dtype=x.dtype), (1, _N_COEFFS))
  )
  dt_dh = dt_dh.reshape(n_faces, _N_COEFFS, n_faces).transpose(1, 2, 0)
  dt_dq = jax.vmap(lambda e: t_lin(jnp.zeros_like(raw), e))(seeds_q)
  _, r_lin = jax.linearize(residual, x, c, q)
  zeros = lambda rows, cols: jnp.zeros((rows, cols), x.dtype)
  seeds = lambda colours, n: jax.nn.one_hot(colours, n, dtype=x.dtype).T
  compressed = jax.vmap(r_lin)(
      jnp.concatenate([seeds(colours_x, n_x), zeros(n_c + k, n_state)]),
      jnp.concatenate([zeros(n_x, c.size), seeds(colours_c, n_c), dt_dq]),
      jnp.concatenate([zeros(n_x + n_c, k), seeds_q]),
  )
  jac = _decompress(compressed[:n_x], colours_x, mask_x)

  # dG/dc dT/dh dh/dx, applied row by row.
  dh_dx = _decompress(jax.vmap(h_lin)(seeds(colours_h, n_h)), colours_h, mask_h)
  dc_dx = jnp.einsum(
      'kfg,kgn->kfn', dt_dh, dh_dx.reshape(_N_COEFFS, n_faces, n_state)
  )
  rows = jnp.arange(n_state)
  cell = rows % n_cells
  for i in range(_N_COEFFS):
    left = compressed[n_x + 2 * i + cell % 2, rows]
    right = compressed[n_x + 2 * i + (cell + 1) % 2, rows]
    jac += left[:, None] * dc_dx[i][cell] + right[:, None] * dc_dx[i][cell + 1]

  # (dG/dq + dG/dc dT/dq) dq/dx, of rank k.
  if k:
    q_vjp = jax.linear_transpose(q_lin, x)
    jac += compressed[n_x + n_c :].T @ jax.vmap(lambda v: q_vjp(v)[0])(seeds_q)

  if jax_utils.errors_enabled():
    # A coupling missing from the stencils shows up in its row, against a JVP
    # of the full residual along a random probe.
    probe = jnp.asarray(
        np.random.default_rng(0).uniform(1.0, 2.0, n_state), x.dtype
    )
    jv = jax.jvp(check_against, (x,), (probe,))[1]
    tol = jnp.sqrt(jnp.finfo(x.dtype).eps) * (jnp.abs(jac) @ probe)
    jac = jax_utils.error_if(
        jac,
        jnp.abs(jac @ probe - jv) > tol,
        'The structured Jacobian misses a coupling of the residual; use'
        " jacobian_mode='dense'.",
    )
  return jac


def _user_defined_models(models, runtime_params) -> list[str]:
  """The models in the residual that are not TORAX's, and so assumed local.

  The pedestal model is not listed: its output is a global quantity.
  """

  def outside_torax(obj):
    if isinstance(obj, functools.partial):
      obj = obj.func
    return not getattr(obj, '__module__', 'torax.').startswith('torax.')

  candidates = {
      'transport model': models.transport_model,
      'internal boundary condition model': (
          models.internal_boundary_condition_model
      ),
      'neoclassical conductivity model': (
          models.neoclassical_models.conductivity
      ),
      'bootstrap current model': models.neoclassical_models.bootstrap_current,
      'neoclassical transport model': models.neoclassical_models.transport,
  }
  transport = models.transport_model
  for name, model in (
      *transport.core_transport_models.items(),
      *transport.pedestal_transport_models.items(),
  ):
    candidates[f'transport model {name}'] = model
  for name, source in models.source_models.standard_sources.items():
    params = runtime_params.sources.get(name)
    if (
        params is not None
        and params.mode == sources_runtime_params_lib.Mode.MODEL_BASED
        and not isinstance(source.model_func, source_lib.SplitModelFunction)
    ):
      candidates[f'source {name}'] = source
      candidates[f'model function of source {name}'] = source.model_func
  return [name for name, obj in candidates.items() if outside_torax(obj)]


def _transport_reach(transport_params) -> tuple[int, int]:
  """Cells left and right of face f that its raw coefficients depend on."""
  params = (
      *transport_params.core_transport_model_params.values(),
      *transport_params.pedestal_transport_model_params.values(),
  )
  # Face values reach one cell either side and the three-point face gradients
  # one more to the right, and the magnetic shear uses q on the neighbouring
  # faces. The E x B shear of rotating models is a face gradient of a
  # face-to-cell average of a face value built from these; the inputs of
  # TGLF-based models reach one cell further on either side (measured). On a
  # uniform grid the third weight of the face gradients vanishes and each
  # reach is one cell less to the right, but the grid is only known at run
  # time.
  if any(
      isinstance(p, tglf_based_transport_model.RuntimeParams) and p.use_rotation
      for p in params
  ):
    return (4, 5)
  if any(
      isinstance(p, qualikiz_based_transport_model.RuntimeParams)
      and p.rotation_mode != qualikiz_based_transport_model.RotationMode.OFF
      for p in params
  ):
    return (3, 4)
  return (2, 2)


def _colouring(n_cells, n_channels, period):
  """Colours the channel-major state cyclically with `period` per channel."""
  idx = np.arange(n_cells * n_channels)
  return (idx // n_cells) * period + (
      idx % n_cells
  ) % period, n_channels * period


def _decompress(compressed, colours, mask):
  return jnp.where(mask, compressed[colours].T, 0.0)


def _coeffs_to_vec(coeffs):
  return jnp.concatenate([
      coeffs.chi_face_ion,
      coeffs.chi_face_el,
      coeffs.d_face_el,
      coeffs.v_face_el,
  ])


def _vec_to_coeffs(vec, n_faces):
  c = vec.reshape(_N_COEFFS, n_faces)
  return transport_coeffs_lib.TransportCoeffs(
      chi_face_ion=c[0], chi_face_el=c[1], d_face_el=c[2], v_face_el=c[3]
  )
