"""Harness for the structured-Jacobian stress tests.

Builds residual_fun exactly as structured_jacobian_test does, captures the
closures jacobian_fn hands to _assemble (via a spy), and computes the true
factors with jax.jacfwd to check each sparsity assumption.
"""

import copy
import dataclasses
import functools
import json
import os
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

if os.environ.get('JAX_PRECISION', 'f64') == 'f64':
  jax.config.update('jax_enable_x64', True)

from torax._src import jax_utils  # noqa: E402
from torax._src.config import build_runtime_params  # noqa: E402
from torax._src.core_profiles import convertors  # noqa: E402
from torax._src.core_profiles import updaters  # noqa: E402
from torax._src.fvm import calc_coeffs  # noqa: E402
from torax._src.fvm import enums  # noqa: E402
from torax._src.fvm import fvm_conversions  # noqa: E402
from torax._src.fvm import residual_and_loss  # noqa: E402
from torax._src.orchestration import initial_state as initial_state_lib  # noqa: E402
from torax._src.orchestration import run_simulation  # noqa: E402
from torax._src.orchestration import step_function_processing  # noqa: E402
from torax._src.solver import structured_jacobian as sj  # noqa: E402
from torax._src.sources.ion_cyclotron_source import toric_nn  # noqa: E402
from torax._src.torax_pydantic import model_config  # noqa: E402
from torax.examples import iterhybrid_predictor_corrector  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def write_dummy_toric_nn(path: str) -> None:
  # pylint: disable=protected-access
  network = toric_nn._ToricNN(
      hidden_sizes=[3],
      pca_coeffs=4,
      input_dim=10,
      radial_nodes=toric_nn._TORIC_GRID_SIZE,
  )
  _, params = network.init_with_output(jax.random.PRNGKey(0), jnp.ones(10))
  config = dataclasses.asdict(network)
  weights = jax.tree_util.tree_map(lambda x: x.tolist(), params['params'])
  for name in (
      toric_nn._HELIUM3_ID,
      toric_nn._TRITIUM_SECOND_HARMONIC_ID,
      toric_nn._ELECTRON_ID,
  ):
    config[name] = weights
  with open(path, 'w') as f:
    json.dump(config, f)


TORIC_PATH = os.path.join(HERE, 'toric_nn.json')
if not os.path.exists(TORIC_PATH):
  write_dummy_toric_nn(TORIC_PATH)


def base_config(n_rho: int, global_sources: bool = False) -> dict:
  config = copy.deepcopy(iterhybrid_predictor_corrector.CONFIG)
  config['geometry']['n_rho'] = n_rho
  config['solver'] = dict(
      solver_type='newton_raphson',
      use_predictor_corrector=True,
      n_corrector_steps=2,
      use_pereverzev=True,
  )
  if global_sources:
    config['sources'].update({
        'cyclotron_radiation': {},
        'impurity_radiation': {
            'model_name': 'P_in_scaled_flat_profile',
            'fraction_P_heating': 0.1,
        },
        'icrh': {'model_path': TORIC_PATH, 'P_total': 10e6},
    })
    config['geometry'] = {'geometry_type': 'circular', 'n_rho': n_rho}
  return config


def solve_block_kwargs(config: dict, dt: float = 0.05, state_override=None):
  torax_config = model_config.ToraxConfig.from_dict(config)
  step_fn = run_simulation.make_step_fn(torax_config)
  state, _ = initial_state_lib.get_initial_state_and_post_processed_outputs(
      step_fn=step_fn
  )
  if state_override is not None:
    state = state_override(state)
  models = step_fn.solver.models
  runtime_params_t, geo_t, explicit_source_profiles, edge_outputs, pts = (
      step_function_processing.pre_step(
          input_state=state,
          runtime_params_provider=step_fn.runtime_params_provider,
          geometry_provider=step_fn.geometry_provider,
          models=models,
      )
  )
  dt = jnp.asarray(dt)
  runtime_params_t_plus_dt, geo_t_plus_dt = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=state.t + dt,
          runtime_params_provider=step_fn.runtime_params_provider,
          geometry_provider=step_fn.geometry_provider,
          edge_outputs=edge_outputs,
          core_profiles=state.core_profiles,
      )
  )
  evolving_names = runtime_params_t.numerics.evolving_names
  return dict(
      dt=dt,
      runtime_params_t=runtime_params_t,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t=geo_t,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=convertors.core_profiles_to_solver_x_tuple(
          state.core_profiles, evolving_names
      ),
      core_profiles_t=state.core_profiles,
      core_profiles_t_plus_dt=updaters.provide_core_profiles_t_plus_dt(
          dt=dt,
          runtime_params_t=runtime_params_t,
          runtime_params_t_plus_dt=runtime_params_t_plus_dt,
          geo_t_plus_dt=geo_t_plus_dt,
          core_profiles_t=state.core_profiles,
      ),
      explicit_source_profiles=explicit_source_profiles,
      models=models,
      coeffs_callback=calc_coeffs.CoeffsCallback(
          models=models, evolving_names=evolving_names
      ),
      evolving_names=evolving_names,
      initial_guess_mode=enums.InitialGuessMode.LINEAR,
      maxiter=30,
      tol=1e-8,
      coarse_tol=1e-2,
      delta_reduction_factor=0.5,
      tau_min=0.01,
      pedestal_transition_state=pts,
      max_linesearch_steps=100,
  ), torax_config


def residual_fun_from_kwargs(kwargs):
  return functools.partial(
      residual_and_loss.theta_method_block_residual,
      coeffs_old=kwargs['coeffs_callback'](
          kwargs['runtime_params_t'],
          kwargs['geo_t'],
          kwargs['core_profiles_t'],
          prev_core_profiles=None,
          dt=None,
          x=kwargs['x_old'],
          explicit_source_profiles=kwargs['explicit_source_profiles'],
          explicit_call=True,
          pedestal_transition_state=kwargs['pedestal_transition_state'],
      ),
      **{
          k: kwargs[k]
          for k in (
              'dt',
              'runtime_params_t_plus_dt',
              'geo_t_plus_dt',
              'x_old',
              'core_profiles_t',
              'core_profiles_t_plus_dt',
              'explicit_source_profiles',
              'models',
              'evolving_names',
              'pedestal_transition_state',
          )
      },
  )


def capture(residual_fun):
  """Returns (captured kwargs of _assemble, jitted jacobian_fn)."""
  captured = {}
  orig = sj._assemble

  def spy(x, **kw):
    captured.update(kw)
    return orig(x, **kw)

  sj._assemble = spy
  try:
    jac_fn = sj.jacobian_fn(residual_fun)
  finally:
    sj._assemble = orig
  return captured, jac_fn


def masks(n_cells, n_channels, axis_columns=(), bandwidth=2, h_reach=(2, 2)):
  n_state, n_faces = n_cells * n_channels, n_cells + 1
  cells = np.arange(n_state) % n_cells
  faces = np.arange(4 * n_faces) % n_faces
  mask_x = np.abs(cells[:, None] - cells[None, :]) <= bandwidth
  mask_x[:, list(axis_columns)] = True
  mask_h = (cells[None, :] >= faces[:, None] - h_reach[0]) & (
      cells[None, :] <= faces[:, None] + h_reach[1]
  )
  mask_c = (faces[None, :] == cells[:, None]) | (
      faces[None, :] == cells[:, None] + 1
  )
  return mask_x, mask_h, mask_c


def outside(mat, mask):
  """Returns (max |entry| outside mask, list of (row, col, value))."""
  mat = np.asarray(mat)
  bad = (~mask) & (mat != 0)
  if not bad.any():
    return 0.0, []
  idx = np.argwhere(bad)
  vals = np.abs(mat[bad])
  order = np.argsort(-vals)
  return float(vals.max()), [
      (int(idx[k][0]), int(idx[k][1]), float(mat[idx[k][0], idx[k][1]]))
      for k in order[:8]
  ]


def label(i, n_cells, names):
  return f'{names[i // n_cells]}[{i % n_cells}]'


def analyse(residual_fun, x, tag='', verbose=True, names=None, check_struct=True):
  """Checks every factor of the structured Jacobian at x."""
  t0 = time.time()
  cap, jac_fn = capture(residual_fun)
  out = {'tag': tag}
  if check_struct:
    J_struct = np.asarray(jac_fn(x))
  else:
    J_struct = None
  n_cells = cap['n_cells']
  n_ch = cap['n_channels']
  axis_columns = []
  if cap['psi_channel'] is not None:
    first = int(np.argmax(np.asarray(cap['rho_norm']) >= cap['min_rho_norm']))
    start = min(max(first - 1, 0), n_cells - sj._AXIS_CELLS)
    axis_columns = [
        cap['psi_channel'] * n_cells + start + i for i in range(sj._AXIS_CELLS)
    ]
  n_faces = n_cells + 1
  names = names or residual_fun.keywords['evolving_names']
  residual, raw_transport, postprocess, state_globals = (
      cap['residual'],
      cap['raw_transport'],
      cap['postprocess'],
      cap['state_globals'],
  )
  J_dense = np.asarray(jax.jit(jax.jacfwd(residual_fun))(x))
  q = state_globals(x)
  raw = raw_transport(x, q)
  c = postprocess(raw, q)
  dGdx = np.asarray(jax.jit(jax.jacfwd(lambda x: residual(x, c, q)))(x))
  dGdc = np.asarray(jax.jit(jax.jacfwd(lambda c: residual(x, c, q)))(c))
  dhdx = np.asarray(jax.jit(jax.jacfwd(lambda x: raw_transport(x, q)))(x))
  dTdh = np.asarray(jax.jit(jax.jacfwd(lambda h: postprocess(h, q)))(raw))
  k = int(q.shape[0])
  if k:
    dGdq = np.asarray(jax.jit(jax.jacfwd(lambda q: residual(x, c, q)))(q))
    dTdq = np.asarray(jax.jit(jax.jacfwd(lambda q: postprocess(raw, q)))(q))
    dqdx = np.asarray(jax.jit(jax.jacfwd(state_globals))(x))
  else:
    dGdq = np.zeros((x.size, 0))
    dTdq = np.zeros((c.size, 0))
    dqdx = np.zeros((0, x.size))
  J_chain = dGdx + dGdc @ dTdh @ dhdx + (dGdq + dGdc @ dTdq) @ dqdx
  # R(x) == G(x, T(h(x), q(x)), q(x)) as values
  r_full = np.asarray(residual_fun(x))
  r_split = np.asarray(residual(x, c, q))
  mask_x, mask_h, mask_c = masks(
      n_cells, n_ch, axis_columns, h_reach=cap['h_reach']
  )
  ox = outside(dGdx, mask_x)
  oh = outside(dhdx, mask_h)
  oc = outside(dGdc, mask_c)
  nd = np.linalg.norm(J_dense)
  out.update(
      n_cells=n_cells,
      n_channels=n_ch,
      axis_columns=axis_columns,
      k=k,
      names=names,
      value_split_maxdiff=float(np.max(np.abs(r_full - r_split))),
      value_scale=float(np.max(np.abs(r_full))),
      chain_vs_dense=float(np.linalg.norm(J_chain - J_dense) / nd),
      struct_vs_dense=(
          float(np.linalg.norm(J_struct - J_dense) / nd)
          if J_struct is not None
          else None
      ),
      struct_vs_chain=(
          float(np.linalg.norm(J_struct - J_chain) / nd)
          if J_struct is not None
          else None
      ),
      dGdx_outside=ox[0],
      dGdx_outside_rel=ox[0] / max(np.abs(dGdx).max(), 1e-300),
      dGdx_outside_entries=[
          (label(r, n_cells, names), label(cc, n_cells, names), v)
          for r, cc, v in ox[1]
      ],
      dhdx_outside=oh[0],
      dhdx_outside_entries=[
          (f'coef{r // n_faces}[face {r % n_faces}]', label(cc, n_cells, names), v)
          for r, cc, v in oh[1]
      ],
      dGdc_outside=oc[0],
      dGdc_outside_entries=[
          (label(r, n_cells, names), f'coef{cc // n_faces}[face {cc % n_faces}]', v)
          for r, cc, v in oc[1]
      ],
      time=time.time() - t0,
  )
  if J_struct is not None:
    diff = np.abs(J_struct - J_dense)
    i, j = np.unravel_index(np.argmax(diff), diff.shape)
    out['struct_maxdiff'] = float(diff[i, j])
    out['struct_maxdiff_at'] = (
        label(i, n_cells, names),
        label(j, n_cells, names),
        float(J_dense[i, j]),
        float(J_struct[i, j]),
    )
  out['_mats'] = dict(
      J_dense=J_dense,
      J_struct=J_struct,
      J_chain=J_chain,
      dGdx=dGdx,
      dGdc=dGdc,
      dhdx=dhdx,
      dGdq=dGdq,
      dTdh=dTdh,
      dTdq=dTdq,
      dqdx=dqdx,
  )
  if verbose:
    print_result(out)
  return out


def print_result(out):
  print(f"=== {out['tag']} ===")
  for key, val in out.items():
    if key.startswith('_') or key == 'tag':
      continue
    print(f'  {key}: {val}')
  sys.stdout.flush()


def perturbed_x(kwargs, scale=0.05, seed=0):
  x = fvm_conversions.cell_variable_tuple_to_vec(kwargs['x_old'])
  if scale == 0:
    return x
  return x * (1.0 + scale * np.random.RandomState(seed).standard_normal(x.shape))
