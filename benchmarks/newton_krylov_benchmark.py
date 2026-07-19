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

r"""Benchmark: dense-Jacobian Newton-Raphson vs Jacobian-free Newton-Krylov.

Compares `jax_root_finding.root_newton_raphson` (materializes the full N x N
Jacobian via `jax.jacfwd` and solves densely) against the prototype
`jax_root_finding.root_newton_krylov` (matrix-free preconditioned GMRES with
J v products from `jax.linearize`) on a single implicit theta-method step of
one or more TORAX configs, starting both solvers from the identical LINEAR
initial guess.

The JFNK preconditioner is the block-tridiagonal LHS matrix of the theta-method
discretization, frozen at the initial guess and applied with the Thomas
algorithm (O(N)). This matrix is the exact Jacobian minus the
transport-coefficient sensitivity terms, so it captures the dominant
diffusion/convection structure.

Usage:
  # Built-in STEP flat-top + TGLFNN-ukaea stress case (N=400):
  python benchmarks/newton_krylov_benchmark.py

  # Any config file(s) exposing a module-level CONFIG dict:
  python benchmarks/newton_krylov_benchmark.py \
      --configs torax/tests/test_data/test_psi_heat_dens.py --csv results.csv

Each config is benchmarked on one implicit step at the dt chosen by its own
time-step calculator (scaled by --dt_mult). Results are printed and optionally
appended to a CSV for aggregation across many configs. Run each config in its
own process when sweeping a large suite (jit caches accumulate memory).
"""

import argparse
import csv
import functools
import importlib.util
import os
import pathlib
import time
import traceback

import jax
import jax.numpy as jnp

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_BUILTIN_STEP_TGLFNN = 'step_tglfnn'


def _load_config_dict(config_spec: str):
  """Loads a CONFIG dict from a builtin name or a config file path."""
  if config_spec == _BUILTIN_STEP_TGLFNN:
    path = _REPO_ROOT / 'torax' / 'examples' / 'step_flattop_bgb.py'
  else:
    path = pathlib.Path(config_spec)
  spec = importlib.util.spec_from_file_location(path.stem, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  if not hasattr(mod, 'CONFIG'):
    raise ValueError(f'{path} does not define a module-level CONFIG dict.')
  config = mod.CONFIG
  if config_spec == _BUILTIN_STEP_TGLFNN:
    config['transport'] = {
        'model_name': 'tglfnn-ukaea',
        # Public weights. The 'step' weights are not publicly distributed.
        'machine': 'multimachine',
        'chi_min': 0.15,
        'chi_max': 100.0,
        'D_e_min': 1e-3,
        'D_e_max': 100.0,
        'V_e_min': -50.0,
        'V_e_max': 50.0,
        'smooth_everywhere': True,
        'smoothing_width': 0.05,
    }
  return config


def _build_problem(config_dict, dt_mult: float):
  """Builds the residual function and solver inputs for one implicit step."""
  # Imported here so --help works without a full TORAX import.
  import torax  # pylint: disable=g-import-not-at-top
  from torax._src.config import build_runtime_params  # pylint: disable=g-import-not-at-top
  from torax._src.core_profiles import convertors  # pylint: disable=g-import-not-at-top
  from torax._src.core_profiles import updaters  # pylint: disable=g-import-not-at-top
  from torax._src.fvm import calc_coeffs  # pylint: disable=g-import-not-at-top
  from torax._src.fvm import fvm_conversions  # pylint: disable=g-import-not-at-top
  from torax._src.fvm import residual_and_loss  # pylint: disable=g-import-not-at-top
  from torax._src.orchestration import initial_state as initial_state_lib  # pylint: disable=g-import-not-at-top
  from torax._src.orchestration import run_simulation  # pylint: disable=g-import-not-at-top
  from torax._src.orchestration import step_function_processing  # pylint: disable=g-import-not-at-top
  from torax._src.solver import predictor_corrector_method  # pylint: disable=g-import-not-at-top

  torax_config = torax.ToraxConfig.from_dict(config_dict)
  step_fn = run_simulation.make_step_fn(torax_config)
  state0, _ = initial_state_lib.get_initial_state_and_post_processed_outputs(
      step_fn=step_fn,
  )

  rpp = step_fn.runtime_params_provider
  geo_provider = step_fn.geometry_provider
  models = step_fn.solver.models

  (runtime_params_t, geo_t, explicit_source_profiles, edge_outputs,
   pedestal_transition_state) = step_function_processing.pre_step(
       input_state=state0,
       runtime_params_provider=rpp,
       geometry_provider=geo_provider,
       models=models,
   )

  if not runtime_params_t.numerics.evolving_names:
    raise ValueError('Config evolves no PDE variables; nothing to benchmark.')

  dt = step_fn.time_step_calculator.next_dt(runtime_params_t, state0) * dt_mult
  runtime_params_t_plus_dt, geo_t_plus_dt = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=state0.t + dt,
          runtime_params_provider=rpp,
          geometry_provider=geo_provider,
          edge_outputs=edge_outputs,
          core_profiles=state0.core_profiles,
      )
  )
  core_profiles_t = state0.core_profiles
  core_profiles_t_plus_dt = updaters.provide_core_profiles_t_plus_dt(
      dt=dt,
      runtime_params_t=runtime_params_t,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      core_profiles_t=core_profiles_t,
  )
  evolving_names = runtime_params_t.numerics.evolving_names
  x_old = convertors.core_profiles_to_solver_x_tuple(
      core_profiles_t, evolving_names
  )

  coeffs_callback = calc_coeffs.CoeffsCallback(
      models=models, evolving_names=evolving_names
  )
  coeffs_old = coeffs_callback(
      runtime_params_t, geo_t, core_profiles_t,
      prev_core_profiles=None, dt=None, x=x_old,
      explicit_source_profiles=explicit_source_profiles,
      explicit_call=True,
      pedestal_transition_state=pedestal_transition_state,
  )

  residual_fun = functools.partial(
      residual_and_loss.theta_method_block_residual,
      dt=dt,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      core_profiles_t=core_profiles_t,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      models=models,
      explicit_source_profiles=explicit_source_profiles,
      coeffs_old=coeffs_old,
      evolving_names=evolving_names,
      pedestal_transition_state=pedestal_transition_state,
  )

  # Initial guess: single linear implicit solve (LINEAR mode, no
  # predictor-corrector), as in newton_raphson_solve_block.
  coeffs_exp_linear = coeffs_callback(
      runtime_params_t, geo_t, core_profiles_t,
      prev_core_profiles=None, dt=None, x=x_old,
      explicit_source_profiles=explicit_source_profiles,
      allow_pereverzev=True, explicit_call=True,
      pedestal_transition_state=pedestal_transition_state,
  )
  x_new_guess = convertors.core_profiles_to_solver_x_tuple(
      core_profiles_t_plus_dt, evolving_names
  )
  x_init_tuple = predictor_corrector_method.predictor_corrector_method(
      dt=dt,
      runtime_params_t_plus_dt=runtime_params_t_plus_dt,
      geo_t_plus_dt=geo_t_plus_dt,
      x_old=x_old,
      x_new_guess=x_new_guess,
      core_profiles_t=core_profiles_t,
      core_profiles_t_plus_dt=core_profiles_t_plus_dt,
      coeffs_exp=coeffs_exp_linear,
      coeffs_callback=coeffs_callback,
      explicit_source_profiles=explicit_source_profiles,
      pedestal_transition_state=pedestal_transition_state,
  )
  x_init_vec = fvm_conversions.cell_variable_tuple_to_vec(x_init_tuple)

  def _make_precond_apply(x_tuple, allow_pereverzev: bool):
    """Builds v -> M(x)^{-1} v from the theta-method LHS matrix at x."""
    coeffs_x = coeffs_callback(
        runtime_params_t_plus_dt, geo_t_plus_dt, core_profiles_t_plus_dt,
        prev_core_profiles=core_profiles_t, dt=dt, x=x_tuple,
        explicit_source_profiles=explicit_source_profiles,
        allow_pereverzev=allow_pereverzev,
        pedestal_transition_state=pedestal_transition_state,
    )
    solver_params = runtime_params_t_plus_dt.solver
    lhs, lhs_vec, rhs, rhs_vec = (
        residual_and_loss.theta_method_matrix_equation(
            dt=dt,
            x_old=x_old,
            x_new_guess=x_tuple,
            coeffs_old=coeffs_old,
            coeffs_new=coeffs_x,
            theta_implicit=solver_params.theta_implicit,
            convection_dirichlet_mode=solver_params.convection_dirichlet_mode,
            convection_neumann_mode=solver_params.convection_neumann_mode,
        )
    )
    if coeffs_x.has_internal_boundary_conditions:
      lhs, lhs_vec, rhs, rhs_vec = (
          residual_and_loss.apply_internal_boundary_conditions(
              lhs, lhs_vec, rhs, rhs_vec,
              coeffs_x.internal_boundary_condition_mask,
              coeffs_x.internal_boundary_condition_target_vec,
          )
      )
    num_cells = lhs.num_blocks
    num_channels = lhs.block_size

    def precond_apply(v):
      # The flattened residual/state vector is channel-major:
      # v.reshape(C, n).T gives the (num_cells, num_channels) layout used by
      # the block-tridiagonal matvec/solve.
      from torax._src import tridiagonal  # pylint: disable=g-import-not-at-top
      arr = v.reshape(num_channels, num_cells).T
      out = tridiagonal.thomas_solve(lhs, arr)
      return out.T.reshape(-1)

    return precond_apply

  def build_precond_fn(refresh: bool, allow_pereverzev: bool = False):
    """Preconditioner builder: frozen at the initial guess, or refreshed."""
    if refresh:

      def precond_fn(x_vec):
        x_tuple = fvm_conversions.vec_to_cell_variable_tuple(
            x_vec, core_profiles_t_plus_dt, evolving_names
        )
        return _make_precond_apply(x_tuple, allow_pereverzev)

      return precond_fn
    frozen_apply = _make_precond_apply(x_init_tuple, allow_pereverzev)
    return lambda x_vec: frozen_apply

  return residual_fun, x_init_vec, dt, build_precond_fn


def _benchmark_config(config_spec: str, args) -> list[dict]:
  """Runs all solver variants on one config. Returns CSV-ready row dicts."""
  from torax._src.solver import jax_root_finding  # pylint: disable=g-import-not-at-top

  name = pathlib.Path(config_spec).stem
  print(f'=== {name}: building problem... ===', flush=True)
  config_dict = _load_config_dict(config_spec)
  residual_fun, x_init_vec, dt, build_precond_fn = _build_problem(
      config_dict, args.dt_mult
  )
  residual_fun = jax.tree_util.Partial(residual_fun)
  n = int(x_init_vec.shape[0])
  print(f'N = {n}, dt = {float(dt):.4e} s', flush=True)

  common = dict(maxiter=args.maxiter, tol=args.tol)

  def dense(x0):
    return jax_root_finding.root_newton_raphson(
        fun=residual_fun, x0=x0, use_jax_custom_root=False, **common
    )

  def jfnk(x0, precond_fn, restart):
    return jax_root_finding.root_newton_krylov(
        fun=residual_fun, x0=x0,
        gmres_rtol=args.gmres_rtol,
        gmres_restart=restart,
        gmres_maxiter=args.gmres_maxiter,
        precond_fn=precond_fn,
        **common,
    )

  variants = [
      ('dense_newton', jax.jit(dense)),
      ('jfnk_noprecond', jax.jit(
          functools.partial(
              jfnk, precond_fn=None, restart=2 * args.gmres_restart))),
      ('jfnk_thomas', jax.jit(
          functools.partial(
              jfnk, precond_fn=build_precond_fn(refresh=False),
              restart=args.gmres_restart))),
      ('jfnk_refresh', jax.jit(
          functools.partial(
              jfnk, precond_fn=build_precond_fn(refresh=True),
              restart=args.gmres_restart))),
  ]
  if args.include_pereverzev_precond:
    variants.append(
        ('jfnk_pereverzev', jax.jit(
            functools.partial(
                jfnk,
                precond_fn=build_precond_fn(
                    refresh=False, allow_pereverzev=True),
                restart=args.gmres_restart))))

  rows = []
  print(f'{"variant":18} {"iters":>5} {"error":>5} {"resid":>10} '
        f'{"compile+run":>12} {"run":>9}')
  for variant, fn in variants:
    t0 = time.perf_counter()
    _, meta = jax.block_until_ready(fn(x_init_vec))
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, meta = jax.block_until_ready(fn(x_init_vec))
    t_run = time.perf_counter() - t0
    resid = float(jnp.mean(jnp.abs(meta.residual)))
    print(f'{variant:18} {int(meta.iterations):5d} {int(meta.error):5d} '
          f'{resid:10.2e} {t_first:11.1f}s {t_run:8.2f}s', flush=True)
    rows.append({
        'config': name,
        'N': n,
        'dt': float(dt),
        'variant': variant,
        'iters': int(meta.iterations),
        'error': int(meta.error),
        'residual': resid,
        'compile_run_s': round(t_first, 2),
        'run_s': round(t_run, 3),
    })
  return rows


_CSV_FIELDS = [
    'config', 'N', 'dt', 'variant', 'iters', 'error', 'residual',
    'compile_run_s', 'run_s',
]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--configs', nargs='+', default=[_BUILTIN_STEP_TGLFNN],
      help='Config .py files exposing a CONFIG dict, or the builtin name '
      f"'{_BUILTIN_STEP_TGLFNN}' (STEP flat-top with TGLFNN-ukaea transport).",
  )
  parser.add_argument(
      '--dt_mult', type=float, default=1.0,
      help='Multiplier on each config\'s own nominal dt.',
  )
  parser.add_argument('--maxiter', type=int, default=30)
  parser.add_argument('--tol', type=float, default=1e-5)
  parser.add_argument('--gmres_restart', type=int, default=20)
  parser.add_argument('--gmres_rtol', type=float, default=1e-2)
  parser.add_argument('--gmres_maxiter', type=int, default=1,
                      help='Number of GMRES restart cycles.')
  parser.add_argument(
      '--include_pereverzev_precond', action='store_true',
      help='Also run the Pereverzev-stiffened preconditioner variant '
      '(counterproductive in benchmarks to date; off by default).',
  )
  parser.add_argument(
      '--csv', type=str, default='',
      help='Append per-variant results to this CSV file.',
  )
  args = parser.parse_args()

  all_rows = []
  for config_spec in args.configs:
    try:
      all_rows.extend(_benchmark_config(config_spec, args))
    except Exception as e:  # pylint: disable=broad-except
      print(f'FAILED to benchmark {config_spec}: {e}', flush=True)
      traceback.print_exc()
      all_rows.append({
          'config': pathlib.Path(config_spec).stem,
          'N': -1, 'dt': -1.0, 'variant': 'BUILD_FAIL',
          'iters': -1, 'error': -1, 'residual': float('nan'),
          'compile_run_s': -1.0, 'run_s': -1.0,
      })

  if args.csv:
    write_header = not os.path.exists(args.csv)
    with open(args.csv, 'a', newline='') as f:
      writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
      if write_header:
        writer.writeheader()
      writer.writerows(all_rows)


if __name__ == '__main__':
  main()
