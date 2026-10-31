#!/usr/bin/env python
"""Run one TORAX simulation step-by-step in a fresh process and record timings.

Usage:
  python run_one.py --config torax.examples.iterhybrid_rampup \
      --overrides '{"numerics": {"t_final": 2.0}}' --label NAME --out results/NAME.json \
      [--max-steps N] [--aot] [--capture-log] [--n-timed-steps N]

Outputs a JSON file with process-level timings (import, config, prepare,
optional AOT lower/compile), per-step rows (t, dt, wall, inner, outer, err,
optional Newton log), summary statistics, and final profiles.
"""
import argparse
import contextlib
import copy
import importlib
import io
import json
import os
import re
import sys
import time

import numpy as np

ITER_RE = re.compile(
    r'Iteration:\s*(\d+)\.\s*Residual:\s*([0-9.eE+-]+)\.\s*tau\s*=\s*([0-9.eE+-]+)'
)


def deep_update(d, u):
  """Recursive dict update; a sub-dict with '__replace__': true replaces."""
  for k, v in u.items():
    if isinstance(v, dict) and v.pop('__replace__', False):
      d[k] = v
    elif isinstance(v, dict) and isinstance(d.get(k), dict):
      deep_update(d[k], v)
    else:
      d[k] = v
  return d


def summarize(rows):
  """Summary statistics over per-step rows."""
  s = {}
  n = len(rows)
  s['n_steps'] = n
  if n == 0:
    return s
  walls = np.array([r['wall'] for r in rows])
  inner = np.array([r['inner'] for r in rows])
  outer = np.array([r['outer'] for r in rows])
  err = np.array([r['err'] for r in rows])
  dts = np.array([r['dt'] for r in rows])
  s['first_step_wall_s'] = float(walls[0])
  rest = walls[1:] if n > 1 else walls
  med = float(np.median(rest))
  s['median_step_wall_excl_first_s'] = med
  # compile events: any step > 5x median of steps after the first
  compile_events = [int(i) for i in range(n) if walls[i] > 5 * med]
  if 0 not in compile_events:
    compile_events = [0] + compile_events
  s['compile_event_steps'] = compile_events
  s['compile_event_walls_s'] = [float(walls[i]) for i in compile_events]
  mask = np.ones(n, dtype=bool)
  mask[compile_events] = False
  clean = walls[mask]
  s['n_clean_steps'] = int(mask.sum())
  s['run_time_excl_first_s'] = float(walls[1:].sum()) if n > 1 else 0.0
  s['run_time_clean_s'] = float(clean.sum()) if clean.size else 0.0
  if clean.size:
    s['step_wall_mean_s'] = float(clean.mean())
    s['step_wall_median_s'] = float(np.median(clean))
    s['step_wall_max_s'] = float(clean.max())
    s['step_wall_min_s'] = float(clean.min())
    s['step_wall_std_s'] = float(clean.std())
  s['total_inner_iterations'] = int(inner.sum())
  s['total_inner_iterations_clean'] = int(inner[mask].sum())
  s['inner_iterations_mean'] = float(inner.mean())
  s['inner_iterations_max'] = int(inner.max())
  vals, counts = np.unique(inner, return_counts=True)
  s['inner_iterations_histogram'] = {int(v): int(c) for v, c in zip(vals, counts)}
  vals, counts = np.unique(outer, return_counts=True)
  s['outer_iterations_histogram'] = {int(v): int(c) for v, c in zip(vals, counts)}
  s['n_steps_outer_gt1'] = int((outer > 1).sum())
  s['n_steps_err2'] = int((err == 2).sum())
  s['n_steps_err1'] = int((err == 1).sum())
  s['dt_min'] = float(dts.min())
  s['dt_median'] = float(np.median(dts))
  s['dt_max'] = float(dts.max())
  s['t_reached'] = float(rows[-1]['t'])
  if s['total_inner_iterations_clean'] > 0 and clean.size:
    s['time_per_inner_iteration_s'] = (
        s['run_time_clean_s'] / s['total_inner_iterations_clean']
    )
  # Newton log statistics if present
  if 'log' in rows[0]:
    all_tau = []
    n_backtrack_iters = 0
    n_iters = 0
    iters_per_step = []
    for r in rows:
      log = r.get('log', [])
      iters_per_step.append(len(log))
      for (_, _, tau) in log:
        n_iters += 1
        all_tau.append(tau)
        if tau < 0.999999:
          n_backtrack_iters += 1
    s['log_total_iterations'] = n_iters
    s['log_iterations_per_step'] = iters_per_step
    s['log_n_iterations_with_tau_lt_1'] = n_backtrack_iters
    tvals, tcounts = np.unique(np.round(np.array(all_tau), 6), return_counts=True) if all_tau else ([], [])
    s['log_tau_histogram'] = {float(v): int(c) for v, c in zip(tvals, tcounts)}
    # number of extra residual evaluations due to backtracking
    # (tau = 0.5^k -> k extra evaluations)
    extra = 0
    for tau in all_tau:
      if tau < 0.999999 and tau > 0:
        extra += int(round(np.log(tau) / np.log(0.5)))
    s['log_extra_residual_evals_from_backtracking'] = int(extra)
  return s


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--config', required=True)
  ap.add_argument('--overrides', default='{}')
  ap.add_argument('--out', required=True)
  ap.add_argument('--label', default='')
  ap.add_argument('--max-steps', type=int, default=0)
  ap.add_argument('--aot', action='store_true')
  ap.add_argument('--capture-log', action='store_true')
  args = ap.parse_args()

  proc_t0 = time.perf_counter()
  T = {}
  t0 = time.perf_counter()
  import jax
  import torax
  from torax._src.orchestration import run_simulation as rs
  T['import_s'] = time.perf_counter() - t0

  t0 = time.perf_counter()
  mod = importlib.import_module(args.config)
  T['config_module_import_s'] = time.perf_counter() - t0
  cfg_dict = copy.deepcopy(mod.CONFIG)
  overrides = json.loads(args.overrides)
  deep_update(cfg_dict, overrides)

  t0 = time.perf_counter()
  cfg = torax.ToraxConfig.from_dict(cfg_dict)
  T['from_dict_s'] = time.perf_counter() - t0

  t0 = time.perf_counter()
  initial_state, ppo, step_fn = rs.prepare_simulation(cfg)
  jax.block_until_ready((initial_state, ppo))
  T['prepare_simulation_s'] = time.perf_counter() - t0

  meta = {
      'label': args.label,
      'config': args.config,
      'overrides': overrides,
      'jax_version': jax.__version__,
      'torax_version': getattr(torax, '__version__', 'n/a'),
      'devices': [str(d) for d in jax.devices()],
      'cpu_count': os.cpu_count(),
  }
  try:
    meta['n_rho'] = int(initial_state.core_profiles.T_i.value.shape[0])
  except Exception as e:  # pylint: disable=broad-except
    meta['n_rho'] = str(e)
  for name in ('solver', 'numerics', 'time_step_calculator'):
    try:
      meta[name] = json.loads(getattr(cfg, name).model_dump_json())
    except Exception as e:  # pylint: disable=broad-except
      meta[name] = str(getattr(cfg, name))
  try:
    meta['geometry_type'] = str(cfg_dict['geometry'].get('geometry_type'))
  except Exception:  # pylint: disable=broad-except
    pass

  if args.aot:
    f = type(step_fn).__call__
    t0 = time.perf_counter()
    lowered = f.lower(step_fn, initial_state, ppo)
    T['aot_lower_s'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    compiled = lowered.compile()
    T['aot_compile_s'] = time.perf_counter() - t0
    try:
      ca = compiled.cost_analysis()
      if isinstance(ca, list):
        ca = ca[0]
      T['aot_cost_analysis'] = {
          k: float(v) for k, v in ca.items()
          if k in ('flops', 'transcendentals', 'bytes accessed')
      }
    except Exception as e:  # pylint: disable=broad-except
      T['aot_cost_analysis'] = str(e)
    try:
      T['aot_hlo_text_bytes'] = len(compiled.as_text())
    except Exception as e:  # pylint: disable=broad-except
      T['aot_hlo_text_bytes'] = str(e)
    try:
      ma = compiled.memory_analysis()
      T['aot_memory_analysis'] = str(ma)
    except Exception as e:  # pylint: disable=broad-except
      T['aot_memory_analysis'] = str(e)

  rows = []
  state = initial_state
  sim_error = 'NO_ERROR'
  k = 0

  def dump(partial):
    """Write (possibly partial) results so they survive a kill/timeout."""
    cp = state.core_profiles
    profiles = {}
    for name in ('T_i', 'T_e', 'n_e', 'psi'):
      try:
        profiles[name] = np.asarray(getattr(cp, name).value).tolist()
      except Exception as e:  # pylint: disable=broad-except
        profiles[name] = str(e)
    try:
      profiles['rho_norm'] = np.asarray(state.geometry.rho_norm).tolist()
    except Exception:  # pylint: disable=broad-except
      pass
    out = {
        'meta': meta,
        'timings': T,
        'summary': summarize(rows),
        'sim_error': sim_error,
        'partial': partial,
        'rows': rows,
        'final_profiles': profiles,
        'final_t': float(state.t),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + '.tmp'
    with open(tmp, 'w') as fh:
      json.dump(out, fh, indent=1)
    os.replace(tmp, args.out)
    return out

  loop_t0 = time.perf_counter()
  last_dump = time.perf_counter()
  while not bool(step_fn.is_done(state.t)):
    if args.max_steps and k >= args.max_steps:
      break
    buf = io.StringIO()
    t0 = time.perf_counter()
    if args.capture_log:
      with contextlib.redirect_stdout(buf):
        state, ppo = step_fn(state, ppo)
        jax.block_until_ready((state, ppo))
    else:
      state, ppo = step_fn(state, ppo)
      jax.block_until_ready((state, ppo))
    wall = time.perf_counter() - t0
    sno = state.solver_numeric_outputs
    row = dict(
        k=k,
        t=float(state.t),
        dt=float(state.dt),
        wall=wall,
        inner=int(sno.inner_solver_iterations),
        outer=int(sno.outer_solver_iterations),
        err=int(sno.solver_error_state),
    )
    if args.capture_log:
      row['log'] = [
          (int(i), float(r), float(tau)) for i, r, tau in ITER_RE.findall(buf.getvalue())
      ]
    rows.append(row)
    err = step_fn.check_for_errors(state, ppo)
    if err != torax.SimError.NO_ERROR:
      sim_error = str(err)
      print('SIM ERROR', err, flush=True)
      break
    k += 1
    if k % 50 == 0:
      print(f'[{args.label}] step {k} t={float(state.t):.4f} '
            f'wall_last={wall:.3f}s', file=sys.stderr, flush=True)
    if time.perf_counter() - last_dump > 20.0:
      # periodic partial dump (outside the timed region of a step)
      T['loop_total_s'] = time.perf_counter() - loop_t0
      dump(True)
      last_dump = time.perf_counter()
  T['loop_total_s'] = time.perf_counter() - loop_t0
  T['process_total_s'] = time.perf_counter() - proc_t0

  out = dump(False)
  s = out['summary']
  print(
      f"[{args.label}] DONE steps={s.get('n_steps')} t={out['final_t']:.4f} "
      f"first_step={s.get('first_step_wall_s', 0):.2f}s "
      f"median_step={s.get('step_wall_median_s', 0):.4f}s "
      f"run_clean={s.get('run_time_clean_s', 0):.2f}s "
      f"inner_total={s.get('total_inner_iterations')} "
      f"outer>1={s.get('n_steps_outer_gt1')} err2={s.get('n_steps_err2')} "
      f"err1={s.get('n_steps_err1')} sim_error={sim_error} "
      f"proc_total={T['process_total_s']:.1f}s",
      flush=True,
  )


if __name__ == '__main__':
  main()
