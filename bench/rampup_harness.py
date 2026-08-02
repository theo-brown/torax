"""Shared timing + accuracy harness for solver experiments on iterhybrid_rampup.

All solver experiments (Anderson, JFNK, TR-BDF2, psi splitting) must report
through this harness so their numbers are comparable.

Usage
-----
  # Run the config and record trajectory + timings.
  python bench/rampup_harness.py run <tag> [--overrides '<json>'] [--n-rho N]
                                          [--t-final T] [--reps R]

  # Compare two recorded runs (accuracy).
  python bench/rampup_harness.py cmp <tag_a> <tag_b>

  # Print the timing table for one or more tags.
  python bench/rampup_harness.py time <tag> [<tag> ...]

Design notes
------------
Timing: wall clock on this machine has a large per-process spread, and
consecutive trajectory steps do genuinely different work (different dt,
different Newton iteration counts), so a "total trajectory time" is not a
comparable sample. We therefore report two numbers:

  * ``fixed_ms``  -- median over ``reps`` re-executions of the *same* step from
    the *same* input state. Every sample is identical work, so the spread is
    machine noise only. This is the number to compare across variants.
  * ``traj_s``    -- total wall time for the full trajectory. This is what a
    user feels, and it also captures changes in the *number* of steps (which is
    the whole point of TR-BDF2), but it is noisier.

Deterministic work proxies (``flops``, ``bytes``, ``hlo_lines``, ``fusions``)
come from the compiled executable and are identical run to run, so they
distinguish a real change from noise even when wall clock cannot.

Accuracy: ``cmp`` interpolates both trajectories onto a common time grid before
comparing, because variants that change the time integration do not land on the
same time points. Report ``max_rel`` per profile.
"""

import argparse
import copy
import json
import os
import time

import jax
import numpy as np

_OUT_DIR = os.environ.get('RAMPUP_BENCH_DIR', '/tmp/rampup_bench')
_CONFIG_PATH = 'torax/tests/test_data/test_iterhybrid_rampup.py'
_PROFILES = ('T_i', 'T_e', 'n_e', 'psi')
# Face-grid diagnostics. `q_face` is a *gradient* of psi, so it amplifies error
# in the psi solution rather than inheriting it one-for-one -- which makes it
# the sensitive test for the psi-splitting variants. `s_face` (magnetic shear)
# differentiates once more again.
_FACE_DIAGNOSTICS = ('q_face', 's_face')


def _build(overrides, n_rho, t_final):
  """Loads the rampup config, applies overrides, returns a prepared sim."""
  from torax._src.config import config_loader
  from torax._src.orchestration import run_simulation
  from torax._src.torax_pydantic import model_config

  repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  cfg = copy.deepcopy(
      config_loader.import_module(os.path.join(repo, _CONFIG_PATH))['CONFIG']
  )
  if n_rho is not None:
    cfg['geometry']['n_rho'] = n_rho
  if t_final is not None:
    cfg['numerics']['t_final'] = t_final
  _deep_update(cfg, overrides or {})
  torax_config = model_config.ToraxConfig.from_dict(cfg)
  return torax_config, run_simulation.prepare_simulation(torax_config)


def _deep_update(dst, src):
  for k, v in src.items():
    if isinstance(v, dict) and isinstance(dst.get(k), dict):
      _deep_update(dst[k], v)
    else:
      dst[k] = v


def run(tag, overrides, n_rho, t_final, reps):
  """Records trajectory and timings for one variant."""
  os.makedirs(_OUT_DIR, exist_ok=True)
  torax_config, (s0, p0, step_fn) = _build(overrides, n_rho, t_final)

  t0 = time.perf_counter()
  lowered = jax.jit(step_fn.__call__).lower(s0, p0)
  t_trace = time.perf_counter() - t0
  t0 = time.perf_counter()
  compiled = lowered.compile()
  t_compile = time.perf_counter() - t0

  cost = compiled.cost_analysis()
  if isinstance(cost, (list, tuple)):
    cost = cost[0]
  text = compiled.as_text()

  # --- fixed-work timing: same step, same input, repeated ---
  s, p = step_fn(s0, p0)
  jax.block_until_ready((s, p))
  fixed = []
  for _ in range(reps):
    a = time.perf_counter()
    s, p = step_fn(s0, p0)
    jax.block_until_ready((s, p))
    fixed.append((time.perf_counter() - a) * 1e3)

  # --- full trajectory: accuracy record + felt runtime + step count ---
  traj = {name: [] for name in _PROFILES + _FACE_DIAGNOSTICS}
  times, dts, inner, outer, err = [], [], [], [], []
  s, p = s0, p0
  t_start = time.perf_counter()
  n_steps = 0
  while not step_fn.is_done(s.t):
    s, p = step_fn(s, p)
    jax.block_until_ready((s, p))
    n_steps += 1
    for name in _PROFILES:
      traj[name].append(np.asarray(getattr(s.core_profiles, name).value))
    for name in _FACE_DIAGNOSTICS:
      traj[name].append(np.asarray(getattr(s.core_profiles, name)))
    times.append(float(s.t))
    dts.append(float(s.dt))
    sno = s.solver_numeric_outputs
    inner.append(int(sno.inner_solver_iterations))
    outer.append(int(sno.outer_solver_iterations))
    err.append(int(sno.solver_error_state))
    if n_steps > 5000:
      raise RuntimeError('step cap exceeded')
  traj_s = time.perf_counter() - t_start

  summary = dict(
      tag=tag,
      n_rho=int(s0.geometry.torax_mesh.nx),
      t_final=float(torax_config.numerics.t_final),
      overrides=overrides or {},
      n_steps=n_steps,
      fixed_ms=float(np.median(fixed)),
      fixed_min_ms=float(np.min(fixed)),
      fixed_iqr_pct=float(
          (np.percentile(fixed, 75) - np.percentile(fixed, 25))
          / np.median(fixed) * 100
      ),
      fixed_all_ms=fixed,
      traj_s=traj_s,
      ms_per_step=traj_s / n_steps * 1e3,
      total_inner_iterations=int(sum(inner)),
      total_outer_iterations=int(sum(outer)),
      max_error_state=int(max(err)),
      t_trace_s=t_trace,
      t_compile_s=t_compile,
      flops=float(cost.get('flops', 0)),
      bytes=float(cost.get('bytes accessed', 0)),
      temp_mb=compiled.memory_analysis().temp_size_in_bytes / 1e6,
      hlo_lines=len(text.splitlines()),
      fusions=text.count(' fusion('),
  )
  np.savez(
      os.path.join(_OUT_DIR, f'{tag}.npz'),
      t=np.array(times), dt=np.array(dts),
      inner=np.array(inner), outer=np.array(outer), err=np.array(err),
      **{name: np.stack(traj[name])
         for name in _PROFILES + _FACE_DIAGNOSTICS},
  )
  with open(os.path.join(_OUT_DIR, f'{tag}.json'), 'w') as f:
    json.dump(summary, f, indent=1)
  print('RESULT ' + json.dumps({
      k: v for k, v in summary.items() if k != 'fixed_all_ms'}))
  return summary


def cmp(tag_a, tag_b):
  """Compares two recorded trajectories on a common time grid."""
  a = np.load(os.path.join(_OUT_DIR, f'{tag_a}.npz'))
  b = np.load(os.path.join(_OUT_DIR, f'{tag_b}.npz'))
  ta, tb = a['t'], b['t']
  lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
  grid = np.linspace(lo, hi, 64)
  print(f'{tag_a} ({len(ta)} steps) vs {tag_b} ({len(tb)} steps)')
  print(f'  common time window [{lo:.4g}, {hi:.4g}]')
  worst = 0.0
  for name in _PROFILES:
    xa, xb = a[name], b[name]
    ia = np.stack([np.interp(grid, ta, xa[:, j]) for j in range(xa.shape[1])], 1)
    ib = np.stack([np.interp(grid, tb, xb[:, j]) for j in range(xb.shape[1])], 1)
    scale = max(np.abs(ia).max(), 1e-30)
    rel = np.abs(ia - ib).max() / scale
    worst = max(worst, rel)
    print(f'  {name:>4}: max_rel = {rel:.3e}')
  print(f'  WORST max_rel = {worst:.3e}')
  return worst


def show(tags):
  cols = ('n_steps', 'fixed_ms', 'fixed_iqr_pct', 'traj_s', 'ms_per_step',
          'total_inner_iterations', 'max_error_state', 't_compile_s',
          'flops', 'hlo_lines')
  print(f'{"tag":<28}' + ''.join(f'{c:>14}' for c in cols))
  base = None
  for tag in tags:
    with open(os.path.join(_OUT_DIR, f'{tag}.json')) as f:
      s = json.load(f)
    if base is None:
      base = s
    row = ''
    for c in cols:
      v = s[c]
      row += f'{v:>14.4g}' if isinstance(v, float) else f'{v:>14}'
    print(f'{tag:<28}' + row)
    if s is not base:
      d = lambda c: (s[c] / base[c] - 1) * 100 if base[c] else float('nan')
      print(f'{"  vs " + base["tag"]:<28}'
            + ''.join(f'{d(c):>13.1f}%' for c in cols))


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest='cmd', required=True)
  r = sub.add_parser('run')
  r.add_argument('tag')
  r.add_argument('--overrides', default=None)
  r.add_argument('--n-rho', type=int, default=None)
  r.add_argument('--t-final', type=float, default=None)
  r.add_argument('--reps', type=int, default=7)
  c = sub.add_parser('cmp')
  c.add_argument('tag_a')
  c.add_argument('tag_b')
  t = sub.add_parser('time')
  t.add_argument('tags', nargs='+')
  args = ap.parse_args()

  if args.cmd == 'run':
    run(args.tag, json.loads(args.overrides) if args.overrides else None,
        args.n_rho, args.t_final, args.reps)
  elif args.cmd == 'cmp':
    cmp(args.tag_a, args.tag_b)
  else:
    show(args.tags)


if __name__ == '__main__':
  main()
