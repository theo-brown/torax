#!/usr/bin/env python
"""Aggregate results/*.json into all_results.json and markdown tables (tables.md)."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = {}
for fn in sorted(glob.glob(os.path.join(HERE, 'results', '*.json'))):
  with open(fn) as fh:
    d = json.load(fh)
  RES[d['meta']['label']] = d

with open(os.path.join(HERE, 'all_results.json'), 'w') as fh:
  json.dump(RES, fh, indent=1)


def g(label):
  return RES.get(label)


def f(x, nd=3):
  if x is None:
    return 'n/a'
  if isinstance(x, str):
    return x
  return f'{x:.{nd}f}'


def hist_str(h):
  if not h:
    return 'n/a'
  items = sorted(h.items(), key=lambda kv: float(kv[0]))
  return ', '.join(f'{k}:{v}' for k, v in items)


def solver_desc(m):
  s = m.get('solver', {})
  if not isinstance(s, dict):
    return str(s)
  st = s.get('solver_type')
  if st == 'newton_raphson':
    return (f"newton (pc={s.get('use_predictor_corrector')}, "
            f"n_corr={s.get('n_corrector_steps')}, guess={s.get('initial_guess_mode')}, "
            f"vmap_ls={s.get('vmap_linesearch')}, maxiter={s.get('n_max_iterations')})")
  return (f"{st} (pc={s.get('use_predictor_corrector')}, "
          f"n_corr={s.get('n_corrector_steps')}, pereverzev={s.get('use_pereverzev')})")


def dt_desc(m):
  return m.get('time_step_calculator', {}).get('calculator_type', 'n/a')


lines = []
P = lines.append


def table(header, rows):
  P('| ' + ' | '.join(header) + ' |')
  P('|' + '|'.join(['---'] * len(header)) + '|')
  for r in rows:
    P('| ' + ' | '.join(str(x) for x in r) + ' |')
  P('')


# ---------------------------------------------------------------- baselines
def baseline_row(label, note=''):
  d = g(label)
  if d is None:
    return [label, 'MISSING'] + [''] * 14
  m, s, T = d['meta'], d['summary'], d['timings']
  return [
      label + (f' ({note})' if note else ''),
      m['config'].split('.')[-1],
      solver_desc(m),
      m.get('n_rho'),
      dt_desc(m),
      s.get('n_steps'),
      f(d['final_t'], 3),
      f(s.get('first_step_wall_s'), 2),
      str(s.get('compile_event_steps')) + ' -> ' + str([round(x, 2) for x in s.get('compile_event_walls_s', [])]),
      f(s.get('run_time_clean_s'), 2),
      f(s.get('step_wall_mean_s'), 4) + ' / ' + f(s.get('step_wall_median_s'), 4) + ' / ' + f(s.get('step_wall_max_s'), 4),
      s.get('total_inner_iterations'),
      hist_str(s.get('inner_iterations_histogram')),
      s.get('n_steps_outer_gt1'),
      f"{s.get('n_steps_err2')} / {s.get('n_steps_err1')}",
      f(s.get('dt_min'), 5) + ' / ' + f(s.get('dt_median'), 5) + ' / ' + f(s.get('dt_max'), 5),
      f(s.get('time_per_inner_iteration_s'), 4),
      d['sim_error'],
  ]


BASE_HDR = ['run', 'config', 'solver', 'n_rho', 'dt calc', 'steps', 't_end [s]',
            'first step (compile+1 step) [s]', 'compile events (step -> wall s)',
            'run time excl. compile [s]', 'step wall mean/median/max [s]',
            'total inner iters', 'inner-iter histogram (iters:steps)',
            'steps outer>1', 'steps err2 / err1', 'dt min/median/max [s]',
            'time per inner iter [s]', 'sim_error']

P('## T1. Baseline end-to-end runs (as-is configs unless noted)')
P('')
table(BASE_HDR, [baseline_row(l, n) for l, n in [
    ('base_rampup', ''), ('base_pc', ''), ('base_basic', ''), ('base_step', ''),
    ('base_small_dt', 'benchmark, capped steps'), ('rampup_log', 'log_iterations=True'),
] if g(l) is not None])

P('### T1b. Process-level timings for the baseline runs [s]')
P('')
table(['run', 'import torax+jax', 'config module import', 'ToraxConfig.from_dict',
       'prepare_simulation (initial state)', 'step loop total', 'process total'],
      [[l, f(d['timings'].get('import_s'), 2), f(d['timings'].get('config_module_import_s'), 2),
        f(d['timings'].get('from_dict_s'), 2), f(d['timings'].get('prepare_simulation_s'), 2),
        f(d['timings'].get('loop_total_s'), 2), f(d['timings'].get('process_total_s'), 2)]
       for l, d in RES.items() if l.startswith('base_') or l == 'rampup_log'])

# ---------------------------------------------------------------- newton log
d = g('rampup_log')
if d is not None:
  s = d['summary']
  P('## T2. Newton-Raphson iteration log analysis (iterhybrid_rampup, log_iterations=True)')
  P('')
  P(f"- total Newton iterations logged: {s.get('log_total_iterations')} over {s.get('n_steps')} steps")
  P(f"- iterations per step: {s.get('log_iterations_per_step')}")
  P(f"- iterations with tau < 1 (line-search backtracking): {s.get('log_n_iterations_with_tau_lt_1')}")
  P(f"- tau histogram (tau: count): {hist_str(s.get('log_tau_histogram'))}")
  P(f"- extra residual evaluations caused by backtracking (sum of log2(1/tau)): {s.get('log_extra_residual_evals_from_backtracking')}")
  P('')
  # per-step table
  rows = []
  for r in d['rows']:
    log = r.get('log', [])
    taus = [t for (_, _, t) in log]
    res = [x for (_, x, _) in log]
    rows.append([r['k'], f(r['t'], 1), f(r['dt'], 2), f(r['wall'], 3), r['inner'], r['outer'], r['err'],
                 len(log), sum(1 for t in taus if t < 0.999999),
                 ' '.join(f'{x:.1e}' for x in res)])
  table(['step', 't', 'dt', 'wall [s]', 'inner', 'outer', 'err', 'logged iters', 'iters w/ tau<1',
         'residual sequence (mean-abs norm)'], rows)
  # representative sequences with convergence ratios
  P('### T2b. Representative residual sequences and convergence-order diagnostics')
  P('')
  P('For a sequence r_k, linear convergence gives r_{k+1}/r_k ~ const, quadratic gives r_{k+1}/r_k^2 ~ const.')
  P('')
  cand = sorted(d['rows'], key=lambda r: -len(r.get('log', [])))
  picked = []
  seen = set()
  for r in cand[:2] + [rr for rr in d['rows'] if any(t < 0.999999 for (_, _, t) in rr.get('log', []))][:1] + d['rows'][len(d['rows'])//2:len(d['rows'])//2+1]:
    if r['k'] in seen:
      continue
    seen.add(r['k'])
    picked.append(r)
  for r in picked:
    log = r.get('log', [])
    if not log:
      continue
    P(f"step {r['k']} (t={r['t']:.1f}, dt={r['dt']:.2f}, outer={r['outer']}, err={r['err']}):")
    P('')
    rows = []
    prev = None
    for (i, res, tau) in log:
      lin = f'{res/prev:.3g}' if prev and prev > 0 else ''
      quad = f'{res/prev**2:.3g}' if prev and prev > 0 else ''
      rows.append([i, f'{res:.3e}', f'{tau:.4f}', lin, quad])
      prev = res
    table(['iter', 'residual', 'tau', 'r_k/r_{k-1}', 'r_k/r_{k-1}^2'], rows)

# ---------------------------------------------------------------- solver comparison
P('## T3. Solver comparison on identical physics (iterhybrid_predictor_corrector, t_final=2.0 s)')
P('')
cmp_labels = [('cmp_a_linear', '(a) linear as-is'), ('cmp_a10_linear', "(a') linear, n_corrector_steps=10"),
              ('cmp_b_newton', '(b) newton, chi dt'), ('cmp_c_newton_xold', "(c) newton, initial_guess_mode='x_old'"),
              ('cmp_d_newton_nopc', '(d) newton, use_predictor_corrector=False')]
rows = []
for l, note in cmp_labels:
  d = g(l)
  if d is None:
    continue
  s = d['summary']
  rows.append([note, s.get('n_steps'), f(d['final_t'], 3), f(s.get('first_step_wall_s'), 2),
               f(s.get('run_time_clean_s'), 2),
               f(s.get('step_wall_mean_s'), 4) + ' / ' + f(s.get('step_wall_median_s'), 4) + ' / ' + f(s.get('step_wall_max_s'), 4),
               s.get('total_inner_iterations'), f(s.get('inner_iterations_mean'), 2),
               hist_str(s.get('inner_iterations_histogram')), s.get('n_steps_outer_gt1'),
               f"{s.get('n_steps_err2')} / {s.get('n_steps_err1')}",
               f(s.get('dt_min'), 5) + ' / ' + f(s.get('dt_median'), 5) + ' / ' + f(s.get('dt_max'), 5),
               f(s.get('time_per_inner_iteration_s'), 4), d['sim_error']])
table(['run', 'steps', 't_end', 'first step [s]', 'run excl. compile [s]', 'step mean/median/max [s]',
       'total inner', 'mean inner/step', 'inner histogram', 'outer>1', 'err2 / err1',
       'dt min/median/max', 'time per inner iter [s]', 'sim_error'], rows)

ref = g('cmp_a_linear')
if ref is not None:
  P('### T3b. Final-profile differences vs (a) linear solver (max over grid)')
  P('')
  rows = []
  for l, note in cmp_labels[1:]:
    d = g(l)
    if d is None:
      continue
    r = []
    for name in ('T_i', 'T_e', 'n_e', 'psi'):
      a = np.array(ref['final_profiles'][name])
      b = np.array(d['final_profiles'][name])
      if a.shape != b.shape:
        r.append('shape mismatch')
        continue
      rel = np.max(np.abs(b - a) / np.maximum(np.abs(a), 1e-300))
      relnorm = np.max(np.abs(b - a)) / np.max(np.abs(a))
      r.append(f'{rel:.2e} (scaled: {relnorm:.2e})')
    rows.append([note, f(d['final_t'], 4)] + r)
  table(['run', 't_end', 'T_i max|rel diff|', 'T_e max|rel diff|', 'n_e max|rel diff|',
         'psi max|rel diff|'], rows)
  P('`rel` = max_i |b_i - a_i| / |a_i|; `scaled` = max_i |b_i - a_i| / max_i |a_i| (robust to near-zero values).')
  P('')

# ---------------------------------------------------------------- scaling
P('## T4. Grid-size scaling (iterhybrid_predictor_corrector physics, t_final=1.0 s, chi dt, step-capped)')
P('')
for solver, pref in (('newton_raphson', 'scale_newton'), ('linear', 'scale_linear')):
  rows = []
  base = None
  for n in (25, 50, 100, 200):
    d = g(f'{pref}_{n}')
    if d is None:
      continue
    s = d['summary']
    T = d['timings']
    if base is None:
      base = s
    rows.append([n, 4 * n, s.get('n_steps'), f(d['final_t'], 4),
                 f(s.get('dt_median'), 5),
                 f(s.get('first_step_wall_s'), 2),
                 f(T.get('aot_lower_s'), 2) if 'aot_lower_s' in T else 'n/a',
                 f(T.get('aot_compile_s'), 2) if 'aot_compile_s' in T else 'n/a',
                 f(s.get('step_wall_mean_s'), 4), f(s.get('step_wall_median_s'), 4),
                 f(s.get('step_wall_mean_s', 0) / base.get('step_wall_mean_s', 1), 2),
                 f(s.get('inner_iterations_mean'), 2),
                 f(s.get('time_per_inner_iteration_s'), 4),
                 f(s.get('time_per_inner_iteration_s', 0) / base.get('time_per_inner_iteration_s', 1), 2),
                 s.get('n_steps_outer_gt1'), f"{s.get('n_steps_err2')} / {s.get('n_steps_err1')}",
                 d['sim_error']])
  P(f'### {solver}')
  P('')
  table(['n_rho', 'N unknowns', 'steps run', 't reached', 'dt median', 'first step [s]',
         'AOT lower [s]', 'AOT compile [s]', 'step mean [s]', 'step median [s]', 'step mean / n_rho=25',
         'mean inner/step', 'time per inner iter [s]', 'per-iter / n_rho=25',
         'outer>1', 'err2 / err1', 'sim_error'], rows)

# ---------------------------------------------------------------- compile breakdown
P('## T5. Compile-time breakdown (Newton config (b), n_rho=50, fresh process each)')
P('')
rows = []
for l, note in [('compile_vmapF_pcT', 'vmap_linesearch=False, use_predictor_corrector=True (default (b))'),
                ('compile_vmapT_pcT', 'vmap_linesearch=True, use_predictor_corrector=True'),
                ('compile_vmapF_pcF', 'vmap_linesearch=False, use_predictor_corrector=False'),
                ('compile_vmapT_pcF', 'vmap_linesearch=True, use_predictor_corrector=False'),
                ('compile_linear', 'linear solver (a) for reference')]:
  d = g(l)
  if d is None:
    continue
  T, s = d['timings'], d['summary']
  ca = T.get('aot_cost_analysis', {})
  rows.append([note, f(T.get('prepare_simulation_s'), 2), f(T.get('aot_lower_s'), 2), f(T.get('aot_compile_s'), 2),
               f(T.get('aot_lower_s', 0) + T.get('aot_compile_s', 0), 2),
               f(s.get('first_step_wall_s'), 2), f(s.get('step_wall_median_s'), 4),
               f'{ca.get("flops", float("nan")):.3e}' if isinstance(ca, dict) else str(ca),
               T.get('aot_hlo_text_bytes'), s.get('total_inner_iterations'), s.get('n_steps')])
table(['config', 'prepare_simulation [s]', 'AOT trace+lower [s]', 'AOT XLA compile [s]', 'AOT total [s]',
       'first step_fn call [s]', 'median step [s]', 'XLA flops estimate', 'compiled HLO text bytes',
       'inner iters', 'steps'], rows)

# ---------------------------------------------------------------- decomposition
P('## T6. Per-step cost decomposition (Newton config (b), n_rho=50, adaptive_dt=False, chi dt)')
P('')
rows = []
for l, note in [('decomp_newton_full', 'full Newton step (reference)'),
                ('decomp_newton_maxiter0', 'n_max_iterations=0: pre-step + linear initial guess (PC) + 1 residual + post'),
                ('decomp_newton_maxiter0_xold', "n_max_iterations=0, x_old guess: pre-step + 1 residual + post"),
                ('decomp_linear_nopc', 'linear solver, use_predictor_corrector=False: pre-step + 1 linear solve + post'),
                ('decomp_linear_pc1', 'linear solver, PC with n_corrector_steps=1: pre-step + 2 linear solves + post'),
                ('decomp_linear_pc10', 'linear solver, PC with n_corrector_steps=10: pre-step + 11 linear solves + post')]:
  d = g(l)
  if d is None:
    continue
  s = d['summary']
  rows.append([note, s.get('n_steps'), f(s.get('first_step_wall_s'), 2), f(s.get('step_wall_mean_s'), 4),
               f(s.get('step_wall_median_s'), 4), f(s.get('inner_iterations_mean'), 2),
               f(s.get('time_per_inner_iteration_s'), 4), d['sim_error']])
table(['run', 'steps', 'first step [s]', 'step mean [s]', 'step median [s]', 'mean inner/step',
       'time per inner iter [s]', 'sim_error'], rows)

with open(os.path.join(HERE, 'tables.md'), 'w') as fh:
  fh.write('\n'.join(lines))
print('\n'.join(lines))
