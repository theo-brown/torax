#!/usr/bin/env python
"""Markdown table for the prod_* dense-vs-structured end-to-end runs."""
import json
import os
import sys

import numpy as np

res_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
cases = [
    ('prodg_rampup50', '`iterhybrid_rampup` + cyclotron, constant-fraction radiation, ToricNN ICRH', 50),
    ('prodg_rampup100', 'same, n_rho = 100', 100),
    ('prod_rampup50', '`iterhybrid_rampup` (dt = 2 s, 40 steps)', 50),
    ('prod_rampup100', '`iterhybrid_rampup`, n_rho = 100', 100),
    ('prod_scale25', '`iterhybrid_predictor_corrector`, Newton, chi dt, t_final = 1 s', 25),
    ('prod_scale50', 'same, n_rho = 50', 50),
    ('prod_scale100', 'same, n_rho = 100', 100),
]
print('| case | n_rho | N | mode | steps | compile (first step) | run (excl. compile) | median step | Newton its total | ms per Newton it | final-profile rel. diff vs dense |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for key, desc, n_rho in cases:
  rows = {}
  for mode in ('dense', 'structured'):
    path = os.path.join(res_dir, f'{key}_{mode}.json')
    if not os.path.exists(path):
      continue
    rows[mode] = json.load(open(path))
  if 'dense' not in rows:
    continue
  ref = rows['dense']['final_profiles']
  for mode, d in rows.items():
    s, T = d['summary'], d['timings']
    n = s.get('n_steps', 0)
    run = s.get('run_time_clean_s', 0.0)
    its = s.get('total_inner_iterations_clean', 0)
    per_it = 1e3 * run / its if its else float('nan')
    if mode == 'dense':
      diff = '-'
    else:
      diffs = []
      for name in ('T_i', 'T_e', 'n_e', 'psi'):
        a = np.asarray(d['final_profiles'][name]); b = np.asarray(ref[name])
        diffs.append(np.max(np.abs(a - b)) / np.max(np.abs(b)))
      diff = f'{max(diffs):.1e}'
    err = d.get('sim_error', 'NO_ERROR')
    flag = '' if err in ('NO_ERROR', 'SimError.NO_ERROR') else f' ({err})'
    print(f"| {desc if mode == 'dense' else ''} | {n_rho} | {4 * n_rho} | {mode} | {n}{flag} | {s.get('first_step_wall_s', 0):.1f} s | {run:.2f} s | {1e3 * s.get('step_wall_median_s', 0):.0f} ms | {s.get('total_inner_iterations')} | {per_it:.0f} | {diff} |")
