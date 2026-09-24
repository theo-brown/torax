"""Compares the dense and structured runs of e2e_gaps.py: python e2e_compare.py OUT.jsonl"""
import json
import sys

import numpy as np

runs = {}
for line in open(sys.argv[1]):
  r = json.loads(line)
  runs[(r['case'], r['n_rho'], r['mode'])] = r
for (case, n_rho, mode), s in sorted(runs.items()):
  if mode != 'structured' or (case, n_rho, 'dense') not in runs:
    continue
  d = runs[(case, n_rho, 'dense')]
  diff = max(
      float(np.max(np.abs(np.subtract(s[k], d[k])) / np.maximum(np.abs(d[k]), 1e-300)))
      for k in ('T_i', 'T_e', 'n_e', 'psi')
  )
  h_mode = sum(v < 0.99 for v in s['chi_e_pedestal_min'])
  print(
      f'{case:28s} n_rho={n_rho:3d} steps {d["steps"]}/{s["steps"]}'
      f' newton {d["newton_iterations"]}/{s["newton_iterations"]}'
      f' coarse {d["coarse_tol_steps"]}/{s["coarse_tol_steps"]}'
      f' t_end {d["t_end"]:.2f}/{s["t_end"]:.2f} max rel diff {diff:.1e}'
      f' steps with chi_e<0.99 above rho 0.9: {h_mode}'
      f' wall {d["wall_s"]:.0f}/{s["wall_s"]:.0f} s'
  )
