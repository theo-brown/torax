"""Table of first vs repeated run_simulation calls from first_vs_repeat.jsonl.

python summarize_first_vs_repeat.py NEW.jsonl [OLD.jsonl]
"repeat" is the mean of the calls after the first. With OLD, the structured
columns of OLD are shown next to the new ones.
"""
import json
import sys

LABELS = {
    'rampup': '`iterhybrid_rampup`',
    'rampup_global': '+ cyclotron, constant-fraction radiation, ToricNN ICRH',
    'predictor_corrector': '`iterhybrid_predictor_corrector`, Newton, t_final = 1 s',
}


def load(path):
  out = {}
  for line in open(path):
    r = json.loads(line)
    runs = r['runs']
    out[(r['case'], r['n_rho'], r['mode'])] = dict(
        first=runs[0]['wall_s'],
        repeat=sum(x['wall_s'] for x in runs[1:]) / (len(runs) - 1),
        steps=runs[0]['steps'],
        compiles=runs[0]['compiles'],
        compile_s=runs[0]['compile_s'],
        T_e0=runs[0]['T_e0'],
    )
  return out


new = load(sys.argv[1])
old = load(sys.argv[2]) if len(sys.argv) > 2 else None
header = '| case | n_rho | steps | dense first | structured first | dense repeat | structured repeat | repeat speedup |'
if old:
  header += ' structured first / repeat before |'
print(header)
print('|' + '---|' * (header.count('|') - 1))
for case in ('rampup', 'rampup_global', 'predictor_corrector'):
  for n_rho in (25, 50, 100):
    d, s = new.get((case, n_rho, 'dense')), new.get((case, n_rho, 'structured'))
    if not d or not s:
      continue
    row = (
        f'| {LABELS[case]} | {n_rho} | {d["steps"]} | {d["first"]:.1f} s |'
        f' {s["first"]:.1f} s | {d["repeat"]:.2f} s | {s["repeat"]:.2f} s |'
        f' {d["repeat"] / s["repeat"]:.1f}x |'
    )
    if old and (case, n_rho, 'structured') in old:
      o = old[(case, n_rho, 'structured')]
      row += f' {o["first"]:.1f} s / {o["repeat"]:.2f} s |'
    print(row)
    assert d['steps'] == s['steps'], (case, n_rho)
    rel = abs(d['T_e0'] - s['T_e0']) / abs(d['T_e0'])
    if rel > 1e-10:
      print(f'  (T_e0 differs by {rel:.1e})')
