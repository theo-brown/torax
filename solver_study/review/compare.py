"""python compare.py BASE.npz BRANCH.npz [--verbose]: bitwise comparison."""

import collections
import sys

import numpy as np


def main():
  a = np.load(sys.argv[1], allow_pickle=False)
  b = np.load(sys.argv[2], allow_pickle=False)
  verbose = '--verbose' in sys.argv
  ka, kb = set(a.files), set(b.files)
  only_a, only_b = sorted(ka - kb), sorted(kb - ka)
  if only_a:
    print('ONLY IN BASE:', len(only_a))
    for k in only_a[:50]:
      print('   ', k)
  if only_b:
    print('ONLY IN BRANCH:', len(only_b))
    for k in only_b[:50]:
      print('   ', k)
  n_equal = 0
  diffs = []
  per_prefix = collections.defaultdict(lambda: [0, 0])
  for k in sorted(ka & kb):
    x, y = a[k], b[k]
    prefix = k.split('/')[0]
    if x.shape != y.shape or x.dtype != y.dtype:
      diffs.append((k, f'shape/dtype {x.shape}{x.dtype} vs {y.shape}{y.dtype}',
                    np.inf))
      per_prefix[prefix][1] += 1
      continue
    if x.dtype.kind in 'fc':
      eq = np.array_equal(x, y, equal_nan=True)
    else:
      eq = np.array_equal(x, y)
    if eq:
      n_equal += 1
      per_prefix[prefix][0] += 1
      continue
    per_prefix[prefix][1] += 1
    if x.dtype.kind in 'fc':
      d = np.abs(x.astype(np.float64) - y.astype(np.float64))
      scale = np.maximum(np.abs(x), np.abs(y)).astype(np.float64)
      with np.errstate(invalid='ignore', divide='ignore'):
        rel = np.where(scale > 0, d / scale, 0.0)
      nan_mismatch = np.sum(np.isnan(x) != np.isnan(y))
      diffs.append((
          k,
          f'max_abs={np.nanmax(d):.3e} max_rel={np.nanmax(rel):.3e} '
          f'n_diff={int(np.sum(d > 0))}/{x.size} nan_mismatch={nan_mismatch}',
          float(np.nanmax(rel)),
      ))
    else:
      diffs.append((k, f'non-float mismatch {x.ravel()[:5]} vs {y.ravel()[:5]}',
                    np.inf))
  print(f'{n_equal} bitwise-equal arrays, {len(diffs)} differing arrays')
  for prefix, (ne, nd) in sorted(per_prefix.items()):
    print(f'  {prefix:30s} equal={ne:5d} differ={nd:5d}')
  diffs.sort(key=lambda t: -t[2])
  for k, msg, _ in diffs[: (None if verbose else 60)]:
    print('DIFF', k, msg)


if __name__ == '__main__':
  main()
