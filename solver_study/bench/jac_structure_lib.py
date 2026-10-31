import numpy as np

def greedy_column_coloring(S):
  """Distance-2 (column) coloring: columns sharing a row cannot share a colour."""
  S = S.astype(bool)
  n = S.shape[1]
  # conflict graph: cols j,k conflict if they share any nonzero row
  C = (S.T.astype(np.int32) @ S.astype(np.int32)) > 0
  np.fill_diagonal(C, False)
  colors = -np.ones(n, dtype=int)
  for j in range(n):
    used = set(colors[C[j]][colors[C[j]] >= 0].tolist())
    c = 0
    while c in used:
      c += 1
    colors[j] = c
  return int(colors.max() + 1)

