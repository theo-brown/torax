"""Jacobian structure / preconditioner-quality analysis at a mid-sim state."""
import json, sys, time
import numpy as np
import jax, jax.numpy as jnp
import scipy.linalg
import scipy.sparse.linalg
from torax.examples import iterhybrid_rampup
from torax._src.solver import jax_root_finding
import bench_common as bc

OUT = {}
n_rho = int(sys.argv[1]) if len(sys.argv) > 1 else 50
smooth = sys.argv[2] if len(sys.argv) > 2 else 'on'
dt = float(sys.argv[3]) if len(sys.argv) > 3 else None
n_warm = int(sys.argv[4]) if len(sys.argv) > 4 else 3

extra = None
if smooth == 'off':
  extra = {'transport': {'smoothing_zones': [], 'smoothing_width': 0.0}}
p = bc.build_problem(iterhybrid_rampup.CONFIG, n_warm_steps=n_warm, dt=dt,
                     n_rho=n_rho, extra_overrides=extra)
N, n_ch, n_cells = p.n, len(p.evolving_names), p.n_rho
print(f'N={N} channels={p.evolving_names} n_rho={n_cells} t={float(p.state.t)} dt={float(p.dt)} smoothing={smooth}')
OUT.update(N=N, n_ch=n_ch, n_rho=n_cells, t=float(p.state.t), dt=float(p.dt), smoothing=smooth)

res = jax.jit(p.residual_fun)
jac = jax.jit(jax.jacfwd(p.residual_fun))
x0 = p.x0
r0 = res(x0)
J0 = np.asarray(jac(x0))
P0, b0, _ = p.picard_matrix(x0)
P0 = np.asarray(P0); b0 = np.asarray(b0)
# validate the picard matrix reconstruction: R(x0) == P0 x0 - b0
err_id = np.max(np.abs(P0 @ np.asarray(x0) - b0 - np.asarray(r0))) / (np.max(np.abs(r0)) + 1e-30)
print('picard identity check (rel max err):', err_id)
OUT['picard_identity_err'] = float(err_id)
print('initial residual mean-abs norm:', bc.mean_abs(r0))

# --- converged solution with TORAX's own Newton ---
x_star, meta = jax_root_finding.root_newton_raphson(
    p.residual_fun, x0, maxiter=30, tol=1e-5, coarse_tol=1e-2, use_jax_custom_root=False)
print('TORAX newton: iterations', int(meta.iterations), 'residual', float(meta.residual.__abs__().mean()) if hasattr(meta.residual,'mean') else None, 'error', int(meta.error))
OUT['torax_newton_iterations'] = int(meta.iterations)
Js = np.asarray(jac(x_star))
Ps, bs, _ = p.picard_matrix(x_star)
Ps = np.asarray(Ps)

def block_offset_profile(J):
  """max |J_ij| per (row ch, col ch, |cell offset|), normalised by block max."""
  prof = {}
  for a in range(n_ch):
    for b in range(n_ch):
      B = J[a*n_cells:(a+1)*n_cells, b*n_cells:(b+1)*n_cells]
      bmax = np.max(np.abs(B))
      offs = []
      for d in range(n_cells):
        vals = np.abs(np.diagonal(B, offset=d)).max() if d < n_cells else 0
        vals2 = np.abs(np.diagonal(B, offset=-d)).max() if d < n_cells else 0
        offs.append(max(vals, vals2) / (bmax + 1e-300))
      prof[(p.evolving_names[a], p.evolving_names[b])] = (bmax, np.array(offs))
  return prof

def bandwidth_stats(J, label):
  prof = block_offset_profile(J)
  print(f'\n[{label}] per-block max|J| by cell offset (relative to block max):')
  rows = {}
  for (ra, cb), (bmax, offs) in prof.items():
    bw6 = int(np.max(np.where(offs > 1e-6)[0])) if np.any(offs > 1e-6) else -1
    bw3 = int(np.max(np.where(offs > 1e-3)[0])) if np.any(offs > 1e-3) else -1
    bw2 = int(np.max(np.where(offs > 1e-2)[0])) if np.any(offs > 1e-2) else -1
    s = ' '.join(f'{v:.0e}' for v in offs[:12])
    print(f'  ({ra:>3},{cb:>3}) blockmax={bmax:.2e} bw(>1e-2)={bw2:3d} bw(>1e-3)={bw3:3d} bw(>1e-6)={bw6:3d}  offsets0..11: {s}')
    rows[f'{ra},{cb}'] = dict(blockmax=float(bmax), bw_1e2=bw2, bw_1e3=bw3, bw_1e6=bw6, offsets=[float(v) for v in offs[:40]])
  # global: fraction of Frobenius norm within band k
  cell_of = np.arange(N) % n_cells
  D = np.abs(cell_of[:, None] - cell_of[None, :])
  fro = np.linalg.norm(J)
  fr = {}
  for k in [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
    fr[k] = float(np.linalg.norm(J * (D <= k)) / fro)
  print('  fraction of ||J||_F within cell-band k:', {k: round(v, 6) for k, v in fr.items()})
  nnz = int(np.sum(np.abs(J) > 1e-12 * np.max(np.abs(J))))
  print(f'  nnz(|J|>1e-12 max) = {nnz} of {N*N} ({nnz/N/N:.3f}); dense fraction')
  # greedy coloring of the structural pattern above threshold -> number of colours
  for thr in [1e-12, 1e-6, 1e-3]:
    S = np.abs(J) > thr * np.max(np.abs(J))
    ncol = greedy_column_coloring(S)
    print(f'  greedy column-coloring colours for pattern |J|>{thr:g}*max: {ncol}')
    rows[f'colors_thr_{thr:g}'] = ncol
  rows['fro_frac_band'] = fr
  rows['nnz_frac'] = nnz / N / N
  return rows

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

OUT['J_x0'] = bandwidth_stats(J0, 'J at x0 (linear initial guess)')
OUT['J_xstar'] = bandwidth_stats(Js, 'J at converged x*')

# --- Picard matrix as preconditioner / approximate Newton matrix ---
def analyze_pair(J, P, label):
  d = {}
  d['relfro_JmP'] = float(np.linalg.norm(J - P) / np.linalg.norm(J))
  d['cond_J'] = float(np.linalg.cond(J))
  d['cond_P'] = float(np.linalg.cond(P))
  PinvJ = np.linalg.solve(P, J)
  ev = np.linalg.eigvals(PinvJ)
  d['rho_I_minus_PinvJ'] = float(np.max(np.abs(1 - ev)))
  d['eig_PinvJ_real_range'] = [float(np.min(ev.real)), float(np.max(ev.real))]
  d['eig_PinvJ_imag_max'] = float(np.max(np.abs(ev.imag)))
  d['cond_PinvJ'] = float(np.linalg.cond(PinvJ))
  # GMRES iteration counts with left preconditioner P^{-1}
  rhs = -np.asarray(res(x0) if 'x0' in label else res(x_star))
  Pinv = scipy.sparse.linalg.LinearOperator(J.shape, matvec=lambda v: np.linalg.solve(P, v))
  its = {}
  for tol in [1e-2, 1e-4, 1e-6, 1e-8]:
    cnt = [0]
    def cb(rk): cnt[0] += 1
    xg, info = scipy.sparse.linalg.gmres(J, rhs, M=Pinv, rtol=tol, atol=0, restart=200, maxiter=200, callback=cb, callback_type='pr_norm')
    its[tol] = (cnt[0], int(info))
  d['gmres_iters_precP'] = {str(k): v for k, v in its.items()}
  cnt = [0]
  def cb2(rk): cnt[0] += 1
  # unpreconditioned but diagonally scaled
  Dg = scipy.sparse.linalg.LinearOperator(J.shape, matvec=lambda v: v / np.diag(J))
  xg, info = scipy.sparse.linalg.gmres(J, rhs, M=Dg, rtol=1e-6, atol=0, restart=400, maxiter=400, callback=cb2, callback_type='pr_norm')
  d['gmres_iters_jacobi_1e-6'] = (cnt[0], int(info))
  print(f'\n[{label}] ||J-P||/||J||={d["relfro_JmP"]:.3f} cond(J)={d["cond_J"]:.2e} cond(P)={d["cond_P"]:.2e} '
        f'rho(I-P^-1J)={d["rho_I_minus_PinvJ"]:.3f} eig(P^-1J) real in {d["eig_PinvJ_real_range"]} |imag|max={d["eig_PinvJ_imag_max"]:.3f}')
  print(f'  GMRES iters (M=P^-1): {d["gmres_iters_precP"]}   Jacobi-scaled GMRES to 1e-6: {d["gmres_iters_jacobi_1e-6"]}')
  return d

OUT['pair_x0'] = analyze_pair(J0, P0, 'x0: J vs Picard P')
OUT['pair_xstar'] = analyze_pair(Js, Ps, 'x*: J vs Picard P')

# --- lagged / truncated Jacobians as approximate Newton matrices ---
def rho_approx(A, J):
  ev = np.linalg.eigvals(np.linalg.solve(A, J))
  return float(np.max(np.abs(1 - ev)))
lag = {}
lag['rho_Jx0_vs_Jxstar'] = rho_approx(J0, Js)
lag['rho_Jxold_vs_Jxstar'] = rho_approx(np.asarray(jac(p.x_old_vec)), Js)
print('\nchord convergence factor rho(I - J(x0)^-1 J(x*)) =', lag['rho_Jx0_vs_Jxstar'], ' with J(x_old):', lag['rho_Jxold_vs_Jxstar'])
for k in [1, 2, 3, 4, 6, 8, 12, 16, 24]:
  Jt = np.asarray(bc.truncate_band(Js, k, n_ch, n_cells))
  lag[f'rho_band{k}'] = rho_approx(Jt, Js)
print('band-truncated J(x*) as Newton matrix: rho(I - Jk^-1 J) =', {k: round(v, 4) for k, v in lag.items() if 'band' in k})
OUT['lagged'] = lag

json.dump(OUT, open(f'jac_structure_n{n_rho}_{smooth}_dt{float(p.dt):g}.json', 'w'), indent=1, default=str)
print('DONE')
