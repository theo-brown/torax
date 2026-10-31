"""Diagnose the Newton stall at small dt: residual distribution, Taylor
(linearisation) test along the Newton direction, and near-singular directions."""
import sys, json
import numpy as np
import jax, jax.numpy as jnp
from torax.examples import iterhybrid_rampup
import bench_common as bc

dts = [float(a) for a in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0.02, 0.05, 0.2]
n_warm = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT = {}
for dt in dts:
  p = bc.build_problem(iterhybrid_rampup.CONFIG, n_warm_steps=n_warm, dt=dt, solver_overrides={'use_pereverzev': True})
  N, n_ch, n_cells = p.n, len(p.evolving_names), p.n_rho
  res = jax.jit(p.residual_fun); jac = jax.jit(jax.jacfwd(p.residual_fun))
  r = bc.newton_generic(res, jac, p.x0, maxiter=6)
  x = r['x']; rx = np.asarray(res(x)); J = np.asarray(jac(x))
  print(f'\n##### dt={dt} t={float(p.state.t)}: newton it={r["iterations"]} conv={r["converged"]} final={r["final_norm"]:.2e} taus={r["taus"]}')
  R2 = np.abs(rx).reshape(n_ch, n_cells)
  print('  mean|R| by channel:', {n: '%.2e' % v for n, v in zip(p.evolving_names, R2.mean(axis=1))})
  for i, n in enumerate(p.evolving_names):
    top = np.argsort(R2[i])[::-1][:5]
    print(f'   {n}: largest |R| at cells {top.tolist()} values {[float("%.1e" % R2[i][c]) for c in top]}')
  # Taylor test along Newton direction
  d = np.linalg.solve(J, -rx)
  print('  |d|_max=%.3e (rel to x max %.3e)' % (np.max(np.abs(d)), np.max(np.abs(d)) / np.max(np.abs(np.asarray(x)))))
  rows = []
  for s in [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]:
    rs = np.asarray(res(jnp.asarray(x + s * d)))
    lin = rx + s * (J @ d)
    err = np.linalg.norm(rs - lin) / np.linalg.norm(rs - rx + 1e-300)
    rows.append((s, bc.mean_abs(rs), err))
    print(f'   s={s:<6g} mean|R(x+sd)|={bc.mean_abs(rs):.3e}  (predicted {bc.mean_abs(lin):.3e})  linearisation err/|dR| = {err:.3e}')
  # where does the linearisation fail at s=1e-2?
  s = 1e-2
  rs = np.asarray(res(jnp.asarray(x + s * d))); lin = rx + s * (J @ d)
  dev = np.abs(rs - lin).reshape(n_ch, n_cells)
  for i, n in enumerate(p.evolving_names):
    top = np.argsort(dev[i])[::-1][:4]
    print(f'   nonlinearity (s=1e-2) {n}: cells {top.tolist()} dev {[float("%.1e" % dev[i][c]) for c in top]} vs |dR| {[float("%.1e" % abs(rs-rx).reshape(n_ch,n_cells)[i][c]) for c in top]}')
  # singular values
  U, S, Vt = np.linalg.svd(J)
  print('  cond(J)=%.2e; smallest sing. values %s' % (S[0] / S[-1], ['%.2e' % v for v in S[-4:]]))
  v = Vt[-1].reshape(n_ch, n_cells)
  print('  smallest right singular vector: channel weights', {n: '%.2f' % np.linalg.norm(v[i]) for i, n in enumerate(p.evolving_names)}, 'peak cells', [int(np.argmax(np.abs(v[i]))) for i in range(n_ch)])
  # transport coefficient clipping status at x (chi_min/chi_max hits) via core_transport at solution
  OUT[str(dt)] = dict(it=r['iterations'], conv=r['converged'], final=r['final_norm'], taylor=rows)
json.dump(OUT, open('stall_diag.json', 'w'), indent=1, default=str)
print('DONE')
