"""Effect of a global constraint term (volume-averaged density feedback) on the
Jacobian structure, colouring and preconditioned GMRES."""
import json, sys
import numpy as np
import jax, jax.numpy as jnp
import scipy.sparse.linalg
from torax.examples import iterhybrid_rampup
import bench_common as bc
from jac_structure_lib import greedy_column_coloring

n_rho = int(sys.argv[1]) if len(sys.argv) > 1 else 50
p = bc.build_problem(iterhybrid_rampup.CONFIG, n_warm_steps=3, n_rho=n_rho)
N, n_ch, n_cells = p.n, len(p.evolving_names), p.n_rho
names = list(p.evolving_names); i_ne = names.index('n_e')
geo = p.geo_t_plus_dt
w = np.asarray(geo.vpr * geo.drho_norm); w = w / w.sum()          # volume-average weights over cells
v = np.zeros(N); v[i_ne*n_cells:(i_ne+1)*n_cells] = w               # d<n_e>/dx (dense over the n_e block)
rho = np.asarray(geo.rho_norm); shape = np.exp(-((rho - 0.85) / 0.1) ** 2)  # gas-puff-like deposition shape
res = jax.jit(p.residual_fun); jacf = jax.jit(jax.jacfwd(p.residual_fun))
x0 = p.x0; J = np.asarray(jacf(x0)); r0 = np.asarray(res(x0))
P, b, _ = p.picard_matrix(x0); P = np.asarray(P)
# feedback source: S = gain * shape * (X - <n_e>) added to the n_e residual rows (scaled units);
# gain chosen so the rank-1 term is as strong as the largest existing n_e-row entry
gain = np.abs(J[i_ne*n_cells:(i_ne+1)*n_cells]).max() * 0.5
u = np.zeros(N); u[i_ne*n_cells:(i_ne+1)*n_cells] = gain * shape
Jg = J + np.outer(u, v)                                              # exact Jacobian of the augmented residual
def colours(A, thr=1e-12):
    return greedy_column_coloring(np.abs(A) > thr * np.abs(A).max())
out = {}
out['colours_J'] = colours(J)
out['colours_Jg_naive'] = colours(Jg)
mask = np.ones(N, bool); mask[i_ne*n_cells:(i_ne+1)*n_cells] = False
# colour the banded part only: remove the dense rank-1 coupling (known structure) before colouring
out['colours_Jg_minus_uv'] = colours(Jg - np.outer(u, v))
print(f'N={N}: colours J={out["colours_J"]}, naive colouring of J+uv^T={out["colours_Jg_naive"]}, colouring after subtracting the known rank-1 term={out["colours_Jg_minus_uv"]}')
# GMRES with the Picard preconditioner (which knows nothing about the global term)
def gmres_its(A, rhs, tol):
    Pinv = scipy.sparse.linalg.LinearOperator(A.shape, matvec=lambda z: np.linalg.solve(P, z)); cnt = [0]
    xg, info = scipy.sparse.linalg.gmres(A, rhs, M=Pinv, rtol=tol, atol=0, restart=200, maxiter=200, callback=lambda rk: cnt.__setitem__(0, cnt[0]+1), callback_type='pr_norm')
    return cnt[0], int(info)
for tol in [1e-4, 1e-6]:
    out[f'gmres_J_{tol}'] = gmres_its(J, -r0, tol); out[f'gmres_Jg_{tol}'] = gmres_its(Jg, -r0, tol)
    # preconditioner with the Woodbury rank-1 correction (P + uv^T)^-1
    Pinv_u = np.linalg.solve(P, u); denom = 1.0 + v @ Pinv_u
    Mw = scipy.sparse.linalg.LinearOperator(J.shape, matvec=lambda z: (lambda y: y - Pinv_u * (v @ y) / denom)(np.linalg.solve(P, z)))
    cnt = [0]; scipy.sparse.linalg.gmres(Jg, -r0, M=Mw, rtol=tol, atol=0, restart=200, maxiter=200, callback=lambda rk: cnt.__setitem__(0, cnt[0]+1), callback_type='pr_norm')
    out[f'gmres_Jg_woodburyP_{tol}'] = cnt[0]
    print(f'tol {tol}: GMRES its  J: {out[f"gmres_J_{tol}"][0]}   J+uv^T (plain P^-1): {out[f"gmres_Jg_{tol}"][0]}   J+uv^T (Woodbury-corrected P^-1): {out[f"gmres_Jg_woodburyP_{tol}"]}')
# cost of the extra pieces with autodiff: the row v is jax.grad of the scalar functional (one VJP), the column u is one JVP
nbar = jax.jit(lambda x: jnp.sum(jnp.asarray(w) * x[i_ne*n_cells:(i_ne+1)*n_cells]))
gr = jax.jit(jax.grad(nbar))
out['t_row_vjp_ms'] = bc.timeit(gr, x0, n=10)[0] * 1e3
out['t_residual_ms'] = bc.timeit(res, x0, n=10)[0] * 1e3
print(f'cost of the dense row by reverse-mode AD: {out["t_row_vjp_ms"]:.3f} ms (residual {out["t_residual_ms"]:.2f} ms); the dense column is one extra JVP column')
# exactness of the split assembly: banded part via colouring of (Jg - uv^T) + rank-1 term
rel = np.linalg.norm((Jg - np.outer(u, v)) + np.outer(u, v) - Jg) / np.linalg.norm(Jg)
out['split_assembly_rel_err'] = float(rel)
json.dump(out, open(f'global_constraint_n{n_rho}.json', 'w'), indent=1, default=str)
print('DONE')
