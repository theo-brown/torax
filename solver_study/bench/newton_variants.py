"""Iteration-count experiment: full Newton vs chord/lagged, Newton-Picard,
band-truncated, and Jacobian-free Newton-Krylov with Picard preconditioner."""
import json, sys
import numpy as np
import jax, jax.numpy as jnp
import scipy.sparse.linalg
from torax.examples import iterhybrid_rampup
from torax._src.fvm import calc_coeffs, fvm_conversions, residual_and_loss
from torax._src.core_profiles import updaters
import bench_common as bc

n_rho = int(sys.argv[1]) if len(sys.argv) > 1 else 50
dts = [float(a) for a in sys.argv[2].split(',')] if len(sys.argv) > 2 else [2.0, 0.5, 0.1, 0.02]
n_warm = int(sys.argv[3]) if len(sys.argv) > 3 else 3
OUT = {}

def picard_matrix_pereverzev(p, x_vec):
  x_new_guess = fvm_conversions.vec_to_cell_variable_tuple(x_vec, p.core_profiles_t_plus_dt, p.evolving_names)
  cp = updaters.update_core_profiles_during_step(x_new_guess, p.runtime_params_t_plus_dt, p.geo_t_plus_dt,
      p.core_profiles_t_plus_dt, prev_core_profiles=p.core_profiles_t, dt=p.dt, evolving_names=p.evolving_names)
  coeffs_new = calc_coeffs.calc_coeffs(runtime_params=p.runtime_params_t_plus_dt, geo=p.geo_t_plus_dt,
      core_profiles=cp, explicit_source_profiles=p.explicit_source_profiles, models=p.step_fn.solver.models,
      evolving_names=p.evolving_names, use_pereverzev=True, pedestal_transition_state=p.pedestal_transition_state)
  sp = p.runtime_params_t_plus_dt.solver
  lhs, lhs_vec, rhs, rhs_vec = residual_and_loss.theta_method_matrix_equation(
      dt=p.dt, x_old=p.x_old, x_new_guess=x_new_guess, coeffs_old=p.coeffs_old, coeffs_new=coeffs_new,
      theta_implicit=sp.theta_implicit, convection_dirichlet_mode=sp.convection_dirichlet_mode,
      convection_neumann_mode=sp.convection_neumann_mode)
  if coeffs_new.has_internal_boundary_conditions:
    lhs, lhs_vec, rhs, rhs_vec = residual_and_loss.apply_internal_boundary_conditions(lhs, lhs_vec, rhs, rhs_vec,
        coeffs_new.internal_boundary_condition_mask, coeffs_new.internal_boundary_condition_target_vec)
  n_cells, n_ch = p.n_rho, len(p.evolving_names)
  perm = np.arange(n_cells * n_ch).reshape(n_cells, n_ch).T.reshape(-1)
  return np.asarray(lhs.to_dense())[np.ix_(perm, perm)]

for dt in dts:
  p = bc.build_problem(iterhybrid_rampup.CONFIG, n_warm_steps=n_warm, dt=dt, n_rho=n_rho,
                       solver_overrides={'use_pereverzev': True})
  N, n_ch, n_cells = p.n, len(p.evolving_names), p.n_rho
  res = jax.jit(p.residual_fun)
  jac = jax.jit(jax.jacfwd(p.residual_fun))
  jvp = jax.jit(lambda x, v: jax.jvp(p.residual_fun, (x,), (v,))[1])
  x0 = p.x0
  r0 = res(x0)
  print(f'\n##### dt={dt} t={float(p.state.t)} N={N}: initial residual norm {bc.mean_abs(r0):.3e}, |x0-x_old|/|x_old|={float(jnp.linalg.norm(x0-p.x_old_vec)/jnp.linalg.norm(p.x_old_vec)):.3e}')
  R = {}
  base = bc.newton_generic(res, jac, x0)
  R['full_newton'] = base
  print(f"full Newton: it={base['iterations']} conv={base['converged']} res={base['n_res']} jac={base['n_jac']} ls_extra={base['n_ls_extra']} hist={['%.1e'%h for h in base['hist']]}")
  # from x_old as initial guess
  bo = bc.newton_generic(res, jac, p.x_old_vec)
  R['full_newton_from_xold'] = bo
  print(f"full Newton from x_old: it={bo['iterations']} conv={bo['converged']} res={bo['n_res']} jac={bo['n_jac']} ls_extra={bo['n_ls_extra']} hist={['%.1e'%h for h in bo['hist']]}")
  # distance to converged solution after each iteration, relative to the step change |x*-x_old|
  xstar = base['x']; dstep = float(jnp.linalg.norm(xstar - p.x_old_vec))
  dist = [float(jnp.linalg.norm(xx - xstar)) / dstep for xx in base['xhist']]
  R['full_newton']['rel_dist_to_xstar'] = dist
  print('  full Newton: |x_k - x*| / |x* - x_old| per iteration:', ['%.1e' % d for d in dist])
  print('  residual norm per iteration:', ['%.1e' % h for h in base['hist']])
  # linear extrapolation predictor (free initial guess, as in BDF/Nordsieck predictors)
  if p.prev_x_vec is not None:
    ratio = float(p.dt) / p.prev_dt
    x_ex = p.x_old_vec + ratio * (p.x_old_vec - p.prev_x_vec)
    be = bc.newton_generic(res, jac, x_ex)
    R['full_newton_from_extrap'] = be
    print(f"full Newton from extrapolated guess (dt ratio {ratio:.2f}): it={be['iterations']} conv={be['converged']} res={be['n_res']} jac={be['n_jac']} ls_extra={be['n_ls_extra']} hist={['%.1e'%h for h in be['hist']]}")
    print(f"  initial-guess quality |x_guess - x*|/|x*-x_old|: linear PC {float(jnp.linalg.norm(x0-xstar))/dstep:.3e}, x_old 1.0, extrapolation {float(jnp.linalg.norm(x_ex-xstar))/dstep:.3e}")
  for je in [2, 3]:
    r = bc.newton_generic(res, jac, x0, jac_every=je, maxiter=60)
    R[f'jac_every_{je}'] = r
    print(f"Jacobian every {je}: it={r['iterations']} conv={r['converged']} res={r['n_res']} jac={r['n_jac']} ls_extra={r['n_ls_extra']} final={r['final_norm']:.1e}")
  r = bc.newton_generic(res, jac, x0, fixed_jac=jac(x0), maxiter=100)
  R['chord_J_x0'] = r
  print(f"chord (J at x0 fixed): it={r['iterations']} conv={r['converged']} res={r['n_res']} ls_extra={r['n_ls_extra']} final={r['final_norm']:.1e}")
  r = bc.newton_generic(res, jac, x0, fixed_jac=jac(p.x_old_vec), maxiter=100)
  R['chord_J_xold'] = r
  print(f"chord (J at x_old fixed): it={r['iterations']} conv={r['converged']} res={r['n_res']} ls_extra={r['n_ls_extra']} final={r['final_norm']:.1e}")
  # Newton-Picard: use the frozen-coefficient matrix P(x) (no dC/dx terms) as Newton matrix
  Pfn = lambda x: p.picard_matrix(x)[0]
  r = bc.newton_generic(res, Pfn, x0, maxiter=100)
  R['newton_picard_P'] = r
  print(f"Newton with Picard P(x): it={r['iterations']} conv={r['converged']} res={r['n_res']} ls_extra={r['n_ls_extra']} final={r['final_norm']:.1e}")
  Ppc = lambda x: jnp.asarray(picard_matrix_pereverzev(p, x))
  r = bc.newton_generic(res, Ppc, x0, maxiter=100)
  R['newton_picard_P_pereverzev'] = r
  print(f"Newton with Pereverzev-augmented P(x): it={r['iterations']} conv={r['converged']} res={r['n_res']} ls_extra={r['n_ls_extra']} final={r['final_norm']:.1e}")
  # band-truncated Jacobian
  for k in [2, 4, 8]:
    r = bc.newton_generic(res, jac, x0, band_truncate=k, n_ch=n_ch, n_cells=n_cells, maxiter=60)
    R[f'band_trunc_{k}'] = r
    print(f"band-truncated J (k={k} cells): it={r['iterations']} conv={r['converged']} res={r['n_res']} jac={r['n_jac']} final={r['final_norm']:.1e}")
  # JFNK: inexact Newton with GMRES (exact JVPs), preconditioner = P(x) (plain / pereverzev), Eisenstat-Walker-like forcing
  for pname, pfun in [('P', Pfn), ('P_pereverzev', Ppc)]:
    x = x0; r_ = np.asarray(res(x)); norm = bc.mean_abs(r_); it = 0; n_jvp = 0; n_res = 1; hist=[norm]; n_prec = 0
    eta = 1e-1
    while norm > 1e-5 and it < 30:
      P = np.asarray(pfun(x)); n_prec += 1
      Pinv = scipy.sparse.linalg.LinearOperator((N, N), matvec=lambda v: np.linalg.solve(P, v))
      cnt = [0]
      def mv(v):
        cnt[0] += 1
        return np.asarray(jvp(x, jnp.asarray(v)))
      A = scipy.sparse.linalg.LinearOperator((N, N), matvec=mv)
      d, info = scipy.sparse.linalg.gmres(A, -r_, M=Pinv, rtol=eta, atol=0, restart=100, maxiter=100)
      n_jvp += cnt[0]
      # backtracking line search as TORAX
      step = 1.0; k = 0
      while True:
        xt = x + step * jnp.asarray(d); rt = np.asarray(res(xt)); n_res += 1; nt = bc.mean_abs(rt); k += 1
        if (nt <= (1 - 1e-4*step)*norm) or k >= 20 or step*np.max(np.abs(d)) <= 1e-7: break
        step *= 0.5
      old_norm = norm
      x, r_, norm = xt, rt, nt; it += 1; hist.append(norm)
      eta = max(min(0.9 * (norm/old_norm)**2, 0.1), 1e-6)  # Eisenstat-Walker choice 2 (capped)
    R[f'jfnk_{pname}'] = dict(iterations=it, converged=bool(norm <= 1e-5), n_jvp=n_jvp, n_res=n_res, n_prec=n_prec, hist=hist)
    print(f"JFNK (prec {pname}): newton it={it} conv={norm<=1e-5} total JVPs={n_jvp} res evals={n_res} prec builds={n_prec} hist={['%.1e'%h for h in hist]}")
  OUT[str(dt)] = {k: {kk: (vv if not hasattr(vv, 'shape') else None) for kk, vv in v.items() if kk not in ('x', 'xhist')} for k, v in R.items()}
json.dump(OUT, open(f'newton_variants_n{n_rho}.json', 'w'), indent=1, default=str)
print('DONE')
