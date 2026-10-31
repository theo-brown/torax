"""Linear-solver (predictor-corrector / Picard) convergence with and without
Anderson acceleration (SUNDIALS SUNNonlinSol_FixedPoint / PETSc SNESANDERSON style),
measured against the fully converged Newton solution of the same step."""
import json, sys
import numpy as np
import jax, jax.numpy as jnp
from torax.examples import iterhybrid_rampup, iterhybrid_predictor_corrector
from torax._src.fvm import implicit_solve_block, fvm_conversions
from torax._src.solver import jax_root_finding
import bench_common as bc

which = sys.argv[1] if len(sys.argv) > 1 else 'rampup'
dts = [float(a) for a in sys.argv[2].split(',')] if len(sys.argv) > 2 else [2.0, 0.5, 0.1, 0.02]
CONFIG = iterhybrid_rampup.CONFIG if which == 'rampup' else iterhybrid_predictor_corrector.CONFIG
OUT = {}
for dt in dts:
  p = bc.build_problem(CONFIG, n_warm_steps=3, dt=dt, solver_overrides={'use_pereverzev': True, 'solver_type': 'newton_raphson'})
  sp = p.runtime_params_t_plus_dt.solver
  res = jax.jit(p.residual_fun)
  # exact solution of the nonlinear step
  x_star, meta = jax_root_finding.root_newton_raphson(p.residual_fun, p.x0, maxiter=40, tol=1e-9, coarse_tol=1e-2, use_jax_custom_root=False)
  x_star = np.asarray(x_star)
  dstep = np.linalg.norm(x_star - np.asarray(p.x_old_vec))
  def picard_map(x_vec, pereverzev):
    x_guess = fvm_conversions.vec_to_cell_variable_tuple(jnp.asarray(x_vec), p.core_profiles_t_plus_dt, p.evolving_names)
    coeffs_new = p.coeffs_callback(p.runtime_params_t_plus_dt, p.geo_t_plus_dt, p.core_profiles_t_plus_dt,
        prev_core_profiles=p.core_profiles_t, dt=p.dt, x=x_guess, explicit_source_profiles=p.explicit_source_profiles,
        pedestal_transition_state=p.pedestal_transition_state, allow_pereverzev=pereverzev)
    coeffs_exp = p.coeffs_callback(p.runtime_params_t, p.geo_t, p.core_profiles_t, prev_core_profiles=None, dt=None, x=p.x_old,
        explicit_source_profiles=p.explicit_source_profiles, allow_pereverzev=pereverzev, explicit_call=True,
        pedestal_transition_state=p.pedestal_transition_state)
    x_new = implicit_solve_block.implicit_solve_block(dt=p.dt, x_old=p.x_old, x_new_guess=x_guess, coeffs_old=coeffs_exp,
        coeffs_new=coeffs_new, theta_implicit=sp.theta_implicit, convection_dirichlet_mode=sp.convection_dirichlet_mode,
        convection_neumann_mode=sp.convection_neumann_mode, implicit_solver_type=sp.implicit_solver_type)
    return fvm_conversions.cell_variable_tuple_to_vec(x_new)
  gmaps = {True: jax.jit(lambda x: picard_map(x, True)), False: jax.jit(lambda x: picard_map(x, False))}
  x_guess0 = np.asarray(fvm_conversions.cell_variable_tuple_to_vec(
      __import__('torax._src.core_profiles.convertors', fromlist=['x']).core_profiles_to_solver_x_tuple(p.core_profiles_t_plus_dt, p.evolving_names)))
  R = {}
  for pere in [True, False]:
    g = gmaps[pere]
    # plain Picard
    x = x_guess0; errs = []
    for k in range(25):
      x = np.asarray(g(x)); errs.append(float(np.linalg.norm(x - x_star) / dstep))
      if not np.isfinite(errs[-1]) or errs[-1] > 1e3: break
    R[f'picard_pereverzev{pere}'] = errs
    # Anderson acceleration (m=3), damping 1, as in SUNDIALS fixed point
    for m in [2, 4]:
      x = x_guess0; errs_a = []
      X = []; F = []
      for k in range(25):
        gx = np.asarray(g(x)); f = gx - x
        X.append(x.copy()); F.append(f.copy())
        if len(F) > m + 1: X.pop(0); F.pop(0)
        if len(F) >= 2:
          dF = np.diff(np.array(F), axis=0).T   # N x (mk)
          dX = np.diff(np.array(X), axis=0).T
          if not np.all(np.isfinite(dF)) or not np.all(np.isfinite(f)):
            break
          try:
            gamma, *_ = np.linalg.lstsq(dF, f, rcond=1e-10)
          except np.linalg.LinAlgError:
            break
          x_new = gx - (dX + dF) @ gamma
        else:
          x_new = gx
        x = x_new; errs_a.append(np.linalg.norm(x - x_star) / dstep)
        if not np.isfinite(errs_a[-1]) or errs_a[-1] > 1e3: break
      R[f'anderson_m{m}_pereverzev{pere}'] = errs_a
  print(f'\n### {which} dt={dt}: newton its={int(meta.iterations)}, |x*-x_old|={dstep:.3e}; relative error |x_k - x*|/|x*-x_old| after k=1..:')
  for k, v in R.items():
    print(f'  {k:28s}', ' '.join('%.1e' % e for e in v[:12]))
  OUT[str(dt)] = R
json.dump(OUT, open(f'picard_anderson_{which}.json', 'w'), indent=1)
print('DONE')
