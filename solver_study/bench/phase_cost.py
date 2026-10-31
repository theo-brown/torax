"""Why do early ramp-up steps cost 0.7-1.4 s while late ones cost 0.2 s?
Time the pieces of a step at two states of iterhybrid_rampup (t = 2*n_warm)."""
import sys, json, time, functools
import numpy as np
import jax, jax.numpy as jnp
from torax.examples import iterhybrid_rampup
from torax._src.orchestration import step_function_processing
from torax._src.solver import jax_root_finding
import bench_common as bc

OUT = {}
for n_warm in [int(a) for a in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['10', '30'])]:
  p = bc.build_problem(iterhybrid_rampup.CONFIG, n_warm_steps=n_warm)
  t = float(p.state.t); print(f'\n##### state t={t} (n_warm={n_warm}) dt={float(p.dt)} N={p.n}')
  res = jax.jit(p.residual_fun); jacf = jax.jit(jax.jacfwd(p.residual_fun))
  T = {}
  T['residual'] = bc.timeit(res, p.x0, n=10)
  T['jacfwd'] = bc.timeit(jacf, p.x0, n=5)
  nr = jax.jit(lambda x: jax_root_finding.root_newton_raphson(p.residual_fun, x, maxiter=30, tol=1e-5, coarse_tol=1e-2, use_jax_custom_root=False))
  xs, meta = nr(p.x0); T['newton_solve'] = bc.timeit(nr, p.x0, n=3); OUT[f'{t}_newton_its'] = int(meta.iterations); OUT[f'{t}_newton_err'] = int(meta.error)
  sf = p.step_fn
  T['full_step'] = bc.timeit(lambda: sf(p.state, p.ppo), n=3)
  pre = jax.jit(lambda st: step_function_processing.pre_step(input_state=st, runtime_params_provider=sf.runtime_params_provider, geometry_provider=sf.geometry_provider, models=sf.solver.models))
  T['pre_step'] = bc.timeit(pre, p.state, n=5)
  # the solver call alone (as the step function calls it), including the linear PC initial guess
  solver = sf.solver
  sv = lambda: solver(t=p.state.t, dt=p.dt, runtime_params_t=p.runtime_params_t, runtime_params_t_plus_dt=p.runtime_params_t_plus_dt, geo_t=p.geo_t, geo_t_plus_dt=p.geo_t_plus_dt, core_profiles_t=p.core_profiles_t, core_profiles_t_plus_dt=p.core_profiles_t_plus_dt, explicit_source_profiles=p.explicit_source_profiles, pedestal_transition_state=p.pedestal_transition_state)
  x_new, sno = sv(); T['solver_call'] = bc.timeit(sv, n=3); OUT[f'{t}_solver_inner'] = int(sno.inner_solver_iterations)
  fin = jax.jit(lambda x_new, sno: step_function_processing.finalize_outputs(t=p.state.t, dt=p.dt, x_new=x_new, solver_numeric_outputs=sno, runtime_params_t_plus_dt=p.runtime_params_t_plus_dt, geometry_t_plus_dt=p.geo_t_plus_dt, core_profiles_t=p.core_profiles_t, core_profiles_t_plus_dt=p.core_profiles_t_plus_dt, explicit_source_profiles=p.explicit_source_profiles, edge_outputs=p.edge_outputs, models=solver.models, evolving_names=p.evolving_names, input_post_processed_outputs=p.ppo, time_step_calculator_state_t=p.state.time_step_calculator_state, pedestal_transition_state=p.pedestal_transition_state))
  T['finalize_outputs'] = bc.timeit(fin, x_new, sno, n=5)
  # count NaNs/denormals in residual & jacobian at x0 and along the newton path
  J = np.asarray(jacf(p.x0)); r = np.asarray(res(p.x0))
  OUT[f'{t}_nan_J'] = int(np.isnan(J).sum()); OUT[f'{t}_denormal_J'] = int(np.sum((np.abs(J) > 0) & (np.abs(J) < 2.3e-308)))
  OUT[f'{t}_min_abs_nonzero_J'] = float(np.min(np.abs(J[np.abs(J) > 0])))
  for k, (b, m) in T.items(): print(f'  {k:18s} {b*1e3:9.2f} ms (median {m*1e3:.2f})')
  print('  newton its', OUT[f'{t}_newton_its'], 'solver inner', OUT[f'{t}_solver_inner'], 'nan in J', OUT[f'{t}_nan_J'], 'denormals in J', OUT[f'{t}_denormal_J'], 'min|J|>0', OUT[f'{t}_min_abs_nonzero_J'])
  OUT[f'{t}_timings_ms'] = {k: b*1e3 for k, (b, m) in T.items()}
json.dump(OUT, open('phase_cost.json', 'w'), indent=1)
print('DONE')
