"""Per-step budget at N=100 (n_rho=25): Newton iterations vs everything else."""
import copy, time, functools, json
import jax, jax.numpy as jnp, numpy as np
import torax
from torax.examples import iterhybrid_predictor_corrector as pc
from torax._src.solver import jax_root_finding
from torax._src.fvm import newton_raphson_solve_block, enums
from torax._src.orchestration import run_simulation as rs
import bench_common as bc

cfg = copy.deepcopy(pc.CONFIG)
p = bc.build_problem(cfg, n_warm_steps=3, solver_overrides={'solver_type': 'newton_raphson'})
print(f'N={p.n} n_rho={p.n_rho} t={float(p.state.t):.3f} dt={float(p.dt):.4f}')
res = jax.jit(p.residual_fun); jacf = jax.jit(jax.jacfwd(p.residual_fun))
T = {}
T['residual'] = bc.timeit(res, p.x0, n=10)[0]
T['jacfwd'] = bc.timeit(jacf, p.x0, n=10)[0]
nr = jax.jit(lambda x: jax_root_finding.root_newton_raphson(p.residual_fun, x, maxiter=30, tol=1e-5, coarse_tol=1e-2))
xs, meta = nr(p.x0); its = int(meta.iterations)
T['root_newton_raphson'] = bc.timeit(nr, p.x0, n=5)[0]
sp = p.runtime_params_t.solver
blk = jax.jit(lambda dt: newton_raphson_solve_block.newton_raphson_solve_block(
    dt=dt, runtime_params_t=p.runtime_params_t, runtime_params_t_plus_dt=p.runtime_params_t_plus_dt, geo_t=p.geo_t,
    geo_t_plus_dt=p.geo_t_plus_dt, x_old=p.x_old, core_profiles_t=p.core_profiles_t, core_profiles_t_plus_dt=p.core_profiles_t_plus_dt,
    explicit_source_profiles=p.explicit_source_profiles, models=p.step_fn.solver.models, coeffs_callback=p.coeffs_callback,
    evolving_names=p.evolving_names, initial_guess_mode=enums.InitialGuessMode(sp.initial_guess_mode), maxiter=sp.maxiter, tol=sp.residual_tol,
    coarse_tol=sp.residual_coarse_tol, delta_reduction_factor=sp.delta_reduction_factor, tau_min=sp.tau_min,
    pedestal_transition_state=p.pedestal_transition_state, max_linesearch_steps=sp.max_linesearch_steps, vmap_linesearch=sp.vmap_linesearch))
T['newton_raphson_solve_block (guess + newton)'] = bc.timeit(blk, p.dt, n=5)[0]
sf = p.step_fn
T['full step (newton config)'] = bc.timeit(lambda: sf(p.state, p.ppo), n=5)[0]
# same state, linear solver
cfg2 = copy.deepcopy(pc.CONFIG); cfg2['solver']['solver_type'] = 'linear'
cfg2c = torax.ToraxConfig.from_dict(cfg2)
_, ppo2, step_fn2 = rs.prepare_simulation(cfg2c)
T['full step (linear config, same state)'] = bc.timeit(lambda: step_fn2(p.state, p.ppo), n=5)[0]
# dispatch cost proxy: pytree size and flatten time
leaves = jax.tree_util.tree_leaves((p.state, p.ppo)); t0 = time.perf_counter()
for _ in range(20): jax.tree_util.tree_flatten((p.state, p.ppo))
T['tree_flatten(state, ppo) x1'] = (time.perf_counter() - t0) / 20
print('newton iterations in this step:', its, '| leaves in (state, ppo):', len(leaves))
for k, v in T.items(): print(f'  {k:45s} {v*1e3:8.2f} ms')
json.dump({k: v*1e3 for k, v in T.items()} | {'its': its, 'leaves': len(leaves)}, open('overhead_n25.json', 'w'), indent=1)
