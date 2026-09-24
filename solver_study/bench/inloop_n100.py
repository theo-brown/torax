"""At N=400: cost of one Newton iteration as executed inside the while_loop
(jax_root_finding._body) vs the standalone pieces."""
import functools, json
import jax, jax.numpy as jnp
from torax.examples import iterhybrid_rampup
from torax._src.solver import jax_root_finding, linesearch
import bench_common as bc

p = bc.build_problem(iterhybrid_rampup.CONFIG, n_warm_steps=3, n_rho=100)
print(f'N={p.n} n_rho={p.n_rho} dt={float(p.dt)}')
f = p.residual_fun
res = jax.jit(f); jacf = jax.jit(jax.jacfwd(f))
T = {}
T['residual'] = bc.timeit(res, p.x0, n=10)[0]
T['jacfwd'] = bc.timeit(jacf, p.x0, n=5)[0]
J = jacf(p.x0); r = res(p.x0)
T['dense_solve'] = bc.timeit(jax.jit(lambda A, b: jnp.linalg.solve(A, b)), J, -r, n=10)[0]
# one Newton iteration exactly as in the loop body
jacobian_fun = jax.jit(jax.jacfwd(f), inline=jax.Inline.XLA_LATE)
body = jax.jit(functools.partial(jax_root_finding._body, jacobian_fun=jacobian_fun, residual_fun=f, log_iterations=False,
    delta_reduction_factor=0.5, sufficient_decrease=1e-4, linesearch_norm=jax_root_finding._mean_abs_norm,
    convergence_norm=jax_root_finding._mean_abs_norm, vmap_linesearch=False, max_linesearch_steps=100))
state = {'x': p.x0, 'iterations': jnp.array(0.0), 'residual': r, 'last_tau': jnp.array(1.0), 'residual_norm': jax_root_finding._mean_abs_norm(r)}
out = body(state); print('tau after one body call:', float(out['last_tau']), 'norm', float(out['residual_norm']))
T['one_newton_iteration_body (jit)'] = bc.timeit(body, state, n=5)[0]
# the same body without the line search while_loop: direction + one residual
def body_nols(st):
    a = jacobian_fun(st['x']); d = jnp.linalg.solve(a, -st['residual']); x = st['x'] + d; rr = f(x)
    return x, rr
T['jacfwd+solve+residual (jit, no linesearch loop)'] = bc.timeit(jax.jit(body_nols), state, n=5)[0]
# full loop
nr = jax.jit(lambda x: jax_root_finding.root_newton_raphson(f, x, maxiter=30, tol=1e-5, coarse_tol=1e-2, use_jax_custom_root=False))
xs, meta = nr(p.x0); its = int(meta.iterations)
T[f'root_newton_raphson ({its} iterations)'] = bc.timeit(nr, p.x0, n=3)[0]
for k, v in T.items(): print(f'  {k:50s} {v*1e3:8.1f} ms')
json.dump({k: v*1e3 for k, v in T.items()}, open('inloop_n100.json', 'w'), indent=1)
