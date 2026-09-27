import jax, jax.numpy as jnp, json, os
from torax.examples import iterhybrid_rampup
import bench_common as bc
p = bc.build_problem(iterhybrid_rampup.CONFIG, n_warm_steps=3, n_rho=100)
f = p.residual_fun; res = jax.jit(f); jacf = jax.jit(jax.jacfwd(f))
def body_nols(x, r):
    a = jacf(x); d = jnp.linalg.solve(a, -r); return f(x + d)
r = res(p.x0)
T = {'residual': bc.timeit(res, p.x0, n=10)[0], 'jacfwd': bc.timeit(jacf, p.x0, n=5)[0],
     'jacfwd+solve+residual fused': bc.timeit(jax.jit(body_nols), p.x0, r, n=5)[0],
     'jvp_single': bc.timeit(jax.jit(lambda x, v: jax.jvp(f, (x,), (v,))[1]), p.x0, jnp.ones_like(p.x0), n=10)[0],
     'batched_jvp_k32': bc.timeit(jax.jit(lambda x, V: jax.vmap(lambda v: jax.jvp(f, (x,), (v,))[1])(V)), p.x0, jnp.eye(p.n)[:32], n=5)[0]}
print(os.environ.get('XLA_FLAGS', 'default flags'), {k: round(v*1e3, 1) for k, v in T.items()})
