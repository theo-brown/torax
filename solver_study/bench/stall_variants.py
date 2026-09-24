"""Test which non-smooth switch causes the small-dt Newton stall: rerun the
Newton solve + Taylor test at dt=0.02/0.05 with candidate switches disabled."""
import copy, sys, json
import numpy as np
import jax, jax.numpy as jnp
from torax.examples import iterhybrid_rampup
import bench_common as bc

variants = {
  'baseline': {},
  'DV_effective_off': {'DV_effective': False},
  'avoid_big_negative_s_off': {'avoid_big_negative_s': False},
  'An_min_0': {'An_min': 0.0},
}
dts = [float(a) for a in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0.02, 0.05]
OUT = {}
for name, qlknn_over in variants.items():
  cfg = copy.deepcopy(iterhybrid_rampup.CONFIG)
  cfg['transport']['core_transport_models']['qlknn'].update(qlknn_over)
  for dt in dts:
    p = bc.build_problem(cfg, n_warm_steps=3, dt=dt, solver_overrides={'use_pereverzev': True})
    n_ch, n_cells = len(p.evolving_names), p.n_rho
    res = jax.jit(p.residual_fun); jac = jax.jit(jax.jacfwd(p.residual_fun))
    r = bc.newton_generic(res, jac, p.x0, maxiter=8)
    x = r['x']; rx = np.asarray(res(x)); J = np.asarray(jac(x)); d = np.linalg.solve(J, -rx)
    tay = []
    for s in [1e-4, 1e-3, 1e-2]:
      rs = np.asarray(res(jnp.asarray(x + s * d))); lin = rx + s * (J @ d)
      tay.append(float(np.linalg.norm(rs - lin) / np.linalg.norm(rs - rx + 1e-300)))
    print(f'{name:26s} dt={dt:<5} newton it={r["iterations"]} conv={r["converged"]} final={r["final_norm"]:.2e} ls_extra={r["n_ls_extra"]} taylor err(s=1e-4,1e-3,1e-2)={["%.2e" % t for t in tay]} hist={["%.1e" % h for h in r["hist"]]}')
    OUT[f'{name}_{dt}'] = dict(it=r['iterations'], conv=r['converged'], final=r['final_norm'], taylor=tay, hist=r['hist'])
json.dump(OUT, open('stall_variants.json', 'w'), indent=1, default=str)
print('DONE')
