"""Per-row check ratio |J p - jvp| / (|J| |p|) of the structured Jacobian.

python check_ratio.py VARIANT N_RHO [break]
break: 'reach' forces the no-rotation transport stencil.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import numpy as np
import harness
import variants
from torax._src.solver import structured_jacobian as sj

name, n_rho = sys.argv[1], int(sys.argv[2])
if len(sys.argv) > 3 and sys.argv[3] == 'reach':
  sj._transport_reach = lambda *_: (2, 1)
if name == 'global':
  config = harness.base_config(n_rho, global_sources=True)
else:
  config = variants.variant(name, n_rho)
kwargs, _ = harness.solve_block_kwargs(config)
residual_fun = harness.residual_fun_from_kwargs(kwargs)
structured = jax.jit(sj.jacobian_fn(residual_fun))
n = sum(v.value.size for v in kwargs['x_old'])
probe = np.random.default_rng(0).uniform(1.0, 2.0, n)
jvp = jax.jit(lambda x: jax.jvp(residual_fun, (x,), (probe,))[1])
for scale, seed in ((0.0, 0), (0.05, 1), (0.2, 2)):
  x = harness.perturbed_x(kwargs, scale, seed)
  j = np.asarray(structured(x))
  jv = np.asarray(jvp(x))
  ratio = np.abs(j @ probe - jv) / (np.abs(j) @ probe)
  if not np.all(np.isfinite(ratio)):
    print(f'{name} n={n_rho} noise={scale}: non-finite ({int((~np.isfinite(ratio)).sum())} rows)')
    continue
  print(f'{name} n={n_rho} noise={scale}: max row ratio {ratio.max():.1e} (row {ratio.argmax()}), median {np.median(ratio):.1e}', flush=True)
