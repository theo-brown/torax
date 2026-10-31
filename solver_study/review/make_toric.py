"""python make_toric.py toricnn.json: the dummy ToricNN of components.py/sims.py.

The small random network of 3.10-3.13 (its hidden layer is inactive at the
states used, see 3.14 (4)). It serves here only to run the ICRH code path in
the dense mode of both versions, which read the same file.
"""
import dataclasses, json, sys
import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from torax._src.sources.ion_cyclotron_source import toric_nn
network = toric_nn._ToricNN(hidden_sizes=[3], pca_coeffs=4, input_dim=10, radial_nodes=toric_nn._TORIC_GRID_SIZE)
_, params = network.init_with_output(jax.random.PRNGKey(0), jnp.ones(10))
config = dataclasses.asdict(network)
weights = jax.tree_util.tree_map(lambda x: x.tolist(), params['params'])
for name in (toric_nn._HELIUM3_ID, toric_nn._TRITIUM_SECOND_HARMONIC_ID, toric_nn._ELECTRON_ID):
  config[name] = weights
json.dump(config, open(sys.argv[1], 'w'))
print('ok')
