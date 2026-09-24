"""Writes a small random ToricNN surrogate for end-to-end checks of the ICRH source."""
import dataclasses
import json
import sys

import jax
import jax.numpy as jnp
from torax._src.sources.ion_cyclotron_source import toric_nn


def write(path: str) -> None:
  # pylint: disable=protected-access
  network = toric_nn._ToricNN(
      hidden_sizes=[3], pca_coeffs=4, input_dim=10,
      radial_nodes=toric_nn._TORIC_GRID_SIZE,
  )
  _, params = network.init_with_output(
      jax.random.PRNGKey(0), jnp.ones(10, dtype=jnp.float32)
  )
  config = dataclasses.asdict(network)
  weights = jax.tree_util.tree_map(lambda x: x.tolist(), params['params'])
  for name in (toric_nn._HELIUM3_ID, toric_nn._TRITIUM_SECOND_HARMONIC_ID,
               toric_nn._ELECTRON_ID):
    config[name] = weights
  with open(path, 'w') as f:
    json.dump(config, f)


if __name__ == '__main__':
  write(sys.argv[1])
  print('written', sys.argv[1])
