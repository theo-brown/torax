"""Writes a small random ToricNN surrogate for end-to-end checks of the ICRH source."""
import dataclasses
import json
import sys

import jax
import jax.numpy as jnp
from torax._src.sources.ion_cyclotron_source import toric_nn


def write(path: str) -> None:
  """Writes ToricNN weights whose output depends on every input.

  As in structured_jacobian_test: no hidden layer (the ReLU layer of the
  earlier dummy, hidden_sizes=[3], was inactive at the states used, which made
  the ICRH profile independent of its inputs), inputs scaled to order one and
  positive outputs, which the model would otherwise clip at zero.
  """
  # pylint: disable=protected-access
  network = toric_nn._ToricNN(
      hidden_sizes=[], pca_coeffs=4, input_dim=10,
      radial_nodes=toric_nn._TORIC_GRID_SIZE,
  )
  _, params = network.init_with_output(
      jax.random.PRNGKey(0), jnp.ones(10, dtype=jnp.float32)
  )
  params = dict(
      params['params'],
      scaler_mean=jnp.zeros(10),
      scaler_scale=jnp.array([1e8, 10.0, 1.0, 10.0] + [1.0] * 5 + [10.0]),
      pca_mean=jnp.full(toric_nn._TORIC_GRID_SIZE, 10.0),
  )
  config = dataclasses.asdict(network)
  weights = jax.tree_util.tree_map(lambda x: x.tolist(), params)
  for name in (toric_nn._HELIUM3_ID, toric_nn._TRITIUM_SECOND_HARMONIC_ID,
               toric_nn._ELECTRON_ID):
    config[name] = weights
  with open(path, 'w') as f:
    json.dump(config, f)


if __name__ == '__main__':
  write(sys.argv[1])
  print('written', sys.argv[1])
