"""Structured vs dense Jacobian for sources that couple the state globally.

Usage: sources_split.py gaspuff_fb|custom_cyclo [n_rho] [line|volume]
  gaspuff_fb:   the experimental gas-puff feedback model
                (torax.experimental.gas_puff_feedback_source), registered as the
                docs describe; its S_total depends on the averaged n_e.
  custom_cyclo: a user-registered cyclotron_radiation model with a local
                model function, overriding only the config's model_func.
"""
import sys
from typing import Annotated, Literal

from harness import *  # noqa: F403
from torax._src.sources import cyclotron_radiation_heat_sink as crhs
from torax._src.sources import register_model
from torax._src.torax_pydantic import torax_pydantic

name = sys.argv[1]
n_rho = int(sys.argv[2]) if len(sys.argv) > 2 else 12
c = base_config(n_rho)

if name == 'gaspuff_fb':
  from torax.experimental import gas_puff_feedback_source as gpf
  register_model.register_source_model_config(gpf.GasPuffFeedbackSourceConfig, 'gas_puff')
  c['sources']['gas_puff'] = {
      'model_name': 'feedback',
      'puff_decay_length': 0.3,
      'S_feedforward': 6.0e21,
      'feedback_gain': 1.0e2,
      'target_average_n_e': 1.5e20,
      'average_type': sys.argv[3] if len(sys.argv) > 3 else 'volume',
  }
elif name == 'custom_cyclo':

  def local_sink(runtime_params, geo, source_name, core_profiles, unused_csp,
                 unused_cond):
    del runtime_params, geo, source_name
    return (-1e3 * core_profiles.T_e.value**2 * core_profiles.n_e.value / 1e20,)

  class LocalCycloConfig(crhs.CyclotronRadiationHeatSinkConfig):
    model_name: Annotated[Literal['local_test'], torax_pydantic.JAX_STATIC] = 'local_test'

    @property
    def model_func(self):
      return local_sink

  register_model.register_source_model_config(LocalCycloConfig, 'cyclotron_radiation')
  c['sources']['cyclotron_radiation'] = {'model_name': 'local_test'}
else:
  raise ValueError(name)

c['solver']['jacobian_mode'] = 'structured'
kwargs, tc = solve_block_kwargs(c)
src = kwargs['models'].source_models.standard_sources
for k, s in src.items():
  if k in ('gas_puff', 'cyclotron_radiation'):
    print(f'  source {k}: model_func={type(s.model_func).__name__}')
rf = residual_fun_from_kwargs(kwargs)
x = perturbed_x(kwargs)
out = analyse(rf, x, tag=f'{name} n_rho={n_rho}')
np.savez(f'out_{name}_{n_rho}.npz', **{k: v for k, v in out['_mats'].items() if v is not None})
