# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark of electrostatic vs electromagnetic TGLF on ITER scenarios.

Runs matched pairs of ITER simulations that differ only in the TGLF settings
controlling the electromagnetic fluctuations, records the wall-clock cost of
each, and writes TORAX comparison plots.

Electromagnetic TGLF retains the transverse magnetic fluctuation
:math:`\\delta B_\\perp` (``USE_BPER``) and searches a wider space of Gaussian
mode widths (negative ``WIDTH_MIN``) with a larger Hermite basis, all of which
increase the eigenvalue work per call. The physics response comes through the
finite electron beta: TORAX already supplies ``BETAE`` and ``P_PRIME_LOC``
per flux surface, so the electromagnetic drive is active as soon as the flags
are set.

Requires the optional ``tglf2py`` extension module; see
https://torax.readthedocs.io/en/latest/installation.html#optional-install-tglf.

This is expensive. On 4 cores with ``n_processes=4``, the predictor-corrector
case to ``t_final=5`` took ~31 min electrostatic (56 steps, ~33 s/step) and
~4 h 11 min electromagnetic (52 steps, ~289 s/step). Use ``--max_steps`` to
measure per-step cost on a new machine before committing to a full run.

Example::

  python torax/benchmarks/tglf_es_vs_em.py --output_dir=/tmp/tglf_es_vs_em
"""

import copy
import json
import os
import time
from typing import Any

from absl import app
from absl import flags
from absl import logging
import torax
from torax._src import simulation_app
from torax._src.plotting import plotruns_lib
from torax.examples import iterhybrid_predictor_corrector
from torax.plotting.configs import default_plot_config
from torax.plotting.configs import transport_plot_config

# pylint: disable=invalid-name

# Radial extent over which TGLF is evaluated. Inside `_TGLF_RHO_MIN` ad-hoc
# MHD/EM transport is prescribed, outside `_TGLF_RHO_MAX` an L-mode-like near
# edge is prescribed. Matches the layout used by the TGLF sim test config
# (torax/tests/test_data/test_iterhybrid_predictor_corrector_tglf.py).
_TGLF_RHO_MIN = 0.15
_TGLF_RHO_MAX = 0.95

# TGLF settings shared by both variants. Taken from the `common.tglf` files
# distributed with the UKAEA TGLFNN surrogates
# (https://github.com/ukaea/tglfnn-ukaea), which define this project's
# electrostatic and electromagnetic TGLF conventions.
#
# Two settings deliberately depart from those files:
#   * UNITS: the surrogate namelists use 'GYRO'. TORAX denormalizes the TGLF
#     fluxes with Q_GB = n_e * T_e * c_s * (rho_s / a)^2 built on B_unit, i.e.
#     the CGYRO convention, so 'CGYRO' is required here for the fluxes to be
#     converted correctly. UNITS is identical between the two variants, so
#     holding it fixed does not affect the electrostatic/electromagnetic
#     contrast.
#   * NS is kept at the surrogates' value of 2 (electrons + main ion). TORAX
#     populates a third species (ZS_3, MASS_3, TAUS_3, AS_3, RLNS_3, RLTS_3)
#     but TGLF ignores it at NS=2. Main ion dilution still reaches TGLF via
#     AS_2 = n_i / n_e and the impurity charge via ZEFF; what is dropped is
#     the impurity's own density and temperature gradient drive.
_COMMON_TGLF_SETTINGS = {
    'UNITS': 'CGYRO',
    'USE_TRANSPORT_MODEL': True,
    'GEOMETRY_FLAG': 1,  # Miller/MXH geometry.
    'NS': 2,
    'SAT_RULE': 2,
    'KYGRID_MODEL': 4,
    'XNU_MODEL': 2,
    'NXGRID': 16,
    'NKY': 12,
    'NMODES': 2,
    'NBASIS_MAX': 6,
    'NWIDTH': 21,
    'FIND_WIDTH': True,
    # Avoids picking the lowest ky as the maximum gamma/ky when setting the
    # intensity spectrum shape in the saturation rule.
    'ALPHA_ZF': -1,
    'ETG_FACTOR': 1.25,
}

# Electrostatic: MultiMachineHyperES_11Mar26/common.tglf.
ES_TGLF_SETTINGS = _COMMON_TGLF_SETTINGS | {
    'USE_BPER': False,  # No transverse magnetic fluctuations.
    'USE_BPAR': False,  # No compressional magnetic fluctuations.
    'USE_MHD_RULE': False,  # Retain the pressure gradient curvature drift.
    'WIDTH': 1.65,
    'WIDTH_MIN': 0.3,  # Positive: electrostatic mode width search.
    'NBASIS_MIN': 2,
    'FILTER': 2.0,
}

# Electromagnetic: MultiMachineHyper_15Jan26_MAE/common.tglf.
EM_TGLF_SETTINGS = _COMMON_TGLF_SETTINGS | {
    'USE_BPER': True,  # Transverse magnetic fluctuations, delta B_perp.
    'USE_BPAR': False,
    # With USE_BPER active the ideal MHD ballooning rule is used for the
    # curvature drift rather than the local pressure gradient.
    'USE_MHD_RULE': True,
    'WIDTH': 3.0,
    'WIDTH_MIN': -0.3,  # Negative: electromagnetic mode width search.
    'NBASIS_MIN': 6,
    'FILTER': -0.1,
}

TGLF_SETTINGS_BY_VARIANT = {
    'es': ES_TGLF_SETTINGS,
    'em': EM_TGLF_SETTINGS,
}

# Only base scenarios whose solver does not differentiate through the transport
# model can be used here. The TGLF interface dispatches to Fortran through
# `jax.pure_callback`, which has no JVP rule, so a `newton_raphson` solver fails
# when it forms the Jacobian of the residual. This rules out
# `torax.examples.iterhybrid_rampup` unless its solver is changed.
BASE_CONFIGS = {
    'predictor_corrector': iterhybrid_predictor_corrector.CONFIG,
}

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    '/tmp/torax_tglf_es_vs_em',
    'Directory for NetCDF outputs, plots and the runtime summary.',
)
_CASES = flags.DEFINE_multi_string(
    'cases',
    list(BASE_CONFIGS),
    f'Base ITER scenarios to run. One or more of {list(BASE_CONFIGS)}.',
)
_VARIANTS = flags.DEFINE_multi_string(
    'variants',
    list(TGLF_SETTINGS_BY_VARIANT),
    f'TGLF variants to run. One or more of {list(TGLF_SETTINGS_BY_VARIANT)}.',
)
_MAX_STEPS = flags.DEFINE_integer(
    'max_steps',
    None,
    'If set, stop each simulation after this many steps. Useful for measuring'
    ' per-step cost before committing to a full run.',
)
_T_FINAL = flags.DEFINE_float(
    't_final',
    None,
    'If set, override the final simulation time of every case.',
)
_N_PROCESSES = flags.DEFINE_integer(
    'n_processes',
    4,
    'Number of parallel TGLF processes, one radial face per process.',
)


def build_config(
    base_config: dict[str, Any],
    tglf_settings: dict[str, Any],
    n_processes: int,
    t_final: float | None = None,
) -> dict[str, Any]:
  """Returns a copy of `base_config` with TGLF transport installed.

  Args:
    base_config: A TORAX config dict to use as the starting point.
    tglf_settings: TGLF namelist parameters, either `ES_TGLF_SETTINGS` or
      `EM_TGLF_SETTINGS`.
    n_processes: Number of parallel TGLF processes.
    t_final: If set, overrides the final simulation time in seconds.

  Returns:
    A TORAX config dict.
  """
  config = copy.deepcopy(base_config)
  config['transport'] = {
      'model_name': 'combined',
      'transport_models': [
          # Ad-hoc MHD/EM transport in the inner core, where TGLF is not
          # evaluated.
          {
              'model_name': 'constant',
              'rho_max': _TGLF_RHO_MIN,
              'chi_i': 1.0,
              'chi_e': 1.0,
              'D_e': 0.25,
              'V_e': 0.0,
          },
          {
              'model_name': 'tglf',
              'rho_min': _TGLF_RHO_MIN,
              'rho_max': _TGLF_RHO_MAX,
              'DV_effective': True,
              # Start from the defaults distributed with TGLF so that
              # `tglf_settings` below is the only source of physics settings.
              'use_legacy_torax_defaults': False,
              'tglf_settings': tglf_settings,
              'n_processes': n_processes,
              'n_cores_per_process': 1,
          },
          # L-mode-like near edge region.
          {
              'model_name': 'constant',
              'rho_min': _TGLF_RHO_MAX,
              'rho_max': 1.0,
              'chi_i': 2.0,
              'chi_e': 2.0,
              'D_e': 0.1,
              'V_e': 0.0,
          },
      ],
      'chi_min': 0.05,
      'chi_max': 100,
      'D_e_min': 0.05,
      'smoothing_width': 0.1,
  }
  if t_final is not None:
    config['numerics'] = copy.deepcopy(config['numerics'])
    config['numerics']['t_final'] = t_final
  return config


def _strip_tglf_settings(config: dict[str, Any]) -> dict[str, Any]:
  """Returns a copy of `config` with the TGLF settings dict removed."""
  stripped = copy.deepcopy(config)
  for model in stripped['transport']['transport_models']:
    model.pop('tglf_settings', None)
  return stripped


def assert_variants_are_matched(configs: dict[str, dict[str, Any]]) -> None:
  """Checks that the variant configs differ only in their TGLF settings.

  Any difference in the benchmark outputs is only attributable to the
  electrostatic/electromagnetic settings if everything else is held fixed, so
  this is asserted rather than assumed.

  Args:
    configs: Mapping of variant name to TORAX config dict.

  Raises:
    ValueError: If two variants differ outside of `tglf_settings`, or if two
      variants share identical TGLF settings.
  """
  (reference_name, reference), *others = configs.items()
  for name, config in others:
    if _strip_tglf_settings(reference) != _strip_tglf_settings(config):
      raise ValueError(
          f"Configs '{reference_name}' and '{name}' differ outside of"
          ' tglf_settings; the comparison would not be controlled.'
      )
    if reference['transport']['transport_models'][1]['tglf_settings'] == (
        config['transport']['transport_models'][1]['tglf_settings']
    ):
      raise ValueError(
          f"Configs '{reference_name}' and '{name}' have identical TGLF"
          ' settings.'
      )


def run_case(
    label: str,
    config: dict[str, Any],
    max_steps: int | None,
) -> tuple[Any, dict[str, Any]]:
  """Builds and runs a single simulation, timing each stage.

  Args:
    label: Human readable name used for logging and plot legends.
    config: TORAX config dict to run.
    max_steps: If set, stop the simulation after this many steps.

  Returns:
    A tuple of the output DataTree and a dict of timing statistics.

  Raises:
    RuntimeError: If the simulation did not complete successfully.
  """
  logging.info('Running %s', label)

  start = time.perf_counter()
  torax_config = torax.ToraxConfig.from_dict(config)
  build_time = time.perf_counter() - start

  # Confirm the settings survived config parsing and reached the model.
  resolved = torax_config.transport.transport_models[1].tglf_settings
  expected = config['transport']['transport_models'][1]['tglf_settings']
  for key, value in expected.items():
    if isinstance(value, bool):
      value = '.true.' if value else '.false.'
    if resolved[key] != value:
      raise RuntimeError(
          f'TGLF setting {key} was not applied for {label}: expected'
          f' {value!r}, resolved to {resolved[key]!r}.'
      )

  start = time.perf_counter()
  data_tree, state_history = torax.run_simulation(
      torax_config, progress_bar=False, max_steps=max_steps
  )
  simulation_time = time.perf_counter() - start

  # A run truncated by `max_steps` legitimately stops short of t_final; any
  # other non-zero error means the simulation actually failed.
  acceptable = {torax.SimError.NO_ERROR}
  if max_steps is not None:
    acceptable.add(torax.SimError.DID_NOT_REACH_T_FINAL)
  if state_history.sim_error not in acceptable:
    raise RuntimeError(f'{label} failed with {state_history.sim_error}.')

  n_steps = int(data_tree.coords['time'].size)
  stats = {
      'build_time_s': build_time,
      'simulation_time_s': simulation_time,
      'n_steps': n_steps,
      'simulation_time_per_step_s': simulation_time / max(n_steps, 1),
      't_final_reached_s': float(data_tree.coords['time'].values[-1]),
  }
  logging.info('%s: %s', label, stats)
  return data_tree, stats


def _write_plots(
    case: str,
    data_trees: dict[str, Any],
    output_dir: str,
) -> list[str]:
  """Writes ES/EM comparison plots for one case and returns their paths."""
  paths = []
  plot_configs = {
      'overview': default_plot_config.PLOT_CONFIG,
      'transport': transport_plot_config.PLOT_CONFIG,
  }
  for plot_name, plot_config in plot_configs.items():
    fig = plotruns_lib.plot_run_from_data_tree(
        plot_config,
        data_trees,
        interactive=False,
        fig_title=f'TGLF electrostatic vs electromagnetic: {case}',
    )
    path = os.path.join(output_dir, f'{case}_{plot_name}.html')
    fig.write_html(path)
    paths.append(path)
    logging.info('Wrote %s', path)
  return paths


def main(_) -> None:
  output_dir = _OUTPUT_DIR.value
  os.makedirs(output_dir, exist_ok=True)

  summary = {}
  for case in _CASES.value:
    configs = {
        variant: build_config(
            BASE_CONFIGS[case],
            TGLF_SETTINGS_BY_VARIANT[variant],
            n_processes=_N_PROCESSES.value,
            t_final=_T_FINAL.value,
        )
        for variant in _VARIANTS.value
    }
    if len(configs) > 1:
      assert_variants_are_matched(configs)

    data_trees = {}
    for variant, config in configs.items():
      label = f'{case}_{variant}'
      data_tree, stats = run_case(label, config, _MAX_STEPS.value)
      simulation_app.write_output_to_file(
          os.path.join(output_dir, f'{label}.nc'), data_tree
      )
      data_trees[variant.upper()] = data_tree
      summary[label] = stats

    if len(data_trees) > 1:
      _write_plots(case, data_trees, output_dir)

  summary_path = os.path.join(output_dir, 'runtime_summary.json')
  with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
  logging.info('Wrote %s', summary_path)

  for label, stats in summary.items():
    print(
        f'{label}: {stats["simulation_time_s"]:.1f} s over'
        f' {stats["n_steps"]} steps'
        f' ({stats["simulation_time_per_step_s"]:.2f} s/step)'
    )


if __name__ == '__main__':
  app.run(main)
