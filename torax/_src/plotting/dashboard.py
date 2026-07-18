# Copyright 2024 DeepMind Technologies Limited
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

"""Plots TORAX runs in the React dashboard (see the dashboard/ directory).

Public API:
  plot_run: Loads output files and opens them in the dashboard.
  plot_run_from_data_tree: Same, for in-memory xr.DataTrees.
  run_to_dict: Converts one run to the dashboard's JSON document format.
  write_dashboard_html: Writes a self-contained dashboard HTML page.

The dashboard itself is a self-contained HTML template (dashboard.html,
rebuilt with `npm run bundle` in the dashboard/ directory) into which the
exported run documents are injected. This module deliberately imports only
numpy/xarray, so it can also be used as a standalone exporter script:

  python torax/_src/plotting/dashboard.py run.nc [-o run.json]
"""

import json
import math
import pathlib
import tempfile
from typing import Any, Mapping, Sequence
import webbrowser

import numpy as np
import xarray as xr

# Format identifier expected by the dashboard app (see dashboard/src/data.ts).
RUN_DATA_FORMAT = 'torax-dashboard-v1'

_TEMPLATE_PATH = pathlib.Path(__file__).parent / 'dashboard.html'
# Marker in the template replaced with a JSON array of run documents.
_RUNS_PLACEHOLDER = '/*__TORAX_RUNS__*/ []'

# Unit transformations applied on export, so the dashboard receives display
# units. Maps variable name -> (divisor, display units).
_TRANSFORMATIONS: dict[str, tuple[float, str]] = {
    'j_total': (1e6, 'MA/m²'),
    'j_ohmic': (1e6, 'MA/m²'),
    'j_bootstrap': (1e6, 'MA/m²'),
    'j_external': (1e6, 'MA/m²'),
    'j_generic_current': (1e6, 'MA/m²'),
    'j_ecrh': (1e6, 'MA/m²'),
    'I_bootstrap': (1e6, 'MA'),
    'Ip_profile': (1e6, 'MA'),
    'Ip': (1e6, 'MA'),
    'p_icrh_i': (1e6, 'MW/m³'),
    'p_icrh_e': (1e6, 'MW/m³'),
    'p_generic_heat_i': (1e6, 'MW/m³'),
    'p_generic_heat_e': (1e6, 'MW/m³'),
    'p_ecrh_e': (1e6, 'MW/m³'),
    'p_alpha_i': (1e6, 'MW/m³'),
    'p_alpha_e': (1e6, 'MW/m³'),
    'p_ohmic_e': (1e6, 'MW/m³'),
    'p_bremsstrahlung_e': (1e6, 'MW/m³'),
    'p_cyclotron_radiation_e': (1e6, 'MW/m³'),
    'p_impurity_radiation_e': (1e6, 'MW/m³'),
    'ei_exchange': (1e6, 'MW/m³'),
    'P_ohmic_e': (1e6, 'MW'),
    'P_aux_total': (1e6, 'MW'),
    'P_alpha_total': (1e6, 'MW'),
    'P_bremsstrahlung_e': (1e6, 'MW'),
    'P_cyclotron_e': (1e6, 'MW'),
    'P_ecrh': (1e6, 'MW'),
    'P_radiation_e': (1e6, 'MW'),
    'I_ecrh': (1e6, 'MA'),
    'I_aux_generic': (1e6, 'MA'),
    'W_thermal_total': (1e6, 'MJ'),
    'n_e': (1e20, '10²⁰ m⁻³'),
    'n_i': (1e20, '10²⁰ m⁻³'),
    'n_impurity': (1e20, '10²⁰ m⁻³'),
    'n_e_volume_avg': (1e20, '10²⁰ m⁻³'),
    'n_i_volume_avg': (1e20, '10²⁰ m⁻³'),
    'n_e_line_avg': (1e20, '10²⁰ m⁻³'),
    'n_i_line_avg': (1e20, '10²⁰ m⁻³'),
    's_gas_puff': (1e20, '10²⁰ m⁻³ s⁻¹'),
    's_generic_particle': (1e20, '10²⁰ m⁻³ s⁻¹'),
    's_pellet': (1e20, '10²⁰ m⁻³ s⁻¹'),
}

_TIME = 'time'
_RHO_COORDS = ('rho_cell_norm', 'rho_face_norm', 'rho_norm')


def _clean(values: np.ndarray) -> Any:
  """Converts an array to nested lists with NaN/inf replaced by None."""

  def convert(x):
    if isinstance(x, list):
      return [convert(v) for v in x]
    if x is None or math.isnan(x) or math.isinf(x):
      return None
    # Round to 6 significant digits to keep the JSON compact.
    return float(f'{x:.6g}')

  return convert(values.astype(np.float64).tolist())


def _transform(name: str, da: xr.DataArray) -> tuple[np.ndarray, str]:
  """Applies unit transformation, returning (values, display units)."""
  values = da.to_numpy()
  if name in _TRANSFORMATIONS:
    divisor, units = _TRANSFORMATIONS[name]
    return values / divisor, units
  return values, str(da.attrs.get('units', ''))


def run_to_dict(data_tree: xr.DataTree, label: str) -> dict[str, Any]:
  """Converts a TORAX output DataTree to the dashboard's JSON document."""
  top = data_tree.dataset

  if _TIME not in top:
    raise ValueError(f"No '{_TIME}' variable found in the output.")

  coords: dict[str, Any] = {}
  for coord_name in _RHO_COORDS:
    if coord_name in top.coords or coord_name in top:
      coords[coord_name] = _clean(top[coord_name].to_numpy())

  profiles: dict[str, Any] = {}
  scalars: dict[str, Any] = {}

  datasets = [top] + [
      data_tree.children[name].dataset
      for name in ('profiles', 'scalars', 'numerics')
      if name in data_tree.children
  ]

  for ds in datasets:
    for name in ds.data_vars:
      name = str(name)
      if name == _TIME or name in profiles or name in scalars:
        continue
      da = ds[name]
      dims = tuple(str(d) for d in da.dims)
      if dims == (_TIME,):
        values, units = _transform(name, da)
        scalars[name] = {'values': _clean(values), 'units': units}
      elif len(dims) == 2 and dims[0] == _TIME and dims[1] in coords:
        values, units = _transform(name, da)
        profiles[name] = {
            'values': _clean(values),
            'coord': dims[1],
            'units': units,
        }
      # Variables with other dimensions (e.g. config strings) are skipped.

  return {
      'format': RUN_DATA_FORMAT,
      'label': label,
      'time': _clean(top[_TIME].to_numpy()),
      'coords': coords,
      'profiles': profiles,
      'scalars': scalars,
  }


def _open_output_file(path: pathlib.Path) -> xr.DataTree:
  """Opens an output file, upgrading legacy formats when TORAX is available."""
  try:
    # pylint: disable=g-import-not-at-top
    from torax._src.output_tools import output  # Heavy import; keep lazy.
  except ImportError:
    # Standalone use without a TORAX installation: no legacy-format handling.
    return xr.open_datatree(path)
  return output.load_state_file(str(path))


def load_run(
    path: str | pathlib.Path, label: str | None = None
) -> dict[str, Any]:
  """Loads an output .nc file and converts it to a dashboard run document."""
  path = pathlib.Path(path)
  return run_to_dict(_open_output_file(path), label or path.stem)


def write_dashboard_html(
    runs: Sequence[dict[str, Any]],
    output_path: str | pathlib.Path | None = None,
) -> pathlib.Path:
  """Writes a self-contained dashboard HTML page with the runs embedded.

  Args:
    runs: Run documents from run_to_dict / load_run.
    output_path: Where to write the page. Defaults to a temporary file.

  Returns:
    The path of the written HTML file.
  """
  template = _TEMPLATE_PATH.read_text(encoding='utf-8')
  if _RUNS_PLACEHOLDER not in template:
    raise RuntimeError(
        f'Dashboard template {_TEMPLATE_PATH} has no runs placeholder;'
        ' rebuild it with `npm run bundle` in the dashboard/ directory.'
    )
  # Escaping '<' keeps '</script>' sequences impossible inside the JSON.
  runs_json = json.dumps(list(runs)).replace('<', '\\u003c')
  html = template.replace(_RUNS_PLACEHOLDER, runs_json)

  if output_path is None:
    fd = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.html',
        prefix='torax_dashboard_',
        delete=False,
        encoding='utf-8',
    )
    with fd:
      fd.write(html)
    return pathlib.Path(fd.name)
  output_path = pathlib.Path(output_path)
  output_path.write_text(html, encoding='utf-8')
  return output_path


def _open_dashboard(
    runs: Sequence[dict[str, Any]],
    output_path: str | pathlib.Path | None,
    open_browser: bool,
) -> pathlib.Path:
  """Writes the dashboard page and optionally opens it in the browser."""
  html_path = write_dashboard_html(runs, output_path)
  if open_browser:
    webbrowser.open(html_path.resolve().as_uri())
  return html_path


def _label_by_file_name(paths: Sequence[str]) -> dict[str, str]:
  """Maps unique legend labels to paths, disambiguating equal file names."""
  labels: dict[str, str] = {}
  for i, path in enumerate(paths):
    stem = pathlib.Path(path).stem
    labels[stem if stem not in labels else f'{stem} ({i + 1})'] = path
  return labels


def plot_run(
    outfiles: Mapping[str, str] | Sequence[str],
    output_path: str | pathlib.Path | None = None,
    open_browser: bool = True,
) -> pathlib.Path:
  """Opens one or more runs from output files in the dashboard.

  Args:
    outfiles: Either a mapping ``{label: filepath}`` where *label* appears in
      the dashboard legend, or a plain sequence of file paths (labeled by
      file name).
    output_path: Where to write the dashboard HTML page. Defaults to a
      temporary file.
    open_browser: If True, opens the written page in the default browser.

  Returns:
    The path of the written HTML file.
  """
  if not isinstance(outfiles, Mapping):
    outfiles = _label_by_file_name(outfiles)
  runs = [load_run(path, label) for label, path in outfiles.items()]
  return _open_dashboard(runs, output_path, open_browser)


def plot_run_from_data_tree(
    data_trees: Mapping[str, xr.DataTree],
    output_path: str | pathlib.Path | None = None,
    open_browser: bool = True,
) -> pathlib.Path:
  """Opens one or more in-memory runs in the dashboard.

  Args:
    data_trees: A mapping ``{label: xr.DataTree}`` where *label* appears in
      the dashboard legend.
    output_path: Where to write the dashboard HTML page. Defaults to a
      temporary file.
    open_browser: If True, opens the written page in the default browser.

  Returns:
    The path of the written HTML file.
  """
  runs = [run_to_dict(dt, label) for label, dt in data_trees.items()]
  return _open_dashboard(runs, output_path, open_browser)


def _main():
  """Standalone exporter CLI (needs only numpy + xarray + netcdf4)."""
  import argparse  # pylint: disable=g-import-not-at-top

  parser = argparse.ArgumentParser(
      description='Export a TORAX output .nc file to dashboard JSON.'
  )
  parser.add_argument('outfile', help='Path to the TORAX output .nc file.')
  parser.add_argument(
      '-o', '--output', default=None, help='Output JSON path.'
  )
  parser.add_argument(
      '--label', default=None, help='Run label shown in the dashboard legend.'
  )
  args = parser.parse_args()

  document = load_run(args.outfile, args.label)
  output = pathlib.Path(
      args.output or pathlib.Path(args.outfile).with_suffix('.json')
  )
  output.write_text(
      json.dumps(document, separators=(',', ':')), encoding='utf-8'
  )
  print(f'Wrote {output}')


if __name__ == '__main__':
  _main()
