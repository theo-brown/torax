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

"""Unit tests for torax._src.plotting.dashboard."""

import json

from absl.testing import absltest
from absl.testing import parameterized
from torax._src import path_utils
from torax._src.plotting import dashboard
from torax._src.test_utils import paths
import xarray as xr


def _all_test_data_files():
  """All output files in test_data, mirroring the old plotruns coverage."""
  test_data_dir = path_utils.torax_path() / 'tests' / 'test_data'
  return sorted(f.name for f in test_data_dir.glob('*.nc'))


class DashboardTest(parameterized.TestCase):

  @parameterized.parameters(_all_test_data_files())
  def test_export_all_test_data_files(self, filename: str):
    """Every output file exports to a valid, JSON-serializable document."""
    path = paths.test_data_dir() / filename
    document = dashboard.load_run(path)

    self.assertEqual(document['format'], dashboard.RUN_DATA_FORMAT)
    self.assertEqual(document['label'], path.stem)
    self.assertNotEmpty(document['time'])
    self.assertNotEmpty(document['profiles'])
    self.assertNotEmpty(document['scalars'])

    # Every exported profile is (time, rho)-shaped against a known coord.
    n_time = len(document['time'])
    for name, profile in document['profiles'].items():
      self.assertIn(profile['coord'], document['coords'], msg=name)
      self.assertLen(profile['values'], n_time, msg=name)
      self.assertLen(
          profile['values'][0],
          len(document['coords'][profile['coord']]),
          msg=name,
      )
    for name, scalar in document['scalars'].items():
      self.assertLen(scalar['values'], n_time, msg=name)

    # The document is JSON-serializable (no NaN/inf leftovers).
    json.dumps(document)

  def test_export_applies_unit_transformations(self):
    path = paths.test_data_dir() / 'test_iterhybrid_rampup.nc'
    document = dashboard.load_run(path)
    # Plasma current is exported in MA, not A.
    ip_values = [v or 0 for v in document['scalars']['Ip']['values']]
    self.assertLess(max(ip_values), 1e3)
    self.assertEqual(document['scalars']['Ip']['units'], 'MA')

  def test_run_to_dict_from_data_tree(self):
    path = paths.test_data_dir() / 'test_iterhybrid_rampup.nc'
    document = dashboard.run_to_dict(xr.open_datatree(path), 'my label')
    self.assertEqual(document['label'], 'my label')

  def test_plot_run_writes_dashboard_html(self):
    """End-to-end: files in, self-contained page with embedded runs out."""
    output = self.create_tempfile('dashboard.html').full_path
    html_path = dashboard.plot_run(
        {
            'Run 1': str(paths.test_data_dir() / 'test_iterhybrid_rampup.nc'),
            'Run 2': str(paths.test_data_dir() / 'test_psi_and_heat.nc'),
        },
        output_path=output,
        open_browser=False,
    )

    html = html_path.read_text(encoding='utf-8')
    self.assertIn('window.TORAX_EMBEDDED_RUNS = [{', html)
    self.assertIn('"label": "Run 1"', html)
    self.assertIn('"label": "Run 2"', html)
    # The injected JSON must not be able to terminate its script tag.
    self.assertNotIn('</script>', html.split('TORAX_EMBEDDED_RUNS')[1][:1000])

  def test_plot_run_disambiguates_equal_file_names(self):
    """Sequence inputs with colliding file names keep both runs."""
    path = str(paths.test_data_dir() / 'test_iterhybrid_rampup.nc')
    output = self.create_tempfile('dashboard.html').full_path
    html_path = dashboard.plot_run(
        [path, path], output_path=output, open_browser=False
    )
    html = html_path.read_text(encoding='utf-8')
    self.assertIn('"label": "test_iterhybrid_rampup"', html)
    self.assertIn('"label": "test_iterhybrid_rampup (2)"', html)


if __name__ == '__main__':
  absltest.main()
