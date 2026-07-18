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

"""Post-run plotting tool: opens TORAX output files in the React dashboard.

Loads one or more output files (several files enable run comparison) into a
self-contained HTML page and opens it in the default browser. Which variables
are plotted is configured interactively in the dashboard's settings popup;
plot configurations can be saved to and loaded from JSON files there.
"""

import json
import pathlib

from absl import app
from absl.flags import argparse_flags
from torax._src.plotting import dashboard


def parse_flags(_):
  """Parses flags for the plotting tool."""
  parser = argparse_flags.ArgumentParser(description='Plot finished run(s)')
  parser.add_argument(
      '--outfile',
      nargs='+',
      required=True,
      help=(
          'Relative location of output files (pass several to compare runs)'
      ),
  )
  parser.add_argument(
      '--output',
      default=None,
      help=(
          'Path to write the dashboard HTML page to (default: a temporary'
          ' file).'
      ),
  )
  parser.add_argument(
      '--no_open',
      action='store_true',
      help='Do not open the written HTML page in a browser.',
  )
  parser.add_argument(
      '--export_json',
      action='store_true',
      help=(
          'Instead of building an HTML page, write each run as a dashboard'
          ' JSON file next to the output file (loadable via the dashboard'
          " app's Open runs button)."
      ),
  )
  return parser.parse_args()


def main(args):
  if args.export_json:
    for outfile in args.outfile:
      document = dashboard.load_run(outfile)
      json_path = pathlib.Path(outfile).with_suffix('.json')
      json_path.write_text(
          json.dumps(document, separators=(',', ':')), encoding='utf-8'
      )
      print(f'Wrote {json_path}')
    return

  html_path = dashboard.plot_run(
      args.outfile,
      output_path=args.output,
      open_browser=not args.no_open,
  )
  print(f'Dashboard written to {html_path}')


# Method used by the `plot_torax` binary.
def run():
  app.run(main, flags_parser=parse_flags)


if __name__ == '__main__':
  run()
