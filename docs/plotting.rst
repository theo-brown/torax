.. _plotting:

Plotting simulations
####################

TORAX ships a browser-based dashboard for visualizing simulation output. It
renders spatial profile panels (vs normalized toroidal flux coordinate ρ)
driven by a time slider with playback, and time-series panels with a marker
showing the current slider time. Loading several runs plots them together for
comparison: color identifies the variable and dash pattern identifies the run.

Using the plot_torax script
===========================

To visualize simulation results, use the ``plot_torax`` script. If you have
cloned the repository, an alternative is
``python3 torax/plotting/plotruns.py``.

.. code-block:: console

  plot_torax --outfile <full_path_to_simulation_output>

This writes a self-contained HTML page and opens it in your default browser.
Pass several output files to compare runs:

.. code-block:: console

  plot_torax --outfile <full_path_to_simulation_output1> \
   <full_path_to_simulation_output2>

Additional flags:

``--output`` - Write the dashboard HTML page to a specific path instead of a
temporary file. The page is fully self-contained (data included), so it can
be archived or shared as a single file.

``--no_open`` - Write the page without opening a browser.

``--export_json`` - Instead of building an HTML page, write each run as a
dashboard JSON file next to the output file. These files can be loaded into
any running dashboard via its **Open runs…** button (or dragged onto the
page), which is useful when the dashboard is hosted elsewhere.

Choosing what is plotted
========================

Which variables appear is configured interactively in the dashboard's
**⚙ Settings** popup:

* Toggle variables per panel, add any variable found in the loaded output
  files, rename panels, show/hide them, and add new profile or time-series
  panels.
* **Load preset…** offers the built-in layouts: Default, Simple, Global
  parameters, Sources, and Transport.
* **Save to file… / Load from file…** round-trip the current plot
  configuration as a JSON file, so a configuration can be versioned alongside
  a project and shared between users.
* The **Time slider** tab switches the slider stops between *plasma time*
  (evenly spaced) and *simulation steps* (the timesteps taken by the solver).

The layout persists in the browser's localStorage between sessions.

Plotting programmatically
=========================

``torax.plot_run`` and ``torax.plot_run_from_data_tree`` build and open the
dashboard from Python. See :ref:`running_programmatically`.

Development
===========

The dashboard is a React app living in the ``dashboard/`` directory of the
repository. The Python side (``torax/_src/plotting/dashboard.py``) exports
runs to JSON and injects them into a prebuilt single-file template
(``torax/_src/plotting/dashboard.html``). After changing the app, rebuild the
template with:

.. code-block:: console

  cd dashboard && npm install && npm run bundle
