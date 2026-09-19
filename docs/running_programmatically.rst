.. _running_programmatically:

Running simulations programmatically
####################################

This short guide describes how to integrate Torax into your codebase and allows
you to run it multiple times efficiently.

First, we need a ``torax.ToraxConfig`` object representing the simulation config.
In this example, we will use the ``torax/examples/iterhybrid_rampup.py`` config:

.. code-block:: python

  import torax

  torax_config = torax.build_torax_config_from_file('examples/iterhybrid_rampup.py')

If you already have a ``config_dict`` dictionary in Python, you could
instead use ``torax_config = torax.ToraxConfig.from_dict(config_dict)``.

We can then run the simulation:

.. code-block:: python

  # returns the output XArray DataTree and a torax.StateHistory object.
  data_tree, state_history = torax.run_simulation(torax_config)

  # Check that the simulation completed successfully.
  if state_history.sim_error != torax.SimError.NO_ERROR:
    raise ValueError(
        f'TORAX failed to run the simulation with error: {state_history.sim_error}.'
    )

  # Example below shows how to access the fusion gain at time=2 seconds.
  Q_fusion_t2 = data_tree.scalars.Q_fusion.sel(time=2, method='nearest')

Plotting from an in-memory simulation
######################################

If you have already run a simulation and have a ``data_tree`` in memory, you can
plot it directly without saving to a file first using
``torax.plot_run_from_data_tree``:

.. code-block:: python

  plot_config = torax.import_module('plotting/configs/default_plot_config.py')['PLOT_CONFIG']

  # Plot directly from the in-memory data_tree returned by run_simulation.
  fig = torax.plot_run_from_data_tree(plot_config, {"TORAX": data_tree})

To compare multiple in-memory runs, pass a dictionary mapping labels to
DataTrees. The labels appear in the plot legends:

.. code-block:: python

  fig = torax.plot_run_from_data_tree(
      plot_config,
      {
          "TORAX": torax_data_tree,
          "JINTRAC": jintrac_data_tree,
          "EXPERIMENT": experimental_data_tree,
      },
  )

If you have saved the output to a ``.nc`` file and want to plot from disk,
use ``torax.plot_run`` instead:

.. code-block:: python

  fig = torax.plot_run(plot_config, {"TORAX": PATH_TO_LOCAL_NC_FILE})

Coupling to an external equilibrium code
########################################

TORAX can be loosely coupled to a free-boundary equilibrium code, using the
IMAS ``equilibrium`` IDS as the interchange format in both directions. The
building blocks are all available from ``torax.experimental``:

* ``torax.experimental.torax_state_to_imas_equilibrium(sim_state, post_processed_outputs)``
  writes the TORAX state to an ``equilibrium`` IDS. Besides the flux surface
  geometry it contains the quantities that source the Grad-Shafranov equation,
  ``profiles_1d.dpressure_dpsi`` (:math:`p'`) and ``profiles_1d.f_df_dpsi``
  (:math:`FF'`), and the plasma current ``global_quantities.ip``.
* ``torax.experimental.geometry.IMASConfig(equilibrium_object=ids, ...)``
  builds a ``StandardGeometry`` from an in-memory ``equilibrium`` IDS returned
  by the equilibrium code.
* ``torax.experimental.make_step_fn`` and
  ``torax.experimental.get_initial_state_and_post_processed_outputs`` expose
  the jitted step function and the initial state. Both accept a
  ``GeometryProvider`` override, so the geometry from the equilibrium code
  can be injected at every step without rebuilding the config or recompiling.

A loosely coupled time step from ``t`` to ``t + dt`` then looks like:

.. code-block:: python

  from torax import experimental
  from torax.experimental import geometry as torax_geometry

  step_fn = experimental.make_step_fn(torax_config)

  def geometry_from_ids(ids):
    return torax_geometry.IMASConfig(
        equilibrium_object=ids,
        face_centers=torax_config.geometry.get_face_centers(),
        explicit_convert=False,
    ).build_geometry()

  # Initial state from the equilibrium code's starting equilibrium.
  geo = geometry_from_ids(equilibrium_code.initial_equilibrium_ids())
  state, post_processed = experimental.get_initial_state_and_post_processed_outputs(
      step_fn,
      geometry_overrides=torax_geometry.StandardGeometryProvider.create_provider(
          {t: geo, t + dt: geo}, calcphibdot=True
      ),
  )

  for _ in range(max_iterations):
    # 1. TORAX -> equilibrium code: p', FF' and Ip in an equilibrium IDS.
    torax_ids = experimental.torax_state_to_imas_equilibrium(state, post_processed)
    # 2. Equilibrium code -> TORAX: the new geometry at t + dt in an IDS.
    geo_next = geometry_from_ids(equilibrium_code.solve(t + dt, torax_ids))
    # 3. Transport step with the geometry interpolated between t and t + dt.
    provider = torax_geometry.StandardGeometryProvider.create_provider(
        {t: geo, t + dt: geo_next}, calcphibdot=True
    )
    new_state, new_post_processed = step_fn.jitted_fixed_time_step(
        dt, state, post_processed, geo_overrides=provider
    )
    # 4. Iterate until the exchanged p' and FF' stop changing.
    ...

Note that the same provider type (here ``StandardGeometryProvider``) must be
used for the initial state and for every step, since the geometry is part of
the state carried through the jitted loops and its array types must not
change. A complete implementation of this loop, including the convergence
iteration, is available in the FreeGSNKE package
(``freegsnke.torax_coupling``), both with static equilibria for prescribed
coil currents and with the coil and passive-structure currents evolved on the
vessel timescale inside each coupling interval.
