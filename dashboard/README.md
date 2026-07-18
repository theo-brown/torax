# TORAX dashboard

The React app behind TORAX's plotting tooling. `plot_torax` (and
`torax.plot_run`) export runs to JSON and inject them into a prebuilt
single-file bundle of this app, `torax/_src/plotting/dashboard.html`, which
opens in the browser. The Python side lives in
`torax/_src/plotting/dashboard.py`.

Features:

- **Plotly charts**: spatial profile panels (vs normalized toroidal flux
  coordinate ρ) driven by a global time slider with playback, and time-series
  panels with a marker showing the current slider time.
- **Settings popup** (⚙ Settings) to choose which variables appear in each
  panel, show/hide panels, rename them, add new profile or time-series panels,
  and add any variable found in the loaded output files. It also holds the
  time-slider mode: **Plasma time** (stops evenly spaced in time) or
  **Simulation steps** (the timesteps taken by the sim). Everything is
  persisted in the browser's localStorage.
- **Presets and config files**: "Load preset…" offers the Default, Simple,
  Global parameters, Sources, and Transport layouts, and "Save to file…" /
  "Load from file…" round-trip the current panel configuration as a JSON file
  you can share or keep per project.
- **Run comparison**: load several runs; variables keep their color while
  runs are distinguished by dash pattern.
- Unified hover tooltip listing every series at the cursor, light and dark
  themes, Roboto typography, LaTeX-formatted labels, derived totals
  (`chi_total_i`, `D_total_e`, `V_total_e`, `P_auxiliary`, `P_sink`, …)
  computed on load.

## Everyday use

No node required — `plot_torax` uses the committed bundle:

```bash
plot_torax --outfile run.nc                 # open one run in the browser
plot_torax --outfile a.nc b.nc              # compare runs
plot_torax --outfile run.nc --export_json   # write run.json for a hosted app
```

Exports apply display-unit transformations (A → MA, W → MW,
m⁻³ → 10²⁰ m⁻³, …). The exporter also works standalone, needing only
`xarray`, `netcdf4`, and `numpy`:

```bash
python torax/_src/plotting/dashboard.py run.nc -o run.json
```

## Development

```bash
cd dashboard
npm install
npm run dev      # development server (load runs via Open runs…)
npm run bundle   # rebuild torax/_src/plotting/dashboard.html
```

After changing the app, run `npm run bundle` and commit the regenerated
`torax/_src/plotting/dashboard.html` so the Python package stays in sync.

## Layout

- `src/catalog.ts` — variable labels/units (LaTeX subset, see `src/latex.ts`).
- `src/presets.ts` — panel library and the built-in presets.
- `src/derived.ts` — derived variables computed on load.
- `src/scale.ts` — percentile-based y-limits.
- `src/components/` — Plotly chart wrapper, panel card, time slider, and the
  settings modal.
- `scripts/bundle.mjs` — builds the self-contained template with fonts from
  `fonts/` and a placeholder for embedded runs.
