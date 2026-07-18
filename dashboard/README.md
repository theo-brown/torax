# TORAX dashboard

A React dashboard for visualizing TORAX simulation output: spatial profile
panels (vs normalized toroidal flux coordinate ρ) driven by a global time
slider with playback, and time-series panels with a marker showing the
current slider time.

Features:

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

## Development

```bash
cd dashboard
npm install
npm run dev      # development server
npm run build    # production build in dist/ (static, host anywhere)
```

Open the app, then drag exported run `.json` files onto the page (or use
**Open runs…**). Load more than one file to compare runs.

## Layout

- `src/catalog.ts` — variable labels/units (LaTeX subset, see `src/latex.ts`).
- `src/presets.ts` — panel library and the built-in presets.
- `src/derived.ts` — derived variables computed on load.
- `src/scale.ts` — percentile-based y-limits.
- `src/components/` — Plotly chart wrapper, panel card, time slider, and the
  settings modal.
