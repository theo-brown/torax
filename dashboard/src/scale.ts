/** Robust y-limits for panels (mirroring plotruns_lib._get_y_limits). */

import {isAllZero} from './data';
import type {PanelConfig, RunData} from './types';

function percentile(values: number[], p: number): number {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const frac = idx - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

/** Robust y-domain for a panel across all runs (fixed while sliding time). */
export function computeYDomain(
  runs: RunData[],
  panel: PanelConfig,
): [number, number] | null {
  let ymin = Infinity;
  let ymax = -Infinity;

  for (const run of runs) {
    const values: number[] = [];
    for (const variable of panel.variables) {
      if (!variable.on) continue;
      if (panel.suppressZero && isAllZero(run, panel.type, variable.name))
        continue;
      if (panel.type === 'spatial') {
        const profile = run.profiles[variable.name];
        if (!profile) continue;
        const startT = panel.skipFirstTime && profile.values.length > 1 ? 1 : 0;
        for (let t = startT; t < profile.values.length; t++) {
          for (const v of profile.values[t])
            if (v != null && isFinite(v)) values.push(v);
        }
      } else {
        const scalar = run.scalars[variable.name];
        if (!scalar) continue;
        const startT = panel.skipFirstTime && scalar.values.length > 1 ? 1 : 0;
        for (let t = startT; t < scalar.values.length; t++) {
          const v = scalar.values[t];
          if (v != null && isFinite(v)) values.push(v);
        }
      }
    }
    if (values.length === 0) continue;
    ymin = Math.min(ymin, percentile(values, panel.lowerPercentile));
    ymax = Math.max(ymax, percentile(values, panel.upperPercentile));
  }

  if (!isFinite(ymin) || !isFinite(ymax)) return null;

  let lower = ymin > 0 ? ymin / 1.05 : ymin * 1.05;
  if (panel.yMinZero) lower = Math.min(lower, 0);
  let upper = ymax * 1.05;
  if (lower === upper) {
    const pad = Math.abs(lower) * 0.1 || 1;
    lower -= pad;
    upper += pad;
  }
  // Keep a zero baseline visually distinct from the axis edge.
  if (lower === 0) lower = -0.02 * Math.max(Math.abs(upper), 1e-12);
  return [lower, upper];
}
