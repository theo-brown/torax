/** Loading and querying exported TORAX run data. */

import {addDerivedVariables} from './derived';
import type {PanelType, RunData} from './types';

/** Source of unique run ids for the session (runs have no natural key). */
let runCounter = 0;

/** Parses and validates a JSON document produced by export_data.py. */
export function parseRun(json: unknown, fallbackLabel: string): RunData {
  if (typeof json !== 'object' || json === null) {
    throw new Error('Not a JSON object.');
  }
  const doc = json as Record<string, unknown>;
  if (doc.format !== 'torax-dashboard-v1') {
    throw new Error(
      'Unrecognized file format. Export runs with dashboard/export_data.py ' +
        "(expected format 'torax-dashboard-v1').",
    );
  }
  if (
    !Array.isArray(doc.time) ||
    doc.time.length === 0 ||
    doc.time.some(t => typeof t !== 'number')
  ) {
    throw new Error("Missing or non-numeric 'time' array.");
  }
  const run: RunData = {
    id: `run-${runCounter++}`,
    label:
      typeof doc.label === 'string' && doc.label ? doc.label : fallbackLabel,
    time: doc.time as number[],
    coords: (doc.coords ?? {}) as RunData['coords'],
    profiles: (doc.profiles ?? {}) as RunData['profiles'],
    scalars: (doc.scalars ?? {}) as RunData['scalars'],
  };
  return addDerivedVariables(run);
}

/** All variable names of the given kind present in any loaded run. */
export function availableVariables(runs: RunData[], type: PanelType): string[] {
  const names = new Set<string>();
  for (const run of runs) {
    const source = type === 'spatial' ? run.profiles : run.scalars;
    for (const name of Object.keys(source)) names.add(name);
  }
  return [...names].sort((a, b) => a.localeCompare(b));
}

/** Returns {x, y} for one variable of one run at the given kind, or null. */
export function getSeriesData(
  run: RunData,
  type: PanelType,
  name: string,
  timeIndex: number,
): {x: number[]; y: (number | null)[]} | null {
  if (type === 'spatial') {
    const profile = run.profiles[name];
    if (!profile) return null;
    const x = run.coords[profile.coord];
    if (!x) return null;
    const t = Math.min(timeIndex, profile.values.length - 1);
    return {x, y: profile.values[t]};
  }
  const scalar = run.scalars[name];
  if (!scalar) return null;
  return {x: run.time, y: scalar.values};
}

/** True if a variable's data is identically zero (or missing) in a run. */
export function isAllZero(
  run: RunData,
  type: PanelType,
  name: string,
): boolean {
  if (type === 'spatial') {
    const profile = run.profiles[name];
    if (!profile) return true;
    return profile.values.every(row => row.every(v => v == null || v === 0));
  }
  const scalar = run.scalars[name];
  if (!scalar) return true;
  return scalar.values.every(v => v == null || v === 0);
}

/** Index of the time in `run` nearest to the master run's time value. */
export function nearestTimeIndex(run: RunData, tValue: number): number {
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < run.time.length; i++) {
    const d = Math.abs(run.time[i] - tValue);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}
