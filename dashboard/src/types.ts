/** Shared type definitions for the TORAX dashboard. */

export interface ProfileVar {
  /** 2D array [time][rho]. */
  values: (number | null)[][];
  /** Name of the radial coordinate this profile is defined on. */
  coord: string;
  units: string;
}

export interface ScalarVar {
  /** 1D array [time]. */
  values: (number | null)[];
  units: string;
}

/** One TORAX run, as produced by export_data.py (plus derived variables). */
export interface RunData {
  id: string;
  label: string;
  time: number[];
  coords: Record<string, number[]>;
  profiles: Record<string, ProfileVar>;
  scalars: Record<string, ScalarVar>;
}

export type PanelType = 'spatial' | 'time';

/** Slider stepping mode, matching the original plotly tool's toggle:
 *  'plasma' — stops evenly spaced in plasma time;
 *  'steps' — stops at the timesteps taken by the simulation. */
export type SliderMode = 'plasma' | 'steps';

export interface PanelVariable {
  name: string;
  /** Whether the variable is currently plotted. Order in the panel's list
   *  fixes the color slot, so toggling never repaints survivors. */
  on: boolean;
}

export interface PanelConfig {
  id: string;
  title: string;
  type: PanelType;
  variables: PanelVariable[];
  visible: boolean;
  /** Extend the y-axis to include zero. */
  yMinZero: boolean;
  /** Percentiles used for robust y-limits (mirrors plotruns_lib). */
  upperPercentile: number;
  lowerPercentile: number;
  /** Exclude the first timepoint from y-limit calculation. */
  skipFirstTime: boolean;
  /** Do not draw traces whose data is identically zero. */
  suppressZero: boolean;
}
