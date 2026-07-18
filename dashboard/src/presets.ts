/** Built-in plot presets, mirroring the configs in torax/plotting/configs/,
 *  plus validation for panel lists loaded from files or localStorage.
 *
 *  Panels are defined once in PANEL_LIBRARY and presets are composed from
 *  library keys, so a panel shared by several presets has a single source
 *  of truth.
 */

import type {PanelConfig, PanelType, PanelVariable, SliderMode} from './types';

let panelCounter = 0;

function panel(
  title: string,
  type: PanelType,
  variables: string[],
  overrides: Partial<
    Omit<PanelConfig, 'id' | 'title' | 'type' | 'variables'>
  > = {},
): PanelConfig {
  const vars: PanelVariable[] = variables.map(name => ({name, on: true}));
  return {
    id: `panel-${panelCounter++}`,
    title,
    type,
    variables: vars,
    visible: true,
    yMinZero: true,
    upperPercentile: 100,
    lowerPercentile: 0,
    skipFirstTime: false,
    suppressZero: false,
    ...overrides,
  };
}

/** Robust y-limits for volatile transport coefficients: clip the top (and
 *  optionally bottom) percentiles and skip t=0, where transport is zero. */
const ROBUST = {upperPercentile: 98, skipFirstTime: true, yMinZero: false};
const ROBUST_TAILS = {...ROBUST, lowerPercentile: 2};

const J_VARS = [
  'j_total',
  'j_ohmic',
  'j_bootstrap',
  'j_generic_current',
  'j_ecrh',
];
const POWER_VARS = [
  'P_auxiliary',
  'P_ohmic_e',
  'P_alpha_total',
  'P_bremsstrahlung_e',
  'P_radiation_e',
  'P_cyclotron_e',
];

/** Every panel used by any preset, keyed by a short descriptive name. */
const PANEL_LIBRARY = {
  temperature: () => panel('Temperature [keV]', 'spatial', ['T_i', 'T_e']),
  density: () => panel('Density $[10^{20}~m^{-3}]$', 'spatial', ['n_e', 'n_i']),
  electronDensity: () =>
    panel('Electron density $[10^{20}~m^{-3}]$', 'spatial', ['n_e']),
  heatConductivity: () =>
    panel(
      'Heat conductivity $[m^2/s]$',
      'spatial',
      ['chi_total_i', 'chi_total_e'],
      ROBUST,
    ),
  heatConductivityTurb: () =>
    panel(
      'Heat conductivity $[m^2/s]$',
      'spatial',
      ['chi_turb_i', 'chi_turb_e'],
      ROBUST,
    ),
  diffConv: () =>
    panel(
      'Diff $[m^2/s]$ or Conv $[m/s]$',
      'spatial',
      ['D_total_e', 'V_total_e'],
      ROBUST_TAILS,
    ),
  current: () =>
    panel(
      'Current [MA]',
      'time',
      ['Ip', 'I_bootstrap', 'I_aux_generic', 'I_ecrh'],
      {suppressZero: true},
    ),
  poloidalFlux: () => panel('Poloidal flux [Wb]', 'spatial', ['psi']),
  toroidalCurrent: () =>
    panel('Toroidal current $[MA~m^{-2}]$', 'spatial', J_VARS, {
      suppressZero: true,
    }),
  toroidalCurrentAll: () =>
    panel('Toroidal current $[MA~m^{-2}]$', 'spatial', J_VARS),
  safetyFactor: () => panel('Safety factor', 'spatial', ['q']),
  magneticShear: () => panel('Magnetic shear', 'spatial', ['magnetic_shear']),
  fusionGain: () => panel('Fusion gain', 'time', ['Q_fusion']),
  loopVoltage: () =>
    panel('Loop voltage [V]', 'spatial', ['v_loop'], {upperPercentile: 98}),
  externalHeat: () =>
    panel(
      'External heat source density $[MW~m^{-3}]$',
      'spatial',
      [
        'p_icrh_i',
        'p_icrh_e',
        'p_ecrh_e',
        'p_generic_heat_i',
        'p_generic_heat_e',
      ],
      {suppressZero: true},
    ),
  internalHeat: () =>
    panel(
      'Internal heat source density $[MW~m^{-3}]$',
      'spatial',
      ['p_alpha_i', 'p_alpha_e', 'p_ohmic_e', 'ei_exchange'],
      {suppressZero: true},
    ),
  heatSink: () =>
    panel(
      'Heat sink density $[MW~m^{-3}]$',
      'spatial',
      [
        'p_bremsstrahlung_e',
        'p_impurity_radiation_e',
        'p_cyclotron_radiation_e',
      ],
      {suppressZero: true},
    ),
  totalPowers: () =>
    panel('Total heating/sink powers [MW]', 'time', POWER_VARS, {
      suppressZero: true,
    }),
  impurityCharge: () =>
    panel('Average impurity charge', 'spatial', ['Z_impurity']),
  storedEnergy: () =>
    panel('Total thermal stored energy [MJ]', 'time', ['W_thermal_total']),
  volumeAvgTemps: () =>
    panel('Volume average temperatures [keV]', 'time', [
      'T_e_volume_avg',
      'T_i_volume_avg',
    ]),
  volumeAvgDensities: () =>
    panel('Volume average densities $[10^{20}~m^{-3}]$', 'time', [
      'n_e_volume_avg',
      'n_i_volume_avg',
    ]),
  q95: () => panel('$q$ at 95% of the normalised $\\psi$', 'time', ['q95']),
  particleSources: () =>
    panel(
      'Particle sources $[10^{20}~m^{-3}~s^{-1}]$',
      'spatial',
      ['s_gas_puff', 's_generic_particle', 's_pellet'],
      {suppressZero: true},
    ),
  chiTurb: () =>
    panel(
      'Turbulent heat conductivity $[m^2/s]$',
      'spatial',
      ['chi_turb_i', 'chi_turb_e'],
      ROBUST,
    ),
  dTurb: () =>
    panel(
      'Turbulent particle diffusivity $[m^2/s]$',
      'spatial',
      ['D_turb_e'],
      ROBUST_TAILS,
    ),
  vTurb: () =>
    panel(
      'Turbulent particle convectivity $[m/s]$',
      'spatial',
      ['V_turb_e'],
      ROBUST_TAILS,
    ),
  chiNeo: () =>
    panel(
      'Neoclassical heat conductivity $[m^2/s]$',
      'spatial',
      ['chi_neo_i', 'chi_neo_e'],
      {...ROBUST, suppressZero: true},
    ),
  dNeo: () =>
    panel(
      'Neoclassical particle diffusivity $[m^2/s]$',
      'spatial',
      ['D_neo_e'],
      {...ROBUST_TAILS, suppressZero: true},
    ),
  vNeo: () =>
    panel(
      'Neoclassical particle convectivity $[m/s]$',
      'spatial',
      ['V_neo_total_e', 'V_neo_e', 'V_neo_ware_e'],
      {...ROBUST_TAILS, suppressZero: true},
    ),
} satisfies Record<string, () => PanelConfig>;

type PanelKey = keyof typeof PANEL_LIBRARY;

export interface Preset {
  key: string;
  name: string;
  panels: () => PanelConfig[];
}

function preset(key: string, name: string, panelKeys: PanelKey[]): Preset {
  return {key, name, panels: () => panelKeys.map(k => PANEL_LIBRARY[k]())};
}

/** Built-in presets, one per config in torax/plotting/configs/. */
export const PRESETS: Preset[] = [
  preset('default', 'Default', [
    'temperature',
    'density',
    'heatConductivity',
    'diffConv',
    'current',
    'poloidalFlux',
    'toroidalCurrent',
    'safetyFactor',
    'magneticShear',
    'fusionGain',
    'loopVoltage',
    'externalHeat',
    'internalHeat',
    'heatSink',
    'totalPowers',
    'impurityCharge',
  ]),
  preset('simple', 'Simple', [
    'temperature',
    'electronDensity',
    'heatConductivityTurb',
    'toroidalCurrentAll',
    'safetyFactor',
    'magneticShear',
  ]),
  preset('global_params', 'Global parameters', [
    'current',
    'fusionGain',
    'storedEnergy',
    'volumeAvgTemps',
    'volumeAvgDensities',
    'q95',
  ]),
  preset('sources', 'Sources', [
    'temperature',
    'electronDensity',
    'toroidalCurrent',
    'totalPowers',
    'externalHeat',
    'internalHeat',
    'heatSink',
    'particleSources',
  ]),
  preset('transport', 'Transport', [
    'chiTurb',
    'dTurb',
    'vTurb',
    'chiNeo',
    'dNeo',
    'vNeo',
  ]),
];

/** The layout restored by 'Reset to default' and first launch. */
export const defaultPanels = PRESETS[0].panels;

export const CONFIG_FILE_FORMAT = 'torax-dashboard-config-v1';

export interface DashboardConfigFile {
  format: typeof CONFIG_FILE_FORMAT;
  sliderMode?: SliderMode;
  panels: PanelConfig[];
}

/** Validates and normalizes a panel list from a file or localStorage.
 *  Returns null if the shape is unusable; fills defaults for missing
 *  optional fields so older config files keep working. */
export function validatePanels(value: unknown): PanelConfig[] | null {
  if (!Array.isArray(value)) return null;
  const panels: PanelConfig[] = [];
  for (const item of value) {
    const p = item as Partial<PanelConfig> | null;
    if (
      !p ||
      typeof p.id !== 'string' ||
      typeof p.title !== 'string' ||
      (p.type !== 'spatial' && p.type !== 'time') ||
      !Array.isArray(p.variables)
    ) {
      return null;
    }
    panels.push({
      id: p.id,
      title: p.title,
      type: p.type,
      variables: p.variables
        .filter((v): v is PanelVariable => typeof v?.name === 'string')
        .map(v => ({name: v.name, on: v.on !== false})),
      visible: p.visible !== false,
      yMinZero: p.yMinZero !== false,
      upperPercentile:
        typeof p.upperPercentile === 'number' ? p.upperPercentile : 100,
      lowerPercentile:
        typeof p.lowerPercentile === 'number' ? p.lowerPercentile : 0,
      skipFirstTime: p.skipFirstTime === true,
      suppressZero: p.suppressZero === true,
    });
  }
  return panels;
}
