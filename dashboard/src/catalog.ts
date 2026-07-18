/** Variable metadata: display labels, descriptions, and units.
 *
 * Labels use the `text $math$` LaTeX subset parsed by latex.ts (rendered by
 * VarLabel in the DOM and labelToPlotlyHtml in charts). Panel layouts live
 * in presets.ts. Variables absent from this catalog still plot, shown by
 * their raw name.
 */

export interface VarMeta {
  /** Label in the LaTeX subset understood by latex.ts, e.g. '$T_\\mathrm{i}$'. */
  label: string;
  /** Human-readable description shown in the settings popup. */
  name: string;
  /** Display units after export_data.py's transformations; shown alongside
   *  the description. Omitted for dimensionless quantities. */
  units?: string;
}

function v(label: string, name: string, units?: string): VarMeta {
  return {label, name, units};
}

// One line per variable: v(label, description, units?).
// prettier-ignore
export const VAR_META: Record<string, VarMeta> = {
  // Core profiles.
  T_i: v('$T_\\mathrm{i}$', 'Ion temperature', 'keV'),
  T_e: v('$T_\\mathrm{e}$', 'Electron temperature', 'keV'),
  n_e: v('$n_\\mathrm{e}$', 'Electron density', '10²⁰ m⁻³'),
  n_i: v('$n_\\mathrm{i}$', 'Main ion density', '10²⁰ m⁻³'),
  n_impurity: v('$n_\\mathrm{imp}$', 'Impurity density', '10²⁰ m⁻³'),
  psi: v('$\\psi$', 'Poloidal flux', 'Wb'),
  q: v('$q$', 'Safety factor'),
  magnetic_shear: v('$\\hat{s}$', 'Magnetic shear'),
  v_loop: v('$\\dot{\\psi}$', 'Loop voltage', 'V'),
  Z_impurity: v('$\\langle Z_\\mathrm{impurity} \\rangle$', 'Average impurity charge'),
  Z_eff: v('$Z_\\mathrm{eff}$', 'Effective charge'),
  Z_i: v('$Z_\\mathrm{i}$', 'Main ion charge'),

  // Transport coefficients (derived totals + components).
  chi_total_i: v('$\\chi_\\mathrm{i}$', 'Ion heat conductivity (total)', 'm²/s'),
  chi_total_e: v('$\\chi_\\mathrm{e}$', 'Electron heat conductivity (total)', 'm²/s'),
  chi_turb_i: v('$\\chi_\\mathrm{turb,i}$', 'Ion heat conductivity (turbulent)', 'm²/s'),
  chi_turb_e: v('$\\chi_\\mathrm{turb,e}$', 'Electron heat conductivity (turbulent)', 'm²/s'),
  chi_neo_i: v('$\\chi_\\mathrm{neo,i}$', 'Ion heat conductivity (neoclassical)', 'm²/s'),
  chi_neo_e: v('$\\chi_\\mathrm{neo,e}$', 'Electron heat conductivity (neoclassical)', 'm²/s'),
  D_total_e: v('$D_\\mathrm{e}$', 'Electron particle diffusivity (total)', 'm²/s'),
  D_turb_e: v('$D_\\mathrm{turb,e}$', 'Electron particle diffusivity (turbulent)', 'm²/s'),
  D_neo_e: v('$D_\\mathrm{neo,e}$', 'Electron particle diffusivity (neoclassical)', 'm²/s'),
  V_total_e: v('$V_\\mathrm{e}$', 'Electron particle convection (total)', 'm/s'),
  V_turb_e: v('$V_\\mathrm{turb,e}$', 'Electron particle convection (turbulent)', 'm/s'),
  V_neo_total_e: v('$V_\\mathrm{neo,e}+V_\\mathrm{ware,e}$', 'Electron particle convection (neoclassical total)', 'm/s'),
  V_neo_e: v('$V_\\mathrm{neo,e}$', 'Electron particle convection (neoclassical)', 'm/s'),
  V_neo_ware_e: v('$V_\\mathrm{ware,e}$', 'Electron particle convection (Ware pinch)', 'm/s'),

  // Currents.
  Ip: v('$I_\\mathrm{p}$', 'Plasma current', 'MA'),
  I_bootstrap: v('$I_\\mathrm{bs}$', 'Bootstrap current', 'MA'),
  I_aux_generic: v('$I_\\mathrm{generic}$', 'Generic auxiliary current', 'MA'),
  I_ecrh: v('$I_\\mathrm{ecrh}$', 'ECRH-driven current', 'MA'),
  I_non_inductive: v('$I_\\mathrm{ni}$', 'Non-inductive current', 'MA'),
  j_total: v('$j_\\mathrm{tot}$', 'Total current density', 'MA/m²'),
  j_ohmic: v('$j_\\mathrm{ohm}$', 'Ohmic current density', 'MA/m²'),
  j_bootstrap: v('$j_\\mathrm{bs}$', 'Bootstrap current density', 'MA/m²'),
  j_generic_current: v('$j_\\mathrm{generic}$', 'Generic current density', 'MA/m²'),
  j_ecrh: v('$j_\\mathrm{ecrh}$', 'ECRH current density', 'MA/m²'),
  j_external: v('$j_\\mathrm{ext}$', 'External current density', 'MA/m²'),

  // Heat source densities.
  p_icrh_i: v('$Q_\\mathrm{ICRH,i}$', 'ICRH heating to ions', 'MW/m³'),
  p_icrh_e: v('$Q_\\mathrm{ICRH,e}$', 'ICRH heating to electrons', 'MW/m³'),
  p_ecrh_e: v('$Q_\\mathrm{ECRH,e}$', 'ECRH heating to electrons', 'MW/m³'),
  p_generic_heat_i: v('$Q_\\mathrm{generic,i}$', 'Generic heating to ions', 'MW/m³'),
  p_generic_heat_e: v('$Q_\\mathrm{generic,e}$', 'Generic heating to electrons', 'MW/m³'),
  p_alpha_i: v('$Q_\\mathrm{\\alpha,i}$', 'Alpha heating to ions', 'MW/m³'),
  p_alpha_e: v('$Q_\\mathrm{\\alpha,e}$', 'Alpha heating to electrons', 'MW/m³'),
  p_ohmic_e: v('$Q_\\mathrm{ohmic}$', 'Ohmic heating', 'MW/m³'),
  ei_exchange: v('$Q_\\mathrm{ei}$', 'Ion–electron heat exchange', 'MW/m³'),
  p_bremsstrahlung_e: v('$Q_\\mathrm{brems}$', 'Bremsstrahlung', 'MW/m³'),
  p_impurity_radiation_e: v('$Q_\\mathrm{rad}$', 'Impurity radiation', 'MW/m³'),
  p_cyclotron_radiation_e: v('$Q_\\mathrm{cycl}$', 'Cyclotron radiation', 'MW/m³'),

  // Particle sources.
  s_gas_puff: v('$S_\\mathrm{puff}$', 'Gas puff particle source', '10²⁰ m⁻³ s⁻¹'),
  s_generic_particle: v('$S_\\mathrm{generic}$', 'Generic particle source', '10²⁰ m⁻³ s⁻¹'),
  s_pellet: v('$S_\\mathrm{pellet}$', 'Pellet particle source', '10²⁰ m⁻³ s⁻¹'),

  // Integrated powers and scalars.
  P_auxiliary: v('$P_\\mathrm{aux}$', 'Auxiliary heating power', 'MW'),
  P_ohmic_e: v('$P_\\mathrm{ohm}$', 'Ohmic heating power', 'MW'),
  P_alpha_total: v('$P_\\alpha$', 'Alpha heating power', 'MW'),
  P_bremsstrahlung_e: v('$P_\\mathrm{brems}$', 'Bremsstrahlung power loss', 'MW'),
  P_radiation_e: v('$P_\\mathrm{rad}$', 'Radiation power loss', 'MW'),
  P_cyclotron_e: v('$P_\\mathrm{cycl}$', 'Cyclotron power loss', 'MW'),
  P_sink: v('$P_\\mathrm{sink}$', 'Total electron heat sink', 'MW'),
  Q_fusion: v('$Q_\\mathrm{fusion}$', 'Fusion gain'),
  W_thermal_total: v('$W_\\mathrm{th}$', 'Thermal stored energy', 'MJ'),
  tau_E: v('$\\tau_\\mathrm{E}$', 'Energy confinement time', 's'),
  beta_N: v('$\\beta_\\mathrm{N}$', 'Normalized beta'),
  q95: v('$q_{95}$', 'Safety factor at 95% flux'),
  H98: v('$H_{98}$', 'H-mode confinement factor (H98y2)'),
  n_e_line_avg: v('$\\bar{n}_\\mathrm{e}$', 'Line-averaged electron density', '10²⁰ m⁻³'),
  n_e_volume_avg: v('$\\langle n_\\mathrm{e} \\rangle$', 'Volume-averaged electron density', '10²⁰ m⁻³'),
  n_i_volume_avg: v('$\\langle n_\\mathrm{i} \\rangle$', 'Volume-averaged main ion density', '10²⁰ m⁻³'),
  T_e_volume_avg: v('$\\langle T_\\mathrm{e} \\rangle$', 'Volume-averaged electron temperature', 'keV'),
  T_i_volume_avg: v('$\\langle T_\\mathrm{i} \\rangle$', 'Volume-averaged ion temperature', 'keV'),
};

export function varLabel(name: string): string {
  return VAR_META[name]?.label ?? name;
}

/** Description + units for the settings popup, e.g. 'Ion temperature [keV]'.
 *  Empty for variables not in the catalog (they still plot by raw name). */
export function varDescription(name: string): string {
  const meta = VAR_META[name];
  if (!meta) return '';
  return meta.units ? `${meta.name} [${meta.units}]` : meta.name;
}
