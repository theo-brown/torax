/** Derived variables, mirroring the properties on plotruns_lib.PlotData. */

import type {ProfileVar, RunData, ScalarVar} from './types';

type Num = number | null;

function addArrays(a: Num[], b: Num[]): Num[] {
  return a.map((v, i) => {
    const w = b[i];
    return v == null || w == null ? null : v + w;
  });
}

function addProfiles(a: ProfileVar, b: ProfileVar): ProfileVar | null {
  if (a.coord !== b.coord || a.values.length !== b.values.length) return null;
  return {
    values: a.values.map((row, t) => addArrays(row, b.values[t])),
    coord: a.coord,
    units: a.units,
  };
}

function addScalars(a: ScalarVar, b: ScalarVar): ScalarVar | null {
  if (a.values.length !== b.values.length) return null;
  return {values: addArrays(a.values, b.values), units: a.units};
}

/** Adds derived totals to a run in place (skipping any that can't be built). */
export function addDerivedVariables(run: RunData): RunData {
  const p = run.profiles;
  const s = run.scalars;

  const derivedProfile = (name: string, x?: string, y?: string) => {
    if (name in p || !x || !y || !(x in p) || !(y in p)) return;
    const sum = addProfiles(p[x], p[y]);
    if (sum) p[name] = sum;
  };

  derivedProfile('chi_total_i', 'chi_turb_i', 'chi_neo_i');
  derivedProfile('chi_total_e', 'chi_turb_e', 'chi_neo_e');
  derivedProfile('D_total_e', 'D_turb_e', 'D_neo_e');
  derivedProfile('V_neo_total_e', 'V_neo_e', 'V_neo_ware_e');
  derivedProfile('V_total_e', 'V_turb_e', 'V_neo_total_e');

  if (!('P_auxiliary' in s) && 'P_aux_total' in s) {
    s.P_auxiliary = {
      values: [...s.P_aux_total.values],
      units: s.P_aux_total.units,
    };
  }
  if (
    !('P_sink' in s) &&
    'P_bremsstrahlung_e' in s &&
    'P_radiation_e' in s &&
    'P_cyclotron_e' in s
  ) {
    const partial = addScalars(s.P_bremsstrahlung_e, s.P_radiation_e);
    const total = partial && addScalars(partial, s.P_cyclotron_e);
    if (total) s.P_sink = total;
  }

  return run;
}
