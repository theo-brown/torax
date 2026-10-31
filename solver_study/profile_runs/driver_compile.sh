#!/bin/bash
# Compile-time split (AOT lower vs XLA compile, plus the jit first step) for the
# dense and structured Jacobian, with and without the globally coupled sources.
# One fresh process per run, idle machine. Usage: TAG=name bash driver_compile.sh
cd /tmp/claude-0/-home-user/9ffdac3c-d367-5e9c-a2e7-3a84fcf08450/scratchpad/profile_runs
PY=/tmp/claude-0/-home-user/9ffdac3c-d367-5e9c-a2e7-3a84fcf08450/scratchpad/venv/bin/python
TAG=${TAG:-base}
RES=results_compile; LOGS=logs_compile; mkdir -p $RES $LOGS
TORIC=/tmp/claude-0/-home-user/9ffdac3c-d367-5e9c-a2e7-3a84fcf08450/scratchpad/bench/dummy_toric_nn.json
SOURCES="\"sources\": {\"cyclotron_radiation\": {}, \"impurity_radiation\": {\"model_name\": \"P_in_scaled_flat_profile\", \"fraction_P_heating\": 0.1}, \"icrh\": {\"model_path\": \"$TORIC\", \"P_total\": 10e6, \"wall_inner\": 1.24, \"wall_outer\": 2.43}}"
run() {
  label=$1; config=$2; overrides=$3; shift 3
  echo "=== $(date +%T) START $label overrides=$overrides"
  timeout 900 $PY run_one.py --config $config --overrides "$overrides" --label $label --out $RES/$label.json --aot --max-steps 3 "$@" > $LOGS/$label.log 2>&1
  echo "=== $(date +%T) END $label rc=$?"
  $PY - <<PYEOF
import json
d = json.load(open('$RES/$label.json'))
T = d['timings']; s = d['summary']
print('  %s: lower %.1f s, xla compile %.1f s, first step (jit) %.1f s, hlo %.1f MB, steps %d' % ('$label', T.get('aot_lower_s', 0), T.get('aot_compile_s', 0), s.get('first_step_wall_s', 0), T.get('aot_hlo_text_bytes', 0) / 1e6, s.get('n_steps', 0)))
PYEOF
}
PC=torax.examples.iterhybrid_predictor_corrector
for mode in ${MODES:-dense structured}; do
  run cmp_${TAG}_${mode} $PC "{\"numerics\": {\"t_final\": 2.0}, \"geometry\": {\"n_rho\": 50}, \"solver\": {\"solver_type\": \"newton_raphson\", \"jacobian_mode\": \"$mode\"}}"
  run cmpg_${TAG}_${mode} $PC "{\"numerics\": {\"t_final\": 2.0}, \"geometry\": {\"__replace__\": true, \"geometry_type\": \"circular\", \"n_rho\": 50}, \"solver\": {\"solver_type\": \"newton_raphson\", \"jacobian_mode\": \"$mode\"}, $SOURCES}"
done
echo DRIVER_DONE
