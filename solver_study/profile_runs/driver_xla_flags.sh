#!/bin/bash
# Compile/run time of the step function under XLA CPU optimisation levels.
# Usage: bash driver_xla_flags.sh  (runs after the machine is idle)
cd /tmp/claude-0/-home-user/9ffdac3c-d367-5e9c-a2e7-3a84fcf08450/scratchpad/profile_runs
PY=/tmp/claude-0/-home-user/9ffdac3c-d367-5e9c-a2e7-3a84fcf08450/scratchpad/venv/bin/python
RES=results_compile; LOGS=logs_compile; mkdir -p $RES $LOGS
PC=torax.examples.iterhybrid_predictor_corrector
RU=torax.examples.iterhybrid_rampup
run() {
  label=$1; config=$2; overrides=$3; shift 3
  echo "=== $(date +%T) START $label XLA_FLAGS=$XLA_FLAGS"
  timeout 900 $PY run_one.py --config $config --overrides "$overrides" --label $label --out $RES/$label.json "$@" > $LOGS/$label.log 2>&1
  echo "=== $(date +%T) END $label rc=$?"
  $PY - <<PYEOF
import json
d = json.load(open('$RES/$label.json'))
T = d['timings']; s = d['summary']
print('  %s: lower %.1f s, xla compile %.1f s, first step %.1f s, median step %.1f ms, run_clean %.2f s, steps %d' % ('$label', T.get('aot_lower_s', 0), T.get('aot_compile_s', 0), s.get('first_step_wall_s', 0), 1e3 * s.get('step_wall_median_s', 0), s.get('run_time_clean_s', 0), s.get('n_steps', 0)))
PYEOF
}
for lvl in 3 2 1; do
  export XLA_FLAGS="--xla_backend_optimization_level=$lvl"
  for mode in dense structured; do
    run xla${lvl}_pc50_${mode} $PC "{\"numerics\": {\"t_final\": 2.0}, \"geometry\": {\"n_rho\": 50}, \"solver\": {\"solver_type\": \"newton_raphson\", \"jacobian_mode\": \"$mode\"}}" --aot --max-steps 3
  done
  # run-time effect on the full rampup at n_rho = 100 (no AOT: first step = jit compile)
  run xla${lvl}_rampup100_structured $RU "{\"geometry\": {\"n_rho\": 100}, \"solver\": {\"jacobian_mode\": \"structured\"}}"
done
unset XLA_FLAGS
echo DRIVER_DONE
