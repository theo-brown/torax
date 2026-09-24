#!/bin/bash
# Sequential driver for the production jacobian_mode='structured' benchmark:
# one fresh process per configuration, never in parallel, idle machine.
cd /tmp/claude-0/-home-user/9ffdac3c-d367-5e9c-a2e7-3a84fcf08450/scratchpad/profile_runs
PY=/tmp/claude-0/-home-user/9ffdac3c-d367-5e9c-a2e7-3a84fcf08450/scratchpad/venv/bin/python
RES=${RES:-results}; LOGS=${LOGS:-logs}; mkdir -p $RES $LOGS
run() {
  label=$1; config=$2; overrides=$3; shift 3
  echo "=== $(date +%T) START $label config=$config overrides=$overrides extra=$*"
  timeout 900 $PY run_one.py --config $config --overrides "$overrides" --label $label --out $RES/$label.json "$@" > $LOGS/$label.log 2>&1
  rc=$?
  echo "=== $(date +%T) END $label rc=$rc"
  grep -h "DONE\|SIM ERROR\|Error\|error" $LOGS/$label.log | grep -v "INFO\|solver_error" | tail -3
}
PC=torax.examples.iterhybrid_predictor_corrector
RU=torax.examples.iterhybrid_rampup

for mode in dense structured; do
  # iterhybrid_rampup as shipped (n_rho = 50, dt = 2 s, 40 steps).
  run prod_rampup50_$mode $RU "{\"solver\": {\"jacobian_mode\": \"$mode\"}}"
  # the same at n_rho = 100 (N = 400).
  run prod_rampup100_$mode $RU "{\"geometry\": {\"n_rho\": 100}, \"solver\": {\"jacobian_mode\": \"$mode\"}}"
done
# grid scaling on the predictor_corrector physics with the Newton solver (as 3.1c).
for n in 25 50 100; do
  for mode in dense structured; do
    run prod_scale${n}_$mode $PC "{\"numerics\": {\"t_final\": 1.0}, \"geometry\": {\"n_rho\": $n}, \"solver\": {\"solver_type\": \"newton_raphson\", \"jacobian_mode\": \"$mode\"}}" --max-steps 60
  done
done
echo DRIVER_DONE
