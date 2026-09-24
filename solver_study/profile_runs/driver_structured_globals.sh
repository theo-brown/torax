#!/bin/bash
# Dense vs structured Jacobian with the sources that depend on the state
# globally (cyclotron radiation, constant-fraction impurity radiation, ToricNN
# ICRH with a dummy surrogate). One fresh process per run, idle machine.
cd /tmp/claude-0/-home-user/9ffdac3c-d367-5e9c-a2e7-3a84fcf08450/scratchpad/profile_runs
PY=/tmp/claude-0/-home-user/9ffdac3c-d367-5e9c-a2e7-3a84fcf08450/scratchpad/venv/bin/python
RES=${RES:-results}; LOGS=${LOGS:-logs}; mkdir -p $RES $LOGS
TORIC=/tmp/claude-0/-home-user/9ffdac3c-d367-5e9c-a2e7-3a84fcf08450/scratchpad/bench/dummy_toric_nn.json
SOURCES="\"sources\": {\"cyclotron_radiation\": {}, \"impurity_radiation\": {\"model_name\": \"P_in_scaled_flat_profile\", \"fraction_P_heating\": 0.1}, \"icrh\": {\"model_path\": \"$TORIC\", \"P_total\": 10e6, \"wall_inner\": 1.24, \"wall_outer\": 2.43}}"
run() {
  label=$1; config=$2; overrides=$3; shift 3
  echo "=== $(date +%T) START $label config=$config overrides=$overrides extra=$*"
  timeout 900 $PY run_one.py --config $config --overrides "$overrides" --label $label --out $RES/$label.json "$@" > $LOGS/$label.log 2>&1
  rc=$?
  echo "=== $(date +%T) END $label rc=$rc"
  grep -h "DONE\|SIM ERROR\|Error\|error" $LOGS/$label.log | grep -v "INFO\|solver_error" | tail -3
}
RU=torax.examples.iterhybrid_rampup
for n in 50 100; do
  for mode in dense structured; do
    run prodg_rampup${n}_$mode $RU "{\"geometry\": {\"__replace__\": true, \"geometry_type\": \"circular\", \"n_rho\": $n}, \"solver\": {\"jacobian_mode\": \"$mode\"}, $SOURCES}"
  done
done
echo DRIVER_DONE
