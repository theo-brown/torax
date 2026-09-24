#!/bin/bash
# Public-API compile check of the structured assembly under each inlining mode.
cd "$(dirname "$0")"
PY=${PY:-python}
N=${N:-50}
run() {
  echo "=== $(date +%T) $*"
  s=$(date +%s.%N)
  timeout ${TMO:-300} $PY "$@" > last.log 2>&1; rc=$?
  e=$(date +%s.%N)
  grep -E "RESULT|Error" last.log | tail -2
  echo "rc=$rc process_wall=$(echo "$e - $s" | bc) s"
}
run official_api.py dense $N 4
for inl in ${MODES:-xla_late xla_early auto jax_early none}; do run official_api_patched.py $inl structured $N 4; done
echo "=== $(date +%T) done"
