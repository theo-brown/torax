#!/bin/bash
# First vs repeated torax.run_simulation calls, dense and structured Jacobian.
# One fresh process per case and mode, run strictly one after another.
cd "$(dirname "$0")"
PY=${PY:-/tmp/claude-0/-home-user/9ffdac3c-d367-5e9c-a2e7-3a84fcf08450/scratchpad/venv/bin/python}
OUT=${OUT:-first_vs_repeat.jsonl}
: > $OUT
for spec in "rampup 50" "rampup 100" "rampup_global 50" "rampup_global 100" "predictor_corrector 25" "predictor_corrector 50" "predictor_corrector 100"; do
  for mode in dense structured; do
    echo "=== $(date +%T) $spec $mode (load $(cut -d' ' -f1 /proc/loadavg))"
    timeout 1200 $PY first_vs_repeat.py $spec $mode 3 2>/dev/null | tee -a /dev/stderr | grep "^RESULT" | sed 's/^RESULT //' >> $OUT
  done
done
echo DRIVER_DONE
