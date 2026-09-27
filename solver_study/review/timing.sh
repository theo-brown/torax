#!/bin/bash
# Same-session timing of the working tree against an earlier version of the
# code (REPORT.md 3.14: the code before the review), structured mode,
# interleaved pairs, one process at a time.
# Usage: BASE=/path/to/worktree/of/the/earlier/version bash timing.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=${PY:-python}
cd "$HERE/../profile_runs"
OUT=${OUT:-timing.jsonl}
: > $OUT
for pair in 1 2; do
  for spec in "rampup 100" "rampup_global 50"; do
    echo "=== $(date +%T) final $spec (load $(cut -d' ' -f1 /proc/loadavg))"
    timeout 1200 $PY first_vs_repeat.py $spec structured 3 2>/dev/null | grep "^RESULT" | sed 's/^RESULT {/{"code": "final", /' | tee -a $OUT
    echo "=== $(date +%T) previous $spec (load $(cut -d' ' -f1 /proc/loadavg))"
    timeout 1200 $PY "$HERE/run_old.py" "$BASE" first_vs_repeat.py $spec structured 3 2>/dev/null | grep "^RESULT" | sed 's/^RESULT {/{"code": "before_review", /' | tee -a $OUT
  done
done
echo TIMING_DONE
