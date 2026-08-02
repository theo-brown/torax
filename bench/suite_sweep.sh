#!/bin/bash
# Runs the reference test suite with a solver variant forced on, so the
# committed reference data measures how far each variant moves the answer.
#
# The variants are behind flags that default to off, so running the suite
# unmodified only re-tests the default path. Here each flag's *default* is
# flipped for the duration of the run, then restored.
#
# Reference tests use rtol=1e-9, atol=0 (a pure relative check), so a variant
# that changes the answer at all fails them by construction. The number that
# matters is therefore the magnitude of the deviation, not pass/fail; the
# per-element "Rel Err" column in each failure report carries it.
#
# Tests run in BATCHES, each in a fresh pytest process. Running all 70 in one
# process exhausts memory on this 16 GB, swapless box: JAX retains every
# compiled executable, and XLA eventually aborts inside
# backend_compile_and_load with SIGABRT (exit 134) partway through. Batching
# bounds the number of live executables to one batch's worth.
set -u
cd /home/user/torax
LOG=${SWEEP_LOG:-/tmp/suite_sweep}
BATCH=${SWEEP_BATCH:-6}
mkdir -p "$LOG"

FILES="torax/tests/sim_test.py torax/tests/sim_no_compile_test.py torax/tests/pedestal_test.py torax/tests/transport_test.py torax/tests/sim_time_dependence_test.py"

# Collect once, on the trunk branch; the test list is identical on every branch.
git checkout -q claude/torax-solver-performance-l3qc7e
python -m pytest $FILES -q --collect-only 2>/dev/null \
  | grep -E "^torax/tests/.*::" | sort > "$LOG/tests.txt"
echo "collected $(wc -l < "$LOG/tests.txt") tests, batch size $BATCH"

run_suite () {   # $1 = tag, $2 = branch, $3 = sed expression ('' for none)
  local tag=$1 branch=$2 sedexpr=$3
  echo "=========== $tag (branch $branch) ==========="
  git checkout -q "$branch" || return 1
  if [ -n "$sedexpr" ]; then
    sed -i "$sedexpr" torax/_src/solver/pydantic_model.py
    echo "-- forced default: $(grep -oE "(newton_linear_solver|split_psi).*=.*" torax/_src/solver/pydantic_model.py | head -1)"
  fi
  : > "$LOG/$tag.log"
  local t0=$SECONDS n=0 batch=0
  while IFS= read -r -d '' -u 9 chunk || [ -n "$chunk" ]; do :; done 9< /dev/null
  # Split the test list into batches of $BATCH and run each in its own process.
  split -l "$BATCH" -d "$LOG/tests.txt" "$LOG/${tag}_batch_"
  for bf in "$LOG/${tag}"_batch_*; do
    batch=$((batch+1))
    # shellcheck disable=SC2046
    timeout 1800 python -m pytest $(cat "$bf") -q -p no:randomly --tb=long -rN \
        >> "$LOG/$tag.log" 2>&1
    local rc=$?
    n=$((n+$(wc -l < "$bf")))
    echo "   batch $batch rc=$rc ($n/$(wc -l < "$LOG/tests.txt") tests, $((SECONDS-t0))s)"
    rm -f "$bf"
  done
  echo "-- $tag done in $((SECONDS-t0))s"
  echo "-- passed: $(grep -oE '[0-9]+ passed' "$LOG/$tag.log" | awk '{s+=$1} END {print s+0}')" \
       "failed: $(grep -oE '[0-9]+ failed' "$LOG/$tag.log" | awk '{s+=$1} END {print s+0}')" \
       "aborted batches: $(grep -c 'Fatal Python error' "$LOG/$tag.log")"
  git checkout -q -- torax/_src/solver/pydantic_model.py
}

run_suite base     claude/torax-solver-performance-l3qc7e ''
run_suite jfnk     exp/jfnk     "s/\] = 'direct'/] = 'jfnk'/"
run_suite psisplit exp/psisplit "s/^  split_psi: Annotated\[bool, torax_pydantic.JAX_STATIC\] = False/  split_psi: Annotated[bool, torax_pydantic.JAX_STATIC] = True/"

git checkout -q claude/torax-solver-performance-l3qc7e
echo "=========== DONE ==========="
