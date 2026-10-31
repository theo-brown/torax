#!/bin/bash
# Sequential driver: one fresh process per configuration, never in parallel.
cd /tmp/claude-0/-home-user/9ffdac3c-d367-5e9c-a2e7-3a84fcf08450/scratchpad/profile_runs
PY=/tmp/claude-0/-home-user/9ffdac3c-d367-5e9c-a2e7-3a84fcf08450/scratchpad/venv/bin/python
run() {
  label=$1; config=$2; overrides=$3; shift 3
  echo "=== $(date +%T) START $label config=$config overrides=$overrides extra=$*"
  timeout 480 $PY run_one.py --config $config --overrides "$overrides" --label $label --out results/$label.json "$@" > logs/$label.log 2>&1
  rc=$?
  echo "=== $(date +%T) END $label rc=$rc"
  grep -h "DONE\|SIM ERROR\|Error\|error" logs/$label.log | grep -v "INFO\|solver_error" | tail -3
}
PC=torax.examples.iterhybrid_predictor_corrector

# ---- Batch 1: baselines (as-is)
run base_rampup torax.examples.iterhybrid_rampup '{}'
run rampup_log torax.examples.iterhybrid_rampup '{"solver": {"log_iterations": true}}' --capture-log
run base_basic torax.examples.basic_config '{}'
run base_step torax.examples.step_flattop_bgb '{}'
run base_small_dt torax.benchmarks.iterhybrid_predictor_corrector_fixed_small_dt '{}'

# ---- Batch 2: solver comparison, same physics, t_final=2.0
run cmp_a_linear $PC '{"numerics": {"t_final": 2.0}}'
run cmp_b_newton $PC '{"numerics": {"t_final": 2.0}, "solver": {"solver_type": "newton_raphson"}}'
run cmp_c_newton_xold $PC '{"numerics": {"t_final": 2.0}, "solver": {"solver_type": "newton_raphson", "initial_guess_mode": "x_old"}}'
run cmp_d_newton_nopc $PC '{"numerics": {"t_final": 2.0}, "solver": {"solver_type": "newton_raphson", "use_predictor_corrector": false}}'
run cmp_a10_linear $PC '{"numerics": {"t_final": 2.0}, "solver": {"n_corrector_steps": 10}}'

# ---- Batch 4: compile-time breakdown, n_rho=50, AOT lower/compile, 5 steps
run compile_vmapF_pcT $PC '{"numerics": {"t_final": 2.0}, "geometry": {"n_rho": 50}, "solver": {"solver_type": "newton_raphson", "vmap_linesearch": false, "use_predictor_corrector": true}}' --aot --max-steps 5
run compile_vmapT_pcT $PC '{"numerics": {"t_final": 2.0}, "geometry": {"n_rho": 50}, "solver": {"solver_type": "newton_raphson", "vmap_linesearch": true, "use_predictor_corrector": true}}' --aot --max-steps 5
run compile_vmapF_pcF $PC '{"numerics": {"t_final": 2.0}, "geometry": {"n_rho": 50}, "solver": {"solver_type": "newton_raphson", "vmap_linesearch": false, "use_predictor_corrector": false}}' --aot --max-steps 5
run compile_vmapT_pcF $PC '{"numerics": {"t_final": 2.0}, "geometry": {"n_rho": 50}, "solver": {"solver_type": "newton_raphson", "vmap_linesearch": true, "use_predictor_corrector": false}}' --aot --max-steps 5
run compile_linear $PC '{"numerics": {"t_final": 2.0}, "geometry": {"n_rho": 50}}' --aot --max-steps 5

# ---- Batch 3: grid scaling, t_final=1.0, chi dt, capped at 60 steps
for n in 25 50 100; do
  run scale_linear_$n $PC "{\"numerics\": {\"t_final\": 1.0}, \"geometry\": {\"n_rho\": $n}}" --max-steps 60
  run scale_newton_$n $PC "{\"numerics\": {\"t_final\": 1.0}, \"geometry\": {\"n_rho\": $n}, \"solver\": {\"solver_type\": \"newton_raphson\"}}" --max-steps 60
done
run scale_linear_200 $PC '{"numerics": {"t_final": 1.0}, "geometry": {"n_rho": 200}}' --max-steps 60
run scale_newton_200 $PC '{"numerics": {"t_final": 1.0}, "geometry": {"n_rho": 200}, "solver": {"solver_type": "newton_raphson"}}' --max-steps 60

# ---- Batch 5: per-step cost decomposition, n_rho=50, adaptive_dt=False, 40 steps
D='"numerics": {"t_final": 1.0, "adaptive_dt": false}, "geometry": {"n_rho": 50}'
run decomp_newton_full $PC "{$D, \"solver\": {\"solver_type\": \"newton_raphson\"}}" --max-steps 40
run decomp_newton_maxiter0 $PC "{$D, \"solver\": {\"solver_type\": \"newton_raphson\", \"n_max_iterations\": 0}}" --max-steps 40
run decomp_newton_maxiter0_xold $PC "{$D, \"solver\": {\"solver_type\": \"newton_raphson\", \"n_max_iterations\": 0, \"initial_guess_mode\": \"x_old\"}}" --max-steps 40
run decomp_linear_nopc $PC "{$D, \"solver\": {\"use_predictor_corrector\": false}}" --max-steps 40
run decomp_linear_pc1 $PC "{$D}" --max-steps 40
run decomp_linear_pc10 $PC "{$D, \"solver\": {\"n_corrector_steps\": 10}}" --max-steps 40
echo "=== $(date +%T) ALL DONE"
