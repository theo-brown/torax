#!/bin/bash
# Runs torax/tests/sim_test.py in chunks of separate processes (one process
# cannot hold all 60+ compiled simulations: LLVM section allocation fails).
cd /home/user/torax
PY=/tmp/claude-0/-home-user/9ffdac3c-d367-5e9c-a2e7-3a84fcf08450/scratchpad/venv/bin/python
names=(test_crank_nicolson test_implicit test_bohmgyrobohm_all test_combined_transport test_semiimplicit_convection test_fixed_dt test_psi_and_heat test_psi_heat_dens test_particle_sources_cgm test_prescribed_generic_current_source test_all_transport_fusion_qlknn test_chease test_bremsstrahlung_time_dependent_Zimp test_psichease_prescribed_jtot test_psichease_prescribed_johm test_timedependence test_prescribed_timedependent_ne test_ne_qlknn_defromchie test_ne_qlknn_deff_veff test_iterbaseline_mockup test_iterhybrid_mockup test_iterhybrid_predictor_corrector test_iterhybrid_predictor_corrector_eqdsk test_iterhybrid_predictor_corrector_clip_inputs test_iterhybrid_predictor_corrector_zeffprofile test_iterhybrid_predictor_corrector_timedependent_isotopes test_iterhybrid_predictor_corrector_tungsten test_iterhybrid_predictor_corrector_ec_linliu test_iterhybrid_predictor_corrector_constant_fraction_impurity_radiation test_iterhybrid_predictor_corrector_mavrin_impurity_radiation test_iterhybrid_predictor_corrector_mavrin_n_e_ratios test_iterhybrid_predictor_corrector_mavrin_n_e_ratios_lengyel test_iterhybrid_predictor_corrector_mavrin_n_e_ratios_z_eff test_iterhybrid_predictor_corrector_set_pped_tpedratio_nped test_iterhybrid_predictor_corrector_cyclotron test_iterhybrid_predictor_corrector_neoclassical test_iterhybrid_predictor_corrector_tglfnn_ukaea test_iterhybrid_predictor_corrector_rotation test_iterhybrid_predictor_corrector_Lmode_combined test_iterhybrid_predictor_corrector_tglfnn_ukaea_rotation test_iterhybrid_rampup test_iterhybrid_rampup_sawtooth test_iterhybrid_lh_transition_internal_boundary_condition test_changing_config_before test_changing_config_after test_psichease_ip_parameters_vloop_varying test_psichease_ip_chease_vloop test_psichease_prescribed_jtot_vloop test_implicit_short_optimizer test_iterhybrid_predictor_corrector_imas test_imas_profiles_and_geo test_step_flattop_bgb)
chunk=8
for ((i=0; i<${#names[@]}; i+=chunk)); do
  args=""
  for n in "${names[@]:i:chunk}"; do args="$args SimTest.test_run_simulation_$n"; done
  echo "=== $(date +%T) chunk $i"
  timeout 1800 $PY torax/tests/sim_test.py $args 2>&1 | grep -E "^\[  (FAILED|ERROR)|^(OK|FAILED|Ran )|LLVM|Aborted|Traceback"
done
echo "=== $(date +%T) others"
timeout 1800 $PY torax/tests/sim_test.py SimTest.test_fail SimTest.test_full_output_matches_reference SimTest.test_ip_bc_v_loop_bc_equivalence SimTest.test_low_temperature_error SimTest.test_nans_trigger_error SimTest.test_no_op SimTest.test_prescribed_psidot SimTest.test_simulation_with_restart0 SimTest.test_simulation_with_restart1 SimTest.test_simulation_with_restart2 SimTest.test_simulation_with_restart3 2>&1 | grep -E "^\[  (FAILED|ERROR)|^\[       OK|^(OK|FAILED|Ran )|LLVM|Aborted|Traceback|FATAL"
echo SIMCHUNKS_DONE
