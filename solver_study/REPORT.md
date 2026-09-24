# TORAX solver performance: what PETSc and SUNDIALS do differently, and what would actually pay off

*Study of the TORAX transport solver (branch `claude/torax-solver-optimization-b1p7b8`) against PETSc `main` and SUNDIALS v7.9.0, with measurements on TORAX itself. All numbers: JAX 0.11.2 on a 4-core CPU VM.*

## 0. Shortlist

**Where the time goes.** For the Newton-Raphson solver on QLKNN cases, **the dense forward-mode Jacobian is 94-97% of every Newton iteration**: 26 ms (N = 200) and 116 ms (N = 400) per `jax.jacfwd` versus 0.7-1.0 ms per residual and 0.5-2 ms per dense LU solve. End-to-end a Newton iteration costs 12 / 54 / 169 / 495 ms at N = 100 / 200 / 400 / 800 (4.5x, 3.1x, 2.9x per grid doubling), and steps take 2-4 of them. Compile time exceeds the run time of every example configuration (19-22 s versus 5-8 s of stepping for the Newton examples; 8-12 s versus 0.1-0.2 s for the linear ones). The linear (Picard) solver is 7x cheaper per step at N = 100 and ~30-80x cheaper per inner iteration at N = 400-800, but it does not converge to the implicit solution: at fixed dt = 0.05 s its profiles deviate from the converged Newton solution by 3-11% (0-1 corrector sweeps) and ~1% (10-30 sweeps) after 2 s of simulation, and by 12-31% / 2-9% with the example's own chi-based dt.

**Ranked learnings that translate (evidence in sections 3-4):**

1. **Exploit the Jacobian's structure (PETSc `MatFDColoring`/`DMDA` colouring, CVODE banded DQ Jacobian).** Without transport smoothing the TORAX Jacobian is exactly block-tridiagonal and needs 14 batched JVPs instead of N. With smoothing it needs 86-134 colours because the Gaussian kernel couples 12-23 cells, but the chain rule through the (fixed, linear) smoothing matrix restores an exact assembly with 34 seeds for any grid. **Prototyped and measured (3.9)**: the assembled Jacobian matches `jax.jacfwd` to 1e-14 at every state tested and costs 2.3 / 3.7 / 8.7 / 23 ms instead of 5.1 / 23.5 / 91 / 413 ms at N = 100 / 200 / 400 / 800 (2.2x / 6.3x / 10x / 18x); the full Newton solve goes from 779 to 91 ms at n_rho = 100 with identical iterations. The prototype's compile time is ~4 s longer, not shorter. **Implemented in production as `solver.jacobian_mode = 'structured'` (3.10)**: end-to-end the stepping time of `iterhybrid_rampup` drops 2.8x at n_rho = 50 (7.6 -> 2.7 s) and 4.6x at n_rho = 100 (21.7 -> 4.7 s) with identical Newton iterations and solutions equal to 1e-14, for 5-8 s more compile time. Sources with global state dependences (cyclotron sink, constant-fraction radiation, ToricNN ICRH) are handled exactly through a rank-k chain-rule term over their global scalars (2.5x / 3.6x with all three enabled, at ~15 s more compile). An adversarial stress test (3.12) found a compile blow-up through `torax.run_simulation` (which forced reverting the late-inlining optimisation of 3.11, hence the compile times above), a transport stencil too narrow for rotation, and globally coupled sources without or with the wrong split; all are fixed. **Every configuration is now supported (3.13)**: the ones the mode first rejected - an implicit or `ADAPTIVE_TRANSPORT` pedestal with the L-H transition models, the `MTANH` and `beta_poloidal_prime` internal boundary conditions with an evolving current, non-uniform radial grids and any `min_rho_norm` - get an exact Jacobian by extending the global quantities to the pedestal and the internal boundary conditions and the chain rule to the pedestal's scaling of the transport coefficients, with stencils that hold on any grid.
2. **Jacobian-free Newton-Krylov with the Picard matrix as preconditioner (PETSc `-snes_mf_operator`, CVODE SPGMR + `CVBandPre`).** The frozen-coefficient block-tridiagonal matrix is a poor Newton matrix (eigenvalues of `P^-1 J` up to 17-53) but a good preconditioner: GMRES converges in 15-30 iterations independent of grid and dt. Measured in JAX: **a Newton direction costs 12.7 ms instead of 26.9 ms at N = 200 and 27 ms instead of 118 ms at N = 400** (2.1x / 4.3x; not worth it below N ~ 150), no physics-code changes, unchanged Newton iteration counts. Do not put Pereverzev terms in the preconditioner (2-3x more GMRES iterations).
3. **Stop over-solving (CVODE `crate`/`nlscoef`, ARKODE `nlscoef = 0.1`).** After two Newton iterations the iterate is within 0.2% of the step change and after three within 1e-5, while backward Euler's temporal error at the same dt is 1-10%. Tying the Newton tolerance to the time-discretisation error (WRMS norm, per-channel `rtol/atol`, update-based test) saves 1 of 3-4 iterations (**25-33%**) and replaces the "accept if residual < 1e-2" fallback by SUNDIALS' "refresh Jacobian, then cut dt by 4".
4. **Reuse Jacobians, but with SUNDIALS' safeguards (`msbp/msbj/dgmax`, divergence test).** QLKNN's Jacobian changes 40-70% between predictor and solution even at dt = 0.1 s, so a frozen Jacobian needs 5-8 iterations instead of 3 at dt <= 0.5 s (still 2.5-3x cheaper per step because J = 41 residuals) but 21-29 iterations at dt = 2 s. Recomputing J every 2-3 iterations is the robust version: **2 instead of 3-4 Jacobians per step**. Blind lagging (fixed schedules) is not safe here.
5. **Second-order L-stable stepping (PETSc `TSBDF` order 2 / ARKODE ESDIRK, TR-BDF2).** A BDF2 prototype built from the existing BE solver reaches BE's accuracy at **4-8x larger dt** with the same Newton cost per step. Crank-Nicolson (`theta_implicit = 0.5`, the only second-order option TORAX has) gives 20-45% errors on this stiff problem because it is not L-stable - consistent with SUNDIALS' guidance.
6. **A free local error estimate and a controller (CVODE `tq2 * ||corrector - predictor||`, PETSc Theta LTE, keep-band [1, 1.5)).** TORAX has none; its chi heuristic is an explicit-stability bound that varies 25x within a run and tracks accuracy no better than a fixed dt. Measured caveat: on this dissipative problem error control alone does not cut steps at equal final error (it resolves the initial transient, ~1.5x more Newton work); its value is automatic robust dt selection and enabling 4 and 5.
7. **Cheap hygiene items measured along the way:** `vmap_linesearch` with the default 100 steps makes iterations 1.5x slower; the 10-sweep predictor-corrector initial guess is a net win (saves ~1 Newton iteration per step: 43 vs 61 ms per step at N = 100) even though it lands only 10-25% closer to the solution than `x_old` at dt <= 0.5 s; `jax.lax.custom_root` is free; `jax.jacrev` is only marginally cheaper than `jax.jacfwd`; 35-60 ms per step (50-80 residual evaluations) is spent outside the nonlinear solve.

**Grid-size dependence (section 7).** At n_rho = 100 (N = 400) the ranking above holds and is sharpened by two measurements: inside the Newton loop the dense Jacobian runs ~40% slower than standalone (XLA CPU parallelises the batched JVP poorly once it is fused with the solve and residual), so a loop iteration costs ~170 ms; and the structured-Jacobian prototype cuts the Newton solve at this size 8.6x (779 -> 91 ms). At n_rho = 25 (N = 100) the Jacobian is only ~40% of a 49 ms step, JFNK is slower than the dense Jacobian, and compile time, per-step orchestration overhead and fewer Newton iterations/steps take over.

**What does not translate:** banded/block LU instead of dense LU (2% of an iteration at N <= 800); polynomial line searches (backtracking occurs in 3% of iterations); Anderson acceleration of the Picard mode (the Pereverzev map is not contractive; no gain measured); super-time-stepping/ExtSTS and multirate (the stiffness is the diffusion itself); finite-difference Jacobian machinery (JAX JVPs are exact); Crank-Nicolson.

**One robustness finding:** at dt = 0.02 s every Newton variant, including TORAX's, stalls at a residual of 3e-4 to 5e-4 (TORAX accepts it via the coarse tolerance). A Taylor test shows the residual is *discontinuous* there, not ill-conditioned, and the cause is confirmed: the QLKNN `DV_effective` sign switch between pure-D and pure-V particle-transport representations. With it disabled the same solves converge quadratically in 2 iterations. Smoothing or freezing that switch within a Newton solve (and an update-based convergence test as in SUNDIALS/IDA or PETSc `stol`) removes the stalls and the coarse-tolerance acceptances.

## 1. What was done

1. **Code study** of the TORAX solver stack (`torax/_src/solver`, `fvm`, `orchestration`, `time_step_calculator`, `transport_model`) and of the corresponding machinery in PETSc `main` (SNES line searches and convergence tests, Jacobian lagging, `MatMFFD`, `MatFDColoring`/`DMDA` colouring, KSP/PC defaults, `TSAdapt`, `TSBDF`, `TSTHETA`, `TSARKIMEX`, `TSROSW`, `TSPSEUDO`, `TSEvent`, adjoints) and SUNDIALS v7.9.0 (CVODE/CVODES BDF machinery and modified Newton, banded DQ Jacobians, SPGMR tolerances, `SUNNonlinSol_FixedPoint` Anderson acceleration, KINSOL line search, IDA DAE handling, ARKODE DIRK/IMEX tables, predictors, controllers, LSRKStep/ExtSTS, MRIStep, constraints, root-finding). Two subagents produced citation-level reports (`report_petsc.md`, `report_sundials.md`); a third mapped the TORAX residual's coupling structure and non-smooth operations.
2. **Measurements on TORAX itself** (JAX 0.11.2, CPU, 4 cores, Python 3.12 venv), all scripts under `bench/` and `profile_runs/`:
   - end-to-end per-step profiles of the example configurations and solver/grid variants (`profile_runs/run_one.py`);
   - component micro-timings of one Newton iteration at n_rho = 25/50/100 (`bench/timings.py`);
   - dense Jacobian structure, colouring counts, preconditioner quality, GMRES iteration counts, lag/truncation contraction factors (`bench/jac_structure.py`);
   - Newton-variant iteration counts with the real residual: full/lagged/chord/Newton-Picard/band-truncated/JFNK, initial-guess quality, over-solving (`bench/newton_variants.py`);
   - temporal accuracy of BE vs Crank-Nicolson vs a BDF2 prototype against a fine-dt reference, and chi-heuristic vs fixed vs error-controlled step selection (`bench/time_accuracy.py`, `bench/adaptive_dt.py`);
   - Picard vs Anderson-accelerated Picard for the linear solver (`bench/picard_anderson.py`);
   - a diagnostic of the Newton stall at small dt (`bench/stall_diag.py`).
3. **Translation assessment** of each library design choice against TORAX's specifics: N = 100-800 unknowns, four channels, QLKNN critical-gradient stiffness, transport smoothing (non-local coupling), row-replacement (algebraic) rows, JAX fixed-shape loops, CPU execution, compile time.

Caveats. Measurements are from a 4-core cloud VM; absolute times will differ elsewhere but the ratios (Jacobian vs residual, batched-JVP scaling, iteration counts) are the point. A methodological lesson worth recording: XLA's CPU thread pool spin-waits on all cores, so an unrelated single-threaded job pinned to one core inflated TORAX step times by 4-8x (0.75 s instead of 0.15 s per ramp-up step) and produced a spurious "state-dependent cost" pattern; every end-to-end and micro-benchmark number in this report was therefore re-measured with nothing else running (`results/profile_runs/*_idle.json`, `results/logs/timings_idle_*.log`). Iteration counts, Jacobian structure and accuracy results are unaffected by contention.

## 2. What the TORAX solver does today

All paths are relative to `/home/user/torax/torax/_src` (branch `claude/torax-solver-optimization-b1p7b8`, HEAD `f6cdec4`).

**Discretisation.** Four coupled 1-D parabolic equations (`T_i`, `T_e`, `n_e`, `psi`) on `n_rho` finite-volume cells (25-200 in the examples), unknown vector `x` of size `N = 4 n_rho`, channel-major ordering (`fvm/fvm_conversions.py:25`), scaled to O(1) (`core_profiles/convertors.py:42`). The residual is scaled by the transient coefficient so it is also O(1) in `x` units (`fvm/residual_and_loss.py:59-115`).

**Time stepping.** Theta method, `theta_implicit = 1` (backward Euler) by default (`solver/pydantic_model.py:57`). One implicit solve per step, no local error estimate. `dt` comes from `time_step_calculator`: either fixed (`fixed_dt`), or the chi heuristic `dt = prefactor * 0.75 * drho_min^2 / chi_max` capped by `max_dt` (`time_step_calculator/chi_time_step_calculator.py:44-52`). That heuristic is an explicit-stability (CFL-type) bound; it carries no accuracy information for an L-stable implicit method. On Newton failure the step is retried with `dt / dt_reduction_factor` (default 3) inside `orchestration/adaptive_step.py:83`, down to `min_dt`; there is no step growth logic beyond the heuristic, and an unconverged solve with residual below `residual_coarse_tol = 1e-2` is *accepted* with a warning (`solver/jax_root_finding.py:168-186`).

**Nonlinear solve (`newton_raphson`).** `fvm/newton_raphson_solve_block.py` builds the initial guess with the *linear* predictor-corrector solver (10 Picard sweeps with Pereverzev-Corrigan terms, `n_corrector_steps = 10`), then calls `solver/jax_root_finding.py:root_newton_raphson`:
- every iteration recomputes the **dense** Jacobian with `jax.jacfwd` of the full physics residual (`jax_root_finding.py:118`, `:262`), i.e. `N` batched JVPs through transport model (QLKNN), sources, neoclassical conductivity, ion updates, and the FVM assembly;
- solves the `N x N` dense system with `jnp.linalg.solve` (`:265`);
- backtracks geometrically (`step *= 0.5`) until the mean-absolute residual satisfies a weak Armijo condition (`sufficient_decrease = 1e-4`, `solver/linesearch.py`), up to `max_linesearch_steps = 100`;
- stops when `mean(|R|) <= residual_tol = 1e-5`, or after `n_max_iterations = 30`, or when the accepted step fraction drops below `tau_min = 0.01`;
- wraps everything in `jax.lax.custom_root` for implicit-function differentiation.

**Linear solve (`linear`, the default solver).** Picard / predictor-corrector: coefficients frozen at the previous iterate, Pereverzev-Corrigan artificial diffusion + compensating convection to stabilise stiff transport, block-tridiagonal Thomas solve (`tridiagonal.py:thomas_solve`, `O(n_rho C^3)`), fixed number of sweeps (`solver/predictor_corrector_method.py`, `solver/jax_fixed_point.py`). No acceleration.

**Residual scaling and norm.** The residual is in solver-state units (T in keV, n_e in 1e20 m^-3, psi in Wb), and both the convergence test and the line search use the mean absolute value over all `N` entries (`solver/jax_root_finding.py:33-35`). This is not scale-consistent across channels (psi increments per step are orders of magnitude smaller than T increments) and dilutes localised residuals; a per-channel scaling helper exists (`core_profiles/convertors.py:161-220`, `compute_residual_scaling_vector`) but is never called. The `custom_jac` hook of `root_newton_raphson` is rejected whenever `use_jax_custom_root=True` (the default).

**What one residual evaluation computes** (from the structure map in `reports/report_torax_structure.md`): ion/charge-state update (Mavrin fits), q and shear from psi, Sauter conductivity, *all configured sources* (every source is implicit by default, `sources/base.py:53`), bootstrap current, the pedestal model when not explicit, the transport models (QLKNN_7_11: one MLP with 5 tanh layers of width 133, 74k parameters, evaluated on all faces at once), clipping, Gaussian smoothing as a dense face-by-face matmul, Pereverzev terms only in the linear path, then the block-tridiagonal assembly. The kernel is `2^-(drho/sigma)^2` with HWHM `sigma = smoothing_width`, truncated where the row-normalised weight is below 0.01, which gives a half-width of 8-10 faces at n_rho = 50 and 14-17 at n_rho = 100 for `sigma = 0.1`.

**JAX constraints that matter for the comparison.** All loops are `lax.while_loop`s with fixed-shape carries; the whole step (`orchestration/step_function.py:SimulationStepFn.__call__`) is one `jax.jit`; the first call compiles the entire step including the Jacobian graph. Anything that PETSc/SUNDIALS keep as mutable solver state (lag counters, factorisations, step history) has to be carried explicitly through the loop state.

## 3. Measured bottlenecks and evidence

### 3.1 End-to-end profiles (all four cores, JAX 0.11.2 CPU, Python 3.12)

Per-step wall time measured around `step_fn(state, ppo)` with `block_until_ready`; the first step is the JIT compile. Scripts: `profile_runs/run_one.py`, `profile_runs/driver.sh`. All numbers below are from the idle-machine runs (`results/profile_runs/*_idle.json`).

| config | solver | n_rho | steps | compile (first step) | run (excl. compile) | median step | Newton/Picard iterations per step | ms per inner iteration |
|---|---|---|---|---|---|---|---|---|
| `iterhybrid_rampup` (QLKNN, fixed dt = 2 s, t_final = 80 s) | newton_raphson | 50 | 40 | 19.0 s | 6.1 s | 146 ms | 2.7 (max 5) | 58 |
| same, `log_iterations=True` | newton_raphson | 50 | 40 | 20.3 s | 7.6 s | 177 ms | 2.7 | 72 |
| `iterhybrid_predictor_corrector` (QLKNN, chi dt, t_final = 5 s) | linear | 25 | 29 | 11.8 s | 0.17 s | 6.0 ms | 2 | 3.1 |
| `basic_config` | linear | 25 | 21 | 7.5 s | 0.11 s | 5.4 ms | 1 | 5.5 |
| `step_flattop_bgb` (Bohm-gyroBohm, fixed dt = 10 s) | newton_raphson | 100 | 40 | 21.6 s | 4.7 s | 106 ms | 1.5 (max 4) | 82 |
| `benchmarks/..._fixed_small_dt` (dt = 1 ms) | linear | 25 | 5000 | 12.6 s* | 32.3 s* | 5.8 ms* | 2 | 3.2* |

Observations.
- **Compile time dominates every example**: 19-22 s of compile versus 5-8 s of stepping for the Newton cases, 8-12 s versus 0.1-0.2 s for the linear cases (the 5000-step small-dt benchmark, marked *, was not re-run and is the only case where stepping exceeds compile). The Newton path produces a 20-22 MB HLO module, the linear path 10 MB (section 3.7).
- **A Newton iteration costs 12 / 54 / 169 / 495 ms end-to-end at N = 100 / 200 / 400 / 800, a Picard sweep 3.5-6 ms at any N** (3.1c). On the same physics the Newton solver is 7x more expensive per step than the linear solver at N = 100 (43 vs 6 ms, 3.1b) and the gap widens with N (54 vs 4 ms per inner iteration at N = 200, 495 vs 6 at N = 800); the gap is the Jacobian (section 3.2).
- **Line-search backtracking is rare**: in the 109 Newton iterations of `iterhybrid_rampup` only 3 had `tau < 1` (11 extra residual evaluations in total). PETSc's cubic `bt` line search would therefore change almost nothing here.
- **Newton converges quadratically and is over-solved**: typical residual histories are `6e-1, 3e-2, 6e-4, 2e-5, 2e-10` (dt = 2 s); the last iteration takes the residual from just above `1e-5` to `1e-10`.
- `vmap_linesearch=True` with the default `max_linesearch_steps=100` evaluates 100 trial residuals per Newton iteration: 76 vs 49 ms per iteration (1.5x) and +2.5 s compile (`compile_vmapT_*` rows, 3.7). It should not be used with the default step count.

#### 3.1b Same physics, four solver settings (`iterhybrid_predictor_corrector`, n_rho = 25, t_final = 2 s, chi dt)

| setting | steps | compile | run | median step | inner iterations total | ms per inner iteration | dt backtracks |
|---|---|---|---|---|---|---|---|
| linear (PC, 1 corrector) | 7 | 12.3 s | 0.04 s | 6.2 ms | 14 | 3.4 | 0 |
| linear (PC, 10 correctors) | 8 | 13.0 s | 0.07 s | 10.5 ms | 88 | 1.0 | 0 |
| newton, linear initial guess | 10 | 19.0 s | 0.40 s | 43 ms | 40 | 11.7 | 0 |
| newton, `initial_guess_mode = x_old` | 10 | 16.9 s | 0.52 s | 61 ms | 53 | 11.3 | 0 |
| newton, no predictor-corrector guess | 11 | 18.0 s | 0.56 s | 55 ms | 57 | 11.3 | 1 |

The 10-sweep linear initial guess saves 13 Newton iterations over 10 steps (40 vs 53) for 110 Picard sweeps and is a net win end-to-end (43 vs 61 ms per step, 30% faster) at the cost of ~2 s of compile; without any predictor-corrector guess one step also needed a dt backtrack.

#### 3.1c Grid scaling (`iterhybrid_predictor_corrector` physics, t_final = 1 s, chi dt)

| n_rho | N | linear: ms/sweep | newton: ms/iteration | newton: ms/step (iterations/step) | newton compile |
|---|---|---|---|---|---|
| 25 | 100 | 3.5 | 12 | 62 (5.2) | 19 s |
| 50 | 200 | 4.2 | 54 | 174 (3.5) | 20 s |
| 100 | 400 | 4.1 | 169 | 359 (2.6) | 21 s |
| 200 | 800 | 5.8 | 495 | 922 (1.8) | 23 s |

The linear solver's cost per sweep is nearly flat in N (dominated by fixed per-call overhead), while the Newton iteration cost grows 4.5x, 3.1x and 2.9x per doubling of N (the chi heuristic shrinks dt as dx^2, so the n_rho = 200 run only reached t = 0.18 s in 60 steps). Section 3.2 decomposes the Newton per-iteration cost.

### 3.2 Where a Newton iteration's time goes (`bench/timings.py`, best of 5-10 after warm-up)

Measured on the idle machine (`results/logs/timings_idle_n*.log`); the earlier 3-core measurement (`results/bench/timings_n*.json`) agrees within 10% for every row except the reverse-mode Jacobian.

State: `iterhybrid_rampup` at t = 6 s, dt = 2 s (the Newton solve takes 4 iterations). All functions jitted; times exclude compilation.

| component (N = 200, n_rho = 50) | ms | relative to one residual |
|---|---|---|
| residual `theta_method_block_residual` | 0.72 | 1 |
| of which: `calc_coeffs` (transport 0.54, sources 0.18, conductivity 0.07) | 0.94* | - |
| `update_core_profiles_during_step` (ions, q, s) | 0.25 | 0.4 |
| single JVP `jax.jvp` | 1.04 | 1.4 |
| batched JVP, k = 16 / 32 / 64 / 128 columns | 2.7 / 3.7 / 6.8 / 12.3 | 3.8 / 5.2 / 9.5 / 17 |
| **dense Jacobian `jax.jacfwd` (200 columns)** | **26.4** | **36** |
| dense Jacobian `jax.jacrev` | 20.7 | 29 |
| dense LU factor / solve (`jnp.linalg.solve`) | 0.42 / 0.50 | 0.7 |
| block-tridiagonal Thomas solve (`tridiagonal.thomas_solve`) | 0.21 | 0.3 |
| Picard matrix build + LU | 0.73 + 0.38 | 1.7 |
| one JFNK Newton direction: `jax.scipy.sparse.linalg.gmres` (restart 40, tol 1e-4) on exact JVPs, M = Picard LU | 12.7 | 18 |
| predictor-corrector initial guess (11 sweeps) | 9.3 | 13 |
| full Newton solve `root_newton_raphson` (4 iterations), with / without `custom_root` | 177 / 189 | - |
| full `SimulationStepFn` step (pre-step, guess, Newton, finalize, post-processing) | 252 | - |

\* `calc_coeffs` timed standalone includes work the jitted residual shares/fuses; treat as an upper bound.

**Scaling with grid size** (same state, dt = 2 s; `timings_n25/n50/n100`):

| N (n_rho) | residual | single JVP | batched JVP k = 16 / 32 / 64 | `jacfwd` (N columns) | `jacrev` | dense solve | JFNK direction (GMRES + Picard LU) | Newton solve (4-5 it.) | full step | overhead outside solve |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 (25) | 0.54 ms | 0.70 | 2.3 / 2.5 / 3.8 | **5.4** | 7.7 | 0.14 | 8.9 | 23 (4 it.) | 58 | 35 ms |
| 200 (50) | 0.72 | 1.04 | 2.7 / 3.7 / 6.8 | **26.4** | 20.7 | 0.50 | 12.7 | 189 (4 it.) | 252 | 63 ms |
| 400 (100) | 0.99 | 1.76 | 4.2 / 6.5 / 14.0 | **116.5** | 109.8 | 1.92 | 27.1 | 845 (5 it.) | 864 | 19 ms |

The dense Jacobian grows 4.4-4.9x per doubling of N (N columns times an N-proportional residual, plus the smoothing matmul), whereas the JFNK direction grows 1.4-2.1x per doubling and a fixed-colour-count batched JVP ~1.5-2x. Crossover for JFNK is N ~ 150; a 36-column structured Jacobian (between the k = 32 and k = 64 rows) wins at every size: ~3 / ~5 / ~9 ms versus 5.4 / 26 / 116 ms. End-to-end (section 3.1c) a Newton iteration costs 12 / 54 / 169 / 495 ms at N = 100 / 200 / 400 / 800, i.e. the Jacobian plus 10-30% of line-search and bookkeeping.

Findings.
1. **The Jacobian is 94% of a Newton iteration** (26.4 ms of ~28 ms at N = 200; 116 of ~120 ms at N = 400); the dense LU is 2% and a residual evaluation 2-3%. Dense linear algebra is irrelevant at N <= 800; the PETSc/SUNDIALS "banded LU instead of dense LU" lesson does *not* translate at this size (it would at N ~ 2000+).
2. **Batched JVPs are strongly sublinear in the number of columns**: 16 columns cost 2.6x one JVP, 36 columns would cost ~4-5 ms. A 36-colour structured Jacobian (3.3, item 3) therefore cuts the Jacobian cost ~5x at n_rho = 50 (26 -> ~5 ms) and the whole Newton iteration ~4x; at n_rho = 100 the saving is ~12x because the colour count is grid-independent while `jacfwd` columns double.
3. **`jax.jacrev` is only marginally cheaper than `jax.jacfwd`** on the idle machine (20.7 vs 26.4 ms at N = 200, 110 vs 116 ms at N = 400, and slower at N = 100); the 1.45x seen in the first measurement did not survive re-measurement. Not worth pursuing on its own.
4. **JFNK (GMRES on exact JVPs with the Picard preconditioner) halves the cost of a Newton direction at N = 200 (12.7 vs 26.9 ms) and cuts it 4.3x at N = 400 (27 vs 118 ms)** with no change to the residual code; its GMRES iteration count is grid-independent (3.3). The direction is accurate to 9e-4 at tol = 1e-4, and the Newton iteration count is unchanged (3.4). At N = 100 it is slower than the dense Jacobian (8.9 vs 5.4 ms).
5. **`jax.lax.custom_root` costs nothing measurable** at runtime or compile time (177 vs 189 ms; 8.8 vs 8.8 s compile).
6. **Inside the Newton loop the Jacobian runs ~40% slower than standalone at N = 400** (`bench/inloop_n100.py`, `bench/jac_threads_n100.py`): one loop-body iteration costs 167-170 ms while `jacfwd` (91-99 ms) + dense solve (2.3 ms) + residual (1 ms) sum to ~103 ms; the line search is not the cause (166.7 ms without it). Single-threaded, the standalone and fused versions cost the same (314 ms), so the loss is XLA CPU intra-op parallelism: the batched JVP gets a 3.2x thread speed-up on its own but only 1.9x when fused into the loop body with the solve and residual. At N = 100 the effect is negligible (5.9 vs 5.3 ms per iteration). Any change that shrinks the Jacobian batch (colouring) or removes it (JFNK) also removes this loss; alternatively XLA CPU scheduling flags (the environment here runs with `--xla_cpu_opt_preset=FAST_COMPILE`) deserve a look.
7. **Tens of milliseconds per step are spent outside the nonlinear solve**: at N = 200 the full step is 252 ms of which the Newton solve is 189 ms and the predictor-corrector guess 9 ms; pre-step (1.7-2.5 ms), `finalize_outputs` (2.5 ms) and the remaining orchestration take the rest (~35-60 ms at N = 100-200, equivalent to 50-80 residual evaluations; `bench/phase_cost.py`). This overhead is fixed per step and becomes the floor once the Newton solve is made cheap; for the linear solver it is comparable to the 11 Picard sweeps.

### 3.3 Jacobian structure: how sparse is it really? (`bench/jac_structure.py`)

Dense `jacfwd` Jacobians of the real residual at a mid-simulation state of `iterhybrid_rampup` (t = 6 s), at the linear initial guess `x0` and at the converged `x*`. "Colours" = greedy distance-2 column colouring of the structural pattern, i.e. the number of batched JVPs an exact compressed Jacobian would need (PETSc `MatFDColoring`, CVODE `cvLsBandDQJac`).

| case | N | nnz fraction | max cell-offset with |J|>1e-6 (T/n rows) | colours needed | ||J||_F within 1-cell band |
|---|---|---|---|---|---|
| n_rho = 50, smoothing on (`smoothing_width = 0.1`), dt = 2 | 200 | 0.18 | 11-12 cells | **86** | 86% |
| n_rho = 100, smoothing on, dt = 2 | 400 | - | ~23 cells | **134** | 72% |
| n_rho = 50, smoothing **off** | 200 | 0.055 | 1 cell | **14** | 99.8% |
| psi rows (any case) | - | - | 1 cell | - | - |

Findings.
1. **Without smoothing the Jacobian is exactly block-tridiagonal** (stencil +-1 cell, all four channels), and 14 colours reproduce it exactly: 14 batched JVPs instead of 200 (n_rho = 50) or 400 (n_rho = 100). This is the PETSc/CVODE banded-Jacobian regime (`DMDA` 1-D colouring gives `nc*(2s+1)` = 12 colours for 4 dof, stencil 1; CVODE's banded DQ Jacobian needs `mupper+mlower+1` = 15 evaluations).
2. **The Gaussian smoothing of transport coefficients** (`transport_model/transport_model.py:_build_smoothing_matrix`, dense `jnp.dot` with a kernel truncated at weight 0.01) **couples every T/n row to ~12 cells (n_rho = 50) or ~23 cells (n_rho = 100) on each side** inside the smoothing zone (rho 0.3-0.9). Consequently exact colouring saves only 2.3-3x, and the saving does not improve with resolution because the kernel width is fixed in rho. The off-band part carries 14-28% of the Frobenius norm; **truncating it is not an option** (see 3.4: band-truncated Newton matrices diverge or converge with contraction factors 0.8-2).
3. The exact remedy is structural, not numerical: the residual is `R(x) = G(x, S h(x))` with `S` the fixed smoothing matrix, `h` the raw (local) transport coefficients and `G` local. Chain rule gives `J = dG/dx + (dG/dc) S (dh/dx)`, where `dG/dx` (14 colours), `dh/dx` (local, ~14 colours) and `dG/dc` (linear in `c`, ~8 colours) are all narrow-banded. **~36 batched JVPs then reproduce the exact Jacobian for any n_rho**, versus N = 4 n_rho today (5.5x fewer at n_rho = 50, 11x at n_rho = 100, 22x at n_rho = 200). This needs the transport-model call to expose the pre-smoothing coefficients; it is the TORAX analogue of PETSc's advice to keep global couplings as a separate low-rank/structured term rather than breaking the colouring.
4. **Picard matrix vs Jacobian.** The block-tridiagonal matrix the linear solver factorises (`I - scale * C(x)`, transport and sources frozen) differs from the true Jacobian by 83-97% in Frobenius norm at every dt tested; the eigenvalues of `P^-1 J` extend to 17-53, i.e. the frozen-coefficient operator underestimates the stiff response of QLKNN by more than an order of magnitude. That is the quantitative reason the Picard iteration needs Pereverzev-Corrigan stabilisation and never converges to the nonlinear solution (3.4), and why PETSc's `SNESSetPicard` documentation recommends using `A(x)` only as a *preconditioner* for Newton.
5. **As a preconditioner it is good.** GMRES on the exact Jacobian preconditioned by the Picard matrix converges in 15-32 iterations to 1e-2..1e-8 relative residual, *independently of n_rho (50 vs 100) and of dt (0.02-2 s)*, versus 65-450 iterations with Jacobi scaling. Adding the Pereverzev terms to the preconditioner makes it worse (2-3x more GMRES iterations, 3.4).

| n_rho, dt | cond(J) | cond(P) | rho(I - P^-1 J) | GMRES its (M = P^-1) to 1e-2 / 1e-4 / 1e-6 | chord factor rho(I - J(x0)^-1 J(x*)) |
|---|---|---|---|---|---|
| 50, 2.0 | 6.6e7 | 6.9e4 | 18-21 | 16-28 / 22-24 / 28-31 | 0.74 |
| 100, 2.0 | 2.5e8 | 4.0e5 | 15-16 | 17-19 / 27-28 / 31-34 | 2.9 |
| 50, 0.5 | 8e6 | 1.2e4 | 6-11 | 17-22 / 22-23 / 26-28 | 0.37 |
| 50, 0.1 | 1.7e6 | 1.5e3 | 2.7-4.0 | 15-16 / 19-22 / 24-26 | 0.59 |
| 50, 0.02 | 3.4e5 | 2.2e2 | 1.1-1.3 | 7-11 / 14-22 / 17-20 | 23 (stalled solve, see 3.4) |

`rho(I - A^-1 J)` is the asymptotic contraction factor of a Newton iteration that uses `A` instead of `J`; `> 1` means divergence. The Jacobian changes substantially between the predictor and the converged solution (chord factors 0.4-0.7 even at dt = 0.1 s), which is why Jacobian lagging is only a partial win here (3.4).

### 3.4 Nonlinear solver variants: iteration counts (`bench/newton_variants.py`)

Same state (`iterhybrid_rampup`, t = 6 s, n_rho = 50, N = 200), Python re-implementation of TORAX's Newton loop (same line search, norm, tolerance 1e-5), so that the Newton matrix can be swapped. "J" = one dense `jacfwd` (N batched JVPs); "R" = one residual evaluation; "P" = one coefficient evaluation + block-tridiagonal assembly (about one R).

| variant | dt = 2.0 | dt = 0.5 | dt = 0.1 | dt = 0.02 |
|---|---|---|---|---|
| TORAX Newton (linear PC guess) | 4 it, 4 J + 5 R | 3 it, 3 J + 4 R | 3 it, 3 J + 4 R | stalls at 4.7e-4 (4 J + 17 R) |
| Newton from `x_old` | 7 it, 7 J + 15 R | 5 it, 5 J + 7 R | 4 it, 4 J + 6 R | stalls (6 J + 21 R) |
| Newton from linear extrapolation `x_n + (dt/dt_prev)(x_n - x_{n-1})` | 5 it | 3 it | 6 it (13 R) | stalls |
| Jacobian every 2 iterations | 5 it, 3 J + 6 R | 3 it, 2 J + 4 R | 5 it, 3 J + 6 R | stalls |
| Jacobian every 3 iterations | 5 it, 2 J + 6 R | 4 it, 2 J + 5 R | 5 it, 2 J + 6 R | stalls |
| chord: J(x0) frozen | 21 it, 1 J + 22 R | 5 it, 1 J + 6 R | 6 it, 1 J + 7 R | stalls |
| chord: J(x_old) frozen (= reuse from previous step end) | 29 it, 1 J + 84 R | 7 it, 1 J + 8 R | 8 it, 1 J + 9 R | stalls |
| Newton with Picard matrix P(x) | fails (line search stalls) | fails | fails | fails |
| Newton with Pereverzev-augmented P(x) | 100 it, not converged (1.8e-2) | 100 it (3e-3) | 100 it (8e-4) | stalls |
| band-truncated J (k = 2 / 4 / 8 cells) | diverge / diverge / 60 it | 60 / 8 fail / 46 it | 28 / fail / 23 it | stalls |
| JFNK: GMRES on exact JVPs, M = P^-1, Eisenstat-Walker forcing | 4 it, 99 JVP + 5 R + 4 P | 4 it, 87 JVP + 5 R + 4 P | 3 it, 55 JVP + 4 R + 3 P | stalls |
| JFNK with Pereverzev-augmented P | 4 it, 304 JVP | 3 it, 142 JVP | 4 it, 227 JVP | stalls |

Quality of the initial guess, `|x_guess - x*| / |x* - x_old|`: linear PC guess 0.30 / 0.75 / 0.90 / 1.9 (dt = 2 / 0.5 / 0.1 / 0.02); extrapolation 1.56 / 1.38 / 1.04 / 0.88. **The 10-sweep predictor-corrector guess is only 25% closer than `x_old` at dt = 0.5 s and 10% closer at dt = 0.1 s**, at the cost of 11 coefficient evaluations and Thomas solves; it saves one Newton iteration at most.

Distance to the converged solution after each TORAX Newton iteration at dt = 2 s (relative to the step change `|x* - x_old|`): 0.30, 8.9e-3, 2.3e-3, 3.6e-6, 0. **After two iterations the solution is within 0.2% of the step change; the third and fourth iterations buy 1e-3 and 1e-6 relative accuracy** while the backward-Euler temporal error at this dt is O(10%) (3.5). The Newton tolerance is not tied to the time-discretisation error, which is exactly what CVODE's `nlscoef = 0.1 * error tolerance` and ARKODE's `nlscoef` do.

Reading of the table.
- **Modified/lagged Newton works at dt <= 0.5 s but not at dt = 2 s**: a frozen Jacobian needs 5-8 iterations instead of 3 (1 J + 6-9 R versus 3 J + 4 R). Whether that is a win depends on the J/R cost ratio measured in 3.2: with J ~ 50-100 R (measured at N = 200) the chord variant is 2.5-3x cheaper per step. At dt = 2 s (rampup with fixed_dt = 2) the chord method needs 21-29 iterations and many backtracks. CVODE-style lagging (recompute J only after a convergence failure or every 20-50 steps) therefore needs the `crate`/divergence safeguards, not a fixed schedule.
- **Jacobian every 2-3 iterations** is the robust middle ground: 2 J instead of 3-4 J per step at every dt >= 0.1 s, with at most one extra residual evaluation (30-50% fewer Jacobians).
- **JFNK with the Picard preconditioner** keeps the Newton iteration count (3-4) with 18-25 JVPs per Newton iteration (55-99 per step) instead of 200-column `jacfwd`s; the crossover depends on the batched-JVP efficiency measured in 3.2.
- **Nothing based on the frozen-coefficient matrix alone converges** (Picard-Newton fails; Pereverzev-Newton converges linearly with factor ~0.95). Truncating the Jacobian band is not viable while smoothing is on.
- **dt = 0.02 s stall**: every variant, including TORAX's own solver (error state 2, "converged within coarse tolerance"), stalls at a residual of 3e-4 to 5e-4 with the line search cutting the step to `tau < 0.01`. The diagnostic (`bench/stall_diag.py`) shows why: a Taylor test along the Newton direction `d` gives a linearisation error `|R(x + s d) - R(x) - s J d| / |R(x + s d) - R(x)|` of 1.5-1.7 already at `s = 1e-3` (and ~1 for all larger `s`) at dt = 0.02 and 0.05 s, versus 4e-3 and decreasing at dt = 0.2 s. The residual change is therefore not a derivative effect but a **jump: the residual is discontinuous at the iterate**, localised in cells 41-43 (rho = 0.82-0.86, the outer QLKNN zone just inside the pedestal), strongest in the `n_e` rows (|R| up to 1e-2 against a mean of 5e-4) and the T rows. The smallest singular values of J (1e-3..1e-2, cond 1.4e6) are not the cause (the direction is fine; the residual jumps). The structure map (section 2, `report_torax_structure.md`) lists the state-dependent `jnp.where` switches on the implicit path; the candidates in this region are the `DV_effective` representation switch (sign of the particle flux / `|Ane| < An_min`), `avoid_big_negative_s`, the `q_sawtooth_proxy` clamp, and the QLKNN flux clip at the critical gradient. **Confirmed cause** (`bench/stall_variants.py`): with `DV_effective = False` in the QLKNN config the same solves converge quadratically in 2 iterations with no backtracking (residual 4e-3 -> 2.7e-5 -> 7e-10 at dt = 0.02 s) and the Taylor test is clean (linearisation error 9e-2, 9e-3, 9e-4 at s = 1e-4, 1e-3, 1e-2); disabling `avoid_big_negative_s` changes nothing (6 iterations, still stalled). The culprit is the `DV_effective` representation switch (`transport_model/quasilinear_transport_model.py:463-482`): a `jnp.where` on the signs of the particle flux and of the density gradient (and `|Ane| >= An_min`) that flips between a pure-diffusion and a pure-convection representation of the same flux. In the flat-density region near the pedestal top the sign flips between Newton iterates, so the residual jumps. Consequences: (i) with a mean-abs residual test the solver cannot distinguish "converged up to a jump" from "not converged", and TORAX's coarse-tolerance acceptance is what rescues it; (ii) the SUNDIALS/IDA-style test on the *update* norm (the iterate moved by only 2% of the step change in the last iteration) or the PETSc `stol` exit would terminate cleanly; (iii) at larger dt the step change dwarfs the jump, which is why the same physics converges quadratically at dt >= 0.1 s; (iv) smoothing the switch (a `smoothstep` blend instead of `jnp.where`) removes the problem at the source.

### 3.5 Time integration: order, error control and step selection (`bench/time_accuracy.py`, `bench/adaptive_dt.py`)

Physics of `iterhybrid_predictor_corrector` (QLKNN, n_rho = 25), Newton solver with `residual_tol = 1e-7` so that the error is purely temporal, t_final = 1 s, reference = backward Euler with dt = 1.56 ms (640 steps; its own error, estimated from a 2x-dt run, is 2e-5 to 1.3e-4). Error = max over cells of |x - x_ref| / max|x_ref|.

| dt | steps | BE (theta = 1): max rel. error T_i / T_e / n_e / psi | Crank-Nicolson (theta = 0.5) | BDF2 prototype |
|---|---|---|---|---|
| 0.2 | 5 | 8.9e-3 / 6.2e-3 / 6.2e-3 / 1.2e-2 | 0.45 / 0.46 / 0.19 / 0.17 | 5.1e-3 / 5.8e-3 / 2.7e-3 / 4.7e-3 |
| 0.1 | 10 | 4.3e-3 / 2.8e-3 / 3.9e-3 / 5.8e-3 | 0.24 / 0.23 / 0.09 / 0.05 | 1.3e-3 / 6.9e-4 / 1.1e-3 / 5.0e-4 |
| 0.05 | 20 | 1.9e-3 / 1.3e-3 / 1.8e-3 / 2.6e-3 | 0.14 / 0.14 / 0.05 / 0.03 | 3.9e-4 / 1.2e-4 / 1.0e-3 / 1.9e-4 |
| 0.025 | 40 | 7.8e-4 / 5.1e-4 / 1.1e-3 / 1.1e-3 | 0.08 / 0.07 / 0.03 / 0.02 | 5.1e-4 / 3.1e-4 / 6.5e-4 / 5.8e-5 |
| 0.0125 | 80 | 3.2e-4 / 1.7e-4 / 5.2e-4 / 4.2e-4 | 0.04 / 0.04 / 0.01 / 0.01 | 3.9e-4 / 4.0e-4 / 1.9e-4 / 5.2e-5 |

Newton iterations per step: 5.0, 3.5, 2.8, 2.5, 2.1 (BE) - the per-step cost falls with dt, so halving dt costs less than 2x.

Findings.
- **Backward Euler is cleanly first order** (error halves with dt). At the dt values the chi heuristic picks for this case (median 0.12 s) the temporal error is ~0.5%.
- **Crank-Nicolson is unusable for this problem**: 20-45% errors at dt = 0.2 s and still 4% at dt = 12.5 ms. It is A-stable but not L-stable, and the stiff QLKNN modes (eigenvalues of `P^-1 J` up to 50) produce the classic non-decaying oscillations. This confirms the SUNDIALS guidance ("cap BDF at order 2 or use an L-stable DIRK table" for parabolic problems) and rules out `theta_implicit = 0.5` as a cheap second-order option.
- **BDF2 (L-stable) gives the accuracy of BE at 4-8x smaller dt**: BDF2 at dt = 0.1 s (1.3e-3) is between BE at 0.025 and 0.0125; BDF2 at dt = 0.2 s matches BE at 0.1. The prototype (built from the existing BE solver by substituting `x_old_eff = 4/3 x_n - 1/3 x_{n-1}` and `dt_eff = 2/3 dt`; the transient-coefficient ratio is evaluated at `x_old_eff`, an O(dt^2)-per-step inconsistency) plateaus at ~4e-4, above the reference error, so the asymptotic order cannot be read off below dt = 0.05; a production BDF2 needs the transient term formed from `tc_n x_n` and `tc_{n-1} x_{n-1}` exactly. Nonlinear cost per step is unchanged (3.7 vs 3.5 Newton iterations at dt = 0.1).

**Step selection.** With the same physics and solver: the chi heuristic (`chi_timestep_prefactor` = 10 / 30 / 50 / 100 / 300) gives 16 / 7 / 5 / 4 / 3 steps with max errors 4.7e-3 / 1.4e-2 / 2.1e-2 / 2.4e-2 / 2.8e-2 and dt spreads of 25x within one run (e.g. 0.003-0.08 s at prefactor 10). Fixed dt with the same number of steps gives the same error (10 steps of 0.1 s: 5.8e-3; 5 steps of 0.2 s: 1.2e-2): **the chi heuristic is not tracking the temporal error; it is just a step counter with a large spread**. 

**Error-controlled backward Euler** (CVODE/PETSc style, `bench/adaptive_dt.py`): local error estimated at no cost from the linear-extrapolation predictor, `LTE = (x_{n+1} - x_pred) / (1 + dt/dt_prev)` in a WRMS norm with `rtol` and `atol = 1e-3` (scaled units), elementary controller `dt_new = dt * clip(0.9 * err^(-1/2), 0.2, 2)`, rejection when `err > 1`, dt_0 = 0.01 s, dt_max = 0.5 s.

| controller | steps (+rejected) | Newton iterations | dt min / median / max | max rel. error at t = 1 s |
|---|---|---|---|---|
| rtol = 3e-2 | 11 (+4) | 39 | 0.004 / 0.032 / 0.37 | 1.4e-2 |
| rtol = 1e-2 | 16 (+5) | 51 | 0.001 / 0.038 / 0.23 | 8.4e-3 |
| rtol = 3e-3 | 25 (+5) | 66 | 6e-5 / 0.022 / 0.18 | 5.8e-3 |
| rtol = 1e-3 | 36 (+5) | 92 | 1e-4 / 0.015 / 0.11 | 3.5e-3 |
| fixed dt = 0.2 / 0.1 / 0.05 / 0.025 | 5 / 10 / 20 / 40 | 25 / 35 / 55 / 102 | - | 1.2e-2 / 5.8e-3 / 2.6e-3 / 1.1e-3 |
| chi heuristic, prefactor 30 / 10 | 7 / 16 | 28 / 50 | 0.009-0.25 / 0.003-0.08 | 1.4e-2 / 4.7e-3 |

**Error control does not reduce the step count for a given final error in this case** - it costs ~1.5x more Newton iterations than fixed dt at equal error. The controller spends its steps on the initial transient (dt falls to 1e-4..4e-3 s in the first steps, then grows geometrically to 0.1-0.4 s), which the L-stable backward Euler damps anyway so that the fixed-dt runs do not pay for skipping it. This is the expected behaviour of local error control on a strongly dissipative problem and is the honest caveat to the PETSc/SUNDIALS recommendation: the benefit of an error estimate for TORAX is *robustness and automatic dt selection* (no hand-tuned `chi_timestep_prefactor`, dt that follows the dynamics through sawteeth/transitions, a principled retry rule), not fewer steps at fixed accuracy. Fewer steps come from the higher-order method (BDF2 above), and a controller is what makes a variable-step BDF2 usable.

### 3.6 The linear (Picard) solver: convergence and Anderson acceleration (`bench/picard_anderson.py`)

Same states as 3.4. Each Picard sweep is exactly TORAX's corrector step (`implicit_solve_block` with coefficients from the previous iterate, with or without Pereverzev-Corrigan terms); the error is measured against the fully converged Newton solution `x*` of the same implicit step, relative to the step change `|x* - x_old|`. Anderson acceleration (SUNDIALS `SUNNonlinSol_FixedPoint` form, depth m = 2 and 4, no damping) is applied to the same map.

| dt | Picard + Pereverzev, error after 1 / 3 / 6 / 12 sweeps | Anderson m = 4 + Pereverzev | Picard without Pereverzev |
|---|---|---|---|
| 2.0 | 0.96 / 0.25 / 0.21 / 0.30 | 0.96 / 0.22 / 0.28 / 0.30 | diverges (1.9, 4.7, 34, ...) |
| 0.5 | 0.93 / 1.2 / 1.2 / 0.72 | 0.93 / 1.2 / 0.52 / 0.55 | diverges |
| 0.1 | 0.96 / 6.4 / 0.70 / 0.99 | 0.96 / 1.5 / 1.4 / 0.84 | diverges |
| 0.02 | 1.1 / 3.1 / 2.2 / 1.8 | 1.1 / 1.0 / 2.0 / 1.4 | 0.2 - 4.8, oscillating |

Findings.
- **Without Pereverzev-Corrigan terms the frozen-coefficient Picard iteration diverges at every dt** (consistent with `rho(I - P^-1 J)` = 1.1-50 in 3.3). With them it is bounded but **does not converge to the implicit solution**: after 10 sweeps the iterate is still 0.2-1.0 step-changes away from `x*` (it oscillates around a point that is not the Newton solution because the Pereverzev terms only cancel at a converged fixed point). The corrector sweeps beyond the first one buy little.
- **Anderson acceleration does not rescue it**: on a non-contractive map with QLKNN threshold behaviour, depths 2-4 give no systematic improvement (sometimes 2x better, sometimes worse). SUNDIALS and PETSc both recommend Anderson for *linearly convergent* fixed-point maps (CVODE: "for nonstiff systems"); this map is not one. Verdict: the SUNDIALS/PETSc fixed-point machinery does not transfer to TORAX's linear mode as-is; a damped Anderson with residual-based selection (PETSc NGMRES) might, but the measured map is far from the regime where it is known to work.
- The practical consequence: the linear solver's speed (3-6 ms per sweep) comes with a per-step solution error comparable to the step change itself. Measured end-to-end (`bench/linear_vs_newton.py`, same physics, fixed dt = 0.05 s, 40 steps to t = 2 s, deviation from the Newton solution converged to 1e-7):

| solver | max relative deviation T_i / T_e / n_e / psi |
|---|---|
| newton, default tolerance 1e-5 | 6e-7 / 6e-7 / 1e-6 / 9e-8 |
| linear, no corrector (single Picard solve) | 3.8e-2 / 9.2e-2 / 1.1e-1 / 3.4e-2 |
| linear, 1 corrector sweep | 3.3e-2 / 6.6e-2 / 7.2e-2 / 2.4e-2 |
| linear, 3 corrector sweeps | 2.0e-2 / 3.3e-2 / 1.6e-2 / 1.3e-2 |
| linear, 10 corrector sweeps | 8.3e-3 / 1.2e-2 / 3.7e-3 / 8.3e-3 |
| linear, 30 corrector sweeps | 4.9e-3 / 1.1e-2 / 1.7e-3 / 1.8e-3 |

At fixed dt = 0.2 s (10 steps) the deviations are 7-20% with 1 sweep, 2-3% with 10 and 1% with 30; with the chi-based dt of the example itself (section 3.1b) they are 12-31% with 1 sweep and 2-9% with 10. The deviation decreases only slowly with the number of sweeps (sublinear), as the oscillating iteration histories above predict. Whether that matters depends on the use; but it means the linear solver should not be used as a "fast approximate Newton" for accuracy-sensitive work, and that speeding up the *Newton* solver (sections 3.2-3.4) is the relevant target.

### 3.7 Compile time

AOT `jax.jit(step_fn).lower(...).compile()` at n_rho = 50, `iterhybrid_predictor_corrector` physics, first 5 steps (`results/profile_runs/compile_*_idle.json`):

| configuration | lower | compile | HLO text | per-iteration runtime |
|---|---|---|---|---|
| linear solver | 1.6 s | 12.1 s | 10.3 MB | 4 ms/sweep |
| newton, sequential line search, PC guess | 3.8 s | 15.8 s | 21.4 MB | 49 ms |
| newton, sequential line search, no PC guess | 3.9 s | 15.6 s | 20.6 MB | 57 ms |
| newton, `vmap_linesearch` (100 steps), PC guess | 5.0 s | 18.3 s | 21.9 MB | 76 ms |
| newton, `vmap_linesearch` (100 steps), no PC guess | 4.9 s | 17.9 s | 21.0 MB | 72 ms |
| `root_newton_raphson` alone (micro-benchmark, n_rho = 50): with / without `custom_root`, and with `vmap_linesearch` (8 steps) | - | 8.8 / 8.8 / 11.4 s | - | - |

End-to-end first-step compile (trace + compile, 4 cores, idle): 7.5-13.6 s for the linear solver, 17-23 s for the Newton solver, growing slowly with n_rho (19, 20, 21, 23 s at n_rho = 25, 50, 100, 200).

The Newton solver roughly doubles the HLO size relative to the linear solver; `jacfwd` of the full physics is the main contributor (its batch dimension is N). Structured/coloured Jacobians (36 columns) or JFNK (no batch) would shrink this graph. `custom_root` is free. The predictor-corrector initial guess adds ~0.2 s of AOT compile (2 s end-to-end) and is a runtime win (3.1b).

### 3.9 Prototype: structured Jacobian assembly, measured (`bench/structured_jacobian.py`)

The recommendation in 3.3 (item 3) was implemented as a stand-alone prototype without modifying TORAX: the residual is split as `R(x) = G(x, S h(x))` by overriding the smoothing hook in a `TransportModel` subclass (`raw` mode returns the pre-smoothing coefficients for `h`; `inject` mode returns coefficients carried in a `PedestalModelOutput` subclass, so they can be JAX tracers), `S` is TORAX's own `_build_smoothing_matrix`, and `J = dG/dx|_c + dG/dc . S . dh/dx` is assembled from three coloured `jax.vmap(jax.jvp)` batches with structural patterns taken as the union of the nonzero patterns at three states (predictor, converged solution, previous time level). Same `iterhybrid_rampup` states as 3.2 (dt = 2 s, four channels, QLKNN with smoothing). The injected residual reproduces the original to 1e-13.

| n_rho | N | seeds (dG/dx + dh/dx + dG/dc) | `jacfwd` | structured | speed-up | rel. error vs `jacfwd` (x0 / x* / midpoint) | Newton solve, dense `jacfwd` -> structured | iterations | solution rel. diff |
|---|---|---|---|---|---|---|---|---|---|
| 25 | 100 | 16 + 14 + 4 = 34 | 5.1 ms | 2.3 ms | 2.2x | 3e-15 / 3e-15 / 4e-15 | 21.3 -> 10.5 ms | 4 / 4 | 4e-15 |
| 50 | 200 | 34 | 23.5 ms | 3.7 ms | 6.3x | 9e-15 / 7e-15 / 7e-15 | 172 -> 32 ms | 4 / 4 | 6e-15 |
| 100 | 400 | 34 | 90.9 ms | 8.7 ms | 10.4x | 2e-14 / 2e-14 / 1e-14 | 779 -> 91 ms | 5 / 5 | 2e-9 |
| 200 | 800 | 34 | 413 ms | 22.9 ms | 18x | 4e-14 / 4e-14 / 3e-14 | 1629 -> 273 ms | 4 / 4 | 6e-9 |

Breakdown at N = 400: `h(x)` 0.7 ms, 16-seed batch of `dG/dx` 0.9 ms (with the transport coefficients frozen the JVP skips QLKNN and is cheaper than a residual), 14-seed batch of `dh/dx` 2.6 ms (through QLKNN), 4-seed batch of `dG/dc` 0.4 ms, the rest is the prototype's dense decompression and the `(dG/dc)(S dh/dx)` product (at N = 800 that dense part is ~15 of 23 ms; a production version would keep it banded plus the smoothing block). Compile time of the standalone assembly is 8.7-9.4 s versus 5.4-6.2 s for `jacfwd`, and the Newton solve compiles in 13.0-13.5 s versus 7.8-10.2 s, because the prototype traces `G` twice and adds the decompression; `jacfwd`'s compile time barely grows with N at these sizes, so the compile-time benefit anticipated in 3.2 did not materialise and would need a leaner implementation (e.g. `jax.linearize` once, shared forward pass).

Reading: the exactness and the seed count are grid-independent, as predicted; the Jacobian cost now grows ~2x per grid doubling instead of ~4.5x; the Newton solve at n_rho = 100 goes from 779 ms to 91 ms (18 ms per iteration, of which ~9 ms is the Jacobian, the rest residuals, the dense solve and the loop-body overhead of 3.2 item 6). What a production implementation needs from TORAX: (i) `calc_coeffs` accepting precomputed turbulent coefficients (the injection point), (ii) the transport model exposing its pre-smoothing output and the smoothing matrix, (iii) structural sparsity patterns derived from the stencils rather than numerically, with any global scalar functionals (section 7.3) handled as explicit rank-1 terms.

### 3.10 Production implementation: `solver.jacobian_mode = 'structured'`

The prototype of 3.9 was turned into an option of the Newton-Raphson solver inside TORAX (branch `claude/torax-solver-optimization-b1p7b8`). Nothing changes for the default `jacobian_mode = 'dense'`. Sections 3.10-3.12 describe the first version; 3.13 generalises the post-processing of the coefficients and the global quantities, removes the restrictions listed under *Guards* and changes the stencils and seed counts.

What was added.
- **Injection point in `calc_coeffs`.** `calc_coeffs`, `_calc_coeffs_full`, `theta_method_block_residual` and `transport_coefficients_builder.calculate_all_transport_coeffs` take an optional `turbulent_transport` (a `TransportCoeffs` of already-smoothed coefficients); when given, the transport model is not evaluated and the rest of the residual is built around the injected coefficients. `calculate_all_transport_coeffs` also takes a static `apply_smoothing` flag, forwarded to `TransportModel.__call__`, which returns the clipped coefficients without the Gaussian smoothing when it is `False`. `TransportModel.smoothing_matrix(...)` and a builder-level `smoothing_matrix(...)` (which applies the same pedestal-transition override as the coefficients) expose `S`.
- **`torax/_src/solver/structured_jacobian.py`.** `jacobian_fn(residual_fun)` builds the Jacobian function from the bound arguments of the residual partial: the raw coefficients `h(x)` (`update_core_profiles_during_step` + `calculate_all_transport_coeffs(apply_smoothing=False)`), the global quantities `g(x)` (`build_source_globals`, flattened with `ravel_pytree`), the residual `G(x, c, g)` with both injected, and `S`. The assembly linearises each factor once (`jax.linearize`), evaluates the linear maps on cyclic column colourings whose period covers the stencil band (half-width 2), a reserved near-axis block for the `_extrapolate_cell_profile_to_axis` coupling and the stencil of the raw transport coefficients (cells f-2..f+1 of face f, or f-3..f+2 when a transport model includes rotation, 3.12), scatters the compressed columns through boolean masks, applies `dG/dc . (S dh/dx)` row-wise (`dG/dc` needs only the two face parities of each coefficient) and adds the rank-`len(g)` term from `jax.linear_transpose` of the linearised `g`. If `TORAX_ERRORS_ENABLED=True`, one extra JVP of the full residual along a random probe `p` is compared with `J p` row by row at every Newton iteration and raises if a row differs by more than `sqrt(eps)` of `|J| p` - this catches any coupling that is not in the stencils (3.12).
- **Sources with global state dependences** (the pattern of 7.3, implemented). Three source models couple every cell to every other through a few scalars: the Albajar-Artaud cyclotron sink (on-axis `n_e` and `T_e`, the fitted profile factor `K` and the volume integral that normalises the Artaud shape), the constant-fraction impurity radiation (the volume-integrated heating power of the other sources) and the ToricNN ICRH model (the volume averages of `T_e` and `n_e`, the on-axis minority concentration and the two peaking factors that are the surrogate's state-dependent inputs). The model function of each is a `source.SplitModelFunction` of a `globals_func` (the vector `g` of those scalars) and a `profile_func` (the same profile given `g`, local in the state), which composes the two when called, so the split is the model function rather than something attached to it (3.12); the experimental gas-puff feedback source (1 scalar, the averaged density) is split the same way. `source_profile_builders.build_source_globals` evaluates `g` in the same order and with the same previously calculated profiles as the source builder, and `build_source_profiles`/`calc_coeffs`/`theta_method_block_residual` accept injected globals. The residual is then `R(x) = G(x, S h(x), g(x))` and the exact Jacobian gains a rank-`len(g)` term, `J = dG/dx|_{c,g} + dG/dc . S . dh/dx + dG/dg . dg/dx`, from one forward seed of `G` and one reverse-mode VJP of `g` per global scalar: 10 scalars for the three sources together (4 + 1 + 5), so the cost stays grid-independent. Plain model functions are assumed local; the JVP check under `TORAX_ERRORS_ENABLED` remains the safety net for user-defined sources.
- **Wiring.** `newton_raphson_solve_block(jacobian_mode=...)` passes `structured_jacobian.jacobian_fn(residual_fun)` as `custom_jac` to `root_newton_raphson`, which accepts a custom Jacobian together with `jax.lax.custom_root` (the implicit-function tangent solve still uses `jacfwd`, it is only traced for differentiation). The assembly is a plain `jax.jit`; unlike the dense Jacobian it is not late-inlined by XLA (3.12). Config: `solver.jacobian_mode: 'dense' | 'structured'` on `newton_raphson` (`docs/configuration.rst`), a static field of `NewtonRaphsonRuntimeParams`.
- **Guards (removed in 3.13).** The stencils were only valid for residuals whose remaining couplings are local. `ToraxConfig` rejects `jacobian_mode='structured'` with an implicit pedestal (`pedestal.explicit_pedestal=False`) or the `adaptive_transport` pedestal mode, with a non-uniform radial grid (to 1e-9), with the `MTANH` pedestal profile form or `beta_poloidal_prime` internal boundary conditions when the current evolves (3.12), and with `numerics.min_rho_norm > 0.05` (the near-axis block is sized from the static cell count for every `min_rho_norm` up to 0.05, because `min_rho_norm` itself is a traced runtime parameter). The JVP check above is the runtime safety net.
- **Tests** (`torax/_src/solver/tests/structured_jacobian_test.py`, mid-level): at n_rho = 25, with and without the three global sources and QLKNN rotation, the assembled matrix equals `jax.jacfwd` at a perturbed state to 1e-10 in every row and `newton_raphson_solve_block` gives the same iterations and solution (rtol 1e-8) in both modes with the per-iteration JVP check enabled; the source builder evaluated from `build_source_globals` reproduces the model-function profiles exactly; the JVP check raises for a narrowed stencil; and the config validation accepts the global sources and rotation and rejects an implicit pedestal and `MTANH`. The existing `fvm`, `solver`, `sources`, `transport_model`, `torax_pydantic` and `orchestration` suites pass, as does the full `sim_test` regression suite (63 tests, reference outputs of the default mode unchanged); the latter has to be run in chunks of separate processes here (`bench/sim_chunks.sh`), because one process cannot hold all the compiled simulations in this container (the LLVM JIT fails to map new code sections after ~17 configurations).

Seed counts. With four channels the production assembly uses `4 x period` seeds for `dG/dx` with `period = max(5, axis_cells + 3)`, 16 for `dh/dx` and 8 for `dG/dc`: 44 seeds at n_rho = 25-50 and 56 at n_rho = 100 (the reserved near-axis block grows with the grid; the prototype's 34 seeds used numerically derived patterns and the exact `min_rho_norm`), plus one JVP and one VJP per global source scalar (10 with all three global sources enabled). 3.13 replaces the reserved block and the uniform-grid transport stencil by stencils valid on any grid: 24 + 20 + 8 = 52 seeds at any n_rho (64 with rotation).

End-to-end measurements (idle machine, one fresh process per run, `profile_runs/driver_structured.sh`, `results/profile_runs/prod_*.json`; dense and structured runs of each case were made back-to-back in the same session, so the dense rows are the reference for this table rather than the 3.1 numbers from an earlier session). The numbers are for the final code, i.e. after the compile-time change E1 of section 3.11 and the fixes of the stress test of 3.12, which reverted E5 (`results/logs/stress_fixes_benchmark.log`); the rows quoted in the text of 3.11 are from the same driver at earlier stages.

| case | n_rho | N | mode | steps | compile (first step) | run (excl. compile) | median step | Newton its total | ms per Newton it | final-profile rel. diff vs dense |
|---|---|---|---|---|---|---|---|---|---|---|
| `iterhybrid_rampup` (dt = 2 s, 40 steps) | 50 | 200 | dense | 40 | 21.4 s | 7.60 s | 192 ms | 109 | 72 | - |
|  | 50 | 200 | structured | 40 | 27.9 s | 2.69 s | 68 ms | 109 | 26 | 2.0e-14 |
| `iterhybrid_rampup`, n_rho = 100 | 100 | 400 | dense | 40 | 22.5 s | 21.65 s | 490 ms | 139 | 160 | - |
|  | 100 | 400 | structured | 40 | 27.9 s | 4.69 s | 111 ms | 139 | 35 | 2.9e-14 |
| `iterhybrid_rampup` + cyclotron, constant-fraction radiation, ToricNN ICRH | 50 | 200 | dense | 40 | 25.3 s | 9.13 s | 214 ms | 128 | 74 | - |
|  | 50 | 200 | structured | 40 | 40.1 s | 3.65 s | 89 ms | 128 | 29 | 1.1e-14 |
| same, n_rho = 100 | 100 | 400 | dense | 40 | 25.5 s | 20.90 s | 494 ms | 130 | 166 | - |
|  | 100 | 400 | structured | 40 | 40.3 s | 5.75 s | 136 ms | 130 | 46 | 1.9e-14 |
| `iterhybrid_predictor_corrector`, Newton, chi dt, t_final = 1 s | 25 | 100 | dense | 5 | 20.0 s | 0.28 s | 73 ms | 26 | 14 | - |
|  | 25 | 100 | structured | 5 | 26.6 s | 0.18 s | 44 ms | 26 | 9 | 2.1e-15 |
| same, n_rho = 50 | 50 | 200 | dense | 13 | 20.1 s | 2.73 s | 207 ms | 46 | 66 | - |
|  | 50 | 200 | structured | 13 | 28.0 s | 0.83 s | 69 ms | 46 | 20 | 1.3e-14 |
| same, n_rho = 100 | 100 | 400 | dense | 49 | 21.4 s | 19.57 s | 341 ms | 127 | 160 | - |
|  | 100 | 400 | structured | 49 | 26.7 s | 4.11 s | 77 ms | 127 | 34 | 5.2e-14 |

Reading.
- **The stepping time drops 2.8-3.3x at n_rho = 50 and 4.6-4.8x at n_rho = 100** (`iterhybrid_rampup`: 7.6 -> 2.7 s and 21.7 -> 4.7 s for the 40 steps; the predictor-corrector physics: 2.7 -> 0.8 s and 19.6 -> 4.1 s), with exactly the same Newton iteration counts and solutions equal to 1e-14. At n_rho = 25 the gain is 1.7x (73 -> 44 ms per step), consistent with 7.2: at N = 100 the Jacobian is no longer the dominant cost. With the late inlining of E5 (3.11, reverted in 3.12) the compiled loop body was 12-26% faster still (2.15 and 4.17 s for the two `iterhybrid_rampup` runs).
- **Per Newton iteration** the end-to-end cost goes from 160 ms to 34-35 ms at N = 400 and from 66-72 ms to 20-26 ms at N = 200. The remaining ~35 ms at N = 400 are the 56 batched JVPs (about 9-12 ms), the residual evaluations of the line search and convergence test, the dense LU solve and the fixed per-iteration overhead of the loop body (3.2, item 6); the per-step orchestration outside the solve (3.2) is now a comparable share of the step.
- **Solutions are identical to round-off** (max relative difference 2e-15 to 5e-14 on `T_i`, `T_e`, `n_e`, `psi` after the full run). The runs with `TORAX_ERRORS_ENABLED=True` (`results/logs/verify_structured_mode.log`, `results/logs/verify_structured_mode_globals.log`) passed the per-iteration JVP check, in its original global-norm form, on all configurations; the row-wise check of 3.12 passes in the tests, with and without the global sources and rotation.
- **With the three globally coupled sources enabled** (cyclotron sink, constant-fraction impurity radiation, ToricNN ICRH with a dummy surrogate; circular geometry because the ICRH model needs the magnetic-axis height that the CHEASE geometry does not carry) the stepping time still drops 2.5x at n_rho = 50 (9.1 -> 3.7 s) and 3.6x at n_rho = 100 (20.9 -> 5.8 s), with identical iterations and solutions to 1e-14. The rank-10 term costs ~11 ms per iteration at N = 400 (46 vs 35 ms), most of it the reverse-mode pass through the source builder (the cyclotron profile fit is a 32-point grid search over vmapped closed-form fits). The compile time is the larger price, ~15 s over dense (40 vs 25 s; 28-29 s with E5, 46 s before 3.11), because the globals function compiles the source builder once more forward and once in reverse; sharing that primal with the residual (3.11, E2) is the obvious follow-up.
- **Compile time is 5-8 s longer than the dense mode** (first step 26.6-28.0 s vs 20.0-22.5 s). The review of 3.11 had brought it within 0.5-2 s with a single `jax.linearize` per factor (E1, kept) and late XLA inlining of the assembly (E5), which had to be reverted because XLA's propagation of the full Python tracebacks into the late-inlined instructions can run for tens of minutes (3.12). With `JAX_INCLUDE_FULL_TRACEBACKS_IN_LOCATIONS=false` the late-inlined assembly compiles normally (31.3 s vs 34.6 s for the plain `jax.jit` through `torax.run_simulation` at n_rho = 50, dense 29.0 s), so E5 could come back under that setting and recover both its compile time and its faster loop body; it is a process-wide JAX setting that removes the source context from HLO dumps and profiles, so it is left as a decision. Total process time of `iterhybrid_rampup`: 2.7 + 27.9 = 30.6 s vs 7.6 + 21.4 = 29.0 s at n_rho = 50 (dense slightly ahead for this 40-step run) and 4.7 + 27.9 = 32.6 s vs 21.7 + 22.5 = 44.2 s at n_rho = 100.

First and repeated runs (`profile_runs/first_vs_repeat.py`, `profile_runs/driver_first_vs_repeat.sh`, `results/profile_runs/first_vs_repeat.jsonl`): one fresh process per case and mode, run one after another on the otherwise idle machine, each calling `ToraxConfig.from_dict` + `torax.run_simulation` three times. Wall time per call, excluding the ~2.5 s Python import; "repeat" is the mean of calls 2 and 3. Every first call makes 53 XLA compilations (60 with the global sources) in both modes and every repeated call none, so the repeat is the whole simulation (all steps, the initial state and the output conversion) without compilation.

| case | steps | dense first | structured first | dense repeat | structured repeat | repeat speedup |
|---|---|---|---|---|---|---|
| `iterhybrid_rampup`, n_rho = 50 | 40 | 44.1 s | 46.0 s | 8.9 s | 3.4 s | 2.7x |
| `iterhybrid_rampup`, n_rho = 100 | 40 | 61.5 s | 48.8 s | 24.1 s | 5.9 s | 4.1x |
| + cyclotron, constant-fraction radiation, ToricNN ICRH, n_rho = 50 | 40 | 50.4 s | 59.6 s | 10.3 s | 4.5 s | 2.3x |
| same, n_rho = 100 | 40 | 63.3 s | 60.8 s | 22.9 s | 6.7 s | 3.4x |
| `iterhybrid_predictor_corrector`, Newton, t_final = 1 s, n_rho = 25 | 5 | 32.6 s | 38.8 s | 0.53 s | 0.37 s | 1.4x |
| same, n_rho = 50 | 13 | 37.0 s | 38.9 s | 3.6 s | 1.05 s | 3.4x |
| same, n_rho = 100 | 49 | 53.2 s | 40.7 s | 21.3 s | 4.8 s | 4.4x |

The first call is dominated by compilation in both modes (25-33 s of XLA time dense, 29-46 s structured); at n_rho = 100 the structured first call is still 2.5-12.7 s shorter because its steps are, and at n_rho <= 50 it is 1.9-9.2 s longer. Repeated calls in the same process, or with JAX's persistent compilation cache across processes (3.11, item 6), get the full stepping gain.

### 3.11 Compile-time review of the structured Jacobian (`bench/compile_breakdown.py`, `profile_runs/driver_compile.sh`)

Method: `jax.jit(step_fn).lower()` (JAX tracing and lowering) and `.compile()` (XLA) timed separately in fresh processes (`--aot`), `iterhybrid_predictor_corrector` physics at n_rho = 50, Newton solver; and the same split for each factor of the assembly compiled standalone from the exact closures that the solve block builds (`compile_breakdown.py`). Run-to-run noise of the full-step numbers is about +-2 s, so the standalone breakdown is the precise instrument. TORAX already runs XLA with `--xla_cpu_opt_preset=FAST_COMPILE` (set in `torax/__init__.py`).

Full step, before this review (`cmp_base_*`):

| Jacobian | sources | JAX trace + lower | XLA compile | HLO text |
|---|---|---|---|---|
| dense | default | 3.9 s | 19.3 s | 21.4 MB |
| structured | default | 4.8 s | 25.6 s | 22.7 MB |
| dense | + cyclotron, constant fraction, ToricNN ICRH | 4.0 s | 19.7 s | 24.0 MB |
| structured | same | 6.7 s | 32.4 s | 27.2 MB |

XLA dominates (80-85% of the first step), and the structured penalty was +7 s (default sources) and +15 s (with the globally coupled sources), while the HLO grew only 6-13%: the extra cost is in *how many distinct transformed copies* of the physics XLA has to optimise, not in the size of any one of them.

Standalone compile of the factors (n_rho = 50, N = 200; trace + XLA, and run time of the compiled function):

| function | default sources | with global sources | run time |
|---|---|---|---|
| residual `R(x)` (plain pjit) | 0.7 + 1.7 s | 0.7 + 2.2 s | 0.9-1.3 ms |
| dense `jacfwd(R)` (200 seeds) | 1.7 + 4.3 s | 2.2 + 5.3 s | 27-29 ms |
| `h(x)` primal (transport model) | 0.2 + 0.9 s | 0.1 + 0.8 s | 0.6-0.8 ms |
| `dh/dx` as `vmap(jvp(h))`, 16 seeds | 0.8 + 1.6 s | 0.6 + 1.7 s | 2.7 ms |
| `dh/dx` as `linearize(h)` + `vmap`, 16 seeds | 0.3 + 1.6 s | 0.2 + 1.8 s | 2.4-2.6 ms |
| `G(x, c, g)` primal (residual with injected coefficients) | 0.2 + 1.5 s | 0.2 + 1.6 s | 0.4 ms |
| `dG/dx` as `vmap(jvp(G))`, 20 seeds | 0.9 + 3.2 s | 1.4 + 3.8 s | 0.8-1.1 ms |
| `dG/dx` as `linearize(G)` + `vmap`, 20 seeds | 0.6 + 3.7 s | 0.7 + 4.4 s | 1.1-1.2 ms |
| `g(x)` primal (source builder) | - | 0.1 + 0.9 s | 0.3 ms |
| `dg/dx` as `vmap(vjp(g))`, 10 seeds | - | 1.3 + 1.7 s | 0.8 ms |
| `dg/dx` as `linearize(g)` + `linear_transpose`, 10 seeds | - | 0.2 + 1.8 s | 0.7 ms |
| whole structured assembly, after E1 below | 2.8 + 5.1 s | 3.3 + 6.8 s | 4.6-4.8 ms |

Findings, in order of impact.

1. **[fixed: E1] Every factor was traced two or three times.** `_batched_jvp` was called on `G` three times (seeds for `dG/dx`, `dG/dc`, `dG/dg`), on `h` once plus one plain primal call, and `g` was evaluated once plus once more inside `jax.vjp`; each call is a separate JVP transformation of the (pjit) residual, i.e. a separate primal-plus-tangent computation for XLA, and the three `G` copies differ only in which argument carries the tangent. The assembly now calls `jax.linearize` once per factor and evaluates the linear map on all seeds of that factor in one `vmap` (the `dG/dx`, `dG/dc` and `dG/dg` seeds are stacked into one batch of 20 + 8 + k rows; `dg/dx` comes from `jax.linear_transpose` of the linearised `g`). Full step: structured 4.8 + 25.6 -> 4.3 + 21.9 s (default sources), 6.7 + 32.4 -> 5.6 + 28.1 s (global sources); the assembly is exact to the same 1e-10 and the run time is unchanged within noise (the c and g seeds now also evaluate the x-tangent with zero input, ~1 ms per Jacobian at N = 400). The remaining standalone gap to `jacfwd` is 1.8 s (7.8 vs 6.0 s).
2. **[open: E2, estimated 2-3 s] Three primals where one would do.** After E1 the assembly still compiles three "known" computations: the transport model for `h` (includes QLKNN, ~1 s), the residual with injected coefficients for `G` (~1.5 s) and the source builder for `g` (~1 s), all evaluating `update_core_profiles_during_step` and most of the sources again. One function `F(x, dc, dg) = (residual with c = stop_gradient(S h(x)) + dc and globals = stop_gradient(g(x)) + dg, h(x), g(x))` linearised once gives one primal, and the three tangent uses (residual tangent on the x/c/g seeds, `h` tangent on the h seeds, transpose of the `g` tangent) are then dead-code-specialised copies of one linear jaxpr. It needs the transport builder to accept a perturbation of frozen coefficients and the residual to return `h` and `g` as auxiliary outputs (a static `return_aux` flag through `calc_coeffs` and `theta_method_block_residual`); the primal is then guaranteed equal to `R(x)` by construction and `build_source_globals` is no longer needed by the solver.
3. **[open: E3, estimated 1-2 s] Two plain residual computations in the Newton loop.** `_newton_raphson` evaluates the residual before the loop and again inside the line search; both are the same pjit, so XLA sees one computation with two call sites, but each is also distinct from the linearised primal of E2. Computing the residual at the accepted iterate from the linearisation (`linearize(F)` returns it for free) and dropping the pre-loop evaluation would leave the line-search residual as the only plain copy; this costs one extra residual evaluation per Newton solve (~1 ms) and touches `jax_root_finding`, so it is a separate change.
4. **[measured, then reverted: E5] Late XLA inlining of the assembly.** The dense path wraps `jacfwd` in `jax.jit(inline=jax.Inline.XLA_LATE)`; the structured `custom_jac` was inlined at trace time. Wrapping it the same way (the assembly is a `jax.jit(..., inline=jax.Inline.XLA_LATE)` inside `structured_jacobian.jacobian_fn`) turns the assembly into one HLO computation that XLA optimises once and inlines into the Newton while-body at the end, instead of a body that XLA has to optimise with the whole assembly already inlined. Measured on top of E1: structured 4.3 + 21.9 -> 4.1-4.5 + 16.9-18.2 s (two runs) with the default sources and 5.6 + 28.1 -> 5.3-5.7 + 21.2-21.3 s with the global sources, i.e. within the run-to-run noise of the dense step (3.5-3.9 + 16.1-19.3 s over the same session). On the full `iterhybrid_rampup` at n_rho = 100 the structured first step went from 31.9 s to 21.8 s (dense: 21.0 s in the same session), and with the global sources from 46.5 s to 28.3 s (dense: 22.8 s); the compiled loop body also runs ~15% faster (rampup n_rho = 100: 5.1 -> 4.3 s of stepping). E1 is applied on the branch; **E5 was reverted** after the stress test of 3.12 found that XLA's late inlining of the assembly can run for tens of minutes, depending on the depth of the Python call stack at trace time (it did through `torax.run_simulation`, not through the benchmark driver used here). The two of five attempts that ran for more than 20 minutes with error checking enabled were most likely the same effect.
5. **[measured, no effect] XLA optimisation level.** `--xla_backend_optimization_level=1/2/3` on top of the `FAST_COMPILE` preset (`profile_runs/driver_xla_flags.sh`): dense 15.8 / 15.2 / 16.1 s and structured 15.9 / 16.0 / 16.6 s of XLA time, run time unchanged (100 / 102 / 106 ms per step for the structured `iterhybrid_rampup` at n_rho = 100). The preset already runs the fast pipeline; there is no further compile/run trade-off to be had from this flag.
6. **[operational, not code] The persistent compilation cache** removes XLA from repeated runs of the same configuration structure: `--jax_compilation_cache_dir=<dir>` on the command line (JAX's own flag, exercised by `torax/tests/persistent_cache_test.py`) makes the second process load the executable in about a second instead of recompiling for 20-30 s; the structured mode's extra compile is then paid once per machine. Note that `TORAX_ERRORS_ENABLED` inserts host callbacks that disable the cache (see `jax_utils/common.py`).
7. **Minor, not worth a change on their own**: the (N x N) boolean masks and the seed matrices are embedded as HLO constants (iota-computed masks would shave a few hundred kB of HLO text); `jax.eval_shape` of the source builder to learn the layout of `g` costs a Python trace that a static `n_globals` declaration on `Source` would avoid; the `dG/dc` colouring could be folded into the x colouring (8 fewer seeds) at the price of a second decompression mask.

What does not help: `jax.lax.custom_root` (3.2 measured it free, since its eager traces reuse the cached pjit jaxprs); reordering the assembly; smaller seed counts (the compile time is independent of the batch size, only the run time scales with it).

### 3.12 Adversarial stress test of the structured Jacobian (`stress/`)

Three independent reviewers attacked the implementation of 3.10-3.11 from different sides, each in fresh processes on the same machine: (1) a breadth sweep comparing the assembled matrix with `jax.jacfwd` on about 110 configurations - every `tests/test_data` config and example forced to the Newton solver, each transport model and option, each source model and mode, pedestal and internal-boundary-condition models, evolving subsets, geometries, n_rho = 4-100 and `min_rho_norm` = 0-0.05 - at the initial state and at 1%, 20% and smooth +-30% perturbations, with a per-row error metric (the relative Frobenius error hides errors in weakly scaled rows; `results/stress/exactness_sweep.md`); (2) whole simulations in both modes through `torax.run_simulation` and step-by-step drivers, gradients and `vmap` through a simulation, `custom_root` toy problems, recompilation and state leakage between simulations, error checking and float32; (3) white-box checks of each invariant of `_assemble` - the true sparsity of `dG/dx`, `dh/dx`, `dG/dc` and `dg/dx` from `jax.jacfwd` against the masks, the state independence of `S`, the near-axis block for n = 4-400 against more than 2000 `min_rho_norm` values, colourings with fewer cells than the colour period, the layout of the globals, fault injection into the runtime check, the validator and the `custom_root` semantics. Every finding below was reproduced before it was fixed; the scripts are in `stress/`.

Findings and fixes, most severe first.

1. **Compile blow-up through the public API (fixed).** With the late XLA inlining of E5 (3.11), `torax.run_simulation` on `iterhybrid_rampup` at n_rho = 50 did not finish compiling in 400 s on the idle machine (dense: 28.5 s); the same happened in other harnesses at n_rho = 16 with QLKNN rotation and with all sources plus Mavrin radiation. The only busy thread was in XLA's late `CallInliner` (`PropagateCallMetadata::UpdateStackFrame` -> `StackFrames::IsPrefix`), which propagates the Python call-stack location metadata into the tens of thousands of inlined instructions; its cost depends on the depth of the caller's stack at trace time, which is why the benchmark driver of 3.10 (`profile_runs/run_one.py`) never triggered it while `torax.run_simulation` does. The hang seen with error checking enabled (the caveat of 3.11 E5) was most likely the same effect rather than the host callbacks. **E5 is reverted**: the assembly is a plain `jax.jit` (automatic inlining), for which the same call completes in 35.4 s. The inlining modes compared through the public API at n_rho = 50 (`stress/inline_sweep.sh`, wall time of `run_simulation` for 2 steps, `results/stress/inline_sweep_n50.log`): dense 28.5 s; structured `XLA_LATE` 31.3 s (through a wrapper with a different stack; it hangs without it), `XLA_EARLY` 47.0 s, `AUTO` 35.3 s, `JAX_EARLY` 38.2 s, no inner jit 37.7 s. The cause is confirmed by the JAX setting that shortens the location metadata: with `XLA_LATE` and `JAX_INCLUDE_FULL_TRACEBACKS_IN_LOCATIONS=false` the same call compiles in 31.3 s, and with the default it was still compiling after 300 s. The dense Jacobian of `jax_root_finding` is still late-inlined upstream and compiled normally in every run here.
2. **Rotation widens the transport stencil (fixed).** With QLKNN `rotation_mode = 'half_radius' | 'full_radius'`, the E x B shearing rate is a face gradient of a face-to-cell average of the face value of the radial electric field, whose poloidal-velocity term is itself a face value of a face-to-cell average: the raw coefficients at face f depend on cells f-3..f+2 (measured: offset -3 up to 2e-2 and +2 up to 1.5e-2 of the largest entry, +3 at 2e-17, -4 exactly zero; `stress/rotation_reach.py`), not f-2..f+1. The period-4 colouring then folded the missing columns onto other entries (row errors 5e-3 to 2e-2), and on `test_iterhybrid_predictor_corrector_rotation` (n_rho = 25, 25 steps) the structured solve needed 733 instead of 268 Newton iterations and 75 instead of 20 time-step cuts, reaching t = 1.43 s instead of 3.34 s. The stencil is now chosen from the static runtime parameters of the transport models - (3, 2) for QLKNN/QuaLiKiz-based models with rotation and TGLF-based models with `use_rotation` (same construction; measured and corrected for TGLF-based models in 3.13), (2, 1) otherwise; the wider stencil makes the assembly ~18% slower at n_rho = 100 (9.2-10.3 vs 7.8-8.7 ms), so it is only used with rotation. After the fix the rotation run is identical to dense (268 Newton iterations, 45 outer iterations, same final time to 1e-12) and steps 1.6x faster (2.25 vs 3.69 s after the first step); the Jacobian matches `jacfwd` to 8e-14 per row at n_rho = 16-50.
3. **State-dependent internal boundary conditions (rejected here, supported exactly in 3.13).** With `evolve_current`, the `MTANH` pedestal profile form sets its targets from psi at the pedestal-top cell and at the separatrix, and the `beta_poloidal_prime` internal boundary conditions from psi at the axis, the separatrix and an interpolated edge point: every pinned row couples to distant psi cells (row errors up to 0.66 and 0.93; the Newton solve took 19 instead of 9 iterations for MTANH at n_rho = 40). Both are now rejected by the validator when the current evolves. Supporting them exactly would need the globals mechanism of the sources extended to the pedestal and internal-boundary-condition models.
4. **A shipped source without the split (fixed).** The experimental gas-puff feedback source (`torax/experimental/gas_puff_feedback_source.py`) sets its total from the line- or volume-averaged density, so every density row depended on every density cell (row error 0.27%, scaling with the feedback gain). It is now split like the others (one global).
5. **A split that could belong to another model (fixed by a redesign).** The split of the cyclotron model was a default of the `Source` class, so a user-registered cyclotron model with its own local model function got the Jacobian of Albajar's model (relative error 6e-4; the assembly was consistent with the wrong physics). Model functions now carry their own split: `source.SplitModelFunction(globals_func, profile_func)` is itself the model function (calling it composes the two), and the builder recognises it by type, so a plain function is local by construction and no config has to attach a split to the source. This also removed the two extra `Source` fields and restored the upstream `build_source` of the configs.
6. **The runtime check (improved).** Under `TORAX_ERRORS_ENABLED` the check compared `J v` with a JVP in the global 2-norm with a fixed tolerance of 1e-6: that norm is dominated by the stiffest rows (row norms span 1 to 1e4), so a 1% error in a unit-norm row at n_rho = 50 passed (the threshold grows ~n^2), and in float32 the tolerance was below round-off (false positive with an exact Jacobian). It is now per row, `|J p - jvp|_i <= sqrt(eps) (|J| p)_i`, with a pseudo-random positive probe (the former smooth probe takes nearly equal values on some pairs of columns, which would hide an error aliased between them). Measured ratio `|J p - jvp|_i / (|J| p)_i` (`stress/check_ratio.py`): at most 2.4e-14 for correct Jacobians (n_rho = 25-100, with and without global sources and rotation, 3 states each) against 8e-4 to 1.3e-3 with the rotation stencil narrowed and 4e-2 to 1.3e-1 for MTANH, i.e. six orders of magnitude of margin on either side of `sqrt(eps) = 1.5e-8`. A new test checks that the error is raised for a narrowed stencil.
7. **Minor.** The uniform-grid test used `np.allclose` (relative tolerance 1e-5) and accepted a grid with 7e-6 relative spacing jitter, on which the Jacobian is off by 2e-8; it now requires uniformity to 1e-9. An unrelated reformatting of `source_profile_builders.py` that had slipped into the change was removed.

Verified to hold, with no change needed: `dG/dx` stays inside its mask in every configuration tested (in fact tridiagonal apart from the near-axis block, so the half-width of 2 is conservative); the near-axis block formula for every n_cells in 4-400 and `min_rho_norm` <= 0.05, including float32 and the cell centres' float neighbours; `S` is independent of the state and equals the smoothing of `calculate_all_transport_coeffs` bitwise; `dG/dc` only uses faces i and i+1; the globals layout and ordering (k = 0 and k = 10); colourings with fewer cells than the period; 54 `test_data` configs and the three examples at n_rho = 16 (row error <= 3.5e-13); `custom_root` gradients, both on toy problems (also with a deliberately wrong custom Jacobian, which only costs iterations) and through a 3-step simulation (gradients equal to 2e-14); one compilation per simulation and none later; no state leaking between two simulations in one process (two ToricNN weight files); error checking in float64 without false positives.

Not caused by this change: TGLFNN-UKAEA with rotation has NaN in `jax.jacfwd` itself at the initial state (in the rows of faces 1 and 2, 3.13), so Newton cannot run with it in either mode; in float32 (`JAX_PRECISION=f32`) TORAX's Newton solve collapses the time step identically in both modes; `jax.grad` through `root_newton_raphson(use_jax_custom_root=False)` aborts inside XLA (`Check failed: has_layout()`) for the dense Jacobian too, and TORAX always uses `custom_root`. The validator was conservative where the sweep showed it over-rejects (an implicit pedestal without a pedestal model, `min_rho_norm` somewhat above 0.05 on coarse grids); 3.13 removed it.

What remains a limitation by design (also after 3.13): the stencils are fixed in code. Physics added later that couples distant cells has to expose the few quantities it couples through (a `SplitModelFunction` for sources, `references` for internal boundary conditions; 3.13), and `TORAX_ERRORS_ENABLED=True` is the way to check a new configuration.

### 3.13 Closing the gaps: every configuration supported (`stress/gaps_compare.py`, `stress/e2e_gaps.py`, `stress/axis_check.py`)

After 3.12 the structured mode still had a validator that rejected an implicit pedestal (`pedestal.explicit_pedestal = False`) and the `ADAPTIVE_TRANSPORT` pedestal mode (with the L-H formation and saturation models), the `MTANH` pedestal profile form and `beta_poloidal_prime` internal boundary conditions with an evolving current, non-uniform radial grids and `numerics.min_rho_norm > 0.05`. All of them are now supported with an exact Jacobian, the validator is gone, and `min_rho_norm` is again an ordinary runtime parameter. Two further caveats are also addressed: the stencil of TGLF-based models with rotation, which 3.12 had inferred rather than measured, is now measured (and was too narrow); and user-defined models, whose locality cannot be checked in advance, are named in a warning.

**Generalised structure.** The residual is now `R(x) = G(x, T(h(x), q(x)), q(x))`, where
- `h` are the raw turbulent coefficients (clipped, before any post-processing), as before;
- `T` is the post-processing of the coefficients, `transport_coefficients_builder.postprocess_turbulent_transport`: the Gaussian smoothing and, in `ADAPTIVE_TRANSPORT` mode, the pedestal's scaling of the coefficients by its transport multipliers, with its clipping and its smoothing around the pedestal top. `calculate_all_transport_coeffs` now calls the transport model without smoothing and applies `T` itself, so the structured and dense paths run the same code;
- `q` are the global quantities of the state, `calc_coeffs.StateGlobals`, evaluated by `calc_coeffs.calc_state_globals` as `calc_coeffs` evaluates them: (i) the globals of the `SplitModelFunction` sources (as before); (ii) the pedestal model output when it depends on the state - all of it for an implicit pedestal (e.g. the pedestal-top temperatures of `set_P_ped_n_ped` depend on the charge states and `Z_eff` at the pedestal-top cell), and the four transport multipliers for `ADAPTIVE_TRANSPORT`, which the formation model (a sigmoid of `P_SOL / P_LH`) and the saturation model recompute from the state in every Newton iteration; (iii) the reference values of internal boundary conditions whose whole profile depends on a few values of the state: for `MTANH`, psi on the axis, at the separatrix and at the pedestal-top cell and `T_i`, `T_e`, `n_e` at the separatrix; for `beta_poloidal_prime`, psi on the axis, at the separatrix and at `rho_norm_edge`, and `n_e`, `B_pol^2` and the total pressure at the separatrix (six scalars each; `InternalBoundaryConditionModel.references`, `PedestalModelOutput.mtanh_references`).

The exact Jacobian is `J = dG/dx + dG/dc . dT/dh . dh/dx + (dG/dq + dG/dc . dT/dq) . dq/dx`. `T` is linearised once; it acts on each coefficient separately, so one pass of its linear map per face gives `dT/dh` for all four coefficients (n_rho + 1 passes of a cheap linear map), and one pass per global gives `dT/dq`. The transport model sees the pedestal only through the location of its top (masks), so `h` is differentiated with `q` frozen; the runtime check would flag a model for which this is not true.

Code changes, besides `structured_jacobian.py`: `StateGlobals` and `calc_state_globals`, the pedestal evaluation of `_calc_coeffs_full` moved into `_evaluate_pedestal` unchanged, and a `state_globals` argument through `calc_coeffs` and `theta_method_block_residual` (replacing `source_globals`); internal-boundary-condition models get `references(...)` (default `None`) and a `references` argument, as do `PedestalModelOutput.to_internal_boundary_conditions` and `PedestalTransitionState.to_internal_boundary_conditions`; `postprocess_turbulent_transport` is split out of `calculate_all_transport_coeffs`, `TransportModel._smooth_coeffs` became public (`smooth_coeffs`) and `PedestalModelOutput.modify_core_transport` became `scale_transport_coeffs`, which scales one set of coefficients, so that the turbulent and the Pereverzev coefficients are scaled separately (same results; its unit test was adapted); `build_source_profiles_and_globals` replaces `build_source_globals`; the `ToraxConfig` validator is removed.

**Stencils without static mesh information.** The first version of this change took the grid spacing and the cells below `min_rho_norm` from the mesh as numpy values. The Jacobian comparisons passed, because they call `jacobian_fn` outside `jax.jit`, but the test through the jitted solve block failed with a `TracerBoolConversionError`: inside a simulation the mesh (`Grid1D.face_centers`) and `min_rho_norm` are traced. The stencils therefore now hold for any grid and any `min_rho_norm`:
- *Transport stencil.* Face gradients are three-point stencils (cells f-1, f and f+1 for face f) whose third weight vanishes only on a uniform grid, so the raw coefficients of face f depend on cells f-2..f+2, or f-3..f+4 with QuaLiKiz-based rotation (`stress/transport_reach.py`, `results/gaps/transport_reach.log`; on a skewed grid, `face_centers = u + 0.1 u (1 - u)`: offset +2 at 5e-4 of the largest entry, and with QLKNN rotation +3 and +4 at 4e-5 and 6e-8; on a uniform grid they are at round-off, 1e-17). This costs 4 more `dh/dx` seeds than the uniform-grid stencil (8 with rotation).
- *Near-axis coupling.* The cells below `min_rho_norm` take the current density of the first cell at or beyond it (`psi_calculations._extrapolate_cell_profile_to_axis`), which depends on psi in that cell, the one before and the two after it. These four psi columns are found at run time and each gets its own colour, instead of a reserved block sized from the static cell count, which is what had limited `min_rho_norm` to 0.05. The current density is the only profile TORAX extrapolates to the axis inside the residual; its other uses are in post-processing and in a state-independent source.
- *TGLF-based models with rotation (the unverified case of 3.12).* 3.12 had given them the QuaLiKiz-based rotation stencil by reading the code, because TGLFNN-UKAEA with rotation has NaN Jacobians (in both modes) and could not be measured. The NaNs are confined to the rows of faces 1 and 2 (an infinite partial derivative near the axis turns every column of those rows into NaN in forward mode), so the stencil can be measured on the other rows: face f depends on cell f-4 at up to 1.1e-5 of the largest entry (2.5e-5 at n_rho = 30) on a uniform grid, and on cell f+5 at 1e-10 on the skewed grid - one cell beyond the QuaLiKiz-based stencil on either side. The narrower stencil would have given a wrong Jacobian had the NaNs been fixed; TGLF-based models with rotation now get (4, 5), and no entry of any finite row lies outside it.

With four channels the assembly uses 24 seeds for `dG/dx` (20 + 4), 20 for `dh/dx` (32 with QuaLiKiz-based rotation, 40 with TGLF-based rotation) and 8 for `dG/dc` - 52 at any n_rho without rotation, against 44 at n_rho = 25-50 and 56 at n_rho = 100 before - plus one JVP and one VJP per global quantity and n_rho + 1 passes of the linearised post-processing.

**User-defined models.** The locality of models that TORAX does not ship cannot be checked in advance. `jacobian_fn` now logs a warning, once per compilation, naming every user-defined source (model function), transport, neoclassical or internal-boundary-condition model it assumes to couple neighbouring cells only, and pointing to `SplitModelFunction` and `TORAX_ERRORS_ENABLED=True`; a source whose model function is a `SplitModelFunction` is not listed, nor is the pedestal model, whose output is a global quantity. None of the 61 shipped configurations that load here triggers it (the other four need QuaLiKiz or command-line flags); a user-registered cyclotron model function does.

**Verification.**

- *Exactness against `jax.jacfwd`* (`stress/gaps_compare.py`, `results/gaps/exactness.log`): the largest entry error relative to its row norm, at the initial state and at 5% and 20% multiplicative noise, for each formerly rejected configuration (Newton solver, `iterhybrid_predictor_corrector` physics unless named):

| configuration | n_rho | worst row error: x_old / 5% / 20% |
|---|---|---|
| reference: no change to the physics | 25 | 1.3e-14 / 2.3e-14 / 2.2e-15 |
| implicit `set_P_ped_n_ped` pedestal, `MTANH` profile, skewed grid, `min_rho_norm = 0.1` | 25 | 1.1e-14 / 2.5e-14 / 4.9e-15 |
| same | 50 | 4.5e-14 / 2.6e-14 / 7.4e-15 |
| `beta_poloidal_prime` internal boundary conditions | 25 | 1.3e-14 / 2.3e-14 / 2.2e-15 |
| same | 50 | 9.0e-14 / 4.7e-14 / 5.3e-15 |
| `test_iterhybrid_lh_transition`: `ADAPTIVE_TRANSPORT`, Martin-scaling formation, profile-value saturation | 25 | 2.4e-15 / 5.1e-15 / 3.5e-16 |
| same | 50 | 4.3e-15 / 1.0e-14 / 6.0e-15 |
| same with an implicit pedestal | 25 | 2.4e-15 / 5.1e-15 / 3.5e-16 |
| `test_prescribed_timedependent_ne` | 25 | 1.4e-14 / 3.1e-15 / 8.8e-16 |
| QLKNN full-radius rotation on the skewed grid, with the implicit `MTANH` pedestal | 25 | 2.0e-14 / 6.4e-14 / 7.9e-15 |
| the three global sources with QLKNN half-radius rotation (circular geometry) | 25 | 1.1e-14 / 3.6e-14 / - (NaN entries in `jax.jacfwd` too) |
| `min_rho_norm = 0.2`, with the ohmic source | 16 | 6.7e-15 / 4.8e-15 / 3.0e-15 |
| `min_rho_norm = 0.5`, with the ohmic source | 10 | 1.3e-15 / 3.5e-15 / 2.8e-15 |
| `min_rho_norm = 1.0` (every cell extrapolated), with the ohmic source | 8 | 8.2e-16 / 9.7e-15 / 5.0e-16 |

  The implicit and explicit L-H cases agree because `set_T_ped_n_ped` gives the same pedestal either way; what is state-dependent there are the transport multipliers, which the formation model recomputes in every Newton iteration (a sigmoid of `P_SOL / P_LH`: 0.99992 in L-mode at the initial state, far enough from 1 for the scaling, and its derivatives, to be active there too).
- *White-box* (`stress/axis_check.py`, `results/gaps/axis_check.log`): with the coefficients and globals frozen, `dG/dx` has no entry outside the band and the four psi columns found at run time, for `min_rho_norm` = 0, 0.2, 0.6 and 1.0 on uniform grids (n_rho = 10-16), 0.3 on the skewed grid and 0.1 on a grid refined towards the axis (`face_centers = u**1.5`, four cells below `min_rho_norm`); `dh/dx` has none outside the (2, 2) stencil; the assembled matrix equals `jax.jacfwd` to 4e-15 (relative Frobenius) in all six cases. The ohmic source, whose heating uses the extrapolated current density, is on in all of them.
- *Unit tests* (`torax/_src/solver/tests/structured_jacobian_test.py`, `results/gaps/unit_tests.log`): the Jacobian equals `jax.jacfwd` row by row to 1e-10 for local physics, the global sources with rotation, the implicit `MTANH` pedestal on the skewed grid with `min_rho_norm = 0.1`, the adaptive pedestal and `beta_poloidal_prime`; the jitted solve block gives the same Newton iterations and solution (rtol 1e-8) as the dense mode, with the per-iteration check on, for the global sources and for the implicit pedestal on the skewed grid; the check raises for a narrowed stencil. The unit suites of every package the change touches pass.
- *Whole simulations* (`stress/e2e_gaps.py`, `results/gaps/e2e.jsonl`): `torax.run_simulation` in both modes, one process per run, the structured runs with `TORAX_ERRORS_ENABLED=True`, so that every Newton iteration of the jitted solver checks each row of the assembled Jacobian against a JVP of the full residual (none raised):

| configuration | n_rho | simulated | steps, dense / structured | Newton iterations | final `T_i`, `T_e`, `n_e`, `psi`: max rel. diff |
|---|---|---|---|---|---|
| reference | 25 | 2 s | 10 / 10 | 50 / 50 | 1.7e-14 |
| implicit `MTANH` pedestal, skewed grid, `min_rho_norm = 0.1` | 25 | 2 s | 11 / 11 | 47 / 47 | 2.0e-14 |
| `beta_poloidal_prime` | 25 | 2 s | 33 / 33 | 69 / 69 | 4.3e-14 |
| three global sources, QLKNN rotation | 25 | 2 s | 17 / 17 | 177 / 177 | 1.7e-13 |
| QLKNN rotation on the skewed grid, implicit `MTANH` pedestal | 25 | 2 s | 9 / 9 | 72 / 72 | 3.8e-14 |
| `min_rho_norm = 0.2`, ohmic source | 16 | 2 s | 9 / 9 | 53 / 53 | 3.3e-14 |
| `min_rho_norm = 0.5`, ohmic source | 10 | 2 s | 5 / 5 | 23 / 23 | 9.5e-15 |
| `test_prescribed_timedependent_ne` | 25 | 10 s | 23 / 23 | 85 / 85 | 1.3e-14 |
| `test_iterhybrid_lh_transition` | 25 | 60 s | 945 / 945 | 5850 / 5850 | 3.6e-13 |
| same with an implicit pedestal | 25 | 60 s | 945 / 945 | 5850 / 5850 | 1.4e-12 |

  No step was accepted at the coarse tolerance in either mode. In the L-H runs the plasma dithers between L- and H-mode (hence the 945 steps for 60 s at dt <= 1 s), and the pedestal's multipliers pull `chi_e` in the pedestal region from its L-mode value of 1.0 to below 0.99 in 344 of the 945 steps; the structured runs follow the dense ones step for step.
- *Shipped configurations* (`stress/shipped_sweep.py`, `results/gaps/shipped_sweep.log`): every `tests/test_data` config and four examples, forced to the Newton solver at n_rho = 16 and compared with `jax.jacfwd` at the initial state and at 5% noise. The 60 that run give a worst row error of 7.5e-13 (`test_iterhybrid_predictor_corrector_Lmode_combined`; all others 2.7e-14 or less). Non-finite entries appear only where `jax.jacfwd` has them too (TGLFNN-UKAEA with rotation; QLKNN rotation at 5% noise). The QuaLiKiz and TGLF configs call external codes without derivatives, so Newton cannot run with them in either mode, and `test_iterhybrid_rampup_restart` needs command-line flags.
- *Regression*: the full `sim_test` suite (63 tests) passes on the final code (`results/gaps/sim_test_chunks.log`), so the restructured post-processing of the transport coefficients leaves the reference outputs of the default mode unchanged.

**Timing: first and repeated runs.** The driver of 3.10 (`profile_runs/driver_first_vs_repeat.sh`) was rerun on the final code (`results/profile_runs/first_vs_repeat_v2.jsonl`): one fresh process per case and mode, one after another on the otherwise idle machine, each calling `ToraxConfig.from_dict` + `torax.run_simulation` three times; "repeat" is the mean of calls 2 and 3, and every first call makes the same 53 XLA compilations (60 with the global sources) as before. The machine ran 19-25% slower in this session than for the table of 3.10 (dense repeats 10.7 vs 8.9 s and 29.5 vs 24.1 s, as for the previous commit timed in this session), so the previous commit was timed again in the same session with the same driver (last column, `results/profile_runs/first_vs_repeat_previous_commit.jsonl`):

| case | n_rho | steps | dense first | structured first | dense repeat | structured repeat | repeat speedup | previous commit: structured first / repeat |
|---|---|---|---|---|---|---|---|---|
| `iterhybrid_rampup` | 50 | 40 | 52.2 s | 55.4 s | 10.68 s | 4.42 s | 2.4x | 53.7 s / 3.97 s |
| `iterhybrid_rampup` | 100 | 40 | 69.0 s | 60.3 s | 29.51 s | 8.02 s | 3.7x | 56.1 s / 8.03 s |
| + cyclotron, constant-fraction radiation, ToricNN ICRH | 50 | 40 | 62.9 s | 74.4 s | 12.99 s | 5.70 s | 2.3x | 71.4 s / 5.53 s |
| + cyclotron, constant-fraction radiation, ToricNN ICRH | 100 | 40 | 75.0 s | 77.7 s | 27.99 s | 8.61 s | 3.3x | 73.5 s / 8.57 s |
| `iterhybrid_predictor_corrector`, Newton, t_final = 1 s | 25 | 5 | 38.2 s | 46.6 s | 0.64 s | 0.53 s | 1.2x | 47.0 s / 0.49 s |
| `iterhybrid_predictor_corrector`, Newton, t_final = 1 s | 50 | 13 | 41.1 s | 48.5 s | 4.28 s | 1.38 s | 3.1x | 48.3 s / 1.41 s |
| `iterhybrid_predictor_corrector`, Newton, t_final = 1 s | 100 | 49 | 63.7 s | 52.8 s | 26.71 s | 6.93 s | 3.9x | 52.1 s / 6.89 s |

- The structured mode still makes repeated calls 2.3-3.1x faster than dense at n_rho = 50, 3.3-3.9x at n_rho = 100 and 1.2x at n_rho = 25. At n_rho = 100 its first call is also shorter than dense's (60.3 vs 69.0 s and 52.8 vs 63.7 s; with the global sources 77.7 vs 75.0 s); at n_rho <= 50 it is 3-12 s longer.
- Against the previous commit, in the same session: at n_rho = 100 the repeats are unchanged (8.02 vs 8.03 s, 8.61 vs 8.57 s, 6.93 vs 6.89 s), since the new assembly has 52 seeds instead of 56. At n_rho = 50 it has 52 instead of 44, which costs about 5%: three interleaved pairs of `iterhybrid_rampup` processes gave 4.56 vs 4.36 s on average (`results/gaps/rampup50_interleaved.log`; the single pair in the table, 4.42 vs 3.97 s, overstates it), and the assembly alone takes 8.7 vs 8.2 ms at n_rho = 50 and 14.6 vs 15.5 ms at n_rho = 100 (medians of four interleaved pairs, `results/gaps/assembly_interleaved.log`). XLA compile time grew by 1-4 s (e.g. 42.4 vs 38.7 s and 57.0 vs 52.9 s for the two `iterhybrid_rampup` cases at n_rho = 100), within the spread of whole first calls (the three interleaved pairs: 55.2 vs 54.9 s on average).

## 4. PETSc and SUNDIALS design choices, and whether they translate to TORAX

Sources: PETSc `main` (`d3fce5f`), SUNDIALS v7.9.0; file:line citations are to those trees (full subagent reports: `report_petsc.md`, `report_sundials.md`). Verdicts use the measurements of section 3. "R" = one residual evaluation, "J" = one dense Jacobian (41 R at N = 200).

### 4.1 Jacobian formation

| Design choice | PETSc / SUNDIALS | Translates? | Evidence |
|---|---|---|---|
| **Compressed (coloured) Jacobian**: perturb all columns of one colour at once; `ncolors = nc(2s+1)` for a 1-D stencil (PETSc `src/dm/impls/da/fdda.c:430-447`, `MatFDColoring`; CVODE `cvLsBandDQJac` needs `mupper+mlower+1` evaluations, `src/cvode/cvode_ls.c:1192-1253`) | In JAX this is exact: seed `jax.vmap(jax.jvp)` with `ncolors` summed unit vectors, scatter into the band. | **Yes, with a caveat**: TORAX's Jacobian is block-tridiagonal *except* for the transport-coefficient smoothing, which widens the band to ~12-23 cells (86/134 colours instead of 14). Both libraries warn that global couplings break colouring (PETSc `SNESSetPicard` docs; CVODE's CPR grouping aliases far columns). The exact fix is the structured chain rule through the smoothing matrix (3.3 item 3): ~36 JVP columns for any grid. | 3.2 (batched-JVP cost curve), 3.3 (colour counts) |
| **Jacobian lagging / modified Newton**: J reused up to 51 steps, matrix refactored every 20 steps or when gamma drifts > 30% (CVODE `cvode_ls_impl.h:43`, `cvode_impl.h:66-68`); PETSc `-snes_lag_jacobian`, `-snes_lag_jacobian_persists` (`src/snes/interface/snes.c:3044-3123`) | Carry `J`/LU in the loop state; recompute on a "bad J" signal. | **Partially.** With QLKNN the Jacobian changes 40-70% between predictor and solution even at dt = 0.1 s (chord factors 0.4-0.7), and a frozen J needs 5-8 iterations instead of 3 at dt <= 0.5 s and 21-29 at dt = 2 s. Because J = 41 R, chord Newton is still 2.5-3x cheaper per step at dt <= 0.5 s but loses at the large steps used in ramp-up scenarios. The CVODE safeguards (`crate` convergence-rate test, divergence test `||delta_m|| > 2||delta_{m-1}||`, refresh J before shrinking dt) are mandatory. "Jacobian every 2-3 iterations" is the robust version: 2 J instead of 3-4 J per step. | 3.3 (chord factors), 3.4 (iteration table) |
| **Matrix-free Newton-Krylov** with a cheap preconditioner (`-snes_mf_operator`, `MatMFFD` `src/mat/impls/mffd/mffd.c:303-370`; CVODE SPGMR + `CVBandPre`, tolerance `eplifac * nlscoef`) | Exact JVPs replace differencing; the Picard block-tridiagonal matrix is the preconditioner. | **Yes.** 15-30 GMRES iterations per Newton direction, independent of n_rho and dt; 13.4 ms vs 27.7 ms per direction at N = 200, and the advantage grows with N because `jacfwd` scales with N while the Krylov count does not. Compile time also drops (no N-wide batch). SUNDIALS' `eplifac = 0.05` and PETSc's Eisenstat-Walker forcing are directly usable. Do not add Pereverzev terms to the preconditioner (2-3x more GMRES iterations). | 3.2, 3.3, 3.4 |
| **Banded / block LU instead of dense** (PETSc `PCLU` on `SeqBAIJ`; SUNDIALS `SUNLinSol_Band`) | TORAX already has `thomas_solve` for the linear path. | **No (not at this N).** Dense LU + solve is 0.9 ms of a 28.5 ms iteration at N = 200 and ~4 ms at N = 400; not a bottleneck below N ~ 2000. | 3.2 |
| **Reverse-mode Jacobian** (not a PETSc/SUNDIALS idea) | `jax.jacrev` instead of `jax.jacfwd` | **Marginal**: 1.27x at N = 200, 1.06x at N = 400, slower at N = 100 on the idle machine. | 3.2 |

### 4.2 Nonlinear solver policy

| Design choice | PETSc / SUNDIALS | Translates? | Evidence |
|---|---|---|---|
| **Stop on the update norm in WRMS units, tied to the error tolerance**: CVODE converges when `min(1, crate) * ||delta||_WRMS <= 0.1 / tq2` (`cvode_nls.c:373-378`), i.e. the nonlinear error is 10% of the local-error tolerance; ARKODE `nlscoef = 0.1`; max 3 iterations. | Replace `mean|R| < 1e-5` (absolute, unit-mixing) with per-channel `rtol/atol` weights and a tolerance derived from the time-integration accuracy. | **Yes.** TORAX over-solves: after 2 iterations the iterate is within 0.2% of the step change, after 3 within 1e-5, while the BE temporal error is ~1-10% at these dt. Stopping one iteration earlier saves 25-33% of the Newton cost with no visible accuracy change. Needs an error estimate to define "accurate enough" (4.3). | 3.4 (distance-to-solution), 3.5 |
| **Never accept an unconverged solve; cut dt by 0.25 and refresh J instead** (CVODE `ETACF = 0.25`, `MXNCF = 10`; PETSc `TSAdaptCheckStage` with `scale_solve_failed = 0.25`; PETSc has no "coarse tolerance" concept) | TORAX accepts `mean|R| < 1e-2` with a warning and halves dt by `dt_reduction_factor` only after 30 iterations or `tau < 0.01`. | **Yes.** The dt = 0.02 s case shows the "coarse" acceptance path in action (error state 2). SUNDIALS' escalation order (retry with fresh J, then shrink dt) is cheaper than 30 iterations of a stalled line search. | 3.4 |
| **Line search: Armijo with quadratic/cubic interpolation, NaN pre-loop, `stol` step test** (PETSc `bt`, `src/snes/linesearch/impls/bt/linesearchbt.c:57-346`; KINSOL adds the curvature condition) | Closed-form scalar updates inside the existing `while_loop`. | **Low impact.** Backtracking happened in 3 of 109 Newton iterations in the ramp-up run; interpolation would save a handful of residual evaluations per run. The `stol`/small-step exit and the divergence test (`divtol`) are the useful parts. SUNDIALS' integrators use no line search at all - they shrink dt. | 3.1 |
| **Predictor from history** (CVODE Nordsieck extrapolation `cvode.c:2779-2803`; PETSc BDF `-ts_bdf_extrapolate`; ARKODE default *trivial* predictor for stiff problems `arkode_arkstep_io.c:758`) | One vector of history in the carry. | **Mixed.** Linear extrapolation is *worse* than `x_old` at dt >= 0.5 s (QLKNN dynamics are not smooth at that scale) and only marginally better at dt = 0.02 s; ARKODE's advice (trivial predictor for stiff/positivity-sensitive problems) is the right default. TORAX's 10-sweep predictor-corrector guess lands only 10-25% closer to the solution than `x_old` at dt <= 0.5 s (0.75-0.9 of the step change away from `x*`), yet it saves about one Newton iteration per step and is a net win end-to-end (43 vs 61 ms per step at N = 100, 3.1b). | 3.4 |
| **Anderson / NGMRES acceleration of fixed-point iteration** (`SUNNonlinSol_FixedPoint`, QR-updated LS, `sunnonlinsol_fixedpoint.c:545-737`; PETSc `SNESANDERSON`/`SNESNGMRES`, `m = 30` default; recommended `m = 3-5` for Picard) | Fixed-shape ring buffer in the `while_loop` carry; applies to TORAX's `linear` solver. | **No gain measured.** The Pereverzev-stabilised Picard map is not contractive on QLKNN (it oscillates 0.2-1 step-changes away from the implicit solution); Anderson depth 2-4 does not change that. Both libraries scope Anderson to linearly convergent maps. | 3.6 |
| **Positivity constraints** (`CVodeSetConstraints`: clamp if the violation is below the Newton tolerance, else shrink dt by `0.9 * min-quotient`, `cvode.c:3197-3320`; PETSc `SNESVINEWTONRSLS`, `TSSetFunctionDomainError`) | Vector mask + scalar dt factor. | **Robustness, not speed.** TORAX relies on NaNs to trigger backtracking; an explicit domain predicate is cheaper and clearer. | - |
| **Newton at most 3-4 iterations per step** (CVODE 3, IDA 4, ARKODE 3) | `n_max_iterations = 30` today | **Yes, once dt is error-controlled**: with a good predictor and an accuracy-tied tolerance, 2-3 iterations are the norm (measured 2.1-3.5 at dt <= 0.1 s); a step needing more is a step that should be shortened. | 3.5 |

### 4.3 Time integration and step control

| Design choice | PETSc / SUNDIALS | Translates? | Evidence |
|---|---|---|---|
| **Local error estimate at zero cost**: CVODE `dsm = tq2 * ||y_corr - y_pred||_WRMS` with `tq2 = 1/2` for BDF1 (`cvode.c:3458`, `:3061-3063`); PETSc `TSEvaluateWLTE_Theta` uses the last three solutions (`theta.c:681-715`). | One extra vector in the carry; WRMS norm with per-channel `atol/rtol`. | **Yes.** TORAX has no error estimate at all; the chi heuristic does not track error (fixed dt with the same step count gives the same error). Measured: an elementary LTE controller on BE matches fixed-dt accuracy at ~1.5x the Newton work on this dissipative problem (it resolves the initial transient), so the payoff is automatic, robust dt selection and the enabling of BDF2 - not fewer steps by itself. | 3.5 |
| **Step-size controller**: `eta = 1/((6 dsm)^(1/(q+1)) + 1e-6)`, keep-step band [1, 1.5), growth <= 10, shrink >= 0.1 (CVODE `cvode.c:3631-3696`); PETSc basic `h * clip(0.9 err^(-1/order), 0.1, 10)` (`adaptbasic.c:59-67`); Soderlind PI/PID filters (ARKODE) | Scalars in the carry. | **Yes.** The keep-step band is important when the Jacobian/LU is reused across steps (avoids refactorisation for tiny dt changes). TORAX's only controller today is "divide by 3 on failure". | 3.5 |
| **Second order, L-stable**: variable-step BDF2 (CVODE q = 2, PETSc `TSBDF` default order 2), TR-BDF2 / ESDIRK (ARKODE `GKC21_ESDIRK_3_1_2`, `TRBDF2_3_3_2`); the repo guidance says cap BDF at 2 for parabolic problems. | BDF2 needs `x_{n-1}`, `dt_{n-1}` in the carry and an exact transient-term treatment; ESDIRK needs 2 implicit stages of the existing form. | **Yes.** BDF2 reaches BE's accuracy at 4-8x larger dt in the prototype; Crank-Nicolson (the one second-order option TORAX already has, `theta_implicit = 0.5`) is unusable (20-45% errors) because it is not L-stable. | 3.5 |
| **Rosenbrock-W** (`TSROSW`: one Jacobian + s linear solves per step, no Newton loop, tolerates approximate Jacobians; default `ra34pw2`, `rosw.c:1157-1180`) | Fixed-shape by construction (no data-dependent iteration). | **Promising but unproven here.** It removes the Newton loop entirely (1 J + 3-4 solves + 3-4 R per step) and its W-property would let the frozen-smoothing or lagged Jacobian be used without losing order. Risk: no globalisation; QLKNN threshold crossings become error-estimate rejections. Needs a prototype; PETSc's own docs steer strongly nonlinear stiff parts to ARKIMEX/Newton. | 3.3, 3.4 |
| **Pseudo-transient continuation** (`TSPSEUDO`, SER `dt_n = 1.1 dt_{n-1} ||F_{n-1}||/||F_n||`) | One scalar of state. | **Yes for steady-state searches** (a common TORAX use); not a transient-speed item. | - |
| **IMEX / multirate / super-time-stepping** (`TSARKIMEX`, `MRIStep`, LSRKStep RKC/RKL, ExtSTS) | - | **No.** Everything stiff in TORAX is the diffusion itself; RKC would need `s ~ sqrt(1.5 h |lambda|)` = 40-125 transport evaluations per step; psi is cheap to keep implicit. | report_sundials.md section 6 |
| **Event location** (`TSEvent`, `CVodeRootInit`) | Bounded secant on the linear interpolant, then reset the controller/history. | **Robustness/accuracy for sawteeth and L-H transitions**, not speed. | - |
| **DAE handling** (IDA: mask algebraic rows out of the error norm, `cj`-ratio Jacobian reuse [0.6, 1.67], `IDACalcIC`) | Row-replaced (pedestal/prescribed) rows are algebraic. | **Yes when an error estimate is added**: algebraic rows must be excluded from the error norm or they dominate it. | - |

### 4.4 Things TORAX does that the libraries do not, and that the measurements support keeping

- **Exact autodiff Jacobians/JVPs** instead of finite differences: SUNDIALS' DQ increments would be unusable with a piecewise-defined NN surrogate; JAX's exactness is why JFNK and colouring are cleaner here than in C.
- **Backtracking line search inside the step**: SUNDIALS deliberately shrinks dt instead; in TORAX the line search is rarely triggered, so keeping it is cheap. When it *is* triggered repeatedly (dt = 0.02 s stall), the SUNDIALS response (fresh J, then dt * 0.25) is the better escalation.
- **Pereverzev-Corrigan stabilisation** of the Picard solver has no analogue in either library; the measurements confirm it is what keeps the frozen-coefficient iteration from diverging (`rho(I - P^-1 J)` = 3-50), but also that it should not be put in a Newton preconditioner.

## 5. Recommendations, expected impact, effort

Impact estimates are for the Newton solver on QLKNN cases at n_rho = 50-100 (N = 200-400), where one Newton iteration is 28 / 120 ms in the micro-benchmark (54 / 169 ms end-to-end) and the Jacobian is 94-97% of it. Effort: S = days, M = weeks, L = a redesign of a component.

| # | Change | Expected effect | Evidence | Effort |
|---|---|---|---|---|
| 1 | **Structured exact Jacobian**: expose the pre-smoothing transport coefficients and assemble `J = dG/dx + (dG/dc) S (dh/dx)` from ~36 coloured batched JVPs (plus block-tridiagonal + low-rank assembly). Fall back to plain colouring (14 colours) when smoothing is off. | **Measured with the prototype (3.9)**: Jacobian 23.5 -> 3.7 ms (N = 200), 91 -> 8.7 ms (N = 400), 413 -> 23 ms (N = 800); Newton solve 5.3x / 8.6x / 6.0x faster with identical iterations and solutions; ~4 s more compile in the prototype. **Done (3.10-3.12)**: `solver.jacobian_mode = 'structured'`; end-to-end stepping 2.8-3.3x faster at n_rho = 50 and 4.6-4.8x at n_rho = 100, for 5-8 s more compile; globally coupled sources handled exactly by a rank-k term (2.5x / 3.6x with all three enabled, +15 s compile). | 3.9, 3.10, 3.11, 7.3 | done |
| 2 | **Jacobian-free Newton-Krylov** as the alternative or interim: `jax.scipy.sparse.linalg.gmres` on `jax.jvp`, preconditioned by the existing block-tridiagonal Picard matrix (Thomas/LU), Eisenstat-Walker forcing (`eta` from 0.1 down), no Pereverzev terms in the preconditioner. | Newton direction 26.9 -> 12.7 ms (N = 200), 118 -> 27 ms (N = 400): **2.1x / 4.3x**, unchanged iteration count; no change to physics code; the GMRES count (15-30) is grid-independent; not useful below N ~ 150. | 3.2, 3.3, 3.4 | S-M |
| 3 | **`jax.jacrev` instead of `jax.jacfwd`** (one line). | Marginal on the idle machine (1.27x at N = 200, 1.06x at N = 400, slower at N = 100); low priority. | 3.2 | S |
| 4 | **Stop Newton when it is as accurate as the time step**: WRMS norm with per-channel `rtol/atol`, convergence on the update with CVODE's `crate` estimate, tolerance = 0.1 x local error tolerance; add the divergence test (`||delta_m|| > 2 ||delta_{m-1}||`) and drop the coarse-tolerance acceptance in favour of "refresh J, then dt x 0.25". | 1 of 3-4 Newton iterations per step is spent going from 1e-5 to 1e-10 while the temporal error is 1e-2..1e-1: **-25-33% Newton cost** and a cleaner failure path. | 3.4 (distance-to-solution), 3.5 | S (norm/tolerance), M (with 5) |
| 5 | **Free local error estimate + controller**: `LTE = 0.5 ||x_{n+1} - x_pred||_WRMS` (predictor = explicit Euler or last-step extrapolation, CVODE `tq2`), accept if <= 1, `dt_new = dt / ((6 dsm)^(1/2) + 1e-6)` with keep-band [1, 1.5), growth <= 10, shrink >= 0.1, dt x 0.25 on Newton failure; exclude row-replaced (algebraic) entries from the norm (IDA). Replace the chi heuristic as the default. | Step count matched to the requested accuracy instead of a stability heuristic whose dt varies 25x within a run; enables 4 and the Jacobian-reuse logic in 6. Measured caveat: on this dissipative problem an LTE controller alone does not cut steps at equal final error (it resolves the initial transient); the win is robustness plus what it enables (4, 6, 7). | 3.5 | M |
| 6 | **Jacobian reuse with safeguards** (CVODE `msbp/msbj/dgmax` logic in the carry): reuse J/LU across Newton iterations and across steps while `crate` is good and dt changed < 30%; refresh on divergence, on a failed step, or after k steps. Cheapest variant: recompute J every 2-3 Newton iterations. | 2 J instead of 3-4 J per step at dt >= 0.1 s (-30-50% Newton cost); chord Newton 2.5-3x cheaper per step at dt <= 0.5 s but 5-7x more iterations at dt = 2 s, hence "with safeguards". Multiplies with 1-3. | 3.3 (chord factors), 3.4 | M |
| 7 | **Second-order L-stable stepping**: variable-step BDF2 (exact transient term, BE start-up step, history in the carry) or TR-BDF2/ESDIRK; keep `theta = 1` as fallback; never recommend `theta = 0.5`. | Same accuracy at 4-8x larger dt in the prototype; per-step Newton cost unchanged. Combined with 5 this is the largest lever for long simulations. | 3.5 | M-L |
| 8 | **Keep the predictor-corrector initial guess** (it is a net win: 43 vs 61 ms per step at N = 100 because it saves ~1 Newton iteration per step), but re-check its sweep count once 1-6 make a Newton iteration cheap: at dt <= 0.5 s the guess is only 10-25% closer to the solution than `x_old`, so 3-5 sweeps may retain most of the benefit. | 3.1b, 3.4 | S |
| 9 | Keep `vmap_linesearch` off, or cap `max_linesearch_steps` at ~6 when on. | Avoids 1.5x slower iterations and +2.5 s compile. | 3.1, 3.7 | S |
| 10 | **Rosenbrock-W prototype** (`ra34pw2` / `2m`): one structured Jacobian per step, 3-4 linear solves, embedded error estimate, no Newton loop. | Potentially the most XLA-friendly stepper (no data-dependent loops); untested on QLKNN threshold crossings. | 4.3 | L |
| 11 | Reduce fixed per-step overhead outside the solve (orchestration around the Newton call: ~35-60 ms per step at n_rho = 25-50, i.e. 50-80 residual evaluations). | Becomes the floor once 1-6 are in (Newton solve -> ~30 ms/step); already comparable to the Picard sweeps for the linear solver. | 3.2 | M |
| 12 | **Make the residual continuous inside a Newton solve**: replace the `DV_effective` `jnp.where` sign switch (`quasilinear_transport_model.py:463-482`) by a smooth blend, or evaluate its mask once at the predictor and hold it fixed during the iterations (frozen active set). | Removes the Newton stalls at small dt (2 iterations instead of 4 + 12 backtracks and a coarse-tolerance acceptance); the same treatment applies to the other state-dependent `jnp.where` switches listed in `reports/report_torax_structure.md` section 3. | 3.4 (stall diagnostic and variant test) | S |
| 13 | For the **linear solver**: document that its per-step answer is 0.2-1 step-changes away from the implicit solution (measured 3-11% profile deviations from the converged implicit solution with 0-1 corrector sweeps and ~1% with 10-30 sweeps at dt = 0.05 s); do not spend effort on Anderson acceleration (no gain measured); use it as a predictor for Newton, not as a stand-alone integrator for accuracy-sensitive runs. | - | 3.6 | - |

What does not transfer (do not spend time on): banded/block LU replacing dense LU at N <= 800 (LU is 2% of an iteration), polynomial line searches (backtracking is 3% of iterations), super-time-stepping/ExtSTS and multirate (stiffness *is* the diffusion; 40-125 transport evaluations per step), Crank-Nicolson, finite-difference Jacobian machinery (JAX JVPs are exact), Anderson on the Pereverzev-Picard map.

Combined picture for a QLKNN run at n_rho = 100 with today's Newton solver (2.6 iterations x 169 ms ~ 0.36-0.44 s per step end-to-end): items 1 + 4 + 6 give ~2 iterations x ~12 ms ~ 25 ms of solve per step (~15x on the solve, ~5x on the full step including the fixed overhead), before any reduction in the number of steps from 5 + 7.

## 6. Reproduction

Environment: `uv venv --python 3.12 venv && uv pip install -e /home/user/torax` (JAX 0.11.2 CPU). All scripts take the TORAX example configs as-is and do not modify the repository.

```
cd solver_study/bench
python jac_structure.py 50 on            # n_rho, smoothing on/off, [dt], [warm steps]; writes jac_structure_*.json
python jac_structure.py 100 on
python jac_structure.py 50 off
python jac_structure.py 50 on 0.1
python timings.py 50                     # component timings; writes timings_n50_dt2.json
python newton_variants.py 50 2,0.5,0.1,0.02
python time_accuracy.py 1.0 0.2,0.1,0.05,0.025,0.0125 0.0015625 be,cn,bdf2
python adaptive_dt.py 1.0 0.0015625 3e-2,1e-2,3e-3,1e-3
python picard_anderson.py rampup 2,0.5,0.1,0.02
python stall_diag.py 0.02,0.05,0.2
cd ../profile_runs && bash driver.sh     # end-to-end profiles (one fresh process per configuration)
python structured_jacobian.py 100        # (in bench/) prototype of the structured Jacobian, section 3.9
python compile_breakdown.py 50 [globals]  # (in bench/) compile time of each Jacobian factor, section 3.11
TORAX_ERRORS_ENABLED=True python verify_structured_mode.py rampup 100   # production jacobian_mode='structured' vs 'dense', section 3.10
python dummy_toric_nn.py dummy_toric_nn.json && TORAX_ERRORS_ENABLED=True python verify_structured_mode.py rampup 100 globals   # same with the globally coupled sources
cd ../profile_runs && bash driver_structured.sh && bash driver_structured_globals.sh && python summarize_structured.py results   # end-to-end table of 3.10
TAG=x bash driver_compile.sh; bash driver_xla_flags.sh   # (in profile_runs/) trace/XLA split and XLA-level sweep of 3.11
OUT=first_vs_repeat.jsonl bash driver_first_vs_repeat.sh   # (in profile_runs/) first vs repeated run_simulation calls, 3.10 and 3.13
cd ../stress
python gaps_compare.py implicit_mtanh_grid 25   # structured vs jax.jacfwd for a formerly unsupported configuration, 3.13
python axis_check.py skewed 16 0.3              # white-box check of the stencils near the axis, 3.13
python transport_reach.py tglfnn_rotation skewed 16   # measured stencil of the raw transport coefficients, 3.13
python e2e_gaps.py lh_transition 25 60 dense e2e.jsonl && TORAX_ERRORS_ENABLED=True python e2e_gaps.py lh_transition 25 60 structured e2e.jsonl && python e2e_compare.py e2e.jsonl   # whole simulations, 3.13
python shipped_sweep.py torax.tests.test_data.test_iterhybrid_rampup 16   # one shipped config forced to Newton, 3.12-3.13
```

`bench/bench_common.py` builds a `Problem` at a mid-simulation state (runs `n_warm_steps` real steps, then exposes the exact `theta_method_block_residual` closure that `newton_raphson_solve_block` would use, the linear predictor-corrector initial guess, and the Picard block-tridiagonal matrix reordered to the residual's channel-major layout; the reconstruction is checked to 5e-13 against the residual).


## 7. Addendum: priorities by grid size

### 7.1 Priorities at n_rho = 100 (N = 400), the size of interest

Per Newton iteration inside the loop: 167-170 ms, of which the dense Jacobian is ~100 ms when run standalone and ~165 ms as fused into the loop body (item 6 of 3.2); dense solve 2.3 ms, residual 1 ms. Per step end-to-end: 359 ms with 2.6 iterations (chi dt), 864 ms with 5 (dt = 2 s). Compile: 21 s = 60 steps.

1. **Shrink or remove the Jacobian batch** - the structured/coloured Jacobian (34 seeds, 8.7 ms, exact to 1e-14; measured Newton solve 779 -> 91 ms, section 3.9) or JFNK with the Picard preconditioner (27 ms per direction). Either turns a ~170 ms iteration into ~18-35 ms and removes the in-loop parallelisation loss. This is 70-80% of the step at this size. **Now available as `solver.jacobian_mode = 'structured'`** (3.10): 160 -> 34-35 ms per Newton iteration end-to-end, 341-490 -> 77-111 ms per step, i.e. 4.6-4.8x on the stepping time at n_rho = 100, for 5-6 s more compile time (3.11, 3.12).
2. **Fewer Newton iterations per step**: accuracy-tied stopping (-1 iteration of 3-5), `DV_effective` continuity fix (no stalls), Jacobian reuse every 2-3 iterations (2 J instead of 3-5). Each is 10-30% of the Newton cost and they compound with 1.
3. **Fewer steps**: BDF2 (4-8x larger dt at equal accuracy) with a step-size controller; N-independent and multiplicative with 1-2.
4. **Compile time** (21 s) and the per-step orchestration overhead (~20-35 ms): secondary at this size unless runs are short.
5. Not worth it at N = 400: banded LU (2.3 ms), `jacrev` (1.06x), Crank-Nicolson, Anderson on the Picard mode.

### 7.2 Priorities if N never exceeds ~100 (n_rho = 25, four channels)

Measured at N = 100 on the idle machine (`bench/overhead_n25.py`, `bench/overhead_n25b.py`, `iterhybrid_predictor_corrector` physics, dt = 0.42 s, a step with 4 Newton iterations):

| item | ms | share of the Newton step |
|---|---|---|
| one residual | 0.5 | - |
| one dense `jacfwd` (100 columns) | 4.7 | - |
| one Newton iteration (Jacobian + solve + residual + line search) | ~5.9 | 12% |
| `root_newton_raphson` (4 iterations) | 23.5 | 48% |
| `newton_raphson_solve_block` (predictor-corrector guess + Newton) | 25.3 | 51% |
| `pre_step` + `finalize_outputs` + dt calculator | 3.8 | 8% |
| unattributed step orchestration (present only on the Newton path) | 15-21 | 30-40% |
| **full Newton step** (adaptive-dt path / fixed-dt path) | **49 / 44** | 100% |
| full linear step on the same state (2 Picard sweeps) | 9.5 | - |
| first-step compile, Newton / linear | 19 s / 12 s | = 400 / 1300 steps |

What this changes relative to the N = 200-800 picture:

1. **The Jacobian is no longer the bottleneck**: 4 x 4.7 ms = 19 ms of a 49 ms step (~40%), not 94-97%. Structured/coloured Jacobians would save ~10 ms per step (36 columns cost ~2.5 ms instead of 4.7); JFNK is *slower* than the dense Jacobian at this size (8.9 vs 5.4 ms per direction) and drops off the list; `jacrev` is slower too; dense LU is 0.14 ms.
2. **Compile time is the dominant cost for anything shorter than a few hundred steps**: 19 s of compile equals 400 Newton steps or 2000 linear steps. Reducing the Newton graph (a 36-column Jacobian instead of 100, no `vmap_linesearch`, fewer Picard sweeps in the guess) and reusing compiled executables across runs (persistent compilation cache, avoiding retraces when only runtime parameters change) matter more than any per-step algorithmic change.
3. **Per-step fixed costs are half of a Newton step**: 15-21 ms of the step is orchestration that the linear path does not pay (its whole step is 9.5 ms), plus 4 ms of pre/finalize. This is the top unknown and deserves an XLA profile of `SimulationStepFn` (candidates: the `whilei_loop`/`custom_root` nesting, output pytree assembly, copies of the large carried state through the adaptive-step loop). Fixing it is worth as much as removing the Jacobian entirely.
4. **Fewer Newton iterations per step is the cheapest algorithmic win**: an accuracy-tied stopping test (one iteration less, ~6 ms) and the `DV_effective` continuity fix (avoids stalls of 4 iterations plus 12 line-search residuals and coarse-tolerance acceptances) each buy 10-15% of the step. Jacobian reuse every 2-3 iterations buys another ~5-10 ms. These are small code changes.
5. **Fewer steps is the lever for long simulations**: per-step cost has a hard floor of ~25-30 ms (overhead + guess + one iteration) at this size, so BDF2 (same accuracy at 4-8x larger dt) and a controller that lets dt grow where the physics is slow are what change the total for 1000-step runs; they are N-independent.
6. **The linear-versus-Newton trade-off shifts**: at N = 100 the Newton step is 5x the linear step (49 vs 9.5 ms), not 100x, and the linear solver's answer is 3-11% off the implicit solution. Newton with the fixes above (~25 ms per step) becomes the sensible default whenever profile accuracy matters; the linear solver remains the choice for compile-bound short runs and scans.

Revised ranking for N <= 100: (i) compile time and the step-orchestration overhead; (ii) accuracy-tied Newton stopping + `DV_effective` continuity fix + Jacobian every 2-3 iterations; (iii) BDF2 with a step-size controller; (iv) coloured Jacobian (still ~20% of a step, and it shrinks the compiled graph); (v) not JFNK, not banded LU, not `jacrev`.

### 7.3 Global constraint terms (e.g. volume-averaged density = X) and the structured Jacobian

A global constraint enters the Jacobian as a *low-rank* term, not a wide band: a scalar functional `s(x) = <n_e>` (volume weights `v`, dense over the n_e block) feeding back into the equations through a column `u = dR/ds` (a feedback source shape, a Lagrange-multiplier column, or a replaced row). The exact Jacobian is `J_local + u v^T` (rank k for k scalars). Measured on the real residual at N = 200 with a rank-1 density-feedback term whose strength equals half the largest n_e-row entry (`bench/global_constraint.py`):

| quantity | J (today) | J + u v^T naive | J + u v^T with the rank-1 term split off |
|---|---|---|---|
| greedy colours (smoothing on) | 86 | 115 | 86 |
| GMRES iterations, M = P^-1, tol 1e-4 / 1e-6 | 22 / 28 | 34 / 29 | 31 / 27 with a Woodbury-corrected P |
| cost of the dense row `v` by `jax.grad` of the scalar | - | 0.007 ms (one VJP) | - |
| cost of the dense column `u` | - | one extra JVP column | - |

Consequences:
- **Naive colouring degrades** (the dense n_e x n_e block forces one colour per n_e column: +29 colours here, +n_rho in general, and it would alias far-column contributions into band entries if not excluded); **the chain-rule assembly handles it exactly** by treating the scalar exactly like the smoothing: `J = dG/dx + (dG/dc) S (dh/dx) + sum_k (dG/ds_k)(ds_k/dx)^T`, where each scalar costs one JVP column plus one VJP row. The requirement is the same as for the smoothing: the residual must expose the global scalars as named intermediates rather than burying them in the physics.
- **The linear solve stays banded**: `(J_b + U V^T)^-1` by Woodbury/bordered Schur complement is k extra banded solves; PETSc's `MatCreateLRC` / `PCFIELDSPLIT` Schur and CVODE's projection method (`CVodeSetProjFn`, enforce the invariant after the step) are the reference designs.
- **JFNK needs no structural knowledge at all** (exact JVPs include the coupling) and its preconditioner degrades gracefully: a rank-k perturbation costs at most about k extra GMRES iterations asymptotically (measured +1 at tol 1e-6, +12 at 1e-4 in this stiff case); a Woodbury correction of the preconditioner recovers most of it.
- If the constraint is imposed as an algebraic equation with a multiplier (index-1 DAE), the BE/BDF2 path is fine (stiffly accurate), the constraint row must be excluded from the error norm and needs its own residual scaling (section 4.3, IDA), and Crank-Nicolson is excluded for one more reason.

**Implemented for the sources (3.10).** TORAX already has three source models of exactly this kind - the cyclotron sink (on-axis values, fitted profile factor and the normalising volume integral: 4 scalars), the constant-fraction impurity radiation (the total heating power: 1 scalar) and the ToricNN ICRH model (the surrogate's five state-dependent scalar inputs) - and the production `jacobian_mode='structured'` handles them with the recipe above: their model function is a `source.SplitModelFunction` of a `globals_func` (the scalars `g(x)`) and a `profile_func` (the profile given `g`, local in `x`), the residual is evaluated with the globals injected, and the assembly adds `(dG/dg)(dg/dx)` from one JVP of `G` and one VJP of `g` per scalar (`torax/_src/solver/structured_jacobian.py`). 3.13 extended the same hook to the pedestal model output and to the reference values of internal boundary conditions (`calc_coeffs.StateGlobals`). A future global constraint term (a volume-averaged density feedback, a multiplier row) would plug into it the same way: expose its scalar functionals as globals, keep the rest of its residual contribution local. Calling the `SplitModelFunction` composes the two parts, so the split profile is by construction the model function's, and the `TORAX_ERRORS_ENABLED` JVP check guards any source that forgets to declare a global dependence.
