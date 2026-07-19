# JFNK vs dense-Jacobian Newton: benchmark results

Reference results for `newton_krylov_benchmark.py`. All numbers are from a
64-core CPU (JAX 0.10.2, f64); GPU behavior will differ (see notes). One
implicit theta-method step per config, both solvers started from the identical
LINEAR initial guess, `tol=1e-5`.

## Headline case: STEP flat-top + TGLFNN-ukaea (N=400, dt=0.3125 s)

| variant | iters | converged | residual | run |
|---|---|---|---|---|
| dense Newton (`jacfwd`) | 16 | yes | 6.0e-09 | 121.4 s |
| JFNK unpreconditioned | 8 | no (line-search collapse) | 4.9e+00 | 8.9 s |
| **JFNK + Thomas preconditioner** | **14** | **yes** | **1.4e-06** | **8.6 s (14x)** |
| JFNK + Pereverzev-stiffened precond | 30 | no | 1.2e+01 | 3.7 s |

At this size, `jacfwd` costs ~372x a residual evaluation (tangent batching is
memory-bound; chunking does not help), so removing it dominates. The
unpreconditioned and Pereverzev rows show the speedup is specifically due to
the plain frozen block-tridiagonal preconditioner.

## Full test_data suite (58 buildable configs, defaults: restart=20, rtol=1e-2)

- 51/58: both solvers converge; median outer-iteration delta **0**
  (max +2). Inexact GMRES directions do not cost outer iterations.
- 3/58 (`test_imas_profiles_and_geo`, `test_iterbaseline_mockup`,
  `test_iterhybrid_makenans`): JFNK stalls at coarse tolerance with default
  GMRES budget; **converges fully with `--gmres_restart 40 --gmres_rtol
  1e-3`** (+1-2 iterations vs dense). A Krylov-budget artifact, arguing for
  adaptive forcing (Eisenstat-Walker) rather than fixed `gmres_rtol`.
- 2/58 genuine failures where dense succeeds:
  - `test_iterhybrid_lh_transition` (diverges; dense itself needs 21
    iterations - hardest config in the suite),
  - `test_step_flattop_bgb` at nominal dt=10 s (fully non-inductive,
    ~90% bootstrap; zero-loop-voltage psi boundary condition).
  Both are cases where the frozen tridiagonal preconditioner misses important
  Jacobian structure. Candidate fixes: preconditioner refresh across
  iterations, more restart cycles, or fallback to the dense path on error.
- 6 configs did not build in this environment (QuaLiKiz/TGLF models
  unavailable, restart/compilation configs) - unrelated to the solver.

### Run-time speedup vs problem size (CPU)

| regime | JFNK speedup |
|---|---|
| N=400 + NN transport (TGLFNN) | **14x** |
| N=100 + NN transport (`..._predictor_corrector_tglfnn`) | 1.7x |
| N<=100, analytic/QLKNN transport (49 configs) | 0.5-1.1x (median ~0.7x, i.e. dense is faster) |

Interpretation: `jacfwd` tangent batching is efficient at small N (the N=100
Jacobian costs only ~8x a residual eval), so JFNK's ~10-20 jvps per iteration
do not pay off there on CPU. The crossover is where tangent batching saturates
memory bandwidth - large N and/or expensive NN transport models. A production
integration should keep the dense path as the small-N default and select JFNK
for large-N/NN-transport configs (or by measured jacfwd-to-residual cost
ratio). On GPU the trade-offs shift in both directions (jacfwd re-amortizes;
GMRES becomes launch-latency-bound) and should be re-measured with these same
scripts.
