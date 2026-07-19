# JFNK vs dense-Jacobian Newton: benchmark results

Reference results for `newton_krylov_benchmark.py`. All numbers are from a
64-core CPU (JAX 0.10.2, f64); GPU behavior will differ (see notes). One
implicit theta-method step per config, both solvers started from the identical
LINEAR initial guess, `tol=1e-5`. JFNK settings: `gmres_restart=40`,
`gmres_rtol=1e-3`, per-iteration preconditioner refresh (the solver config
defaults).

## Convergence: full parity with dense Newton

Across all 58 buildable configs in `torax/tests/test_data`:

- `jfnk_refresh` converges on **56/58 - exactly the same set as dense
  Newton, with an outer-iteration delta of 0 on every config.** The two
  non-converging configs fail identically for both solvers.
- The frozen-preconditioner variant (`gmres_precond_refresh=False`) converges
  on 55/58: it fails on the LH-transition config, where the adaptive-transport
  pedestal switches transport branches between the initial guess and the
  solution, so a preconditioner frozen at the guess encodes the wrong branch.
  Refresh (one extra `calc_coeffs` per Newton iteration) fixes this.

Two fixes were required to reach parity, both in `jax_root_finding.py`:

1. **Left-preconditioning applied explicitly** (solve `(M o J) d = M(-R)`)
   rather than passing `M` to `jax.scipy.sparse.linalg.gmres`: jax's gmres
   measures its stopping criterion on the preconditioned residual but scales
   the tolerance with the unpreconditioned `||b||`; with `mean|R| ~ 1e8`
   (LH-transition initial guess) it accepted the zero vector without
   iterating, silently stalling Newton.
2. **Per-iterate preconditioner refresh** (`precond_fn(x)` API), for
   regime-switching transport.

## Speed: regime-dependent (CPU)

| case | N | dense | JFNK | speedup |
|---|---|---|---|---|
| STEP flat-top + TGLFNN-ukaea | 400 | 121 s (16 it) | ~8-10 s (14-16 it) | **~12-15x** |
| STEP flat-top + BgB, dt=10 s | 400 | 0.28 s (4 it) | 0.07 s (4 it) | **4x** |
| tglfnn ITER-hybrid config | 100 | - | - | ~1.7-2.3x |
| suite median (N<=100, cheap transport) | 25-100 | - | - | **0.4-0.7x (dense faster)** |

Compile time on the N=400 TGLFNN case: dense ~148 s vs JFNK ~31-37 s
(~4.5x) - tracing the 400-tangent `jacfwd` graph is itself expensive.

Mechanism: JFNK's advantage scales with the cost ratio jacfwd/residual. At
N<=100 with analytic or QLKNN transport, `jacfwd`'s tangent batching is
efficient (~8x a residual eval) and dense wins. At N=400 the batching goes
memory-bound (`jacfwd` = 372x a residual eval with TGLFNN; chunking does not
help) and JFNK dominates. The CPU crossover is at roughly N ~ 150-200,
arriving earlier the more expensive the transport model.

Full-simulation check (`test_iterhybrid_lh_transition`, 60 s through the L-H
dithering phase): both solvers complete with identical step counts
(955 at n_rho=25, 262 at n_rho=50) and final profiles agreeing to <=1%;
dense remains faster in wall time at these small N (86 s vs 101 s at
n_rho=50).

## Recommended configuration

- Default (small N, cheap transport): dense Newton (`use_newton_krylov=False`).
- High resolution and/or NN transport surrogates (N >~ 200): enable
  `use_newton_krylov=True`. Defaults (`gmres_restart=40`, `gmres_rtol=1e-3`,
  `gmres_precond_refresh=True`) reproduce dense iteration counts on every
  tested config.
- On GPU the trade-offs shift in both directions (`jacfwd` re-amortizes;
  GMRES becomes launch-latency-bound); re-measure with these scripts.

## Environment notes

6 of 64 test_data configs do not build here (QuaLiKiz/TGLF external models,
restart/compilation configs) - unrelated to the solver.
