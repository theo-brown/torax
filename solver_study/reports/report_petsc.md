
# PETSc SNES / TS / Jacobian machinery — design choices relevant to TORAX

Source: shallow clone of PETSc `main` at commit `d3fce5f` in
`/tmp/claude-0/-home-user/9ffdac3c-d367-5e9c-a2e7-3a84fcf08450/scratchpad/petsc`.
All paths below are relative to that directory; `file:line` anchors point at the exact statement quoted.

**PETSc sign convention (needed to read every formula below).** SNES solves `F(x) = 0`; the linear solve is
`J Y = F` (not `-F`), and the update is `X <- X - lambda*Y` (`src/snes/impls/ls/ls.c:220`, `:234`; comment at
`src/snes/linesearch/impls/nleqerr/linesearchnleqerr.c:70`). So `Y` is *minus* the Newton step.

**Target problem recap.** N = channels × cells = 100–800 unknowns, block-banded Jacobian (block = 2–4 channels,
nearest-neighbour stencil) plus a few global couplings, stiff NN transport (QLKNN), backward-Euler/theta stepping,
dense `jax.jacfwd` Jacobian at every Newton iteration, dense LU, step-halving line search, mean-abs-residual test
(1e-5, coarse fallback 1e-2), no local error estimate, dt halved on Newton failure, JAX/XLA on CPU, compile time
matters.

---

## 1. SNES line searches (`src/snes/linesearch/`)

### 1.1 Shared infrastructure and defaults

`SNESLineSearchCreate` (`src/snes/linesearch/interface/linesearch.c:186-201`):

| field | default | line |
|---|---|---|
| `lambda` (initial) | 1.0 | 186 |
| `damping` (initial lambda on every apply) | 1.0 | 193 |
| `maxlambda` | 1.0 | 194 |
| `minlambda` | 1e-12 | 195 |
| `rtol` / `atol` / `ltol` (iterative searches) | 1e-8 / 1e-15 / 1e-8 | 196-198 |
| `max_it` | 1 (overridden per type) | 201 |

Every `SNESLineSearchApply` resets `lambda = damping` unless `-snes_linesearch_keeplambda` (`linesearch.c:638`).
The order (`-snes_linesearch_order`) is 1, 2 or 3 (`linesearch.c:869`). `SNESNEWTONLS` uses `bt` by default
(`src/snes/impls/ls/ls.c:353`, inside `SNESCreate_NEWTONLS` starting at `:338`).

Domain/NaN protocol shared by all searches: `SNESLineSearchCheckFunctionDomainError`
(`include/petsc/private/linesearchimpl.h:111-124`): if the new `fnorm` is Inf/NaN, raise an error if
`-snes_error_if_not_converged`, else set the line-search reason to `SNES_LINESEARCH_FAILED_FUNCTION_DOMAIN`
(when the user called `SNESSetFunctionDomainError()`, `src/snes/interface/snes.c:148`) or
`SNES_LINESEARCH_FAILED_NANORINF`, and return. The Newton loop turns those into
`SNES_DIVERGED_FUNCTION_DOMAIN` / `SNES_DIVERGED_FUNCTION_NANORINF` (`ls.c:244-249`).

### 1.2 `bt` — polynomial backtracking (default for Newton) — `src/snes/linesearch/impls/bt/linesearchbt.c`

Defaults set in `SNESLineSearchCreate_BT` (`:406-425`): `max_it = 40` (`:421`), `order = CUBIC` (`:422`),
`alpha = 1e-4` (`:423`). Requires a Jacobian matrix or an objective (`:86`).

Algorithm (`SNESLineSearchApply_BT`, `:57-346`), with `f = ½‖F(x)‖²` (`:114`):

1. **Initial slope** (`:118-127`). Without an objective: `initslope = <F, J·Y>` via one `MatMult` (`:123-124`);
   if it comes out positive it is negated (`:125`), and if zero it is set to `-1` (`:126`). (With an objective:
   `initslope = <Y, F>`, `:120`.) PETSc never rejects a direction as non-descent here; it forces a descent slope.
2. **Inf/NaN pre-loop** (`:130-161`). Evaluate `F(X - lambda*Y)`; while `g = ½‖G‖²` is Inf/NaN, halve lambda
   (`:160`) — at `lambda <= minlambda` it calls `SNESCheckFunctionDomainError` and gives up (`:159`). Every
   trial also checks the `max_funcs` budget and returns `SNES_DIVERGED_FUNCTION_COUNT` (`:133-138`).
3. **Armijo acceptance** (`:164`): `g <= f + lambda*alpha*initslope` → accept full step.
4. **Tiny-step escape** (`:175-185`): if the full step failed Armijo but `stol*xnorm > ynorm`, the step is
   accepted anyway with reason `SUCCEEDED` (the Newton loop then declares `SNES_CONVERGED_SNORM_RELATIVE`,
   `ls.c:241-243`).
5. **Quadratic fit** (order ≥ 2, `:189-192`):
   `lambdatemp = -initslope*lambda² / (2(g - f - lambda*initslope))`, then
   `lambda = clip(lambdatemp, 0.1*lambda, 0.5*lambda)` (`:192`). One more function evaluation (`:206`);
   Inf/NaN here aborts with `FAILED_NANORINF` (`:216-220`).
6. **Cubic loop** (`:238-317`, at most `max_it = 40` passes, exit with `FAILED_REDUCT` when
   `lambda <= minlambda`, `:239-251`). With the two most recent points `(lambda, g)` and `(lambdaprev, gprev)`:
   ```
   t1 = g     - f - lambda    *initslope
   t2 = gprev - f - lambdaprev*initslope
   a  = ( t1/lambda²        - t2/lambdaprev²        ) / (lambda - lambdaprev)
   b  = (-lambdaprev*t1/lambda² + lambda*t2/lambdaprev²) / (lambda - lambdaprev)
   d  = b² - 3 a initslope ;  d = max(d, 0)
   lambdatemp = a == 0 ? -initslope/(2b) : (-b + sqrt(d))/(3a)          (:255-262)
   ```
   order 2 reuses the quadratic formula (`:264`); order 1 is plain halving `lambdatemp = 0.5*lambda` (`:266`).
   In every case the result is clipped to `[0.1*lambda, 0.5*lambda]` (`:271`) and Armijo is re-tested (`:295`).
7. Post-check hook (`SNESLineSearchSetPostCheck`) can modify `Y` or `W`; norms are recomputed if so (`:322-338`).

Cost: 1 `MatMult` + 1 function evaluation for the full step; +1 evaluation per shrink.
Reason codes go back to `SNESSolve_NEWTONLS` (`ls.c:239-264`), where a generic failure counts toward
`snes->maxFailures` (default 1, `src/snes/interface/snes.c:1897`) → `SNES_DIVERGED_LINE_SEARCH`, after a
gradient check that may instead report `SNES_DIVERGED_LOCAL_MIN` (`ls.c:256-261`, check at `ls.c:46`).

**Compared with plain geometric halving (what TORAX does):** halving is exactly `-snes_linesearch_order 1`
(`:266`) but with three additions in PETSc: (i) an explicit Armijo sufficient-decrease test with
`alpha = 1e-4` and a directional-derivative slope, instead of "any decrease"; (ii) a hard floor
`minlambda = 1e-12` and `max_it = 40`; (iii) the `stol` escape. The default cubic fit typically lands on a
usable lambda in 1–2 extra evaluations where halving needs 3–6, because the fitted minimizer is allowed
anywhere in `[0.1, 0.5]·lambda` rather than exactly `0.5·lambda`.

### 1.3 `secant` (formerly `l2`) — `src/snes/linesearch/impls/secant/linesearchsecant.c`

Minimises `f(lambda) = ‖F(x - lambda Y)‖²` (or the objective) on `[0, damping]` by a secant/Newton step on
finite-difference derivatives (`:133-152`):
```
delFnrm     = (3 f(λ) - 4 f(λ_mid) + f(λ_old)) / Δλ        # second-order one-sided
delFnrm_old = (-3 f(λ_old) + 4 f(λ_mid) - f(λ)) / Δλ
del2Fnrm    = (delFnrm - delFnrm_old) / Δλ
λ_update    = λ ∓ delFnrm/del2Fnrm   (sign chosen to go downhill, :151-152)
```
Two function evaluations per secant iteration (midpoint and endpoint, `:53-70`), `max_it = 1` by default
(`:249`), stop on `|Δλ| < ltol=1e-8` (`:124`) or `|f'| <= atol=1e-15` (`:141`). Inf/NaN at the endpoint: cap
`maxlambda = 0.95·λ` and shrink toward the last viable λ (`:101-106`); at `λ <= minlambda` return
`FAILED_REDUCT` (`:91-98`). Unlike `bt` it has **no sufficient-decrease test** and can *increase* lambda up to
`maxlambda`; the final `F` is recomputed once more at the chosen λ (`:202`). Docs `:216-238`.

### 1.4 `cp` — critical point — `src/snes/linesearch/impls/cp/linesearchcp.c`

Assumes `F = grad G` and finds a root of the directional derivative `F(x-λY)·Y` by a secant on `λ`
(`:139-152`); default `order = LINEAR` (`:231`), `max_it = 1` (`:233`); the slope of `fty(λ)` is approximated by
1-, 2- or 3-point differences (`:97-137`: 1 extra evaluation for order 2, 2 for order 3). Terminates on
`|Δλ| < ltol`, `|fty|/|fty_init| < rtol`, `|fty| < atol·‖Y‖` (`:60-87`). It ignores `SNESSetObjective`
(`:218`) and is "the preferred line search for `SNESQN` and `SNESNCG`" (`:220`). Not appropriate for a
non-gradient residual such as a transport PDE.

### 1.5 `none` (= "basic") — `src/snes/linesearch/impls/none/linesearchnone.c`

`W = X - damping·Y` (`:23`), one function evaluation for the norm (`:33-35`, skippable with
`-snes_linesearch_norms false`). Docs `:66-83`: "intended for methods with well-scaled updates; i.e. Newton's
method on well-behaved problems".

### 1.6 `nleqerr` — Deuflhard error-oriented affine-covariant search — `src/snes/linesearch/impls/nleqerr/linesearchnleqerr.c`

Per trial λ: evaluate `F(X - λY)` and solve **one extra linear system with the old Jacobian**
(`KSPSolve(snes->ksp, G, W)`, `:168`) to get the simplified Newton step `bar_dx`. Monitoring quantities
(`:183-188`):
```
theta  = ‖bar_dx‖ / ‖dx‖
mudash = 0.5 ‖dx‖ λ² / ‖bar_dx - (1-λ) dx‖
```
`theta >= 1` → `λ = max(min(mudash, 0.5λ), minlambda)` and retry (`:191-200`); otherwise accept
(`:201-227`). The *next* iteration's initial λ is the Lipschitz prediction
`mu = λ_prev ‖dx_prev‖ ‖bar_dx_prev‖ / (‖bar_dx - dx‖ ‖dx‖)`, `λ = min(1, mu)` (`:103-104`).
`max_it = 40` (`:318`). When `λ <= minlambda` it *resets to λ = 1 and takes the full step* ("not what
Deuflhard suggests, but works better in my experience", `:139-160`). Cost per trial: 1 function evaluation +
1 linear solve; cannot be used with `SNESVINEWTONRSLS` (`:298`).

### 1.7 `bisection` — `src/snes/linesearch/impls/bisection/linesearchbisection.c`

Root of `F(x-λY)·Y` by bisection on `[0, damping]` (`:139-141`); if there is no sign change the full step
is taken (`:57-64`). Defaults `max_it = 50`, `rtol 1e-8`, `atol 1e-6`, `ltol 1e-6` (`:221-224`). One
function evaluation per bisection; Inf/NaN aborts with `FAILED_NANORINF` / `FAILED_FUNCTION_DOMAIN` (`:72-85`).

### 1.8 Picard pre-check (`SNESLineSearchPreCheckPicard`, `src/snes/linesearch/interface/linesearch.c:532-587`)

Optional pre-check for Picard iterations (`-snes_linesearch_precheck_picard`, angle default 10°, `:861`):
if the new direction is within 10° of (±) the previous one, the step is rescaled by
`alpha = ‖Y_last‖/‖Y_last - Y‖` (capped at 1000, `:573-576`) — an Aitken-like extrapolation for linearly
convergent Picard sequences.

### 1.9 Assessment for TORAX

* Replacing "halve until residual decreases" with `bt` semantics is cheap to express in JAX: the Armijo
  test needs `initslope = <F, J Y>` — one extra mat-vec you already have (dense `J @ Y`), and the
  quadratic/cubic formulas are scalar closed forms. In a `lax.while_loop` with fixed trip cap (`max_it`), the
  polynomial fit generally cuts the number of residual evaluations per backtrack from O(3–6) to O(1–2).
  With stiff QLKNN residuals the biggest practical win is the **Inf/NaN pre-loop** (`bt:130-161`): NaNs from
  the surrogate are handled by lambda cutting *before* any fit is attempted.
* `alpha = 1e-4` is tiny by design: PETSc accepts almost any decrease. TORAX's "any decrease" criterion is
  the limit `alpha → 0`; the real difference is the polynomial interpolation and the `minlambda`/`max_it`
  floor, both of which map 1:1 onto fixed-shape loops.
* `nleqerr` costs an extra linear solve per trial; with a dense LU already factored that is O(N²) — cheap for
  N ≤ 800 — and its λ-prediction across iterations is attractive for critical-gradient stiffness, but it needs
  state carried across Newton iterations (three scalars), trivial in a `while_loop` carry.
* `cp`, `bisection` assume `F = grad G`: not applicable. `secant` has no descent safeguard: not recommended for
  a residual that can jump by orders of magnitude when a gradient crosses the critical value.

---

## 2. SNES convergence tests

### 2.1 `SNESConvergedDefault` (`src/snes/interface/snesut.c:720-756`)

```
it == 0:  ttol = fnorm*rtol ; rnorm0 = fnorm                       (:729-730)
NaN/Inf fnorm                       -> SNES_DIVERGED_FUNCTION_NANORINF   (:732-734)
fnorm < abstol (and it>0 or !forceiteration) -> SNES_CONVERGED_FNORM_ABS (:735-737)
nfuncs >= max_funcs                 -> SNES_DIVERGED_FUNCTION_COUNT     (:738-740)
it > 0:
  fnorm <= ttol                     -> SNES_CONVERGED_FNORM_RELATIVE     (:744-746)
  snorm < stol*xnorm                -> SNES_CONVERGED_SNORM_RELATIVE     (:747-749)
  fnorm > divtol*rnorm0             -> SNES_DIVERGED_DTOL                (:750-752)
```
Norms are plain 2-norms (`ls.c:172`, `:236`), **unscaled** by N and without per-component weights.
`snorm` is the 2-norm of the *actual* update `lambda·Y` (`ynorm` recomputed after the line search,
`bt:337`). `stol` is used in three places: here (relative step size test), inside `bt` as an escape
(`bt:175`), and in the Newton loop after a line-search failure (`ls.c:241-243`), where a failed search with a
tiny step is *converted into convergence*.

Defaults (`src/snes/interface/snes.c:1832-1841`, `SNESParametersInitialize`):
`max_its 50`, `max_funcs 10000`, `rtol 1e-8`, `abstol 1e-50`, `stol 1e-8`, `divtol 1e4`.
Other failure budgets: `maxFailures = 1` line-search failures (`snes.c:1897`, `-snes_max_fail`),
`maxLinearSolveFailures = 1` (`snes.c:1930`). `SNESConvergedSkip` (declares convergence after `max_its`)
exists as the alternative test (`snesut.c:783`). `NEWTONTR` changes the default `stol` to 0
(`src/snes/impls/tr/tr.c:1167`).

### 2.2 Assessment for TORAX

* TORAX's "mean absolute residual < 1e-5" is an *absolute*, N-scaled 1-norm test. PETSc's default is a
  **relative** 2-norm test (1e-8 reduction) with the absolute test effectively disabled (1e-50). For a
  time-stepping residual the relative test is a poor choice because `rnorm0` (residual of the extrapolated
  predictor) already shrinks with dt; PETSc users with TS routinely set `-snes_atol` and/or `-snes_stol`. The
  step-tolerance test (`snorm < stol·xnorm`) is the one TORAX lacks: with stiff transport the residual norm
  can plateau (Newton stalls at a residual set by the surrogate's non-smoothness) while the iterate is
  converged to machine precision in `x`; PETSc exits cleanly through `stol` there.
* PETSc has **no equivalent of TORAX's "coarse tolerance fallback" (1e-2)**: an unconverged SNES inside TS is
  a negative reason and is *rejected* by `TSAdaptCheckStage` (§6.1) — the design is "cut dt, never accept a
  bad solve" (except `SNES_DIVERGED_MAX_IT` from an inner nonlinear preconditioner, `ls.c:160`).
* The `divtol = 1e4` growth test (`snesut.c:750`) is a cheap divergence detector TORAX could add to its
  Newton `while_loop` predicate to bail early instead of running to `max_it`.

---

## 3. Jacobian lagging, matrix-free operators, and the other SNES types

### 3.1 Lagging (`SNESComputeJacobian`, `src/snes/interface/snes.c:3044-3123`)

```
lagjacobian == -2 : compute now, then set to -1 (never again)          (:3059-3062)
lagjacobian == -1 : reuse (only re-assemble if the operator is MATMFFD) (:3063-3070)
lagjacobian  >  1 : reuse iff (iter + jac_iter) % lagjacobian != 0      (:3071-3078)
```
Preconditioner lagging is identical with `lagpreconditioner`/`pre_iter` and `KSPSetReusePreconditioner`
(`:3111-3123`). Defaults: `lagjacobian = 1`, `lagpreconditioner = 1`, persistence off
(`snes.c:1899-1904`). `jac_iter`/`pre_iter` are zeroed in `SNESSetUp` (`:3541-3542`) and, with
`-snes_lag_jacobian_persists`, accumulated across solves (`snes.c:5023-5024`), so "every k-th Jacobian" counts
across time steps. Docs: `SNESSetLagJacobian` `:3812-3835` ("-1 before the first solve: THE CODE WILL FAIL;
use -2"), `SNESSetLagJacobianPersists` `:3881-3908` ("for implicit time-stepping, Jacobian lagging in the inner
nonlinear solve over several timesteps may present huge efficiency gains").

### 3.2 Matrix-free JVP (`MATMFFD`, `src/mat/impls/mffd/`) and `-snes_mf_operator`

`SNESSetUseMatrixFree(snes, mf_operator, mf)` (`snes.c:1364`): `mf_operator` keeps the user `Pmat` for the
preconditioner but applies the *true* Jacobian by differencing; `mf` drops the preconditioner (PC set to
`PCNONE`). Options `-snes_mf_operator`, `-snes_mf` (`snes.c:1130-1139`).

`MatMult_MFFD` (`mffd.c:303-370`): `y = (F(U + h a) - F(U))/h` (`:351`, `:365-367`), with `F(U)` computed
once per assembly (`:354`). `h` selection (`:325-330`, default type `MATMFFD_WP`):

* **WP** (Walker–Pernice, `wp.c:50-72`): `h = error_rel·sqrt(1 + ‖U‖)/‖a‖` (`:67`); `‖U‖` cached between
  linear iterations (`-mat_mffd_compute_normu`), no collectives in GMRES because `‖a‖ = 1`.
* **DS** (Dennis–Schnabel, `mffddef.c:47-85`): `h = error_rel·(U·a)/‖a‖²` with the safeguard
  `|U·a| < umin·‖a‖₁ → U·a = ±umin·‖a‖₁` (`:76-78`); `umin = 1e-6` (`:197`).
* `error_rel = sqrt(machine eps)` (`mffd.c:588`), `recomputeperiod = 1` (`:589`), option
  `-mat_mffd_check_positivity` (docs `:663`) shrinks `h` until `U + h a > 0`.

The Krylov solver defaults to GMRES (`src/ksp/ksp/interface/itcl.c:389`) with `rtol 1e-5`,
`abstol 1e-50`, `divtol 1e4`, `max_it 10000` (`src/ksp/ksp/interface/itcreate.c:812-817`); Eisenstat–Walker
forcing is off by default (`snes.c:1919`), defaults version 2, `rtol_0 0.3`, `rtol_max 0.9`, `gamma 1`,
`alpha (1+sqrt5)/2`, `threshold 0.1` (`snes.c:1942-1950`).

### 3.3 The SNES types

| type | file (create / solve) | what it does | cost per iteration | PETSc's own guidance |
|---|---|---|---|---|
| `NEWTONLS` (default) | `ls.c:338 / :125` | Jacobian, `KSPSolve`, line search `bt` | 1 J + 1 solve + 1 `MatMult` + (1+backtracks) F | "default nonlinear solver" |
| `KSPONLY` | `ksponly.c:7` | exactly one Newton step, no norms (`:42-59`) | 1 J + 1 solve + 1 F | "solve linear problems using the SNES interface" (`:86-95`); used by `TSROSW`/`TSPSEUDO` |
| `NRICHARDSON` | `snesrichardson.c:165 / :26` | `x <- x - λ F(x)` (or NPC correction), line search on λ (`:88`) | 1 F (+search) | "much slower than Newton"; update "may be ill-scaled" (`:138-163`) |
| `ANDERSON` | `anderson.c:196 / :21` | `x_M = x - β F` (`:114`, or NPC output blended with β, `:111`), then LS-combination of the last `m` `(x_i, F_i)` via LAPACK `gelss` (`ngmresfunc.c:38-76`) | 1 F + `m×m` least squares | `m = 30`, `β = 1.0`, restart `none`, `restart_it 2`, periodic 30 (`anderson.c:224-232`) |
| `NGMRES` | `snesngmres.c:504 / :130` | same subspace combination but with **selection** between candidate `x_M` and combination `x_A`, and restart tests | 2 F (x_M and x_A) + `m×m` LS | `m = 30`, `γ_A = γ_C = 2`, `δ_B = 0.9`, `ε_B = 0.1`, `restart_it = 2`, periodic 30, select/restart = `difference` (`:528-541`) |
| `QN` | `qn.c:507 / :62` | L-BFGS (default), Broyden, bad-Broyden via `MatLMVM`; `m = 10` (`:531`) | 1 F + O(m) vec ops + line search; a Jacobian *only at restarts* if `scale_type jacobian` (`:157-161`) | line search default `cp` for L-BFGS, `none` for Broyden, `secant` otherwise (`:344-350`) |
| `NEWTONTR` | `tr.c:1155 / :519` | Newton step inside a trust region; fallbacks Newton-scaled/Cauchy/dogleg | 1 J + 1 solve + 1 F per trial | `η1 0.001, η2 0.25, η3 0.75, t1 0.25, t2 2.0, δ0 0.2, δmin 1e-12, δmax 1e10` (`:1178-1185`) |
| `VINEWTONRSLS` | `virs.c:765 / :306` | reduced-space active set for `l ≤ x ≤ u`; Jacobian restricted to the inactive set (`:396`, `:505`), projection (`:346`), `bt` with VI norms (`:337`, `:783`) | as NEWTONLS on the reduced system | active set = on a bound *and* residual pushing outward (`:738-760`); `-snes_vi_zero_tolerance 1e-8` (`snes.c:1932`) |
| `FAS` | `fas.c:1070` | nonlinear multigrid; smoother/coarse SNES per level; multiplicative V-cycle default (`:1093-1111`) | levels × smoother sweeps of F | needs `DMCoarsen`/`DMInterpolate` |

Details:

* **NGMRES select/restart** (`ngmresfunc.c:213`, `:255-268`): choose `x_A` iff
  `obj_A < γ_A·obj_min && (ε_B·‖x_A - x_M‖ < d_min || obj_A < δ_B·obj_min)`; restart when the difference
  condition fails for `restart_it` consecutive iterations, when `obj_A > γ_C·obj_min`, or (option) when
  `‖F_M‖` rises. The subspace is the last `m` vectors; each iteration solves a dense `l×l` normal-equation
  system by SVD (`gelss`, `:54-56`), O(l²N + l³).
* **Anderson** has no selection: `x <- x_A` unconditionally (`anderson.c:143-146`), so with a stiff residual it
  can jump; PETSc's intended use is with a nonlinear preconditioner (`-npc_snes_type`) or as an accelerator
  for a fixed-point map (`x_M = x - β F(x)`).
* **QN restarts** (`qn.c:217-246`): Powell test `|F^T H0 F_old| > 0.9999 |F_old^T H0 F_old|` (default for
  L-BFGS, `:270-281`), periodic every `m` for Broyden (`:228-230`); `SNES_QN_SCALE_JACOBIAN` solves with the
  user Jacobian as `H0` (`MatLMVMSetJ0KSP`, `:160`), rebuilt only at restarts.
* **TR update** (`tr.c:773-793`): `ρ = (f_k - f_{k+1})/(m(0) - m(s))`; `ρ < η2 → δ *= t1`;
  `ρ > η3` and step on boundary `→ δ *= t2`; accept iff `ρ > η1`; NaN in `f_{k+1}` sets `ρ = η1` (reject).
  `δ` bounds the step norm (default 2-norm, `:1189`), initial `δ0 = 0.2`.
* **VI RS**: the active-set variables are removed from the linear system (`MatCreateSubMatrix`,
  `virs.c:396`, `:461`) so the Newton direction is computed on the free variables only.

### 3.4 Assessment for TORAX

* **Lagging** is the highest-leverage idea. `lagjacobian = -2`/`persists` = "one Jacobian per time step (or
  per k steps)" with the line search absorbing inexactness — the standard PETSc recipe for implicit stepping.
  For TORAX this is "reuse the `jacfwd` result from the predictor across Newton iterations", requiring the
  factored `J` in the `while_loop` carry — cheap for N ≤ 800 (dense N² floats ≤ 5 MB). With the QLKNN
  critical-gradient nonlinearity a stale Jacobian can fail on iterations where the operating point crosses a
  threshold; PETSc's `maxFailures = 1` / `TSAdaptCheckStage` logic (recompute or cut dt) is the safety net.
* **`mf_operator` / JFNK** does not translate as such: the "matrix-free product" in JAX is `jax.jvp` (exact),
  so the WP/DS differencing machinery is moot. The *structure* does translate: Newton–Krylov (GMRES on exact
  JVPs) preconditioned by a cheap approximate Jacobian (banded, frozen-coefficient, or lagged) keeps
  exactness without N JVPs per iteration. For N = 100–800 on CPU a direct dense factorization is 1e6–3e8
  flops (milliseconds); the payoff of Krylov is only the compile-time/memory reduction of the JVP batch (§4).
* **Picard / NRICHARDSON**: TORAX's "linear" solver (frozen coefficients + Pereverzev–Corrigan) is PETSc's
  Picard *defect-correction* form (`SNESSetPicard` docs, `snes.c:2438`: `A(x^n)(x^{n+1}-x^n) = b(x^n) - A(x^n)x^n`);
  PETSc's advice: "It is often better to provide the nonlinear function F() and some approximation to its
  Jacobian directly and use an approximate Newton solver", and "Run with `-snes_mf_operator` to solve the
  system with Newton's method using `A(x^n)` to construct the preconditioner". The Pereverzev–Corrigan
  stabilised block matrix is a good `Pmat`.
* **Anderson/NGMRES** are cheap (1–2 residual evaluations, no Jacobian), naturally fixed-shape (window `m` as
  a ring buffer); PETSc's default `m = 30` is far too large for 2–4 channels — `m = 3–5` is typical for
  accelerating Picard. They fit TORAX's "linear" Picard mode as an accelerator, not as a replacement for
  Newton on the stiff mode.
* **QN with Jacobian scaling** (`-snes_qn_scale_type jacobian`, Jacobian only at restarts) is the middle
  ground: one `jacfwd` per restart, L-BFGS updates in between, one residual evaluation per iteration; fully
  expressible with fixed-size histories (`m = 10`).
* **VI bounds** for positivity (`T`, `n ≥ 0`) are a real alternative to NaN-driven backtracking; the reduced
  linear system has a data-dependent size, awkward under XLA — implement by masking (zero rows/cols, unit
  diagonal) rather than sub-matrix extraction.
* **FAS/grid sequencing**: irrelevant at 25–200 cells.

---

## 4. Finite-difference Jacobians with coloring

### 4.1 Driver: `SNESComputeJacobianDefaultColor` (`src/snes/interface/snesj2.c:61-117`)

Enabled by `-snes_fd_color` (`snes.c:1120-1128`). If no coloring is attached: use `DMCreateColoring` when the
DM can (`snesj2.c:80-81`), otherwise build one from the matrix nonzero pattern with
`MatColoringSetDistance(mc, 2)` and type `MATCOLORINGSL` (`:83-87`); then `MatFDColoringCreate/SetUp` and
`MatFDColoringApply` (`:90-111`). The base residual `F(x)` is reused from SNES when possible (`:105-110`).
The dense variant `SNESComputeJacobianDefault` (`snesj.c:48`) perturbs **one column at a time (N evaluations)**
with `dx = ε·sqrt(1+‖x‖)` (wp) or `ε·x_i` (ds), `dx_min = 1e-16`, fallback `dx_par = 0.1` (`snesj.c:55`,
`:109-113`), `ε = sqrt(machine eps)` ("1e-8 double", docs `:22-23`).

### 4.2 Coloring (`src/mat/graphops/color/`)

`MatColoringCreate` defaults: `dist = 2` ("default to Jacobian computation case", `matcoloring.c:81`),
`maxcolors = IS_COLORING_MAX` (`:82`), random weights (`:85`). `MatColoringSetFromOptions` picks `GREEDY` by
default, but `SL` when `dist == 2` (`:186`, `:193`). Types (doc blocks in `impls/*`):

* `SL`, `LF`, `ID` — Coleman–Moré sequential heuristics from MINPACK, **distance-2 only** (`impls/minpack/color.c`).
* `GREEDY` — greedy with parallel conflict correction, distance 1 or 2 (`impls/greedy/greedy.c`).
* `JP` — parallel Jones–Plassmann (`impls/jp/jp.c`).
* `NATURAL` — one color per column ("extremely inefficient but useful for testing"); `POWER` — color `A^n`.

The number of function evaluations in `MatFDColoringApply` equals the number of colors (loop
`for (k = 0; k < ncolors; k++)`, `src/mat/impls/aij/mpi/fdmpiaij.c:96`; the blocked variant `:252` processes
`bcols` colors per pass). For a banded/block-tridiagonal matrix a distance-2 coloring needs exactly
`max nonzeros per row` colors, i.e. `3·nc` for a nearest-neighbour stencil with `nc` dense-coupled channels —
6, 9, 12 colors for nc = 2, 3, 4 — independent of the number of cells.

### 4.3 Step size in `MatFDColoring` (`src/mat/matfd/fdmatrix.c`)

Defaults: `error_rel = sqrt(machine eps)`, `umin = 100·sqrt(machine eps)` (≈ 1.5e-6 double), `htype = "wp"`
(`fdmatrix.c:475-478`; `-mat_fd_coloring_err`, `-mat_fd_coloring_umin`, `-mat_fd_type`). Formulas
(docs `:169-178`, code `fdmpiaij.c:61-75`):
```
wp: h = error_rel * sqrt(1 + ||u||)                                     (one scalar for all columns)
ds: h_i = error_rel * u_i        if |u_i| > umin
        = ± error_rel * umin     otherwise (sign of u_i)
J[:, i] = (F(u + h e_i) - F(u)) / h  -- all columns of one color perturbed simultaneously
```

### 4.4 DMDA-based coloring for 1-D stencils (`src/dm/impls/da/fdda.c:388-455`)

`DMCreateColoring_DA_1d_MPIAIJ`: `col = 2*s + 1` (`:403`), color of (cell i, component l) is
`l + nc*(i mod col)` (`:430`), so
```
ncolors = nc + nc*(col - 1) = nc*(2s + 1)                              (:432, :447)
```
→ `dof = 4, s = 1: 12 colors`; `dof = 4, s = 2: 20 colors`; `dof = 2, s = 1: 6`. Periodic BCs require
`m mod col == 0` for a global coloring (`:404`). With `DMDASetBlockFills` (`ofill` sparse coupling pattern
between components) the count drops to `nc + 2*s*tc` where `tc` = number of components with any neighbour
coupling (`:413-426`): e.g. 3 of 4 channels with a stencil → 4 + 2·1·3 = 10.

### 4.5 Assessment for TORAX

* This is the single most quantitative win available: dense `jax.jacfwd` builds N = 100–800 JVP columns per
  Newton iteration; a coloring-compressed Jacobian needs **12 (dof 4, width 1) or 20 (width 2)** seed vectors
  regardless of the grid, i.e. 8×–65× fewer tangent columns. In JAX this is *not* finite differencing:
  `jax.vmap(lambda s: jax.jvp(residual, (x,), (s,))[1])` over the `ncolors` seed vectors
  `s_c = Σ_{i ∈ color c} e_i`, then scatter each compressed column into the banded blocks (static index arrays
  from the stencil). Exactness is preserved (no `h`), compile time and peak memory of the batched JVP scale
  with 12–20 instead of N, and the block-tridiagonal result feeds a banded solve (§5).
* The catch is PETSc's own caveat in the `SNESSetPicard` docs (`snes.c:2438` block): "the nonzero structure of
  the Jacobian is, in general, larger than that of the Picard matrix" — **global couplings break the
  coloring**. PETSc-style options: (a) widen the stencil to the smoothing-kernel support (`s = 2, 3` → 20, 28
  colors); (b) treat the nonlocal term with a frozen/lagged coefficient (approximate Jacobian: Picard on the
  nonlocal part, Newton on the local part) and let the line search handle inexactness; (c) keep the exact
  operator via a few extra dense "global" columns (one JVP per global scalar, e.g. an integral constraint) —
  a "banded + low-rank" Jacobian with a Woodbury solve.
* `umin`/`error_rel` are irrelevant for AD, but the `DMDASetBlockFills` idea (declare which channel pairs
  couple) is reusable to shave colors: `psi` typically couples to `T_e, n_e` only through resistivity/bootstrap
  terms.

---

## 5. Linear solvers for small banded / block-tridiagonal systems

* **Defaults**: KSP `GMRES` (`itcl.c:389`), PC chosen by `PCGetDefaultType_Private`
  (`src/ksp/pc/interface/precon.c:14-58`): on one process `ICC` if the matrix is known symmetric, else `ILU`
  when an ILU factorization exists for the matrix type (`:32-35`), `BJACOBI` for parallel formats. So out of
  the box a 1-D banded problem gets GMRES(30) + ILU(0) — which on a (block-)tridiagonal matrix with natural
  ordering is an *exact* LU (no fill outside the band), so GMRES converges in 1 iteration.
* **Direct**: `-pc_type lu -ksp_type preonly` (`PCLU` docs, `lu.c:223-258`: "Usually this will compute an
  'exact' solution in one iteration and does not need a Krylov method"). Ordering defaults: `nd` for
  LU/Cholesky, `natural` for ILU/ICC (`src/mat/impls/aij/seq/aijfact.c:33-35`; BAIJ `baijfact.c:707-709`).
  For a banded matrix natural ordering is already optimal (`-pc_factor_mat_ordering_type natural`).
* **Block formats**: `MATSEQBAIJ` (`baij.c:3475`) stores dense `bs×bs` blocks; `PCILU` on BAIJ "implements a
  point block ILU" with `-pc_factor_pivot_in_blocks` (`ilu.c:247-282`); block-tridiagonal with block size =
  number of channels is exactly this. `PCBJACOBI` (`bjacobi.c:470`) is the parallel wrapper — irrelevant on
  one CPU.
* **Cost**: banded LU with block size `m` and `K` cells: ~`K·(8/3)m³` flops (block Thomas) versus dense
  `(2/3)N³ = (2/3)(mK)³`. For `m = 4, K = 200` (N = 800): ≈ 3.4e4 vs 3.4e8 flops (10⁴×); for `m = 2, K = 25`
  (N = 50): 5e2 vs 8e4. Solve phase O(N·m²) vs O(N²). PETSc's sparse LU on a banded SeqAIJ matrix achieves the
  banded cost automatically because there is no fill.

**Assessment.** Dense LU at N = 800 is 3.4e8 flops ≈ 10–30 ms on one core; per Newton iteration comparable
to or larger than a batched residual evaluation through QLKNN. A block-Thomas solve (`lax.scan` over cells
with `m×m` dense blocks) brings this to microseconds and avoids materialising an N×N array.
`jax.lax.linalg.tridiagonal_solve` is scalar only; a block variant is ~20 lines with `lax.scan`. A global
coupling is then handled by Woodbury or kept in the residual only (inexact Newton).

---

## 6. Time stepping (TS)

### 6.1 `TSAdapt` infrastructure (`src/ts/adapt/interface/tsadapt.c`)

Defaults (`TSAdaptCreate`, `:1166-1190`):
```
always_accept FALSE ; safety 0.9 ; reject_safety 0.5 ; clip [0.1, 10] ; dt_min 1e-20 ; dt_max 1e20
ignore_max -1 ; scale_solve_failed 0.25 ; matchstepfac [0.01, 2.0] ; wnormtype NORM_2 ; increase delay 0
```
(options `-ts_adapt_{type,always_accept,safety,reject_safety,clip,dt_min,dt_max,max_ignore,
scale_solve_failed,wnormtype,time_step_increase_delay,monitor}`, `TSAdaptSetFromOptions` `:765-815`).

**`TSAdaptCheckStage`** (`:1090-1146`), called by every implicit stepper after each stage/SNES solve:
user `checkstage` callback (`:1102-1108`), `TSSetFunctionDomainError` callback (`:1110-1114`), then the SNES
reason (`:1116-1124`):
```
if snesreason < 0:
   if snesreason != SNES_DIVERGED_FUNCTION_DOMAIN and ++num_snes_failures >= max_snes_failures (≠ UNLIMITED):
        ts->reason = TS_DIVERGED_NONLINEAR_SOLVE           (:1119-1121)
   reject stage
reject: if !ts->reason: dt <- dt * scale_solve_failed (0.25) and retry (:1135-1141)
```
**Important default:** `max_snes_failures = 1` (`src/ts/interface/tscreate.c:53`), so with defaults the
*first* Newton failure ends the run with `TS_DIVERGED_NONLINEAR_SOLVE` and `TSStep` raises "increase
`-ts_max_snes_failures` or use unlimited to attempt recovery" (`ts.c:3609`, `errorifstepfailed` default TRUE,
`tscreate.c:55`). TORAX-style "halve and retry" requires `-ts_max_snes_failures -1`, after which each failure
multiplies dt by **0.25**, not 0.5; `-ts_adapt_time_step_increase_delay k` then suppresses growth for `k`
accepted steps (`adaptbasic.c:61-64`). Each step attempt may be rejected at most `max_reject = 10` times
(`tscreate.c:54`; `TS_DIVERGED_STEP_REJECTED`, e.g. `theta.c:253-259`, `bdf.c:296-302`).

**`TSAdaptChoose`** (`:939-1037`) calls the controller (`:978`) and, with `TS_EXACTFINALTIME_MATCHSTEP`,
adjusts the last steps: overshoot → `h = t_max - t`; if `2h > remaining` → `h = remaining/2`; if
`1.01h > remaining` → `h = remaining` (`:983-1024`). Adaptivity is suspended while an event is processed
(`:956-976`).

**Error norm** (`ts.c:5320-5341` → `VecErrorWeightedNorms_Basic`, `src/vec/vec/interface/vector.c:2542`):
```
tol_i = atol_i + rtol_i * max(|u_i|, |y_i|)     (:2563-2568)
enorm = sqrt( (1/N) Σ (err_i / tol_i)² )        (ts.c:5333)   (or max_i for NORM_INFINITY)
```
components with `|u_i| < ignore_max` skipped. Defaults `atol = rtol = 1e-4` (`tscreate.c:64-65`); vector
tolerances via `TSSetTolerances(vatol, vrtol)` (e.g. infinite tolerance on algebraic components).

Controllers (`src/ts/adapt/impls/`):

* **basic** (`adaptbasic.c:4-70`): reject iff `enorm > 1` (unless `h ≤ (1+√eps)·dt_min` or `always_accept`;
  a second consecutive rejection multiplies `safety` by `reject_safety = 0.5`, `:41-56`);
  `h_new = h · clip(safety · enorm^(-1/order), 0.1, 10)` (`:59`, `:65`), clipped to `[dt_min, dt_max]` (`:67`).
  `order` is the embedded order reported by `TSEvaluateWLTE` or `candidates.order[0]` with
  `TSEvaluateStep(order-1)` (`:19-28`).
* **dsp** (Söderlind digital filters, `adaptdsp.c:68-165`): default filter `PI42` (`:388`), table `:202-222`
  (`PI42 = {5, {3,-1,0}, {0,0}}` → `b = (0.6, -0.2, 0)`, `a = (0,0)`):
  ```
  c_n   = (1/enorm)^(1/k)
  rho   = c_n^b1 · c_{n-1}^b2 · c_{n-2}^b3 · rho_{n-1}^(-a2) · rho_{n-2}^(-a3)      (:135-142)
  rho   = 1 + atan(rho - 1)                                   (Limiter, :38-41)
  accept iff rho >= safety·0.9 = 0.81 (or always_accept, or h < dt_min)             (:74, :145-148)
  h_new = h · clip(rho, 0.1, 10) clipped to [dt_min, dt_max]                          (:161-162)
  ```
  History rolled back on rejection, reset on restart (`:110-117`). With the limiter the accept test is
  `(1/enorm)^(0.6/k)·c_{n-1}^(-0.2/k) ≥ 0.808`; with steady history that accepts `enorm` up to ≈ 2.0 for
  `k = 2` — deliberately less twitchy than `basic`, with smoothed step ratios.
* **glee** (`adaptglee.c:8-97`): for non-GLEE methods `hfac = safety·min(enorma^(-1/order), enormr^(-1/order))`
  (`:82-86`), rejects if *any* of `enorm, enorma, enormr > 1` (`:47`).
* **cfl** (`adaptcfl.c:3-35`): `h_new = safety·cfl_time·c_cfl` (`:24`); rejection "not implemented / unusable"
  (`:12`), requires `always_accept`.
* **history** (`adapthist.c`): replays a stored `(t, dt)` sequence ("for Tangent Linear Model simulations").
* `none`: fixed step.

### 6.2 `TSBDF` (`src/ts/impls/bdf/bdf.c`)

Variable-step BDF, orders 1–6 (`:498`), **default order 2** (`:552`), `default_adapt_type = BASIC` (`:540`),
initial-guess extrapolation on by default (`:547`).

* Coefficients from derivatives of the Lagrange basis on the stored times (`TSBDF_PreSolve`, `:185-215`):
  `Xdot = shift·X + V0`, `shift = alpha[0]` (`:199`) — the SNES Jacobian is `dF/dU + alpha0·dF/dUdot`
  (`SNESTSFormJacobian_BDF`, `:385-400`).
* **Order is not adapted**: `k` ramps `1 → 2 → … → order` one per accepted step (`:259`) and stays there;
  after a restart (`TSBDF_Restart`, `:218-242`) the method takes one **backward-Euler half step**
  (`time[0] = t + dt/2`, `:229`) to seed the history, then continues with `k = min(2, order)` (`:238`).
* **Error estimator** (`TSBDF_VecLTE`, `:132-147`): difference between the order-`k` and order-`k+1`
  Lagrange-derivative predictors, `alpha_i = (a_i - b_i)/a_0` (`:143`), `lte = Σ alpha_i X_i`; reported order
  `k+1` (`:330`); no estimate (`wlte = -1` → accept, keep dt) until enough history exists (`:323-327`).
* Step loop `:248-305`: extrapolated predictor (`:271`, degree lowered by one after a rejection), one
  `SNESSolve` per step (`:274`), `TSAdaptCheckStage` (`:276`), `TSAdaptChoose` (`:282`), `max_reject` (`:296-302`).

### 6.3 `TSTHETA` / `TSBEULER` / `TSCN` (`src/ts/impls/implicit/theta/theta.c`)

* `TSCreate_Theta` (`:1223`): `Theta = 0.5`, `order = 2`, `extrapolate FALSE`, `default_adapt_type = NONE`
  (`:1250`, `:1269-1270`). `TSBEULER` = Theta 1.0, endpoint FALSE (`:1408-1414`); `TSCN` = Theta 0.5, endpoint
  TRUE (`:1451-1457`). Docs `:1176-1222`: midpoint variant is a 1-stage IRK; endpoint variant is the 2-stage
  `[0 | 0 0; 1 | 1-Θ Θ]` (trapezoid for Θ = 0.5); "the midpoint variant is not suitable for DAEs because it is
  not stiffly accurate".
* Step (`TSStep_Theta`, `:197-262`): `shift = 1/(Θ·dt)` (`:210`), stage time `t + Θdt` or `t + dt` (`:211`);
  endpoint form passes an affine term `((Θ-1)/Θ)·F(t, X0, 0)` to the SNES (`:214-219`, "assumes linear
  time-independent mass matrix"); one `SNESSolve` per step (`:221`); `TSAdaptCheckStage` (`:223`), for midpoint
  a second check on the completed step (`:227-236`); `TSAdaptChoose` (`:239`); rollback and `max_reject` (`:253-259`).
* **`-ts_theta_adapt` no longer exists**; the estimate is enabled by choosing any adaptor other than `none`
  (`-ts_adapt_type basic`): `TSSetUp_Theta` allocates `vec_sol_prev`/`vec_lte_work` iff adapt type ≠ `NONE`
  (`:1036-1042`). The estimator (`TSEvaluateWLTE_Theta`, `:681-715`) is an **extrapolation/backward-difference
  formula on the last three solutions with non-constant steps**:
  ```
  h = dt, h_prev = t_n - t_{n-1}, a = 1 + h_prev/h
  LTE ≈ X_{n+1} - [ (1/a) X_{n+1} - (1/(a-1)) X_n + (1/(a(a-1))) X_{n-1} ]      (:699-710)
  ```
  `Y = X + LTE` is compared with `X` through `TSErrorWeightedNorm`; the reported **order is 2 regardless of
  Theta** (`:713`), so for backward Euler the controller uses exponent −1/2 (conservative growth); the estimate
  is unavailable on the first step and after a restart/event (`:688-696`, `wlte = -1` → accept, keep dt).
  No embedded method, no extra solve: the estimate is free.
* Jacobian: `TSComputeIJacobian(ts, stage_time, x, Xdot, shift, ...)` (`:958-973`).

### 6.4 `TSARKIMEX` (`src/ts/impls/arkimex/arkimex.c`)

* Default scheme **`3`** (Kennedy–Carpenter ARK3(2)4L[2]SA, `:18`; docs `:2350-2380`: "Consider trying TSROSW
  if the stiff part is linear or weakly nonlinear"; explicit-first-stage methods need the stiff part in the
  form `Xdot + Ĝ(t,X)`). DIRK family default `ES213SAL` (TR-BDF2, `:19`).
* Tableaux (doc blocks `:50-160`): `ars122, a2, l2, 1bee, 2c, 2d, 2e, prssp2, 3, ars443, bpr3, 4, 5`
  (orders 1–5; `2e` "optimal second order L-stable"; `3/4/5` Kennedy–Carpenter with one explicit + 3/4/5
  implicit stages). Each has an embedded method (`bembedt`) used by `TSEvaluateStep_ARKIMEX(order-1)`
  (`:1225`, called at `:1510`) for the `basic` controller (`:1514`).
* **`1bee`** (`:756-770`): `At = [[1,0,0],[0,½,0],[0,½,½]]`, `b = (0, ½, ½)`, `bembed = (1, 0, 0)` — one full
  backward-Euler step (stage 1) and two half BE steps (stages 2–3); the solution is the two-half-step value
  and the error estimate is the Richardson difference to the full step: 3 implicit solves per step. Used
  automatically as a **start-up stepper** when `equation_type >= IMPLICIT` and the tableau has an explicit
  first stage: PETSc clones the TS, sets `1bee` fully implicit, takes one step, returns (`:1394-1429`).
* Per stage: `shift = 1/(h·a_ii)` (`:1451`), one `SNESSolve` per implicit stage (`:1468`), `TSAdaptCheckStage`
  (`:1474`). With `-ts_arkimex_fully_implicit` (`:2099-2100`, `TSARKIMEXSetFullyImplicit` `:2243`) the RHS is
  folded into the implicit part; stiffly accurate tableaux then use the last stage as the solution (`:1248-1249`).
* **Fast/slow**: `-ts_arkimex_fastslowsplit` (`TSARKIMEXSetFastSlowSplit`, `:2529`) is a *component-wise*
  split by index sets (`TSRHSSplitSetIS`), not a term split. The classical IMEX split is `TSSetIFunction`
  (stiff, implicit) vs `TSSetRHSFunction` (non-stiff, explicit).

### 6.5 `TSROSW` (`src/ts/impls/rosw/rosw.c`)

* Linearly implicit: SNES is `KSPONLY` (`:1488`, `:1539`), so each implicit stage is **exactly one linear
  solve** with `M/(h·γ_ii) - J` (`shift = scoeff/dt`, `scoeff = 1/γ_ii`, `:1157`, `:1423`, `:1443`); an
  `s`-stage method costs `s` linear solves and `s` residual evaluations per step, and **one Jacobian per
  step**: on stage 0 the code sets `SNESSetLagJacobian(snes, -2)` and restores it after the last stage
  (`:1173-1180`, `TSRosWSetRecomputeJacobian` `:1640-1690`, default recompute = FALSE). One LU per step.
* Docs `:1706-1760`: "intended for problems with well-separated time scales … Consider trying TSARKIMEX if
  the stiff part is strongly nonlinear"; "currently only works with autonomous ODE and DAE"; "Since this uses
  a single linear solve per time-step if you wish to lag the Jacobian … you must use also
  `-snes_lag_jacobian_persists true`"; stage reformulation is Hairer–Wanner's `y_i = Σ γ_ij k_j` with
  `(M/(hγ_ii) - J) y_i = f(u0 + Σ a_ij y_j) + M Σ (c_ij/h) y_j`.
* Tableaux and PETSc's characterisation (doc blocks `:1540-1700`): default **`ra34pw2`** (`:18`; "Four stage
  third order L-stable Rosenbrock-W scheme for PDAE of index 1 … Only an approximate Jacobian is needed …
  strongly A-stable with R(∞) = 0, embedded order 2 with R(∞) = 0.48"); `ra3pw` (3 stages, 3rd order,
  R(∞) = 0.73); `r34prw`, `r3prl2` (Rang 2015, B_PR-consistent order 3 — no order reduction on stiff
  problems); `rodas3` (4 stages, "Both the third order and embedded second order methods are stiffly accurate
  and L-stable" — a Rosenbrock, not W: exact Jacobian); `rodaspr`/`rodaspr2` (6 stages, 4th order; `rodaspr2`
  flagged "surprisingly poor results"); `sandu3`; `2m`/`2p` ("Two stage second order L-stable Rosenbrock-W …
  Only an approximate Jacobian is needed. By default, it is only recomputed once per step"); `theta1`/`theta2`;
  `grk4t`, `shamp4`, `veldd4`, `4l` (classical 4th-order Rosenbrock, exact Jacobian).
* W-property: for `*W` methods the order conditions hold for any `W ≈ J`, which is why PETSc can lag the
  Jacobian across stages *and* accept an approximate (FD-colored, frozen-coefficient, banded-only) Jacobian
  without losing design order — the embedded error estimate also sees the inexactness.

### 6.6 `TSPSEUDO` (`src/ts/impls/pseudo/posindep.c`)

Pseudo-transient continuation, SNES `KSPONLY` by default (`:749`; docs `:655-720`: one Newton step per
pseudo-step; the Jacobian `dF/dX + shift·dF/dXdot` still contains the 1/dt term). SER step rule
(`TSPseudoTimeStepDefault`, `:393-410`):
```
dt_n = inc * dt_{n-1} * ||F(X_{n-1},0)|| / ||F(X_n,0)||      (:406)   inc = 1.1  (:757)
   or  inc * dt_0 * ||F_0|| / ||F_n||  with -ts_pseudo_increment_dt_from_initial_dt (:405)
capped by -ts_pseudo_max_dt (:407)
```
Convergence `fatol 1e-50`, `frtol 1e-12` in double (`:764-765`).

### 6.7 `TSIRK`, `TSMPRK`, `TSEIMEX`

* `TSIRK` (`irk.c:824`): Gauss–Legendre collocation, default 3 stages (order 6), fully coupled stage system
  via `MATKAIJ` — for N = 800 the stage system is 2400 unknowns.
* `TSMPRK` (`mprk.c:1246`): multirate *partitioned* explicit RK — explicit, unusable for stiff parabolic systems.
* `TSEIMEX` (`eimex.c:482`): extrapolated W-IMEX, "designed to be linearly implicit on G and can use an
  approximate and lagged Jacobian"; default 3 extrapolation rows; only `Xdot + F̂(t, X)` forms; no adaptivity.

### 6.8 How TS drives SNES

* TS-created SNES objects keep the SNES defaults of §2 (rtol 1e-8, atol 1e-50, stol 1e-8, max_its 50).
  Practitioners set `-snes_atol`, `-snes_stol`, `-snes_lag_jacobian_persists`, `-ts_max_snes_failures -1`.
  Typical Newton counts per step for BDF2 with the extrapolated predictor are 1–3 (`ts->snes_its`, `bdf.c:208-211`).
* `TSSetIJacobian` (`ts.c:1357`): the matrix is `dF/dU + a·dF/dU_t` = "Jacobian of `F(t, U, W + a·U)`", with
  `a = 1/dt`, `W = -a·U_prev` for backward Euler; the integrator supplies `a` (shift) and `W`. When only an RHS
  Jacobian is given, `TSComputeIJacobian_Internal` (`ts.c:894`) reuses the previously computed RHS Jacobian and
  only *re-shifts* it (`MatShift(A, shift - old_shift)`, `:946-950`) when state and time are unchanged, or
  rescales/shifts it (`:951-970`, `TSRHSJacobianSetReuse`) — "assemble `J_F` once, form `J_F + a I` for many
  `a`".
* Initial time step: `time_step = 0.1` (`tscreate.c:44`), **no heuristic**; `TSSetTimeStep` docs: "This is only
  a suggestion". `-ts_max_step_rejections` (`ts.c:5109`), `-ts_max_snes_failures` (`ts.c:5139`),
  `-ts_error_if_step_fails` (default true). Default TS type: `TSBEULER` when an IFunction is set, else
  `TSEULER` (`ts.c:109`, `:2547`).

### 6.9 Assessment for TORAX

* **Add a local error estimate: the Theta estimator is free.** `TSEvaluateWLTE_Theta` needs only
  `X_{n+1}, X_n, X_{n-1}` and two step sizes — no extra solve. With the `basic` law
  `dt_new = dt·clip(0.9·err^(-1/2), 0.1, 10)` (BE → exponent −1/2 as PETSc does) it replaces TORAX's
  `dt ~ dx²/max(chi)` heuristic, which is a stability (explicit CFL-like) bound irrelevant to an L-stable
  implicit method and blind to the true error. State to carry: one extra vector and one scalar (previous
  dt); accept/reject/retry is a small `while_loop`. The WRMS norm with `atol_i + rtol_i·max|u|` per channel
  (different scales for T, n, psi) is the right normalisation; `mean |r|` has none.
* **BDF2 with extrapolated predictor** is the PETSc workhorse for stiff parabolic problems: one solve per
  step like BE, second order, variable-step coefficients = three scalars from Lagrange-basis derivatives. Its
  restart protocol (BE half step) matters after sawtooth crashes. Carrying `order+1` past solutions/times in
  the loop state is straightforward with fixed shapes.
* **Rosenbrock-W (`ra34pw2`, `2m/2p`, `rodas3`)** is the best structural match for TORAX's cost profile: *no
  Newton iteration* — `s` linear solves with one Jacobian/factorisation per step, tolerating an approximate
  Jacobian (W-property). With the coloring-compressed banded Jacobian (§4) and block-Thomas (§5), a 4-stage
  step costs one 12–20-column JVP batch, one banded factorisation, 4 residual evaluations and 4
  back-substitutions — fixed-shape, no data-dependent loop. Risk: no globalisation, so a threshold crossing
  within a step shows up as a large embedded-error estimate and a rejection (dt cut up to 10×). PETSc's docs
  point at ARKIMEX/full Newton "if the stiff part is strongly nonlinear".
* **ARKIMEX** needs a stiff/non-stiff term split; in TORAX essentially everything is stiff, so it would run
  fully implicit — then it is a DIRK (default `ES213SAL` = TR-BDF2) with 2–3 Newton solves per step and an
  embedded estimate. The `1bee` trick (one BE step vs two half steps → Richardson) is the simplest error
  estimate for TORAX's existing BE solver at 3× cost; the free Theta estimate is the better trade.
* **Failure handling**: adopt PETSc's asymmetry — `dt·0.25` on a nonlinear-solve failure (not 0.5), a bounded
  number of rejections per step (10), an `increase_delay` after a failure, and *never* accept an unconverged
  solve (drop the coarse 1e-2 fallback once an error estimate exists).
* **Pseudo-transient continuation** with SER (`dt_n = 1.1·dt_{n-1}·‖F_{n-1}‖/‖F_n‖`) is the natural algorithm
  for TORAX's steady-state use and is trivially JAX-able (one scalar of state).

---

## 7. Other high-impact PETSc mechanisms

* **Grid sequencing** (`SNESSetGridSequence`, `snes.c:3751`; loop in `SNESSolve` `:4989-5046`): solve on a
  coarse DM, `DMRefine` + `MatInterpolate`, re-solve. For 25–200 cells the value would only be as a predictor
  for the first step. Low priority.
* **Nonlinear preconditioning** (`SNESSetNPC` `:5936`, `-npc_snes_type`, `npcside RIGHT` default
  `snes.c:1922`): e.g. NGMRES/Anderson outer with Newton (or Picard) inner, or Newton outer with a few
  NRICHARDSON/Picard sweeps inner (`ls.c:194-214`). "Picard inner, Anderson/NGMRES outer" is a documented PETSc
  recipe for problems where Picard is robust but slow — a direct match for TORAX's Pereverzev-stabilised
  linear mode: inner = existing predictor–corrector, outer = 3–5-vector Anderson/NGMRES with residual-based
  selection.
* **Function-domain errors** (`SNESSetFunctionDomainError` `snes.c:148`, `TSSetFunctionDomainError`): the
  sanctioned way to say "negative temperature/density: cut the step". In SNES it terminates the line-search
  trial (`linesearchimpl.h:111-124`); in TS it is checked per stage and **does not count against
  `max_snes_failures`** (`tsadapt.c:1117-1119`) — the step is retried with `dt·0.25`. TORAX relies on NaNs;
  an explicit domain predicate (`jnp.any(T < 0)`) inside the line-search loop is the faithful translation.
* **VI bounds** (`SNESVINEWTONRSLS`, §3.3) as the alternative to domain errors for positivity.
* **`TSEvent`** (`src/ts/event/tsevent.c`): indicator functions `g_i(t, U)` with direction and terminate flags
  (`TSSetEventHandler` `:309`, docs `:255-350`); zero crossings located by a modified Anderson–Björck regula
  falsi that drifts toward bisection in hard cases (`RefineAndersonBjorck`, `:500-560`); tolerance
  `-ts_event_tol 1e-6` (`:315`); the step is cut to land on the event and `postevent` may modify the state;
  post-event step sizes `dt1`, `dt2` default `PETSC_DECIDE` (`:353-354`, semantics `:90-135`); `TSAdapt` is
  bypassed while processing (`tsadapt.c:956-976`); after the event `steprestart` is set so BDF re-seeds and
  Theta skips its LTE estimate. For sawteeth (`q_0 = 1` crossing) and L–H transitions (power threshold) this
  is exactly what TORAX lacks: locate the crossing to tolerance, apply the crash/pedestal model as a state
  jump, restart the multistep history. In JAX: detect a sign change of `g` after a step, bracket-refine with a
  fixed number of secant/bisection iterations (each re-doing the implicit step with smaller dt — fixed-trip
  `fori_loop`), apply the jump, reset the history.
* **Discrete adjoints** (`src/ts/interface/sensitivity/tssen.c`): `TSAdjointSolve` (`:1574`, "Solves the
  discrete adjoint problem"; must follow `TSSolve`), `TSSetCostGradients` (`:886`: initialise
  `lambda = df/dy|_T`, `mu = df/dp|_T`; on return they hold the sensitivities). The forward trajectory is
  checkpointed by `TSTrajectory` (`TSTRAJECTORYBASIC` to disk by default, `TSTRAJECTORYMEMORY` in memory);
  each `TSAdjointStep` applies the transpose of the step map, requiring transpose Jacobian solves
  (`SNESKSPTRANSPOSEONLY`, `ksponly.c:98-110`) and `dF/dp`. Adjoint stepping is implemented only for `theta`,
  `arkimex` and `rk` (`ops->adjointstep` in `theta.c`, `arkimex.c`, `explicit/rk/rk.c`). Design note: PETSc
  differentiates the *discrete* scheme exactly but through the *converged* Newton solution (implicit-function
  theorem per step), not through the Newton iterations. JAX reverse mode through `lax.while_loop` is not
  supported, so TORAX's differentiability already implies fixed-trip loops or an implicit-function custom VJP —
  the latter is exactly the PETSc design (one transpose solve with the final Jacobian per step), and it also
  removes the Newton iterations from the gradient graph, cutting memory and compile time.
* **Eisenstat–Walker** (`SNESKSPSetUseEW`, `snes.c:5564`): only meaningful with an iterative KSP; irrelevant
  for direct/banded solves.

---

## 8. Consolidated, ranked recommendations for TORAX

1. **Coloring-compressed exact Jacobian** (§4): 12–20 batched JVPs instead of N; banded block-tridiagonal
   result; global couplings by stencil widening, lagging, or a low-rank Woodbury correction.
2. **Block-Thomas / banded LU** (§5) instead of dense LU: 10²–10⁴× fewer flops, no N×N array.
3. **Free local error estimate + PETSc `basic` controller** for the existing BE/theta scheme (§6.3, §6.1):
   `dt_new = dt·clip(0.9·err^(-1/2), 0.1, 10)`, WRMS norm with per-channel `atol/rtol`, reject if `err > 1`,
   `dt·0.25` on Newton failure, ≤ 10 rejections/step. Then consider **BDF2** (§6.2) or **Rosenbrock-W
   `ra34pw2`/`2m`** (§6.5) — the latter eliminates the Newton loop and is the most XLA-friendly.
4. **Jacobian lagging with a proper line search** (§3.1, §1.2): one Jacobian per step (`lag -2`/`persists`),
   `bt` with Armijo + quadratic/cubic fit + NaN pre-loop, `stol` step test and `divtol` divergence test (§2).
5. **Anderson/NGMRES (m = 3–5) as an accelerator of the Picard/linear mode**, possibly as nonlinear
   preconditioning (§3.3, §7).
6. **Event handling** for sawteeth / L–H transitions (§7), with multistep-history restart.
7. **Implicit-function adjoint** per step (§7) if reverse-mode cost/compile time is an issue.

What does *not* transfer: FD differencing-parameter machinery (`MatMFFD`, `umin`, `error_rel` — replaced by
exact JVPs), parallel colorings (JP/greedy conflict resolution), PCBJACOBI/FAS/grid sequencing (N too small),
IRK (stage system 3N), MPRK (explicit), and PETSc's mutable-object conventions (lag counters, `keeplambda`,
DSP histories) which must become explicit loop-carried state in JAX.