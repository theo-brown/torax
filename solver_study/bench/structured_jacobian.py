"""Prototype of the structured (chain-rule + colouring) Jacobian assembly.

J = dG/dx|_c + dG/dc . S . dh/dx, where
  h(x): raw (pre-smoothing) turbulent transport coefficients (4 face arrays),
  S   : the fixed Gaussian smoothing matrix,
  G(x, c): the theta-method residual with the smoothed turbulent coefficients
           injected (everything else - neoclassical, sources, ions, q, s,
           assembly - still depends on x).
Each factor is banded and is computed from a few coloured batched JVPs.
No TORAX source is modified: the smoothing hook is overridden in a subclass of
TransportModel, and the injected coefficients travel inside a subclass of
PedestalModelOutput (a pytree) so they can be JAX tracers.
"""
import dataclasses, functools, json, sys, time
import numpy as np
import jax, jax.numpy as jnp
from torax.examples import iterhybrid_rampup
from torax._src.transport_model import transport_model as tm_lib
from torax._src.transport_model import transport_coeffs as tc_lib
from torax._src.transport_model import transport_coefficients_builder as tcb
from torax._src.pedestal_model import pedestal_model_output as pmo_lib
from torax._src.fvm import residual_and_loss, fvm_conversions
from torax._src.core_profiles import updaters
from torax._src.solver import jax_root_finding
import bench_common as bc
from jac_structure_lib import greedy_column_coloring

n_rho = int(sys.argv[1]) if len(sys.argv) > 1 else 50
OUT = {'n_rho': n_rho}


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PMOInjected(pmo_lib.PedestalModelOutput):
  """PedestalModelOutput carrying injected (already smoothed) coefficients."""
  injected_transport: tc_lib.TransportCoeffs | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class StructuredTransportModel(tm_lib.TransportModel):
  """TransportModel whose smoothing step is either the identity ('raw') or
  replaced by injected coefficients ('inject')."""
  mode: str = 'raw'

  def _smooth_coeffs(self, runtime_params, geo, input_coeffs, pedestal_model_output):
    if self.mode == 'raw':
      return input_coeffs
    assert isinstance(pedestal_model_output, PMOInjected)
    return pedestal_model_output.injected_transport


p = bc.build_problem(iterhybrid_rampup.CONFIG, n_warm_steps=3, n_rho=n_rho)
N, n_ch, n_cells = p.n, len(p.evolving_names), p.n_rho
n_face = n_cells + 1
models = p.step_fn.solver.models
tmod = models.transport_model
mk = lambda mode: dataclasses.replace(models, transport_model=StructuredTransportModel(
    core_transport_models=tmod.core_transport_models, pedestal_transport_models=tmod.pedestal_transport_models, mode=mode))
models_raw, models_inj = mk('raw'), mk('inject')
pmo0 = p.pedestal_transition_state.pedestal_model_output
pmo_fields = {f.name: getattr(pmo0, f.name) for f in dataclasses.fields(pmo0)}
rp, geo, pts = p.runtime_params_t_plus_dt, p.geo_t_plus_dt, p.pedestal_transition_state

# --- S: the smoothing matrix (fixed during the solve) ---
S = np.asarray(tm_lib._build_smoothing_matrix(rp.transport, rp, geo, pmo0))

def vec_to_tc(cvec):
  c = cvec.reshape(4, n_face)
  return tc_lib.TransportCoeffs(chi_face_ion=c[0], chi_face_el=c[1], d_face_el=c[2], v_face_el=c[3])

def h(x):
  """Raw turbulent transport coefficients (4*n_face,) as a function of x."""
  x_cv = fvm_conversions.vec_to_cell_variable_tuple(x, p.core_profiles_t_plus_dt, p.evolving_names)
  cp = updaters.update_core_profiles_during_step(x_cv, rp, geo, p.core_profiles_t_plus_dt, prev_core_profiles=p.core_profiles_t, dt=p.dt, evolving_names=p.evolving_names)
  ct = tcb.calculate_all_transport_coeffs(transport_model=models_raw.transport_model, neoclassical_models=models_raw.neoclassical_models,
      internal_boundary_condition_model=models_raw.internal_boundary_condition_model, runtime_params=rp, geo=geo, core_profiles=cp,
      pedestal_transition_state=pts, use_pereverzev=False)
  t = ct.turbulent.total
  return jnp.concatenate([t.chi_face_ion, t.chi_face_el, t.d_face_el, t.v_face_el])

def G(x, cvec):
  """Residual with the smoothed turbulent coefficients injected."""
  pts_inj = dataclasses.replace(pts, pedestal_model_output=PMOInjected(**pmo_fields, injected_transport=vec_to_tc(cvec)))
  return residual_and_loss.theta_method_block_residual(
      x, dt=p.dt, runtime_params_t_plus_dt=rp, geo_t_plus_dt=geo, x_old=p.x_old, core_profiles_t=p.core_profiles_t,
      core_profiles_t_plus_dt=p.core_profiles_t_plus_dt, explicit_source_profiles=p.explicit_source_profiles, models=models_inj,
      coeffs_old=p.coeffs_old, evolving_names=p.evolving_names, pedestal_transition_state=pts_inj)

def apply_S(cvec):
  return (jnp.asarray(S) @ cvec.reshape(4, n_face).T).T.reshape(-1)

res = jax.jit(p.residual_fun); jacf = jax.jit(jax.jacfwd(p.residual_fun))
h_j, G_j = jax.jit(h), jax.jit(G)
x0 = p.x0
c0 = apply_S(h_j(x0))
r_ref = np.asarray(res(x0)); r_inj = np.asarray(G_j(x0, c0))
OUT['injection_rel_err'] = float(np.max(np.abs(r_inj - r_ref)) / np.max(np.abs(r_ref)))
print(f'N={N} n_rho={n_rho}: residual with injected S.h(x) vs original: max rel diff {OUT["injection_rel_err"]:.2e}')

# --- structural patterns of the three factors, from dense reference Jacobians at two states ---
x_star, meta = jax_root_finding.root_newton_raphson(p.residual_fun, x0, maxiter=30, tol=1e-5, coarse_tol=1e-2, use_jax_custom_root=False)
Jx_f = jax.jit(lambda x, c: jax.jacfwd(lambda xx: G(xx, c))(x))
Hx_f = jax.jit(jax.jacfwd(h))
Gc_f = jax.jit(lambda x, c: jax.jacfwd(lambda cc: G(x, cc))(c))
def pattern_union(fn, states):
  pat = None
  for s in states:
    A = np.asarray(fn(*s)); m = np.abs(A) > 0
    pat = m if pat is None else (pat | m)
  return pat
c_star = apply_S(h_j(x_star))
P_x = pattern_union(Jx_f, [(x0, c0), (x_star, c_star), (p.x_old_vec, apply_S(h_j(p.x_old_vec)))])
P_h = pattern_union(Hx_f, [(x0,), (x_star,), (p.x_old_vec,)])
P_c = pattern_union(Gc_f, [(x0, c0), (x_star, c_star)])
def bandwidth_cells(P, rows_cell, cols_cell):
  r, c = np.nonzero(P); return int(np.max(np.abs(rows_cell[r] - cols_cell[c]))) if len(r) else 0
cell_of_x = np.arange(N) % n_cells; face_of_c = np.arange(4 * n_face) % n_face
OUT['bw_Gx_cells'] = bandwidth_cells(P_x, cell_of_x, cell_of_x)
OUT['bw_Hx_face_vs_cell'] = bandwidth_cells(P_h, face_of_c, cell_of_x)
OUT['bw_Gc_cell_vs_face'] = bandwidth_cells(P_c, cell_of_x, face_of_c)
col_x = greedy_colours = None
def colouring(P):
  S_ = P.astype(np.int32); C = (S_.T @ S_) > 0; np.fill_diagonal(C, False)
  n = P.shape[1]; colours = -np.ones(n, dtype=int)
  for j in range(n):
    used = set(colours[C[j]][colours[C[j]] >= 0].tolist()); c = 0
    while c in used: c += 1
    colours[j] = c
  return colours
col_x, col_h, col_c = colouring(P_x), colouring(P_h), colouring(P_c)
k_x, k_h, k_c = int(col_x.max() + 1), int(col_h.max() + 1), int(col_c.max() + 1)
OUT.update(k_x=k_x, k_h=k_h, k_c=k_c, nnz_Gx=int(P_x.sum()), nnz_Hx=int(P_h.sum()), nnz_Gc=int(P_c.sum()))
print(f'patterns: dG/dx bandwidth {OUT["bw_Gx_cells"]} cells -> {k_x} colours; dh/dx reach {OUT["bw_Hx_face_vs_cell"]} -> {k_h} colours; dG/dc reach {OUT["bw_Gc_cell_vs_face"]} -> {k_c} colours; total seeds {k_x + k_h + k_c} vs N={N}')

# --- coloured assembly ---
def seeds(col, n_in):
  k = int(col.max() + 1); E = np.zeros((k, n_in)); E[col, np.arange(n_in)] = 1.0; return jnp.asarray(E)
E_x, E_h, E_c = seeds(col_x, N), seeds(col_h, N), seeds(col_c, 4 * n_face)
Pj_x, Pj_h, Pj_c = jnp.asarray(P_x), jnp.asarray(P_h), jnp.asarray(P_c)
cj_x, cj_h, cj_c = jnp.asarray(col_x), jnp.asarray(col_h), jnp.asarray(col_c)
Sj = jnp.asarray(S)

def cjvp(f, x, E):
  return jax.vmap(lambda v: jax.jvp(f, (x,), (v,))[1])(E)          # (k, n_out)

def decompress(C, col, P):
  return jnp.where(P, C[col].T, 0.0)                                # (n_out, n_in)

def structured_jacobian(x):
  craw = h(x)
  c = apply_S(craw)
  Jx = decompress(cjvp(lambda xx: G(xx, c), x, E_x), cj_x, Pj_x)            # N x N
  Hx = decompress(cjvp(h, x, E_h), cj_h, Pj_h)                              # 4n_face x N
  Gc = decompress(cjvp(lambda cc: G(x, cc), c, E_c), cj_c, Pj_c)            # N x 4n_face
  SHx = (Sj @ Hx.reshape(4, n_face, N)).reshape(4 * n_face, N)              # S applied per coefficient
  return Jx + Gc @ SHx

sj = jax.jit(structured_jacobian)
t0 = time.perf_counter(); J_s = np.asarray(sj(x0)); OUT['compile_structured_s'] = time.perf_counter() - t0
t0 = time.perf_counter(); J_ref = np.asarray(jacf(x0)); OUT['compile_jacfwd_s'] = time.perf_counter() - t0
def relerr(A, B): return float(np.linalg.norm(A - B) / np.linalg.norm(B))
OUT['relerr_x0'] = relerr(J_s, J_ref)
OUT['relerr_xstar'] = relerr(np.asarray(sj(x_star)), np.asarray(jacf(x_star)))
x_mid = 0.5 * (x0 + p.x_old_vec)
OUT['relerr_xmid'] = relerr(np.asarray(sj(x_mid)), np.asarray(jacf(x_mid)))
print(f'exactness (rel Frobenius error vs jacfwd): x0 {OUT["relerr_x0"]:.2e}, x* {OUT["relerr_xstar"]:.2e}, midpoint {OUT["relerr_xmid"]:.2e}')

# --- timings ---
T = {}
T['jacfwd'] = bc.timeit(jacf, x0, n=5)[0]
T['structured_total'] = bc.timeit(sj, x0, n=5)[0]
T['h(x)'] = bc.timeit(h_j, x0, n=5)[0]
T['cjvp_Gx'] = bc.timeit(jax.jit(lambda x, c: cjvp(lambda xx: G(xx, c), x, E_x)), x0, c0, n=5)[0]
T['cjvp_Hx'] = bc.timeit(jax.jit(lambda x: cjvp(h, x, E_h)), x0, n=5)[0]
T['cjvp_Gc'] = bc.timeit(jax.jit(lambda x, c: cjvp(lambda cc: G(x, cc), c, E_c)), x0, c0, n=5)[0]
T['residual'] = bc.timeit(res, x0, n=10)[0]
OUT['timings_ms'] = {k: v * 1e3 for k, v in T.items()}
print('timings (ms): ' + ', '.join(f'{k}={v*1e3:.2f}' for k, v in T.items()))
print(f'compile: structured {OUT["compile_structured_s"]:.1f} s, jacfwd {OUT["compile_jacfwd_s"]:.1f} s')

# --- Newton solve using the structured Jacobian ---
nr_base = jax.jit(lambda x: jax_root_finding.root_newton_raphson(p.residual_fun, x, maxiter=30, tol=1e-5, coarse_tol=1e-2, use_jax_custom_root=False))
nr_struct = jax.jit(lambda x: jax_root_finding.root_newton_raphson(p.residual_fun, x, maxiter=30, tol=1e-5, coarse_tol=1e-2, use_jax_custom_root=False, custom_jac=structured_jacobian))
t0 = time.perf_counter(); xb, mb = nr_base(x0); jax.block_until_ready(xb); OUT['compile_newton_base_s'] = time.perf_counter() - t0
t0 = time.perf_counter(); xs_, ms = nr_struct(x0); jax.block_until_ready(xs_); OUT['compile_newton_struct_s'] = time.perf_counter() - t0
OUT['newton_its_base'] = int(mb.iterations); OUT['newton_its_struct'] = int(ms.iterations)
OUT['newton_sol_rel_diff'] = float(np.linalg.norm(np.asarray(xb) - np.asarray(xs_)) / np.linalg.norm(np.asarray(xb)))
OUT['t_newton_base_ms'] = bc.timeit(nr_base, x0, n=3)[0] * 1e3
OUT['t_newton_struct_ms'] = bc.timeit(nr_struct, x0, n=3)[0] * 1e3
print(f'Newton solve: baseline {OUT["t_newton_base_ms"]:.1f} ms ({OUT["newton_its_base"]} it, compile {OUT["compile_newton_base_s"]:.1f} s) | structured {OUT["t_newton_struct_ms"]:.1f} ms ({OUT["newton_its_struct"]} it, compile {OUT["compile_newton_struct_s"]:.1f} s) | solution rel diff {OUT["newton_sol_rel_diff"]:.1e}')
json.dump(OUT, open(f'structured_jacobian_n{n_rho}.json', 'w'), indent=1)
print('DONE')
