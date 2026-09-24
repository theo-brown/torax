"""Whole simulations of the formerly unsupported configurations.

python e2e_gaps.py CASE N_RHO T_FINAL MODE OUT.jsonl
Builds CASE as gaps_compare.py does, runs torax.run_simulation with
solver.jacobian_mode = MODE and appends the final profiles, the Newton
iterations and the step count to OUT.jsonl. Run the structured mode with
TORAX_ERRORS_ENABLED=True: every Newton iteration then checks the assembled
Jacobian against a JVP of the residual inside the jitted solver, and raises if
a row disagrees.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torax  # noqa: E402
import gaps_compare  # noqa: E402

case, n_rho, t_final = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
mode, out = sys.argv[4], sys.argv[5]
config = gaps_compare.build(case, n_rho)
config['solver']['jacobian_mode'] = mode
config['numerics']['t_final'] = t_final
t0 = time.perf_counter()
_, history = torax.run_simulation(
    torax.ToraxConfig.from_dict(config), progress_bar=False
)
wall = time.perf_counter() - t0
final = history.core_profiles[-1]
numerics = history.solver_numeric_outputs
# In ADAPTIVE_TRANSPORT H-mode the pedestal multipliers lower chi there.
pedestal = np.asarray(history.rho_face_norm) > 0.9
record = dict(
    case=case,
    n_rho=n_rho,
    mode=mode,
    errors_enabled=os.environ.get('TORAX_ERRORS_ENABLED', ''),
    wall_s=wall,
    sim_error=str(history.sim_error),
    t_end=float(history.times[-1]),
    steps=len(history.times) - 1,
    newton_iterations=int(sum(int(s.inner_solver_iterations) for s in numerics)),
    outer_iterations=int(sum(int(s.outer_solver_iterations) for s in numerics)),
    coarse_tol_steps=int(sum(int(s.solver_error_state) == 2 for s in numerics)),
    chi_e_pedestal_min=[
        float(np.min(np.asarray(t.turbulent.total.chi_face_el)[pedestal]))
        for t in history.core_transport
    ],
    **{
        name: np.asarray(getattr(final, name).value).tolist()
        for name in ('T_i', 'T_e', 'n_e', 'psi')
    },
)
with open(out, 'a') as f:
  f.write(json.dumps(record) + '\n')
print(
    f'RESULT {case} n_rho={n_rho} mode={mode} steps={record["steps"]}'
    f' t_end={record["t_end"]:.3f} newton={record["newton_iterations"]}'
    f' wall={wall:.1f}s {record["sim_error"]}',
    flush=True,
)
