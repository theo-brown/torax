"""Attribute the non-solver part of a Newton step at N=100."""
import copy, time, json
import jax, jax.numpy as jnp
import torax
from torax.examples import iterhybrid_predictor_corrector as pc
from torax._src.orchestration import run_simulation as rs, step_function_processing
import bench_common as bc

p = bc.build_problem(copy.deepcopy(pc.CONFIG), n_warm_steps=3, solver_overrides={'solver_type': 'newton_raphson'})
sf = p.step_fn; T = {}
T['full step newton, adaptive_dt (default)'] = bc.timeit(lambda: sf(p.state, p.ppo), n=5)[0]
cfg = copy.deepcopy(pc.CONFIG); cfg['solver']['solver_type'] = 'newton_raphson'; cfg['numerics']['adaptive_dt'] = False
_, _, sf_fixed = rs.prepare_simulation(torax.ToraxConfig.from_dict(cfg))
T['full step newton, adaptive_dt=False'] = bc.timeit(lambda: sf_fixed(p.state, p.ppo), n=5)[0]
cfg = copy.deepcopy(pc.CONFIG); cfg['solver']['solver_type'] = 'newton_raphson'; cfg['solver']['use_predictor_corrector'] = False
_, _, sf_nopc = rs.prepare_simulation(torax.ToraxConfig.from_dict(cfg))
T['full step newton, no PC guess'] = bc.timeit(lambda: sf_nopc(p.state, p.ppo), n=5)[0]
models = sf.solver.models
pre = jax.jit(lambda st: step_function_processing.pre_step(input_state=st, runtime_params_provider=sf.runtime_params_provider, geometry_provider=sf.geometry_provider, models=models))
T['pre_step'] = bc.timeit(pre, p.state, n=10)[0]
x_new, sno = sf.solver(t=p.state.t, dt=p.dt, runtime_params_t=p.runtime_params_t, runtime_params_t_plus_dt=p.runtime_params_t_plus_dt, geo_t=p.geo_t, geo_t_plus_dt=p.geo_t_plus_dt, core_profiles_t=p.core_profiles_t, core_profiles_t_plus_dt=p.core_profiles_t_plus_dt, explicit_source_profiles=p.explicit_source_profiles, pedestal_transition_state=p.pedestal_transition_state)
fin = jax.jit(lambda x_new, sno: step_function_processing.finalize_outputs(t=p.state.t, dt=p.dt, x_new=x_new, solver_numeric_outputs=sno, runtime_params_t_plus_dt=p.runtime_params_t_plus_dt, geometry_t_plus_dt=p.geo_t_plus_dt, core_profiles_t=p.core_profiles_t, core_profiles_t_plus_dt=p.core_profiles_t_plus_dt, explicit_source_profiles=p.explicit_source_profiles, edge_outputs=p.edge_outputs, models=models, evolving_names=p.evolving_names, input_post_processed_outputs=p.ppo, time_step_calculator_state_t=p.state.time_step_calculator_state, pedestal_transition_state=p.pedestal_transition_state))
T['finalize_outputs'] = bc.timeit(fin, x_new, sno, n=10)[0]
dtc = jax.jit(lambda st: sf.time_step_calculator.next_dt(p.runtime_params_t, st))
T['time_step_calculator.next_dt'] = bc.timeit(dtc, p.state, n=10)[0]
for k, v in T.items(): print(f'  {k:42s} {v*1e3:8.2f} ms')
json.dump({k: v*1e3 for k, v in T.items()}, open('overhead_n25b.json', 'w'), indent=1)
