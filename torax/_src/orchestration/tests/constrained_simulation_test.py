# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Closed-loop tests of constraint/actuator pairs through run_simulation.

These tests exercise the full orchestration plumbing: the constraint config
on ToraxConfig, actuator state carried across steps in SolverNumericOutputs,
injection of the previous step's values into the runtime params, and the
bordered Newton solve inside the step function.
"""

from absl.testing import absltest
import numpy as np
from torax._src.orchestration import run_simulation
from torax._src.test_utils import default_sources
from torax._src.torax_pydantic import model_config

_NUM_CELLS = 16
_DT = 0.2
_T_FINAL = 1.0
_TARGET = 1.05e20


def _build_config(
    constraints=None,
    time_step_calculator=None,
    t_final=None,
    chi_timestep_prefactor=None,
):
  numerics = dict(
      evolve_ion_heat=True,
      evolve_electron_heat=True,
      evolve_density=True,
      evolve_current=True,
      t_final=t_final if t_final is not None else _T_FINAL,
      fixed_dt=_DT,
      exact_t_final=False,
  )
  if chi_timestep_prefactor is not None:
    numerics['chi_timestep_prefactor'] = chi_timestep_prefactor
  return model_config.ToraxConfig.from_dict(
      dict(
          numerics=numerics,
          plasma_composition=dict(),
          profile_conditions=dict(),
          geometry=dict(geometry_type='circular', n_rho=_NUM_CELLS),
          pedestal=dict(),
          sources=default_sources.get_default_source_config(),
          solver=dict(
              solver_type='newton_raphson',
              use_predictor_corrector=False,
              theta_implicit=1.0,
              residual_tol=1e-9,
          ),
          transport=dict(
              model_name='combined',
              transport_models=[dict(model_name='constant', chi_i=1.0)],
              chi_min=0,
          ),
          time_step_calculator=(
              time_step_calculator
              if time_step_calculator is not None
              else dict(calculator_type='fixed')
          ),
          constraints=constraints or (),
      )
  )


def _run(torax_config):
  """Runs the simulation and extracts (times, nbar, u_hat, error) series."""
  _, history = run_simulation.run_simulation(
      torax_config, progress_bar=False
  )
  times, nbars, actuators, errors = [], [], [], []
  for t, cp, geo, sno in zip(
      history.times,
      history.core_profiles,
      history.geometries,
      history.solver_numeric_outputs,
  ):
    # Same quadrature as the constraint row: line average over rho_norm.
    nbar = float(
        np.sum(
            np.asarray(cp.n_e.value) * np.diff(np.asarray(geo.rho_face_norm))
        )
    )
    times.append(float(t))
    nbars.append(nbar)
    actuators.append(
        None if sno.actuators is None else float(sno.actuators[0])
    )
    errors.append(int(sno.solver_error_state))
  return times, nbars, actuators, errors


class ConstrainedSimulationTest(absltest.TestCase):

  def test_relaxed_constraint_tracks_target_where_free_run_drifts(self):
    """Tight relaxation holds nbar near target; the free run drifts away."""
    constrained = _build_config(
        constraints=[
            dict(
                constraint='n_e_line_avg',
                target=_TARGET,
                mode='relaxed',
                tau=1e-4,
            )
        ]
    )
    free = _build_config()

    _, nbars, actuators, errors = _run(constrained)
    _, nbars_free, actuators_free, _ = _run(free)

    self.assertTrue(all(e == 0 for e in errors))
    # The free run has no actuator state.
    self.assertTrue(all(a is None for a in actuators_free))
    # The constrained run records a varying actuator trajectory.
    self.assertIsNotNone(actuators[-1])
    self.assertNotEqual(actuators[1], actuators[-1])
    # From the first solved step onward the density tracks the target...
    for nbar in nbars[1:]:
      self.assertLess(abs(nbar - _TARGET) / _TARGET, 2e-2)
    # ...while the free run drifts far beyond it.
    self.assertGreater(abs(nbars_free[-1] - _TARGET) / _TARGET, 0.15)

  def test_relaxed_constraint_satisfies_identity_across_steps(self):
    """Every step satisfies tau (u_k - u_{k-1}) / dt + g_k = 0 in sequence.

    This is the closed-loop check that the actuator state is threaded
    correctly: step k's relaxation must start from step k-1's converged
    value, so any break in the SimState hand-off shows up as a violated
    identity.
    """
    tau = 0.05
    torax_config = _build_config(
        constraints=[
            dict(
                constraint='n_e_line_avg',
                target=_TARGET,
                mode='relaxed',
                tau=tau,
            )
        ]
    )
    times, nbars, actuators, errors = _run(torax_config)
    self.assertTrue(all(e == 0 for e in errors))
    for k in range(1, len(times)):
      g_hat = (nbars[k] - _TARGET) / _TARGET
      dt = times[k] - times[k - 1]
      identity = tau * (actuators[k] - actuators[k - 1]) / dt + g_hat
      self.assertLess(abs(identity), 1e-6, msg=f'step {k}')
    # While above target the controller must wind the fuelling down.
    self.assertLess(actuators[-1], actuators[1])

  def test_hard_constraint_holds_target_every_step(self):
    torax_config = _build_config(
        constraints=[
            dict(constraint='n_e_line_avg', target=_TARGET, mode='hard')
        ]
    )
    _, nbars, actuators, errors = _run(torax_config)
    self.assertTrue(all(e == 0 for e in errors))
    # The initial state does not satisfy the constraint; every solved step
    # must, exactly.
    for nbar in nbars[1:]:
      self.assertLess(abs(nbar - _TARGET) / _TARGET, 1e-6)
    self.assertIsNotNone(actuators[-1])

  def test_constraint_on_adaptive_timestep_path(self):
    """The adaptive-dt path threads actuator state through its retry loop."""
    torax_config = _build_config(
        constraints=[
            dict(
                constraint='n_e_line_avg',
                target=_TARGET,
                mode='relaxed',
                tau=0.05,
            )
        ],
        time_step_calculator=dict(calculator_type='chi'),
        t_final=0.3,
        chi_timestep_prefactor=10,
    )
    times, nbars, actuators, errors = _run(torax_config)
    self.assertGreater(len(times), 2)
    self.assertTrue(all(e == 0 for e in errors))
    self.assertTrue(all(np.isfinite(a) for a in actuators))
    # The per-step relaxation identity holds with the adaptive dt.
    for k in range(1, len(times)):
      g_hat = (nbars[k] - _TARGET) / _TARGET
      dt = times[k] - times[k - 1]
      identity = 0.05 * (actuators[k] - actuators[k - 1]) / dt + g_hat
      self.assertLess(abs(identity), 1e-6, msg=f'step {k}')


if __name__ == '__main__':
  absltest.main()
