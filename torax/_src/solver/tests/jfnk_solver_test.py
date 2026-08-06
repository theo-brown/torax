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
"""End-to-end tests of the Jacobian-free Newton-Krylov solver option.

The unit tests for the pieces live next to the pieces: the GMRES solve and the
`linear_solver='jfnk'` path in `jax_root_finding_test.py`, the preconditioner
and the vector layout in `fvm/tests/residual_and_loss_test.py`. What is tested
here is the wiring between them -- that the config option reaches the solver,
that the preconditioner is handed the operator in the layout it expects, and
that the result is a trajectory rather than just a converged linear solve.
"""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.orchestration import run_simulation
from torax._src.test_utils import default_sources
from torax._src.torax_pydantic import model_config


# Four evolving channels, so the block structure of the preconditioner is
# non-trivial and a layout error would not cancel out.
_EVOLVED = dict(
    evolve_ion_heat=True,
    evolve_electron_heat=True,
    evolve_density=True,
    evolve_current=True,
    t_final=0.2,
)

_MAX_STEPS = 3


def _config(**solver_overrides: Any) -> model_config.ToraxConfig:
  return model_config.ToraxConfig.from_dict(
      dict(
          numerics=_EVOLVED,
          plasma_composition=dict(),
          profile_conditions=dict(),
          geometry=dict(geometry_type='circular', n_rho=25),
          pedestal=dict(),
          sources=default_sources.get_default_source_config(),
          solver=dict(
              solver_type='newton_raphson',
              use_predictor_corrector=False,
              theta_implicit=1.0,
              **solver_overrides,
          ),
          transport=dict(
              model_name='combined',
              transport_models=[dict(model_name='constant', chi_i=1.0)],
              chi_min=0,
          ),
          time_step_calculator=dict(),
      )
  )


def _run(**solver_overrides: Any):
  _, state_history = run_simulation.run_simulation(
      _config(**solver_overrides), progress_bar=False, max_steps=_MAX_STEPS
  )
  return state_history


class JfnkSolverTest(parameterized.TestCase):

  def test_jfnk_reproduces_the_direct_solver_trajectory(self):
    """A tight Krylov tolerance must give the same answer as a dense solve.

    JFNK differs from the direct solver only in how the Newton system is
    solved, so with a forcing term well below the Newton tolerance the two must
    agree to within the inexactness of the Krylov solve.
    """
    direct = _run(newton_linear_solver='direct')
    jfnk = _run(newton_linear_solver='jfnk', jfnk_rtol=1e-8)

    for name in ('T_i', 'T_e', 'n_e', 'psi'):
      with self.subTest(name):
        np.testing.assert_allclose(
            getattr(direct._stacked_core_profiles, name).value,
            getattr(jfnk._stacked_core_profiles, name).value,
            rtol=1e-6,
        )

  def test_jfnk_takes_the_same_newton_iterations(self):
    """An exact Krylov solve reproduces the exact Newton step.

    The Newton iteration count is the sensitive quantity: it is unchanged only
    if each Krylov solve actually reaches its tolerance within the iteration
    budget, which in turn requires the preconditioner to be effective. Handing
    the operator to `tridiagonal.solve` in the wrong layout, for instance,
    leaves GMRES to exhaust `jfnk_max_krylov` on an unclustered spectrum.
    """
    direct = _run(newton_linear_solver='direct')
    jfnk = _run(newton_linear_solver='jfnk', jfnk_rtol=1e-8)

    direct_outputs = direct._stacked_solver_numeric_outputs
    jfnk_outputs = jfnk._stacked_solver_numeric_outputs
    np.testing.assert_array_equal(
        np.asarray(direct_outputs.inner_solver_iterations),
        np.asarray(jfnk_outputs.inner_solver_iterations),
    )
    self.assertEqual(int(np.asarray(jfnk_outputs.solver_error_state).max()), 0)


if __name__ == '__main__':
  absltest.main()
