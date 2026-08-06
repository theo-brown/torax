.. _constraints-and-actuators:

Adding constraints and actuators to the PDE system
==================================================

This is a design note for extending TORAX with *constrained* transport
problems: cases where we want to impose a physics target and let the solver
find the actuator value that achieves it. Examples:

* "line-averaged density = :math:`Y`", with unknown "gas puff rate = :math:`X`"
* "pedestal height = :math:`Y`", with unknown "pedestal transport = :math:`X`"

It explains how such pairs slot into the Newton solver as extra rows and
columns of the Jacobian, why that formulation is preferred over feedback
inside a source model, and how to soften a hard constraint into a temporal
relaxation when controller-like dynamics are wanted.

The bordered system
-------------------

The nonlinear solver currently finds the profile vector :math:`x` (all
evolving channels stacked, :math:`4 n_\rho` unknowns) satisfying the
theta-method residual :math:`R(x) = 0`. Each constraint/actuator pair adds
one scalar unknown :math:`u_j` (the actuator) and one scalar equation
:math:`g_j(x) = 0` (the constraint). With :math:`m` pairs, the augmented
Newton system has an *arrowhead* (bordered) shape:

.. code-block:: none

            profiles   actuators
          +----------+---------+
          |          |         |
   PDE    |   J_xx   |  J_xu   |     J_xx : (4N x 4N) PDE block, sparse
   rows   |          |         |     J_xu : (4N x m)  actuator columns
          +----------+---------+     G_x  : (m x 4N)  constraint rows
   constr |   G_x    |  G_u    |     G_u  : (m x m)   often zero
   rows   +----------+---------+

* ``J_xx`` is the existing Jacobian: the FVM stencil, the local coefficient
  halo, and the transport smoothing. Its sparsity pattern is unchanged.
* Each **actuator column** ``J_xu[:, j]`` records how actuator :math:`u_j`
  enters the PDE — e.g. the gas puff rate scales a prescribed deposition
  profile, so its column is simply that profile shape.
* Each **constraint row** ``G_x[j, :]`` records how constraint :math:`g_j`
  reads the state — e.g. a line average is a fixed quadrature over the
  :math:`n_e` cells.
* ``G_u`` is zero unless a constraint reads an actuator directly.

Why a border, and not feedback inside a source
----------------------------------------------

The same physics could be written as a feedback controller inside a source
model: "gas puff amplitude = f(line-averaged density)" evaluated inside the
residual. Numerically this is strictly worse. A global quantity evaluated
inside the residual couples *every* affected equation to *every* cell it
integrates over, planting a dense block inside ``J_xx``. That destroys the
sparsity contract that the declared Jacobian pattern
(``torax/_src/solver/jacobian_pattern.py``) and its probe coloring rely on,
and it is invisible until the pattern verification probe catches it. (The
existing ``cyclotron_radiation`` and ``P_in_scaled_flat_profile`` sources
have exactly this structure; see the module docstring.)

Written as a border pair, the *same* global coupling occupies one explicit
row and one explicit column. The PDE block stays sparse, and the border is
the most declarable structure in the whole system — usually analytic:

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Pair
     - Constraint row ``G_x``
     - Actuator column ``J_xu``
   * - line-avg density / gas puff rate
     - quadrature weights of the line average over the :math:`n_e` block —
       linear, state-independent, known from geometry
     - the prescribed deposition shape — the source is linear in its
       amplitude, so the column is the shape itself
   * - pedestal height / pedestal transport
     - interpolation stencil at :math:`\rho_{ped}` over the :math:`T_e`
       block — sparse, a few entries
     - reach of the transport multiplier through the pedestal-zone
       :math:`T` equations — one Jacobian-vector product w.r.t. the scalar

**Design rule: implement global couplings as border pairs, never as
in-block feedback.**

How each solver handles the border
----------------------------------

*Direct / colored solve.* Factorise ``J_xx`` as today. Then a bordered
(block-elimination / Schur complement) solve: :math:`m` extra
back-substitutions to form :math:`J_{xx}^{-1} J_{xu}`, an :math:`m \times m`
dense solve of the Schur complement
:math:`S = G_u - G_x J_{xx}^{-1} J_{xu}`, and one final back-substitution.
For :math:`m` of a few, the extra cost is negligible. This is the standard
bordered-system technique from continuation codes.

*Jacobian-free Newton-Krylov.* The augmented residual's Jacobian-vector
products cost the same as before. The border is a rank-:math:`m`
perturbation, which GMRES absorbs in roughly :math:`m` extra iterations with
the unmodified preconditioner; a constraint-aware preconditioner
(block-eliminating the border through the tridiagonal approximation) removes
even those.

*Declared pattern / coloring.* The border rows and columns are declared
alongside the PDE pattern. Each actuator column receives its own probe color
(it conflicts with everything, but :math:`m` is small); pairs whose border
vectors are analytic need no probes at all. The runtime verification probe
covers the augmented matrix like any other entry.

Hard constraints vs. temporal relaxation
----------------------------------------

A hard constraint row has no time derivative: the system becomes a DAE
(index 1), and the theta method enforces :math:`g(x) = 0` at every
:math:`t + \Delta t`. Two consequences deserve attention:

* the *initial condition* must already satisfy the constraint, so
  initialization needs a consistency solve;
* if the actuator saturates (e.g. gas puff clipped at zero while the density
  target is unreachable), the Schur complement becomes singular and Newton
  fails hard.

Both are avoided by giving the constraint **relaxation dynamics** instead.
Promote the actuator to a dynamic unknown with

.. math::

   \tau \, \frac{du}{dt} = g(x),

i.e. the actuator integrates the constraint violation with time constant
:math:`\tau`. This is precisely an integral feedback controller with gain
:math:`1/\tau`, but implemented *implicitly*: it is discretised by the same
theta method as every other channel, so the constraint row simply gains a
transient term,

.. math::

   \tau \, \frac{u^{n+1} - u^n}{\Delta t}
   = \theta \, g(x^{n+1}) + (1 - \theta) \, g(x^n),

which adds :math:`\tau / \Delta t` to the corresponding diagonal entry of
``G_u``. Numerically this is strictly regularising: the Schur complement
stays nonsingular even at actuator saturation, no consistent initialization
is required (the actuator just starts relaxing from wherever it is), and the
stiffness is controlled by :math:`\tau`.

The two formulations are ends of one dial:

* :math:`\tau \to 0` recovers the hard algebraic constraint (exact tracking,
  DAE semantics, consistency and saturation caveats apply);
* moderate :math:`\tau` gives controller-like dynamics — the constraint is
  approached on the :math:`\tau` timescale, which is usually also the
  physically honest model of a real actuator loop;
* because the term is implicit, choosing a small :math:`\tau` does *not*
  impose a CFL-like limit on :math:`\Delta t` — it only makes the border
  entries larger, which the bordered solve handles exactly.

Relaxation also composes with actuator limits: clip :math:`u` in the update
(or add a barrier term to its equation) and the integrator simply winds to
the limit instead of making the matrix singular. An anti-windup guard (freeze
the relaxation while clipped) is a one-line addition to the actuator row.

Bounded hard constraints (complementarity)
------------------------------------------

A hard constraint with an actuator bound (e.g. a gas puff rate that cannot
go negative) is no longer a plain equation: *either* the target is met with
a feasible actuator, *or* the actuator sits at its bound and the target is
missed in the only direction it can be. This is a complementarity condition,

.. math::

   u - u_{min} \geq 0, \qquad g \geq 0, \qquad (u - u_{min}) \, g = 0,

written as a single equation with the Fischer-Burmeister function

.. math::

   \phi(a, b) = a + b - \sqrt{a^2 + b^2 + \epsilon^2} = 0,
   \qquad a = u - u_{min}, \; b = g,

which replaces the constraint row verbatim (``u_min`` on the constraint
config enables it). With both ``u_min`` and ``u_max`` the two
Fischer-Burmeister forms nest, giving the box condition: :math:`g = 0` in
the interior, :math:`g > 0` at the lower bound, :math:`g < 0` at the upper
bound. Each single-bound form is the limit of the nested one as the other
bound goes to infinity, so all three cases are one expression evaluated
with whichever bounds are configured. The :math:`\epsilon`-smoothing keeps
the row differentiable for the Newton solve; the enforcement error it
introduces is :math:`O(\epsilon^2)`. Two properties make this the right
formulation:
at saturation the row degenerates toward the well-conditioned
:math:`u = u_{min}` rather than toward a vanishing Schur complement, and
the constraint violation is *reported honestly* (the density overshoots an
unreachable target under zero fuelling) instead of being met by an
unphysical actuator value. The pairing assumes the actuator increases the
constrained quantity, so lower-bound saturation can only leave
:math:`g > 0`.

Practical checklist
-------------------

#. **Scale the border.** Profiles are solved in :math:`O(1)` scaled units;
   actuators and constraint residuals must be nondimensionalised the same
   way, or the Newton line search will fight mixed magnitudes.
#. **Check controllability.** The Schur complement is nonsingular only if
   each actuator actually moves its constraint
   (:math:`\partial g / \partial u \neq 0` through the physics). Log its
   conditioning; a near-singular :math:`S` means a saturated or impotent
   actuator.
#. **Declare the border pattern.** Extend the declared Jacobian pattern with
   the constraint-row and actuator-column supports; prefer analytic border
   vectors where they exist.
#. **Keep the verification probe on.** One extra Jacobian-vector product per
   build checks the augmented pattern contract at runtime — it is the guard
   against a constraint quietly acquiring a dependency the pattern does not
   declare.
