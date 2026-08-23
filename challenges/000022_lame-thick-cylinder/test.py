"""Lamé's thick-walled cylinder: axisymmetric equilibrium reduced to the ODE.

Axisymmetric plane strain, no shear, fields of r only: the radial equilibrium
∂_r T_rr + (T_rr − T_θθ)/r = 0 combined with Hooke's law and u = u(r) e_r must
reduce to the displacement ODE  ∂_r((1/r)∂_r(r u)) = 0 , whose solution is the
classical Lamé formula.
"""

from challenges import harness

CHALLENGE = harness.declare(
    title="thick-walled cylinder → Lamé displacement ODE",
    tier="E",
    source="Lurie, Theory of Elasticity §Lamé problem",
)


@harness.level("L1", expected=False, reason="not yet attempted")
def test_reduces_to_the_displacement_ode():
    harness.todo(
        "substitute u = u(r) e_r into the Navier–Lamé operator in the "
        "cylindrical chart and reduce to ∂_r((1/r)∂_r(r u)) = 0"
    )
