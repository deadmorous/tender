"""Lamé's thick-walled cylinder: the elastic operator reduces to one ODE.

The classical problem — a cylinder under internal and external pressure —
turns on a kinematic assumption: axisymmetric, plane, purely radial
displacement,

    u = u(r) e_r .

Substituting that into the Navier–Lamé operator should collapse a vector
partial-differential equation into a *single ordinary* differential equation
for u(r).  Two things have to happen for that, and both are checked here: the
θ and z components must vanish identically (nothing drives motion in those
directions), and the radial component must be the Euler equation

    (λ + 2μ) d/dr [ (1/r) d/dr ( r u ) ]  =  0 ,

whose solution u = Ar + B/r is the Lamé formula.  The compact form matters:
`d/dr[(1/r)d/dr(ru)]` is how the textbook writes it, and it is *equivalent to*
but not syntactically the expanded Euler form u'' + u'/r − u/r², so both are
verified.

Performed on the public surface: build u from the chart's own frame, hand it
to the operators, read the components.  The reduction to an ODE is a result,
not an input — nothing here assumes which terms survive.
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="thick-walled cylinder → Lamé displacement ODE",
    tier="E",
    source="Lurie, Theory of Elasticity §Lamé problem",
)


@harness.level("L2")
def test_reduces_to_the_displacement_ode():
    ws = t.Workspace()
    cyl, (r, th, z) = ws.cylindrical_chart()
    lam = t.tensor(r"\lambda", 0, ctx=ws.ctx)
    mu = t.tensor(r"\mu", 0, ctx=ws.ctx)
    d = td.partial

    # The kinematic assumption: u = u(r) e_r.  `deps=[r]` is what makes it
    # axisymmetric and plane — ∂_θ u and ∂_z u are then identically zero.
    e_r = cyl.physical_frame().direction(0)
    u = ws.field("u", 0, deps=[r])
    displacement = u * e_r

    # The Navier–Lamé operator (challenge 000008's endpoint), applied to it.
    nl = mu * cyl.div(cyl.grad(displacement)) + (lam + mu) * cyl.grad(
        cyl.div(displacement)
    )
    radial, circumferential, axial = (
        td.simplify_scalars(cyl.expand(c)) for c in cyl.components(nl)
    )
    show("radial component", radial)
    show("circumferential component", circumferential)
    show("axial component", axial)

    # A vector equation collapses to a scalar one: two components vanish.
    assert circumferential.latex() == "0", circumferential.latex()
    assert axial.latex() == "0", axial.latex()

    # …and the survivor is the Euler equation, in the compact textbook form…
    harness.assert_algebraic_eq(
        radial,
        (lam + 2 * mu) * d(d(r * u, r) / r, r),
        "radial equation = (λ+2μ) d/dr[(1/r) d/dr(r u)]",
    )
    # …and equivalently expanded, which is how it is usually solved.
    harness.assert_algebraic_eq(
        radial,
        (lam + 2 * mu) * (d(d(u, r), r) + d(u, r) / r - u / (r * r)),
        "radial equation = (λ+2μ)(u'' + u'/r − u/r²)",
    )
