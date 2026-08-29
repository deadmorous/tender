"""Cross-chart consistency: ∇ ⊗ R_cart evaluated in a cylindrical chart is I.

Charts over the same world frame have related coordinates (x = r cosθ …), so
the gradient of the *Cartesian* position vector must come out as the identity
tensor in *any* chart over that frame.  The forward direction (WCS quantity →
curvilinear chart) shipped in vibe 000090 approach A; the reverse (a
curvilinear quantity evaluated in another chart) is approach B, and the point
of interest is that it needs no inverse embedding at all.
"""

import pytest

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="∇ ⊗ R across charts sharing a world frame",
    tier="D",
    source="vibe 000090 (cross-chart coordinate dependencies)",
)


def _charts():
    ws = t.Workspace()
    cart, (x, y, z) = ws.cartesian_chart()
    cyl, (r, th, zc) = ws.cylindrical_chart()
    return ws, cart, cyl


@harness.level("L2")
def test_forward_gradient_of_wcs_position_is_identity():
    """cyl.evaluate(∇ ⊗ R_cart) = I: the foreign x, y, z are reprojected
    through x = r cosθ … and the result folds back to I (approach A).

    Marked L2 rather than L1 (as it shipped in vibe 000090): ∇ goes in abstract
    and `I` comes out, with no component reduction and nothing compared against
    an oracle.  The derivation is *performed* here, which is what L2 means; the
    component-level check lives in the Jacobian test below.
    """
    ws, cart, cyl = _charts()
    nabla = t.nabla(ctx=ws.ctx)

    grad_R = cyl.evaluate(nabla * cart.position())
    show("cyl.evaluate(∇ ⊗ R_cart)", grad_R)
    harness.assert_algebraic_eq(
        grad_R, t.identity(ws.ctx), "∇ ⊗ R_cart in the cylindrical chart"
    )


@harness.level("L2")
def test_reverse_gradient_of_curvilinear_position_is_identity():
    """cart.evaluate(∇ ⊗ R_cyl) = I — approach B, without an inverse embedding.

    The reverse direction looks like it needs `r = √(x²+y²)`, `θ = atan2(y, x)`,
    which tender cannot even write: there is no arctangent, and `cos(atan2(y,x))`
    would then have to simplify back to `x/√(x²+y²)`.

    None of that is necessary, because the inverse embedding itself is never
    used — only its *derivatives* are, and for an orthogonal chart those are the
    contravariant basis vectors the chart already carries:

        ∂q^a/∂x^b  =  (∇q^a)_b  =  (e_a · i_b) / h_a

    giving `∂r/∂x = cos θ` and `∂θ/∂x = −sin θ / r`, both written in the
    *curvilinear* coordinates — which is where the rest of the expression
    already lives, so nothing needs inverting.  `diff` consults that Jacobian
    instead of treating a sibling chart's coordinate as an independent variable,
    and the chain rule does the rest: ∂ₓ(r cos θ) = cos²θ + sin²θ = 1.
    """
    ws, cart, cyl = _charts()
    nabla = t.nabla(ctx=ws.ctx)

    grad_R = cart.evaluate(nabla * cyl.position())
    show("cart.evaluate(∇ ⊗ R_cyl)", grad_R)
    harness.assert_algebraic_eq(
        grad_R, t.identity(ws.ctx), "∇ ⊗ R_cyl in the Cartesian chart"
    )


@harness.level("L2")
def test_both_directions_agree_for_every_built_in_chart():
    """∇R = I is chart-independent, so the pair must close for each of them."""
    ws = t.Workspace()
    cart, _ = ws.cartesian_chart()
    nabla = t.nabla(ctx=ws.ctx)
    I = t.identity(ws.ctx)
    for name in ("cylindrical_chart", "spherical_chart"):
        chart, _ = getattr(ws, name)()
        fwd = chart.evaluate(nabla * cart.position())
        rev = cart.evaluate(nabla * chart.position())
        show(f"{name}: forward", fwd)
        show(f"{name}: reverse", rev)
        harness.assert_algebraic_eq(fwd, I, f"∇R_cart in {name}")
        harness.assert_algebraic_eq(rev, I, f"∇R_{name} in cart")


@harness.level("L1")
def test_the_derivative_of_a_sibling_coordinate_is_the_jacobian():
    """The mechanism, on its own: ∂r/∂x = cos θ, ∂θ/∂x = −sin θ / r.

    These are what makes the round trip close, and they are the *only* thing
    approach B needed — no inverse embedding, no arctangent.
    """
    ws, cart, cyl = _charts()
    x = cart.coords[0]
    r, th = cyl.coords[0], cyl.coords[1]

    dr_dx = td.simplify_scalars(td.partial(r, x))
    dth_dx = td.simplify_scalars(td.partial(th, x))
    show("∂r/∂x", dr_dx)
    show("∂θ/∂x", dth_dx)
    assert r"\cos" in dr_dx.latex(), dr_dx.latex()
    assert r"\sin" in dth_dx.latex() and "r" in dth_dx.latex(), dth_dx.latex()

    # And the chain rule closes on the embedding: ∂ₓ(r cos θ) = ∂ₓx = 1.
    chained = td.simplify_scalars(td.partial(r * t.cos(th), x))
    show("∂ₓ(r cos θ)", chained)
    harness.assert_algebraic_eq(
        chained, t.scalar(1, ctx=ws.ctx), "∂ₓ(r cos θ) = 1"
    )
