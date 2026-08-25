"""Cross-chart consistency: ∇ ⊗ R_cart evaluated in a cylindrical chart is I.

Charts over the same world frame have related coordinates (x = r cosθ …), so
the gradient of the *Cartesian* position vector must come out as the identity
tensor in *any* chart over that frame.  The forward direction (WCS quantity →
curvilinear chart) shipped in vibe 000090 approach A; the reverse direction
(curvilinear quantity → another chart, needing the inverse embedding) is the
deferred approach B and stays enumerated red.
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


@harness.level("L1")
def test_forward_gradient_of_wcs_position_is_identity():
    """cyl.evaluate(∇ ⊗ R_cart) = I: the foreign x, y, z are reprojected
    through x = r cosθ … and the result folds back to I (approach A)."""
    ws, cart, cyl = _charts()
    nabla = t.nabla(ctx=ws.ctx)

    grad_R = cyl.evaluate(nabla * cart.position())
    show("cyl.evaluate(∇ ⊗ R_cart)", grad_R)
    harness.assert_algebraic_eq(
        grad_R, t.identity(ws.ctx), "∇ ⊗ R_cart in the cylindrical chart"
    )


@harness.level(
    "L1",
    expected=False,
    reason="vibe 000090 approach B (inverse embedding) deferred",
)
def test_reverse_gradient_of_curvilinear_position_is_identity():
    """cart.evaluate(∇ ⊗ R_cyl) needs the inverse embedding r = √(x²+y²) …;
    today it raises the clear approach-B error instead of returning I."""
    ws, cart, cyl = _charts()
    nabla = t.nabla(ctx=ws.ctx)

    grad_R = cart.evaluate(nabla * cyl.position())  # raises today (approach B)
    harness.assert_algebraic_eq(
        grad_R, t.identity(ws.ctx), "∇ ⊗ R_cyl in the Cartesian chart"
    )
