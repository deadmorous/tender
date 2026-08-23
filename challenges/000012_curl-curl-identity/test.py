"""The curl-curl identity: ∇×(∇×u) = ∇(∇·u) − Δu.

L1 verifies component-by-component in a Cartesian chart.  L2 (future) is the
invariant derivation via the ε-δ engine — the differential cousin of bac-cab.
"""

import tender as t
import tender.derivation as td

from challenges import harness

CHALLENGE = harness.declare(
    title="∇×(∇×u) = ∇(∇·u) − Δu",
    tier="C",
    source="Lurie, Theory of Elasticity, tensor-calculus appendix",
)


@harness.level("L1")
def test_verified_in_cartesian_components():
    ws = t.Workspace()
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    u = ws.field("u", 1)

    lhs = cart.rot(cart.rot(u))
    rhs = cart.grad(cart.div(u)) - cart.laplacian(u)
    harness.assert_components_equal(cart, lhs, rhs, "curl-curl")


@harness.level("L2", expected=False, reason="needs invariant ∇ algebra on the M2 engine")
def test_performed_invariantly():
    harness.todo("derive ∇×(∇×u) = ∇(∇·u) − Δu by the ε-pair route with ∇ abstract")
