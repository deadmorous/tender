"""The curl-curl identity: ∇×(∇×u) = ∇(∇·u) − Δu.

L1 verifies component-by-component in a Cartesian chart.  L2 proves it
*invariantly*, with ∇ abstract and no coordinate system chosen.
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="∇×(∇×u) = ∇(∇·u) − Δu",
    tier="C",
    source="Lurie, Theory of Elasticity, tensor-calculus appendix",
)


@harness.level("L1")
def test_verified_in_cartesian_components():
    ws = t.Workspace()
    cart, (x, y, z) = ws.cartesian_chart()
    u = ws.field("u", 1)

    lhs = cart.rot(cart.rot(u))
    rhs = cart.grad(cart.div(u)) - cart.laplacian(u)
    harness.assert_components_equal(cart, lhs, rhs, "curl-curl")


@harness.level("L2")
def test_performed_invariantly():
    """One goal-directed call, with ∇ abstract and no chart in sight.

    The L1 test above verifies this by expanding into Cartesian components;
    here it is proved *invariantly* — the statement never leaves direct
    notation and no coordinate system is chosen.  That is what the Leibniz
    rule group buys (vibe 000101), and it is why the xfail this replaces was
    worth keeping honest rather than quietly verifying in components twice.
    """
    ws = t.Workspace()
    nabla = ws.nabla()
    u = ws.field("u", 1)

    result = td.prove_equal(
        nabla % (nabla % u),
        nabla * (nabla @ u) - nabla @ (nabla * u),
        td.rules("leibniz", ctx=ws.ctx),
    )
    show("prove_equal(∇×(∇×u), ∇(∇·u) − Δu)", repr(result))
    assert result.proved
    assert result.fired.get("curl-curl") == 1
