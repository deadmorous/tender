"""Product rules of ∇: how grad, div, and rot act on products.

    ∇(fg)    = f ∇g + g ∇f
    ∇·(f u)  = f ∇·u + u·∇f
    ∇×(f u)  = f ∇×u + ∇f × u
    ∇·(a×b)  = b·(∇×a) − a·(∇×b)

Each rule is verified component-by-component in a Cartesian chart (L1); the
invariant Leibniz derivations are the M2 engine's material (L2, future).
"""

import tender as t
import tender.derivation as td

from challenges import harness

CHALLENGE = harness.declare(
    title="∇ product rules: ∇(fg), ∇·(fu), ∇×(fu), ∇·(a×b)",
    tier="C",
    source="Borisenko–Tarapov §differential identities",
)


def _setup():
    ws = t.Workspace()
    cart, (x, y, z) = ws.cartesian_chart()
    return ws, cart


@harness.level("L1")
def test_gradient_of_scalar_product():
    ws, cart = _setup()
    f, g = ws.field("f", 0), ws.field("g", 0)
    lhs = cart.grad(f * g)
    rhs = f * cart.grad(g) + g * cart.grad(f)
    harness.assert_components_equal(cart, lhs, rhs, "∇(fg)")


@harness.level("L1")
def test_divergence_of_scaled_vector():
    ws, cart = _setup()
    f, u = ws.field("f", 0), ws.field("u", 1)
    lhs = cart.div(f * u)
    rhs = f * cart.div(u) + u @ cart.grad(f)
    harness.assert_chart_zero(
        cart, cart.expand(lhs) - cart.expand(rhs), "∇·(fu)"
    )


@harness.level("L1")
def test_curl_of_scaled_vector():
    ws, cart = _setup()
    f, u = ws.field("f", 0), ws.field("u", 1)
    lhs = cart.rot(f * u)
    rhs = f * cart.rot(u) + cart.grad(f) % u
    harness.assert_components_equal(cart, lhs, rhs, "∇×(fu)")


@harness.level("L1")
def test_divergence_of_cross_product():
    ws, cart = _setup()
    a, b = ws.field("a", 1), ws.field("b", 1)
    lhs = cart.div(a % b)
    rhs = b @ cart.rot(a) - a @ cart.rot(b)
    harness.assert_chart_zero(
        cart, cart.expand(lhs) - cart.expand(rhs), "∇·(a×b)"
    )


@harness.level("L2", expected=False, reason="needs invariant Leibniz on the M2 engine")
def test_performed_invariantly():
    harness.todo("derive the product rules with ∇ abstract (Leibniz rule set)")
