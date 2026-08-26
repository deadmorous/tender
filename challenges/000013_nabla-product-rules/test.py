"""Product rules of ∇: how grad, div, and rot act on products.

    ∇(fg)    = f ∇g + g ∇f
    ∇·(f u)  = f ∇·u + u·∇f
    ∇×(f u)  = f ∇×u + ∇f × u
    ∇·(a×b)  = b·(∇×a) − a·(∇×b)

Each rule is verified component-by-component in a Cartesian chart (L1); the
invariant Leibniz derivations are the M2 engine's material (L2, future).
"""

import tender as t_
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="∇ product rules: ∇(fg), ∇·(fu), ∇×(fu), ∇·(a×b)",
    tier="C",
    source="Borisenko–Tarapov §differential identities",
    proves=["grad-product", "div-cross", "div-scaled", "curl-scaled"],
)


def _setup():
    ws = t_.Workspace()
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


@harness.level("L2")
def test_the_statable_product_rules_performed_invariantly():
    """Two of the four, proved with ∇ abstract — and why only two.

    `∇(fg)` and `∇·(a×b)` are provable invariantly by the Leibniz rule group.
    `∇·(fu)` and `∇×(fu)` are not, and the obstacle is not a missing rule but
    a missing *capability*: canon cannot put `∇·(f u)` in canonical form at
    all, because it holds a ⊗-product inside a contraction operand.  A rule
    for it cannot be written, let alone fire — see the companion test below,
    which pins that.
    """
    ws = t_.Workspace()
    nabla = ws.nabla()
    f, g = ws.field("f", 0), ws.field("g", 0)
    a, b = ws.field("a", 1), ws.field("b", 1)
    rules = td.rules("leibniz", ctx=ws.ctx)

    grad_fg = td.prove_equal(
        nabla * (f * g), f * (nabla * g) + g * (nabla * f), rules
    )
    show("∇(fg) = f∇g + g∇f", repr(grad_fg))
    assert grad_fg.proved

    div_cross = td.prove_equal(
        nabla @ (a % b), b @ (nabla % a) - a @ (nabla % b), rules
    )
    show("∇·(a×b) = b·(∇×a) − a·(∇×b)", repr(div_cross))
    assert div_cross.proved


@harness.level("L2")
def test_all_four_product_rules_performed_invariantly():
    """The two scaled-vector rules — blocked until the fence fix.

    `∇·(fu)` and `∇×(fu)` could not be *stated* before vibe 000101: the
    operand `f⊗u` holds no ∇ itself, so canon's fence condition missed that
    an operator sat beside it, and rejected the expression outright.  Now
    both are provable invariantly, and this challenge's four product rules
    are complete.
    """
    ws = t_.Workspace()
    nabla = ws.nabla()
    f = ws.field("f", 0)
    u = ws.field("u", 1)
    rules = td.rules("leibniz", ctx=ws.ctx)

    div = td.prove_equal(
        nabla @ (f * u), f * (nabla @ u) + u @ (nabla * f), rules
    )
    show("∇·(fu) = f(∇·u) + u·∇f", repr(div))
    assert div.proved

    curl = td.prove_equal(
        nabla % (f * u), f * (nabla % u) - u % (nabla * f), rules
    )
    show("∇×(fu) = f(∇×u) − u×∇f", repr(curl))
    assert curl.proved
