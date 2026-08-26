"""The convective-acceleration identity: (u·∇)u = ∇(u²/2) − u×(∇×u).

The workhorse of fluid mechanics — it is what turns the material derivative
into a gradient plus a rotational term, and hence what makes Bernoulli's
theorem fall out of the momentum equation.  It also matters here as a test of
the directional derivative: `(u·∇)u` is a vector built by contracting `u`
against the *gradient* of `u`, so it exercises the chart's grad with the
resolution of identity left unfolded.

L1 verifies component-by-component in a Cartesian chart.  L2 (future) is the
invariant derivation, which needs the Leibniz rule group the engine does not
yet have.
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="(u·∇)u = ∇(u²/2) − u×(∇×u)",
    tier="C",
    source="Kochin, Vector Calculus §applications",
)


def _setup():
    ws = t.Workspace()
    cart, (x, y, z) = ws.cartesian_chart()
    return ws, cart, ws.field("u", 1)


@harness.level("L1")
def test_verified_in_components():
    ws, cart, u = _setup()
    half = t.scalar(t.Rational(1, 2), ctx=ws.ctx)

    # (u·∇)u — u contracted against ∇⊗u.  `fold_identity=False` keeps the
    # resolution Σ e_i⊗e_i unfolded so the frame dot can reduce it.
    lhs = cart.dot(u, cart.grad(u, fold_identity=False))
    # ∇(u·u/2) − u×(∇×u)
    rhs = cart.grad(half * (u @ u)) - (u % cart.rot(u))

    show("(u·∇)u", lhs)
    show("∇(u²/2) − u×(∇×u)", rhs)
    harness.assert_components_equal(cart, lhs, rhs, "convective acceleration")


@harness.level(
    "L2",
    expected=False,
    reason="the leibniz group now exists (vibe 000101) but does not reach this: "
    "(u·∇)u needs ∇ contracted against a gradient, and ∇(u·v) expands to terms "
    "with u×(∇×v), so it wants the ε-δ identity *and* Leibniz together — a "
    "combined-group proof nobody has attempted yet",
)
def test_performed_invariantly():
    harness.todo(
        "derive (u·∇)u = ∇(u²/2) − u×(∇×u) with ∇ abstract, which needs "
        "Leibniz rules for ∇ over products and the ε-δ identity behind "
        "u×(∇×u)"
    )
