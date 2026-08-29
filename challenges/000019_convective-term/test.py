"""The convective-acceleration identity: (u·∇)u = ∇(u²/2) − u×(∇×u).

The workhorse of fluid mechanics — it is what turns the material derivative
into a gradient plus a rotational term, and hence what makes Bernoulli's
theorem fall out of the momentum equation.  It also matters here as a test of
the directional derivative: `(u·∇)u` is a vector built by contracting `u`
against the *gradient* of `u`, so it exercises the chart's grad with the
resolution of identity left unfolded.

L1 verifies component-by-component in a Cartesian chart.  L2 derives it with ∇
abstract, and the interesting part is *why* it cannot be done by reading the
vector identity a×(b×c) = b(a·c) − c(a·b) with b = ∇: that loses the ½.
Expanding ∇ into its free-index frame form first puts a genuine vector where
bac-cab needs one, and the ∂-mark keeps the bookkeeping straight.
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


@harness.level("L2")
def test_performed_invariantly():
    """Derived with ∇ abstract — no chart components anywhere.

    The classical proof goes through ε-δ in components, and the reason it has
    to is instructive: ∇ is an *operator*, so the vector identity
    a×(b×c) = b(a·c) − c(a·b) cannot simply be read with b = ∇.  Doing that
    gives ∇(u·u) − (u·∇)u and loses the ½, because the ∇ in the first term must
    differentiate only the u that came out of the curl, not both.

    Expanding ∇ into its free-index frame form settles that by construction:
    `∇×u` becomes `e_i × ∂_i u`, and the ∂ is attached to exactly one operand.
    bac-cab then applies to a genuine vector `e_i`, not to an operator, and the
    bookkeeping is the mark's rather than ours.  Both sides reduce to the same
    free-index expression, which is the proof.

    Nothing here re-derives bac-cab; it is cited, and challenge 000001 is what
    entitles us to (it derives it from the ε-δ identity).  So this *is* the
    combined ε-δ-and-Leibniz proof the L2 wanted — with the ε-δ half arriving
    as a proven identity and the Leibniz half falling out of `expand_nabla`.
    """
    ws, cart, u = _setup()
    nab = ws.nabla()
    half = t.scalar(t.Rational(1, 2), ctx=ws.ctx)
    bac_cab = [r for r in td.rules("cross", ctx=ws.ctx) if r.name == "bac-cab"][0]

    # (u·∇)u — u contracted against ∇⊗u.
    lhs = td.canonicalize(cart.expand_nabla(u @ (nab * u)))
    show("(u·∇)u expanded", lhs)

    # ∇(u²/2): expand_nabla applies Leibniz on the way, and the ½ is what
    # makes ∂_i(u·u)/2 come out as the single term u·∂_i u.
    grad = cart.expand_nabla(nab * (half * (u @ u)))
    show("∇(u²/2) expanded", grad)
    assert "frac" not in grad.latex(), grad.latex()

    # u×(∇×u): the inner curl supplies a real vector e_i for bac-cab to act on.
    cross = cart.expand_nabla(u % (nab % u))
    show("u×(∇×u) expanded", cross)
    reduced = td.canonicalize(td.apply_identity(bac_cab)(cross))
    show("…after bac-cab", reduced)
    assert "times" not in reduced.latex(), reduced.latex()

    rhs = td.canonicalize(td.expand_products(grad - reduced))
    show("∇(u²/2) − u×(∇×u)", rhs)
    harness.assert_algebraic_eq(rhs, lhs, "convective acceleration, derived")


@harness.level("L2")
def test_the_naive_operator_substitution_is_wrong():
    """Why the ½ is not a bookkeeping detail — the trap this challenge sets.

    Reading bac-cab with b = ∇ directly gives ∇(u·u) − (u·∇)u, which differs
    from the truth by a factor of two on the gradient term.  Asserting that it
    is *not* equal keeps the challenge honest: the derivation above earns the ½
    by expanding ∇ before applying the identity, and a future change that let
    the naive substitution through would show up here.
    """
    ws, cart, u = _setup()
    nab = ws.nabla()
    half = t.scalar(t.Rational(1, 2), ctx=ws.ctx)

    truth = nab * (half * (u @ u)) - u % (nab % u)
    naive = nab * (u @ u) - u % (nab % u)
    reduce = lambda e: td.canonicalize(cart.expand_nabla(e))
    show("with ½ (true)", reduce(truth))
    show("without ½ (wrong)", reduce(naive))
    assert not td.algebraic_eq(reduce(truth), reduce(naive))
