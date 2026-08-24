"""Cross with the identity tensor commutes: a × I = I × a.

The identity resolves to the dyad I = Σ_m e_m ⊗ e^m, so each side is a vector
crossed with a dyad; both reduce to the same ε-coordinate expression.

L1 verifies by reducing both sides symbolically in the World Cartesian System.
L2 transforms a × I *into* I × a at the invariant level, never expanding into
a basis.
"""

import tender
import tender.basis as tb
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="a×I = I×a",
    tier="A",
    source="Zhilin, Vectors and Second-Rank Tensors; examples/cross_identity.py",
    proves="cross-identity",
)


@harness.level("L1")
def test_both_sides_reduce_identically():
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    a = tender.tensor("a", rank=1, ctx=ctx)
    I = tender.identity(ctx=ctx)

    def coordinate_form(e):
        e = tb.expand_in_basis(e, frame, tb.Variance.Covariant)
        e = tb.simplify_basis_cross(e, frame)  # e_i × e_m → ε e^k
        return td.canonicalize(e)

    lhs, rhs = coordinate_form(a % I), coordinate_form(I % a)
    show("a×I in coordinates", lhs)
    show("I×a in coordinates", rhs)
    harness.assert_algebraic_eq(lhs, rhs, "a×I = I×a")


@harness.level("L2")
def test_performed_at_the_invariant_level():
    """Transformed at the invariant level — never leaving direct notation.

    Two forms, both ways.  `prove_equal` confirms the identity, and
    `engine_simplify` *rewrites* a×I into I×a from the left-hand side alone
    — the user states the expression, not the answer.  Neither call expands
    into a basis: the whole thing happens in direct notation, which is what
    the L1 test below has to give up to verify the same claim.
    """
    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    I = tender.identity(ctx=ctx)
    rules = td.rules("cross", ctx=ctx)

    result = td.prove_equal(a % I, I % a, rules)
    show("prove_equal(a×I, I×a)", repr(result))
    assert result.proved
    assert result.fired.get("cross-identity") == 1

    rewritten, report = td.engine_simplify(a % I, rules)
    show("engine_simplify(a×I)", rewritten)
    harness.assert_algebraic_eq(rewritten, I % a, "a×I rewritten to I×a")
    assert report["complete"]
