"""Commutativity of the dot product: a · b = b · a, from first principles.

L1 reduces both sides through the World Cartesian basis to the same scalar
contraction Σ_i a_i b_i (symbolically — no concrete components).

L2 is the one-liner `prove_equal(a @ b, b @ a)` — which needs no rules at
all, since canonicalization already decides commutativity of a symmetric
contraction.
"""

import tender
import tender.basis as tb
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="a·b = b·a through the basis",
    tier="B",
    source="Gibbs–Wilson, Vector Analysis; examples/basis_dot_product.py",
)


@harness.level("L1")
def test_both_sides_reduce_to_the_same_contraction():
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)

    def coordinate_form(e):
        e = tb.expand_in_basis(e, frame, tb.Variance.Covariant)
        e = tb.simplify_basis_dot(e, frame)  # e_i · e_j → δ_ij
        e = td.contract_delta(td.canonicalize(e))
        return td.simplify(e)

    ab, ba = coordinate_form(a @ b), coordinate_form(b @ a)
    show("a·b in coordinates", ab)
    show("b·a in coordinates", ba)
    harness.assert_algebraic_eq(ab, ba, "a·b = b·a")


@harness.level("L2")
def test_performed_as_one_goal_directed_call():
    """One call, and *no rules at all*.

    Commutativity of the dot product is decided by canonicalization itself
    (theory T0 orders a symmetric contraction's operands), so the engine
    proves it in zero passes with an empty rule set — the strongest form this
    challenge could take: nothing is cited, nothing is assumed.
    """
    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)

    result = td.prove_equal(a @ b, b @ a, [])
    show("prove_equal(a·b, b·a)", repr(result))
    assert result.proved
    assert result.passes == 0, "canonicalization alone should settle this"
    assert result.fired == {}, "no rule should be needed"
