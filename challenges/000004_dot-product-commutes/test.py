"""Commutativity of the dot product: a · b = b · a, from first principles.

L1 reduces both sides through the World Cartesian basis to the same scalar
contraction Σ_i a_i b_i (symbolically — no concrete components).

L2 (future) is the one-liner `prove_equal(a @ b, b @ a)` on the M2/M3 verb
surface.
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


@harness.level("L2", expected=False, reason="needs the M2/M3 prove_equal verb")
def test_performed_as_one_goal_directed_call():
    harness.todo("prove_equal(a @ b, b @ a) on the verb surface")
