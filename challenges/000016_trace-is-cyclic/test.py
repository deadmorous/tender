"""Cyclicity of the trace: tr(A·B) = tr(B·A) for rank-2 tensors.

L1 verifies by expanding both rank-2 tensors in the World Cartesian basis,
unfolding tr and the dot on dyads, and comparing the coordinate forms.
L2 is a `prove_equal` one-liner against the `dyadic` rule group.
"""

import tender
import tender.basis as tb
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="tr(A·B) = tr(B·A)",
    tier="A",
    source="Zhilin, Vectors and Second-Rank Tensors",
)


@harness.level("L1")
def test_both_sides_reduce_identically():
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    A = tender.tensor("A", rank=2, ctx=ctx)
    B = tender.tensor("B", rank=2, ctx=ctx)

    def coordinate_form(e):
        e = tb.expand_in_basis(e, frame, tb.Variance.Covariant)
        e = td.expand_dyad_ops(e)  # tr(a⊗b) → a·b on the dyads
        e = tb.simplify_basis_dot(e, frame)
        e = td.canonicalize(e)
        e = tb.simplify_basis_dot(e, frame)
        e = td.contract_delta(td.canonicalize(e))
        return td.simplify(e)

    lhs = coordinate_form(tender.tr(A @ B))
    rhs = coordinate_form(tender.tr(B @ A))
    show("tr(A·B) in coordinates", lhs)
    show("tr(B·A) in coordinates", rhs)
    harness.assert_algebraic_eq(lhs, rhs, "trace cyclicity")


@harness.level("L2")
def test_performed_as_one_goal_directed_call():
    """One call against the `dyadic` rule group.

    The proof *cites* the library's `trace-cyclic` identity rather than
    deriving it: cyclicity is an axiom of the trace, not a consequence of the
    ε-δ identities, and citing a standard identity from the toolbox is how a
    human works.  It is not circular — the same challenge's L1 test
    independently verifies that identity by reduction to coordinates, and the
    library rule carries its own fire-test.
    """
    ctx = tender.Context()
    A = tender.tensor("A", rank=2, ctx=ctx)
    B = tender.tensor("B", rank=2, ctx=ctx)

    result = td.prove_equal(
        tender.tr(A @ B), tender.tr(B @ A), td.rules("dyadic", ctx=ctx)
    )
    show("prove_equal(tr(A·B), tr(B·A))", repr(result))
    assert result.proved
    assert result.fired.get("trace-cyclic") == 1
