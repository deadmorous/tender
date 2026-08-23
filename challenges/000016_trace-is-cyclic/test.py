"""Cyclicity of the trace: tr(A·B) = tr(B·A) for rank-2 tensors.

L1 verifies by expanding both rank-2 tensors in the World Cartesian basis,
unfolding tr and the dot on dyads, and comparing the coordinate forms.
L2 (future) is a `prove_equal` one-liner on the verb surface.
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


@harness.level("L2", expected=False, reason="needs the M2/M3 prove_equal verb")
def test_performed_as_one_goal_directed_call():
    harness.todo("prove_equal(tr(A@B), tr(B@A)) on the verb surface")
