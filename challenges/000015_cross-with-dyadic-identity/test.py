"""a × (b × I) = b⊗a − (a·b) I — the vibe-000056 usability case.

The inner cross is a vector crossed with a rank-2 fence, so plain bac-cab
refuses; this challenge is *the* motivating case for the goal-directed verb
surface (its manual route required four undiscoverable steps).

L1 verifies by reducing both sides to concrete World-Cartesian components.
L2 (future) must reach the invariant right-hand side without magic ordering.
"""

import tender
import tender.basis as tb
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="a×(b×I) = b⊗a − (a·b)I",
    tier="A",
    source="Zhilin, Vectors and Second-Rank Tensors; vibe 000056",
)


@harness.level("L1")
def test_verified_in_concrete_components():
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    I = tender.identity(ctx=ctx)

    lhs = a % (b % I)
    rhs = b * a - (a @ b) * I  # b⊗a − (a·b)I

    def concrete(e):
        e = tb.expand_in_basis(e, frame, tb.Variance.Covariant)
        e = tb.simplify_basis_cross(e, frame)
        e = tb.simplify_basis_cross(e, frame)
        e = tb.simplify_basis_dot(e, frame)
        e = td.canonicalize(e)
        e = td.unroll_sums(e)
        e = td.eval_eps_concrete(e)
        e = td.eval_delta_concrete(e)
        e = td.fold_arithmetic(e)
        return td.canonicalize(e)

    L, R = concrete(lhs), concrete(rhs)
    show("a×(b×I) in components", L)
    show("b⊗a − (a·b)I in components", R)
    harness.assert_algebraic_eq(L, R, "a×(b×I)")


@harness.level(
    "L2", expected=False, reason="THE vibe-000056 case: no discoverable route yet"
)
def test_performed_without_magic_ordering():
    harness.todo(
        "reach b⊗a − (a·b)I from a×(b×I) via the verb surface "
        "(today needs distribute_contraction + apply_identity + "
        "expand_products + reassemble_completeness in exactly that order)"
    )
