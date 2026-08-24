"""a × (b × I) = b⊗a − (a·b) I — the vibe-000056 usability case.

The inner cross is a vector crossed with a rank-2 fence, so plain bac-cab
refuses; this challenge is *the* motivating case for the goal-directed verb
surface (its manual route required four undiscoverable steps).

L1 verifies by reducing both sides to concrete World-Cartesian components.
L2 reaches the invariant right-hand side in one goal-directed call — no step
ordering to discover, which is the whole point of this challenge.
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
    proves="cross-removal",
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


@harness.level("L2")
def test_performed_without_magic_ordering():
    """THE vibe-000056 case, resolved: one goal-directed call, no ordering.

    This derivation is why the consolidation happened.  Reaching
    b⊗a − (a·b)I from a×(b×I) used to require `distribute_contraction`, then
    `apply_identity`, then `expand_products`, then `reassemble_completeness`,
    **in exactly that order** — a sequence the vibe-000056 note said no user
    could be expected to discover, and whose steps rendered as no-ops even
    when they were load-bearing.  Now the user names the goal and the rule
    group; the engine finds the route and reports which identities it used.

    The proof cites the library's `cross-removal` identity (Zhilin's rank-2
    companion to bac-cab).  That is not circular: the identity is verified
    independently, in components, by this challenge's own L1 test above, and
    the rule carries a fire-test and a soundness test of its own — bac-cab
    must *not* fire across the rank-2 fence where it would be false.
    """
    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    I = tender.identity(ctx=ctx)

    result = td.prove_equal(
        a % (b % I), b * a - (a @ b) * I, td.rules("cross", ctx=ctx)
    )
    show("prove_equal(a×(b×I), b⊗a − (a·b)I)", repr(result))
    show("identities used", result.fired)
    assert result.proved
    assert result.fired.get("cross-removal") == 1
