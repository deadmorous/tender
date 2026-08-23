"""bac-cab: a × (b × c) = b (a·c) − c (a·b).

The classic vector triple-product identity, in tender's direct notation.

L1 verifies the claim by brute force: both sides are reduced to concrete
World-Cartesian components (every ε and δ evaluated numerically) and compared.

L2 performs the textbook ε-route derivation symbolically, never touching a
concrete component: expand both crosses in the frame (each basis cross is an
ε term), contract the ε-pair over its shared summed index into δδ − δδ,
contract the δ's, and reassemble the coordinate form back to the invariant
right-hand side.  This is the vibe-000035 "no creative step" derivation.
"""

import tender
import tender.basis as tb
import tender.derivation as td

from challenges import harness
from challenges.harness import show


CHALLENGE = harness.declare(
    title="bac-cab: a×(b×c) = b(a·c) − c(a·b)",
    tier="A",
    source="Lurie, Theory of Elasticity, tensor-calculus appendix; "
    "vibe 000035",
)


def _setup():
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    c = tender.tensor("c", rank=1, ctx=ctx)
    return frame, a, b, c


@harness.level("L1")
def test_verified_in_concrete_components():
    """Both sides agree component-by-component in the World Cartesian System."""
    frame, a, b, c = _setup()
    lhs = a % (b % c)  # a × (b × c)
    rhs = b * (a @ c) - c * (a @ b)  # b (a·c) − c (a·b)

    def concrete(e):
        e = tb.expand_in_basis(e, frame, tb.Variance.Covariant)
        e = tb.simplify_basis_cross(e, frame)
        e = tb.simplify_basis_cross(e, frame)  # the second, inner cross
        e = tb.simplify_basis_dot(e, frame)
        e = td.canonicalize(e)
        e = td.unroll_sums(e)
        e = td.eval_eps_concrete(e)
        e = td.eval_delta_concrete(e)
        e = td.fold_arithmetic(e)
        return td.canonicalize(e)

    L, R = concrete(lhs), concrete(rhs)
    show("a×(b×c) in components", L)
    show("b(a·c) − c(a·b) in components", R)
    harness.assert_algebraic_eq(L, R, "bac-cab, concrete components")


@harness.level("L2")
def test_performed_by_eps_pair_contraction():
    """The symbolic ε-route: ε ε → δδ − δδ → the invariant rhs, as performed."""
    frame, a, b, c = _setup()
    lhs = a % (b % c)
    rhs = b * (a @ c) - c * (a @ b)
    show("claim: lhs", lhs)
    show("claim: rhs", rhs)

    x = tb.expand_in_basis(lhs, frame, tb.Variance.Covariant)
    x = tb.simplify_basis_cross(x, frame)  # outer cross → ε
    x = tb.simplify_basis_cross(x, frame)  # inner cross → ε
    x = td.canonicalize(x)  # materialize the Einstein sums
    show("both crosses as an ε-pair", x)

    x = td.contract_eps_pair(x)  # Σ_m ε ε → δδ − δδ
    show("ε-pair contracted to δ's", x)

    x = td.contract_delta(td.expand_products(x))  # split the −, eat the δ's
    x = td.simplify(x)
    show("coordinate form", x)

    back = tb.reassemble(x, frame)  # fold Σ a_j c_j → a·c etc.
    show("reassembled invariant", back)

    harness.assert_algebraic_eq(back, rhs, "bac-cab, symbolic derivation")
