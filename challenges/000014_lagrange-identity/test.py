"""The Lagrange identity: (a×b)·(c×d) = (a·c)(b·d) − (a·d)(b·c).

L1 verifies by reducing both sides to concrete World-Cartesian components.
L2 (future) performs the ε-pair derivation symbolically, like bac-cab
(challenge 000001) but with the ε's meeting through a dot contraction.
"""

import tender
import tender.basis as tb
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="(a×b)·(c×d) = (a·c)(b·d) − (a·d)(b·c)",
    tier="A",
    source="Gibbs–Wilson, Vector Analysis",
)


def _setup():
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    vecs = [tender.tensor(n, rank=1, ctx=ctx) for n in "abcd"]
    return frame, vecs


@harness.level("L1")
def test_verified_in_concrete_components():
    frame, (a, b, c, d) = _setup()
    lhs = (a % b) @ (c % d)
    rhs = (a @ c) * (b @ d) - (a @ d) * (b @ c)

    def concrete(e):
        e = tb.expand_in_basis(e, frame, tb.Variance.Covariant)
        e = tb.simplify_basis_cross(e, frame)
        e = tb.simplify_basis_dot(e, frame)
        e = td.canonicalize(e)
        e = td.unroll_sums(e)
        e = td.eval_eps_concrete(e)
        e = td.eval_delta_concrete(e)
        e = td.fold_arithmetic(e)
        return td.canonicalize(e)

    L, R = concrete(lhs), concrete(rhs)
    show("(a×b)·(c×d) in components", L)
    show("(a·c)(b·d) − (a·d)(b·c) in components", R)
    harness.assert_algebraic_eq(L, R, "Lagrange identity")


@harness.level("L2", expected=False, reason="ε-pair route not yet attempted here")
def test_performed_by_eps_pair_contraction():
    harness.todo("the symbolic ε-pair derivation, as in challenge 000001")
