"""The Lagrange identity: (a×b)·(c×d) = (a·c)(b·d) − (a·d)(b·c).

L1 verifies by reducing both sides to concrete World-Cartesian components.
L2 performs the ε-pair derivation symbolically, like bac-cab (challenge
000001) but with the two ε's brought together by contracting the δ that the
frame dot produces.  The identity is *derived*, not cited.
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
    proves="lagrange",
)


def _setup():
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    vecs = [tender.tensor(n, rank=1, ctx=ctx) for n in "abcd"]
    return ctx, frame, vecs


@harness.level("L1")
def test_verified_in_concrete_components():
    ctx, frame, (a, b, c, d) = _setup()
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


@harness.level("L2")
def test_performed_by_eps_pair_contraction():
    """The symbolic ε-route, *derived* — not cited.

    Both crosses become ε symbols in the frame; the frame dots become δ's;
    contracting the δ brings the two ε's onto a shared index, where the
    ε-pair contraction turns them into δδ − δδ; contracting those and
    reassembling gives the invariant right-hand side.  Nothing along the way
    assumes the Lagrange identity — it comes out of the ε-δ machinery, the
    same route challenge 000001 takes for bac-cab.

    Every step is asserted to fire, so a step that stops pulling its weight
    shows up as a failure rather than as silent decoration.
    """
    ctx, frame, (a, b, c, d) = _setup()
    lhs = (a % b) @ (c % d)
    rhs = (a @ c) * (b @ d) - (a @ d) * (b @ c)
    show("claim: lhs", lhs)
    show("claim: rhs", rhs)

    cov = tb.Variance.Covariant
    drv = td.Derivation(lhs)
    drv.step(lambda e: tb.expand_in_basis(e, frame, cov), label="expand in basis")
    drv.step(lambda e: tb.simplify_basis_cross(e, frame), label="crosses → ε")
    drv.step(lambda e: tb.simplify_basis_dot(e, frame), label="frame dots → δ")
    drv.step(td.canonicalize, label="materialize Einstein sums")
    drv.step(td.contract_delta, label="contract δ (joins the ε pair)")
    drv.step(td.contract_eps_pair, label="Σ_m ε ε → δδ − δδ")
    drv.step(td.contract_delta, label="contract the new δ's")
    drv.step(td.simplify, label="simplify")
    drv.step(lambda e: tb.reassemble(e, frame), label="reassemble invariant")

    for (name, fired), result in zip(drv.steps, drv.history[1:]):
        show(f"[{'fired' if fired else 'no-op'}] {name}", result)

    assert all(fired for _, fired in drv.steps), (
        "every step of this route is load-bearing: " + repr(drv.steps)
    )
    harness.assert_algebraic_eq(drv.current, rhs, "Lagrange, derived")

    # And the same endpoint reached by stating the *intent* instead of the
    # answer: "get rid of the crosses" (vibe 000097).  The default cost keeps
    # (a×b)·(c×d), which is smaller — the expansion is preferred only because
    # the user asked for it.
    found, _ = td.engine_simplify(
        lhs, td.rules("cross", ctx=ctx), prefer="fewest_crosses"
    )
    show("engine_simplify(prefer='fewest_crosses')", found)
    harness.assert_algebraic_eq(found, rhs, "Lagrange, discovered by intent")
