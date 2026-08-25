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
    proves="bac-cab",
)


def _setup():
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    c = tender.tensor("c", rank=1, ctx=ctx)
    return ctx, frame, a, b, c


@harness.level("L1")
def test_verified_in_concrete_components():
    """Both sides agree component-by-component in the World Cartesian System."""
    ctx, frame, a, b, c = _setup()
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
    """The symbolic ε-route: ε ε → δδ − δδ → the invariant rhs, as performed.

    Run as a :class:`td.Derivation`, so every step's *fired* status is
    recorded (vibe 000095 increment 3) and narrated — each line below shows
    [fired] (the step changed the expression) next to its result."""
    ctx, frame, a, b, c = _setup()
    lhs = a % (b % c)
    rhs = b * (a @ c) - c * (a @ b)
    show("claim: lhs", lhs)
    show("claim: rhs", rhs)

    # Historical note: this route originally carried two extra steps — a
    # second `simplify_basis_cross` "for the inner cross" and an
    # `expand_products` "to split the −".  The fired/no-op reporting exposed
    # both as no-ops on its first run (one basis-cross pass handles both
    # crosses; contract_delta eats both addends unsplit), so they are gone —
    # the reporting doing exactly its vibe-000056 job.
    cov = tb.Variance.Covariant
    drv = td.Derivation(lhs)
    drv.step(lambda e: tb.expand_in_basis(e, frame, cov), label="expand in basis")
    drv.step(lambda e: tb.simplify_basis_cross(e, frame), label="crosses → ε-pair")
    drv.step(td.canonicalize, label="materialize Einstein sums")
    drv.step(td.contract_eps_pair, label="Σ_m ε ε → δδ − δδ")
    drv.step(td.contract_delta, label="contract the δ's")
    drv.step(td.simplify, label="simplify")
    drv.step(lambda e: tb.reassemble(e, frame), label="reassemble invariant")

    for (name, fired), result in zip(drv.steps, drv.history[1:]):
        show(f"[{'fired' if fired else 'no-op'}] {name}", result)

    assert all(fired for _, fired in drv.steps), (
        "every step of the bac-cab route is load-bearing; a no-op means the "
        "route changed underneath us: " + repr(drv.steps)
    )
    harness.assert_algebraic_eq(drv.current, rhs, "bac-cab, symbolic derivation")

    # The same result, reached the other way: state the *intent* rather than
    # the answer.  Under the default cost the engine keeps a×(b×c) — it has
    # fewer nodes — so asking for fewest crosses is what makes the expansion
    # the preferred reading of the same saturated graph (vibe 000097).
    found, _ = td.engine_simplify(
        lhs, td.rules("cross", ctx=ctx), prefer="fewest_crosses"
    )
    show("engine_simplify(prefer='fewest_crosses')", found)
    harness.assert_algebraic_eq(found, rhs, "bac-cab, discovered by intent")
