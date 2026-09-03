"""A declared constraint is what a symbol *means*, and only that symbol has it.

`ws.rotation("P")` says P·Pᵀ = Pᵀ·P = I; `ws.vector("n", unit=True)` says
n·n = 1.  Neither is a fact about the shape of an expression — an arbitrary
rank-2 tensor is not orthogonal, and saying so must stay refutable — so the
licence is attached to the *symbol*, and this challenge is as much about the
negatives as the positives.

Three things had to be true at once, and each was a separate discovery
(vibe 000110 I4):

  * **The rules must be in force without being passed.**  A constraint is not
    an optional hypothesis to remember at each call site.
  * **The component decision procedure must abstain.**  Its expansion writes P
    as nine independent components, which satisfy no relation; comparing them
    refutes true conditional claims.  Constraints are quadratic in the
    components, so this is the boundary of what a component check can decide,
    not a gap to close later.
  * **The symbol must be literal in the rule's pattern.**  A slot-less abstract
    tensor is a pattern *variable*, so `P·Pᵀ → I` first read as "for any X,
    X·Xᵀ = I" and cheerfully proved the orthogonality of every tensor in sight
    — measured, and the reason the declaration is stamped on the symbol rather
    than only registered beside it.
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="declared constraints: a rotation is orthogonal, an arbitrary tensor is not",
    tier="A",
    source="vibe 000110 I4",
)


def _setup():
    ws = t.Workspace()
    return ws, ws.rotation("P"), ws.vector("n", unit=True), ws.identity()


@harness.level("L2")
def test_a_declared_constraint_holds_without_being_passed():
    """No rule list: the constraint is in force because P is what it is."""
    ws, P, n, I = _setup()
    for label, lhs, rhs in [
        ("P·Pᵀ = I", P @ P.transpose(), I),
        ("Pᵀ·P = I", P.transpose() @ P, I),
        ("n·n = 1", n @ n, t.scalar(1, ctx=ws.ctx)),
    ]:
        result = td.prove_equal(lhs, rhs, [])
        show(label, f"{result.status}, fired={result.fired}")
        assert result.proved, f"{label} should hold for a declared symbol"


@harness.level("L1")
def test_an_undeclared_symbol_gets_no_licence():
    """The same claims about arbitrary symbols are false, and still refuted.

    This is the test that would catch a rule minted as a schema: if `P·Pᵀ → I`
    meant "for any X", these would prove.
    """
    ws, P, n, I = _setup()
    A = ws.tensor("A", rank=2)
    c = ws.tensor("c", rank=1)
    for label, lhs, rhs in [
        ("A·Aᵀ = I", A @ A.transpose(), I),
        ("c·c = 1", c @ c, t.scalar(1, ctx=ws.ctx)),
    ]:
        result = td.prove_equal(lhs, rhs, [])
        show(label, result.status)
        assert result.refuted, f"{label} is false and must stay refutable"


@harness.level("L1")
def test_a_conditional_claim_is_not_refuted():
    """The component check abstains where it cannot represent the condition.

    `(P·a)·(P·b) = a·b` is true *given* the constraint.  Before the abstention
    the procedure expanded P as an arbitrary tensor, found the two sides
    different, and refuted — the wrong answer to a question it cannot ask.
    """
    ws, P, n, I = _setup()
    a, b = ws.vector("a"), ws.vector("b")
    result = td.prove_equal((P @ a) @ (P @ b), a @ b, [])
    show("(P·a)·(P·b) = a·b", result.status)
    assert not result.refuted


@harness.level(
    "L2",
    expected=False,
    reason="a rule does not fire inside a longer contraction chain "
    "(vibe 000100 context-blocking; measured again in vibe 000110 I4)",
)
def test_orthogonality_fires_inside_a_longer_chain():
    """(P·a)·(P·b) = a·b — the enumerated red, and it is not about constraints.

    Canon normalises the left side to the chain `a·Pᵀ·P·b`, and no rule fires
    on the interior run `Pᵀ·P`: a two-factor pattern does not match a
    contiguous sub-run of a longer contraction chain.  Measured with ordinary
    symbols and no constraint anywhere — `A·B → I` does not fire inside
    `a·A·B·b` either — so this is vibe 000100's context-blocking problem, and
    it is where the next increment of it should start.
    """
    ws, P, n, I = _setup()
    a, b = ws.vector("a"), ws.vector("b")
    result = td.prove_equal(
        (P @ a) @ (P @ b), a @ b, td.rules("transpose", ctx=ws.ctx)
    )
    show("(P·a)·(P·b) = a·b", f"{result.status}, fired={result.fired}")
    assert result.proved
