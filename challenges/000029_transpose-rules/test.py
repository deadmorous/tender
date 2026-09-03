"""Moving a transpose through the algebra: the five rules, verified.

    (A·B)ᵀ = Bᵀ·Aᵀ      tr(Aᵀ) = tr(A)      (A·u)·v = u·(Aᵀ·v)
    a·Aᵀ = A·a          vec(Aᵀ) = −vec(A)

Nothing here is deep — each is a line of index algebra — but the library had
none of them, and the gap was *invisible*: before vibe 000110 I0 every one of
these came back `refuted`, so a missing rule read as a false statement rather
than as an absent one.  That is why they arrive with a challenge of their own
rather than as a quiet addition.

`transpose-adjoint` is the one that earns its place beyond bookkeeping: it is
how an orthogonality argument moves, taking `(P·a)·(P·b)` to `a·(Pᵀ·P)·b` and
thence to `a·b`.  The rotation increments of vibe 000110 rest on it.

Two neighbours are deliberately absent, and their absence is checked below:
`(a⊗b)ᵀ = b⊗a` and `(A+B)ᵀ = Aᵀ+Bᵀ` are already decided by canonicalization, so
a rule for either would cost saturation time and never fire.
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="transposing through products, traces, dots and vec",
    tier="A",
    source="vibe 000110 I5/I6 (rotation tensors need them)",
    proves=[
        "transpose-product",
        "transpose-trace",
        "transpose-adjoint",
        "transpose-dot-left",
        "transpose-vec",
    ],
)


def _setup():
    ws = t.Workspace()
    A, B = ws.tensor("A", rank=2), ws.tensor("B", rank=2)
    u, v = ws.tensor("u", rank=1), ws.tensor("v", rank=1)
    return ws, A, B, u, v


def _claims(A, B, u, v):
    At, Bt = A.transpose(), B.transpose()
    return [
        ("(A·B)ᵀ = Bᵀ·Aᵀ", (A @ B).transpose(), Bt @ At),
        ("tr(Aᵀ) = tr(A)", At.tr(), A.tr()),
        ("(A·u)·v = u·(Aᵀ·v)", (A @ u) @ v, u @ (At @ v)),
        ("a·Aᵀ = A·a", u @ At, A @ u),
        ("vec(Aᵀ) = −vec(A)", At.vec(), -(A.vec())),
    ]


@harness.level("L1")
def test_each_rule_is_true_in_components():
    """The component decision procedure, which is what a hand check would do.

    Independent of the e-graph by construction (vibe 000097), so this is not
    the rules confirming themselves.
    """
    ws, A, B, u, v = _setup()
    for label, lhs, rhs in _claims(A, B, u, v):
        result = td.prove_equal(lhs, rhs, [])  # no rules at all
        show(label, f"{result.status}, components_agree={result.components_agree}")
        assert result.components_agree, f"{label} does not hold in components"


@harness.level("L2")
def test_each_rule_fires_and_proves_its_own_statement():
    """A registered rule that never fires is worse than no rule.

    Each is asked to prove exactly what it states, from the shipped library —
    so the check covers the registration and the pattern, not just the claim.
    """
    ws, A, B, u, v = _setup()
    rules = td.rules("transpose", ctx=ws.ctx)
    for label, lhs, rhs in _claims(A, B, u, v):
        result = td.prove_equal(lhs, rhs, rules)
        show(label, f"{result.status}, fired={result.fired}")
        assert result.proved, f"{label} did not prove from the transpose group"
        assert result.fired, f"{label} proved without any rule firing"


@harness.level(
    "L2",
    expected=False,
    reason="the component check refutes a conditional claim (vibe 000110 M1); "
    "needs I4's constrained symbols",
)
def test_the_orthogonality_move_is_what_this_group_is_for():
    """(P·a)·(P·b) = a·b, given only that P·Pᵀ = I — the reason for the group.

    Enumerated red, and it is precisely vibe 000110 I4's acceptance test.  The
    rules are all present now: `transpose-adjoint` takes the left side to
    `a·(Pᵀ·P)·b`, and the two orthogonality rules finish it.  What stops it is
    upstream of the rules — the component decision procedure expands `P` as an
    *arbitrary* tensor, finds the sides differ, and refutes before a single
    rule fires (M1).  A hypothesis given as a rewrite rule is invisible to it.

    I4 fixes that by carrying orthogonality on the *symbol*, where both the
    rules and the refutation can see it.  When it lands, this test starts
    passing and CI turns red until the marker is removed — which is the point
    of a strict xfail.
    """
    ws, A, B, u, v = _setup()
    P = ws.tensor("P", rank=2)
    I = ws.identity()
    orthogonal = [
        td.Identity("P-orthogonal", P @ P.transpose(), I),
        td.Identity("P-orthogonal-T", P.transpose() @ P, I),
    ]
    result = td.prove_equal(
        (P @ u) @ (P @ v),
        u @ v,
        td.rules("transpose", "dyadic", ctx=ws.ctx) + orthogonal,
    )
    show("(P·a)·(P·b) = a·b", f"{result.status}, fired={result.fired}")
    assert result.proved
    assert "transpose-adjoint" in result.fired


@harness.level("L1")
def test_canon_already_knows_the_two_that_are_not_rules():
    """(a⊗b)ᵀ = b⊗a and (A+B)ᵀ = Aᵀ+Bᵀ need no rule — so they have none.

    Registering an inert rule is not free: it is matched on every saturation
    pass and never fires.  This pins the reason they were left out, so a future
    tidy-up does not "complete" the group by adding them.
    """
    ws, A, B, u, v = _setup()
    for label, lhs, rhs in [
        ("(a⊗b)ᵀ = b⊗a", (u * v).transpose(), v * u),
        ("(A+B)ᵀ = Aᵀ+Bᵀ", (A + B).transpose(), A.transpose() + B.transpose()),
    ]:
        result = td.prove_equal(lhs, rhs, [])
        show(label, f"{result.status} with no rules")
        assert result.proved, f"{label} was expected to fall to canon alone"
