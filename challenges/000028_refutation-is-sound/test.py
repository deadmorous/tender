"""`refuted` must mean refuted: no true statement is ever called false.

Of the verdicts `prove_equal` returns, `refuted` is the only one that is a claim
about *the mathematics* rather than about the library's reach.  `exhausted`
says the rules were not enough; `budget` says the search stopped; `unsupported`
says the shape was not understood.  Only `refuted` says: this is false.  So it
is the one verdict that must never be wrong, and a challenge is the right place
to keep it that way — a defect here does not make a derivation fail, it makes
the library confidently contradict a true identity.

The defect this pins (vibe 000110 M8, found while measuring the transport rules
the rotation increments need) had two independent causes, and each gets a test
below:

  * **Ordering.**  A trace or transpose opened into frame dots only *after*
    `simplify_basis_dot` had gone by, so unevaluated `i·j` reached the
    comparison; two sides of a true identity differed term by term.
    `tr(Aᵀ) = tr(A)` was refuted.
  * **A missing residue.**  A tensor of unknown rank cannot expand on a frame
    at all, so its contraction survives whole — and a surviving contraction was
    not counted as "the reduction did not finish".  `tr(X·Y) = tr(Y·X)` was
    refuted.

Both classes share one shape: *the reduction did not finish, and the leftovers
were read as a difference.*  The rule that follows — a complete reduction
leaves a polynomial in the component symbols and nothing else — is what the
library now enforces.
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="`refuted` is sound: no true identity is called false",
    tier="A",
    source="vibe 000110 M8",
)


def _setup():
    ws = t.Workspace()
    return ws, ws.tensor("A", rank=2), ws.tensor("B", rank=2)


def _true_transpose_identities(ws, A, B):
    a, b = ws.tensor("a", rank=1), ws.tensor("b", rank=1)
    At, Bt = A.transpose(), B.transpose()
    return [
        ("tr(Aᵀ) = tr(A)", At.tr(), A.tr()),
        ("a·Aᵀ = A·a", a @ At, A @ a),
        ("a·A = Aᵀ·a", a @ A, At @ a),
        ("Aᵀ·Bᵀ = (B·A)ᵀ", At @ Bt, (B @ A).transpose()),
        ("(a⊗b)·A = a⊗(Aᵀ·b)", (a * b) @ A, a * (At @ b)),
    ]


@harness.level("L1")
def test_no_true_transpose_identity_is_refuted():
    """Each of these is true, and each was refuted before the fix.

    Asked with **no rules at all**, so the component procedure is the only
    thing that can answer and the test is about it alone.  (With the transpose
    group supplied these now prove outright — challenge 000029 — but that would
    test the rules, not the procedure they are independent of.)
    """
    ws, A, B = _setup()
    for label, lhs, rhs in _true_transpose_identities(ws, A, B):
        result = td.prove_equal(lhs, rhs, [])
        show(label, f"{result.status}, components_agree={result.components_agree}")
        assert not result.refuted, f"{label} was refuted, and it is true"
        assert result.components_agree, (
            f"{label}: the component procedure should decide these equal"
        )


@harness.level("L1")
def test_an_unexpandable_contraction_is_undecided_not_refuted():
    """Trace cyclicity, stated for tensors whose rank is not declared.

    Nothing about `X` says it can be written on a frame, so the reduction
    cannot start; the trace and the dot survive whole.  Reading that leftover
    as a difference refutes a theorem.
    """
    ws, _, _ = _setup()
    X, Y = ws.tensor("X"), ws.tensor("Y")
    result = td.prove_equal((X @ Y).tr(), (Y @ X).tr(), [])
    show("tr(X·Y) = tr(Y·X), rank undeclared", result.status)
    assert not result.refuted
    assert not result.components_agree  # undecided, not decided-equal


@harness.level("L1")
def test_a_false_claim_is_still_refuted():
    """Soundness bought by refusing to answer would be no fix at all.

    A general tensor is not its own transpose, a general pair does not commute
    under the dot, and the procedure still says so.
    """
    ws, A, B = _setup()
    for label, lhs, rhs in [
        ("Aᵀ = A", A.transpose(), A),
        ("A·B = B·A", A @ B, B @ A),
        ("tr(A) = tr(A) + 1", A.tr(), A.tr() + t.scalar(1, ctx=ws.ctx)),
    ]:
        result = td.prove_equal(lhs, rhs, td.rules("transpose", ctx=ws.ctx))
        show(label, result.status)
        assert result.refuted, f"{label} is false and should be refuted"
