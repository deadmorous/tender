"""Round-trip: expanding an invariant into a basis and folding it back is the
identity.

Expansion into components and reassembly into invariant form are inverses.
That is the contract the whole coordinate bridge rests on: every derivation
that drops into components to use the ε-δ machinery and then returns to
direct notation (challenges 000001 and 000014 among them) is trusting it.

So this challenge is a *battery*, not a hand-picked pair: a systematic sweep
over the expressible shapes of rank ≤ 2 — bare tensors, ⊗-polyads, sums,
scalar multiples, dots, traces, and their combinations — each expanded and
folded back, each asserted to return what it started as.

L1 sweeps the shapes the bridge supports.  L2 is the same sweep *including
cross products*, which do not yet fold back — see below.
"""

import tender
import tender.basis as tb
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="reassemble ∘ expand_in_basis = id (shape battery)",
    tier="B",
    source="vibes 000053/000061/000063 (reassembly engine)",
)


def _setup():
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    a, b, c = (tender.tensor(n, rank=1, ctx=ctx) for n in "abc")
    A, B = (tender.tensor(n, rank=2, ctx=ctx) for n in "AB")
    return ctx, frame, (a, b, c), (A, B)


def _round_trip(frame, expr):
    """Expand into the frame, reduce the index algebra, and fold back.

    The reduction is not optional decoration, and the reason is sharper than
    it looks: `reassemble` only ever folds a *whole term* of the form
    (coordinate tensor) × (polyad of basis vectors) — it never descends into a
    contraction operand, so `(a_i e_i)·(b_j e_j)` is not folded even here in
    an orthonormal frame.  What makes the dot round-trip is that
    `simplify_basis_dot` + `contract_delta` **remove the basis vectors
    altogether**, leaving `a_i b_i`, which is recognisable.  (Challenge
    000018's meta/l2-route.md follows this through: in an oblique basis the
    metric is not δ, nothing contracts away, and the fold never happens.)
    """
    x = tb.expand_in_basis(expr, frame, tb.Variance.Covariant)
    for _ in range(2):  # one pass per nesting level of contraction
        x = tb.simplify_basis_cross(x, frame)
        x = tb.simplify_basis_dot(x, frame)
        x = td.contract_delta(td.canonicalize(x))
    return tb.reassemble(td.simplify(x), frame)


def _shapes(vectors, matrices):
    a, b, c = vectors
    A, B = matrices
    return [
        ("a", a),
        ("A", A),
        ("a + b", a + b),
        ("A + B", A + B),
        ("a ⊗ b", a * b),
        ("a ⊗ b ⊗ c", a * b * c),
        ("a · b", a @ b),
        ("A · b", A @ b),
        ("tr A", tender.tr(A)),
        ("(a·b) c", (a @ b) * c),
        ("(a·b)(a·c)", (a @ b) * (a @ c)),
        ("a ⊗ b + b ⊗ a", a * b + b * a),
        ("tr(A) a", tender.tr(A) * a),
    ]


@harness.level("L1")
def test_round_trip_battery():
    """Every supported shape returns as itself."""
    ctx, frame, vectors, matrices = _setup()
    failures = []
    for label, expr in _shapes(vectors, matrices):
        back = _round_trip(frame, expr)
        ok = td.algebraic_eq(back, expr)
        show(f"{label:14s} →", back)
        if not ok:
            failures.append(f"{label}: came back as {back.latex()}")
    assert not failures, "round-trip broken for:\n  " + "\n  ".join(failures)


@harness.level(
    "L2",
    expected=False,
    reason="reassemble does not fold an ε back into a cross: "
    "ε_ikj a_k b_j e_i is not recognised as a × b",
)
def test_round_trip_battery_including_crosses():
    """The same battery with cross products — the one shape that does not close.

    Expanding `a × b` produces `ε_{ikj} a_k b_j e_i`, and `reassemble` has no
    rule that reads an ε-weighted component sum back as a cross.  Dots fold
    (via δ), traces fold, ⊗-polyads fold; ε does not.

    This is a real asymmetry in the bridge rather than a missing test: a
    derivation that expands a cross must currently return to invariant form by
    some other route — which is exactly what challenges 000001 and 000014 do,
    contracting the ε-pair into δ's *first* and reassembling from those.
    """
    ctx, frame, vectors, matrices = _setup()
    a, b, c = vectors
    for label, expr in [("a × b", a % b), ("a · (b × c)", a @ (b % c))]:
        back = _round_trip(frame, expr)
        show(f"{label:14s} →", back)
        harness.assert_algebraic_eq(back, expr, f"round-trip of {label}")
