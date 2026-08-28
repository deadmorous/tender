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

L1 sweeps the dot/trace/polyad shapes.  L2 is the same sweep *including
cross products*, which fold back since vibe 000103 — and, because those folds
are driven by the indices rather than the term's shape, they close with
unrelated factors standing alongside.
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

    The reduction is no longer what makes this work — since vibe 000103 the
    folds are driven by what each summed index *connects*, so `(a_i e_i)·(b_j
    e_j)` folds to `a·b` with the basis vectors still standing.  Reducing first
    is kept because it is what a derivation naturally does on the way through
    the index algebra, and the round-trip must close on that form too.

    (Challenge 000018 is a different matter: in an oblique basis the metric is
    not δ, and the metric row of the fold table is still empty — see its
    meta/l2-route.md.)
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


@harness.level("L2")
def test_round_trip_battery_including_crosses():
    """The same battery with cross products — the shape that used to be missing.

    Expanding `a × b` gives `ε_{ikj} a_k b_j e_i`, and folding it back was the
    asymmetry in the bridge: dots closed (via δ), traces closed, ⊗-polyads
    closed, ε did not.  A derivation that expanded a cross had to return by
    another route — contracting the ε-pair into δ's first, as challenges 000001
    and 000014 do.

    Vibe 000103 closed it, and by reading rather than matching.  ε's three
    indices say what it is doing: two of them landing on rank-1 carriers and
    one on a basis vector is a cross realized at that leg; all three on carriers
    is the scalar triple product.  The slot *order* fixes the result — ε is
    totally antisymmetric, so rotating the leg index to the front costs no sign
    (ε_abc = ε_bca = ε_cab), and the remaining two in that rotated order are the
    operands.  Getting it wrong would silently flip a sign, so it is computed.

    Because the fold is driven by the indices and not by the shape of the whole
    term, the crosses below close with arbitrary company around them — a scalar
    coefficient, another factor, a sum.  That is the property being certified
    here, not merely the bare `a × b`.
    """
    ctx, frame, vectors, matrices = _setup()
    a, b, c = vectors
    A, B = matrices
    cases = [
        ("a × b", a % b),
        ("a · (b × c)", a @ (b % c)),
        # The same folds with unrelated factors alongside: what defeated the
        # shape-directed fold, and what an index-directed one is immune to.
        ("(a·b) (a × c)", (a @ b) * (a % c)),
        ("tr(A) (a × b)", tender.tr(A) * (a % b)),
        ("(a × b) ⊗ c", (a % b) * c),
        ("a × b + c", a % b + c),
        ("(a·(b×c)) a", (a @ (b % c)) * a),
    ]
    failures = []
    for label, expr in cases:
        back = _round_trip(frame, expr)
        show(f"{label:16s} →", back)
        if not td.algebraic_eq(back, expr):
            failures.append(f"{label}: came back as {back.latex()}")
    assert not failures, "cross round-trip broken for:\n  " + "\n  ".join(
        failures
    )
