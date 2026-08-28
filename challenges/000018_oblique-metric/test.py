"""Oblique basis: the metric raises and lowers, and the three forms of a·b agree.

In an orthonormal frame the distinction between co- and contravariant
components collapses, and every dot product is Σ aᵢbᵢ.  In an *oblique* basis
it does not, and the metric is what mediates:

    a·b  =  g_ij a^i b^j        (both expanded on the covariant frame gᵢ)
         =  g^ij a_i b_j        (both on the reciprocal frame gⁱ)
         =  a^i b_i             (one of each — the metric folds into δ)

That the same invariant produces all three is the point: the components differ,
the scalar does not.  This is the machinery every non-orthogonal coordinate
system rests on, and the reason tender carries index *levels* rather than
treating all indices alike.

The basis here is genuinely oblique — three arbitrary vectors p, q, s, with
the reciprocal frame derived from them by the cross-product formula, so
nothing about the geometry is assumed.
"""

import tender
import tender.basis as tb
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="a·b = g_ij a^i b^j in an oblique basis",
    tier="B",
    source="Lurie, Theory of Elasticity, tensor-calculus appendix",
)


def _setup():
    ctx = tender.Context()
    # Three arbitrary (non-orthogonal, non-normalised) vectors.  The
    # contravariant cobasis is derived by the reciprocal formula.
    frame = tb.make_oblique_basis(
        [tender.tensor(n, rank=1, ctx=ctx) for n in ("p", "q", "s")],
        tender.space_3d,
    )
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    return ctx, frame, a, b


def _reduce(frame, expr):
    return td.canonicalize(tb.simplify_basis_dot(expr, frame))


@harness.level("L1")
def test_dot_product_through_the_metric():
    """Each expansion produces the metric form the level convention predicts."""
    ctx, frame, a, b = _setup()
    co, contra = tb.Variance.Covariant, tb.Variance.Contravariant

    # Expanding on the covariant frame gives contravariant components, which
    # the covariant metric contracts: a·b = g_ij a^i b^j.
    lower = _reduce(frame, tb.expand_in_basis(a @ b, frame, co))
    show("both on gᵢ  → g_ij a^i b^j", lower)
    assert "g_{ij}" in lower.latex(), lower.latex()
    assert "a^{i}" in lower.latex() and "b^{j}" in lower.latex()

    # …and on the reciprocal frame, the inverse metric: a·b = g^ij a_i b_j.
    upper = _reduce(frame, tb.expand_in_basis(a @ b, frame, contra))
    show("both on gⁱ  → g^ij a_i b_j", upper)
    assert "g^{ij}" in upper.latex(), upper.latex()
    assert "a_{i}" in upper.latex() and "b_{j}" in upper.latex()


@harness.level("L1")
def test_mixed_variance_folds_the_metric_into_delta():
    """One index up, one down: the metric becomes δ and contracts away.

    This is the payoff of the level convention — a·b = a^i b_i needs no metric
    at all, which is why mixed components are the natural pairing.
    """
    ctx, frame, a, b = _setup()
    co, contra = tb.Variance.Covariant, tb.Variance.Contravariant

    mixed = _reduce(
        frame,
        tb.expand_in_basis(a, frame, co) @ tb.expand_in_basis(b, frame, contra),
    )
    show("mixed → δ^i_j a^j b_i", mixed)
    assert "delta" in mixed.latex(), mixed.latex()

    contracted = td.simplify(td.contract_delta(mixed))
    show("after contracting δ", contracted)
    assert "delta" not in contracted.latex(), contracted.latex()
    assert "a^{i}" in contracted.latex() and "b_{i}" in contracted.latex()


@harness.level("L2")
def test_the_three_forms_are_mutually_equal():
    """All three forms converted into one another, inside the index algebra.

    Each form above is correct, and each came from the *same* invariant a·b —
    which is exactly why folding one back to a·b and re-expanding would prove
    nothing.  The content of this claim is the index gymnastics itself: that
    g_ij and g^ij are mutually inverse, and that raising and lowering are
    consistent.  So the derivation below never leaves component form.

    Two steps do it, and they are one operation read in both directions.
    `contract_metric` spends a metric to move an index — the surviving index is
    g's *other* index at g's *other* level — and `insert_metric` pays one to
    move it back:

        g^ij a_i b_j   →   a^i b_i   →   g_ij a^i b^j
                     raise a       lower b

    The mixed form in the middle is the one that needs no metric at all, which
    is the whole point of the level convention.
    """
    ctx, frame, a, b = _setup()
    co, contra = tb.Variance.Covariant, tb.Variance.Contravariant

    contravariant = _reduce(frame, tb.expand_in_basis(a @ b, frame, contra))
    covariant = _reduce(frame, tb.expand_in_basis(a @ b, frame, co))
    show("start:  g^ij a_i b_j", contravariant)

    mixed = td.contract_metric(contravariant)
    show("raise a → a^i b_i", mixed)
    assert "g" not in mixed.latex(), mixed.latex()
    assert "a^{i}" in mixed.latex() and "b_{i}" in mixed.latex()

    lowered = td.insert_metric(mixed, tender.Level.Upper)
    show("lower b → g_ij a^i b^j", lowered)
    harness.assert_algebraic_eq(
        lowered, covariant, "g^ij a_i b_j converted into g_ij a^i b^j"
    )

    # And the round the other way, so neither direction is privileged.
    raised = td.insert_metric(mixed, tender.Level.Lower)
    show("raise b → g^ij a_i b_j", raised)
    harness.assert_algebraic_eq(
        raised, contravariant, "the mixed form converted back to g^ij a_i b_j"
    )


@harness.level("L2")
def test_the_metric_and_its_inverse_are_inverse():
    """g^ij g_jk = δ^i_k — the fact the conversions above rest on.

    Contracting the inverse metric against the metric is the same single
    operation: raising the lower index of g_jk gives g^i_k, and a g whose slots
    straddle the upper/lower divide *is* the Kronecker δ — that is what the
    reciprocal basis means (g^i_j = e^i·e_j).  So the axiom is not a separate
    rule to be postulated; it falls out of what raising does.
    """
    ctx, frame, a, b = _setup()
    i, j, k = (ctx.alloc_index() for _ in range(3))
    U, L = tender.Level.Upper, tender.Level.Lower

    def g(level0, level1, x, y):
        return tender.metric(
            tender.Realm.Oblique, tender.space_3d, level0, level1, x, y, ctx=ctx
        )

    pair = g(U, U, i, j) * g(L, L, j, k)
    show("g^ij g_jk", pair)
    contracted = td.contract_metric(pair)
    show("contracted", contracted)
    assert "delta" in contracted.latex(), contracted.latex()
    assert "g" not in contracted.latex(), contracted.latex()

    # And it then contracts like any δ, so the derivation can carry on:
    # δ^i_k g^kl = g^il.  (The partner's k must sit at the opposite level —
    # a same-level pair is not an Einstein contraction, and nothing fires.)
    l = ctx.alloc_index()
    onto = td.contract_delta(contracted * g(U, U, k, l))
    show("δ^i_k g^kl", onto)
    assert "delta" not in onto.latex(), onto.latex()
    assert "g" in onto.latex(), onto.latex()
