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


@harness.level(
    "L2",
    expected=False,
    reason="needs, in order: the metric exposed to Python at all (no "
    "tender.metric), the axiom g^ij g_jk = δ^i_k, and lowering a_i = g_ij a^j. "
    "The round-trip alternative (reassemble to a·b, re-expand) is separately "
    "blocked and would be a weaker proof — see meta/l2-route.md",
)
def test_the_three_forms_are_mutually_equal():
    """All three expansions are the same scalar — but tender cannot yet say so.

    Each form above is correct, and each comes from the *same* invariant a·b,
    so they are equal by construction.  Showing it *within* the component
    algebra is a different matter: it needs the raising/lowering identity
    g_ij g^jk = δ_i^k to convert one form into another, and there is no step
    or rule for that.  That is the concrete gap this challenge names.
    """
    harness.todo(
        "raise/lower indices with g_ij g^jk = δ_i^k so the covariant, "
        "contravariant and mixed forms of a·b can be converted into one another"
    )
