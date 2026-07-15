"""Tests for tender.render — the path-labeled view (vibe 000054)."""

import pytest

import tender
import tender.basis as tb
import tender.derivation as td
import tender.render as tr


def _abI(ctx):
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    I = tender.identity(ctx=ctx)
    return a, b, I


def test_labeled_latex_matches_plain_render():
    ctx = tender.Context()
    a, b, I = _abI(ctx)
    e = (a * I) * b
    assert tr.labeled(e).latex == e.latex()


def test_legend_round_trips_every_path():
    """Each legend path selects exactly the subexpression its LaTeX shows."""
    ctx = tender.Context()
    a, b, I = _abI(ctx)
    e = (a * I) * b
    for p, sub in tr.labeled(e, which="all").legend:
        assert e.at(p).latex() == sub


def test_legend_round_trips_under_laplacian_folding():
    """The pretty form folds ∇·(∇⊗X) to Δ X, but the legend paths stay exact."""
    ctx = tender.Context()
    X = tender.tensor("X", rank=1, ctx=ctx)
    e = tender.laplacian(X)
    assert e.latex() == r"\Delta \mathbf{X}"  # folded
    labeled = tr.labeled(e, which="all")
    # A path reaches the inner X even though the whole renders as Δ X.
    assert any(sub == r"\mathbf{X}" for _, sub in labeled.legend)
    for p, sub in labeled.legend:
        assert e.at(p).latex() == sub


def test_which_policies_select_the_right_nodes():
    ctx = tender.Context()
    a, b, I = _abI(ctx)
    e = (a * I) * b

    wellknown = [p for p, _ in tr.labeled(e, which="wellknown").legend]
    assert wellknown == [[0, 1]]  # only the I

    atoms = [p for p, _ in tr.labeled(e, which="atoms").legend]
    assert atoms == [[0, 0], [0, 1], [1]]  # a, I, b

    allp = [p for p, _ in tr.labeled(e, which="all").legend]
    assert allp == [[], [0], [0, 0], [0, 1], [1]]


def test_which_terms_uses_addends():
    ctx = tender.Context()
    a, b, I = _abI(ctx)
    e = a + b - I.vec()
    view = tr.labeled(e, which="terms")
    assert [p for p, _ in view.legend] == e.addends()
    assert [sub for _, sub in view.legend] == [
        e.at(p).latex() for p in e.addends()
    ]


def test_index_names_consistent_between_whole_and_parts():
    """Sub-renders share the whole's index map, so names match."""
    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    frame = tb.wcs(ctx)
    e = td.canonicalize(tb.expand_in_basis(a, frame, tb.Variance.Covariant))
    view = tr.labeled(e, which="all")
    # The body's sub-latex (with its dummy index) is a substring of the whole
    # — it would not be if the two renders allocated different index letters.
    body = [sub for p, sub in view.legend if p and r"\sum" not in sub][0]
    assert body in view.latex


def test_invalid_which_raises():
    ctx = tender.Context()
    a, _, _ = _abI(ctx)
    with pytest.raises(ValueError):
        a.paths("nonsense")


def test_repr_html_contains_expression_and_paths():
    ctx = tender.Context()
    a, b, I = _abI(ctx)
    e = (a * I) * b
    h = tr.labeled(e, which="wellknown")._repr_html_()
    assert "table" in h
    assert "[0, 1]" in h  # the I's path
    assert r"\mathbf{I}" in h
