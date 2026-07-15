"""Tests for positional addressing — selective application and extraction (vibe 000054)."""

import pytest

import tender
import tender.basis as tb
import tender.derivation as td


def _abI(ctx):
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    I = tender.identity(ctx=ctx)
    return a, b, I


# ---- find ------------------------------------------------------------------

def test_find_identity_paths():
    ctx = tender.Context()
    a, b, I = _abI(ctx)
    e = (a * I) * b  # (a ⊗ I) ⊗ b — I at [0, 1]
    assert e.find(kind="Identity") == [[0, 1]]


def test_find_multiple_preorder():
    ctx = tender.Context()
    a, b, I = _abI(ctx)
    e = ((a * I) * b) * I  # I's at [0, 0, 1] and [1]
    assert e.find(kind="Identity") == [[0, 0, 1], [1]]


def test_find_by_name():
    ctx = tender.Context()
    a, b, I = _abI(ctx)
    e = (a * I) * b
    assert e.find(name="a") == [[0, 0]]
    assert e.find(name="b") == [[1]]


def test_find_requires_a_selector():
    ctx = tender.Context()
    a, _, _ = _abI(ctx)
    with pytest.raises(ValueError):
        a.find()


# ---- at (extraction) -------------------------------------------------------

def test_at_extracts_subexpression():
    ctx = tender.Context()
    a, b, I = _abI(ctx)
    e = (a * I) * b
    p = e.find(kind="Identity")[0]
    assert td.structural_eq(e.at(p), I)
    assert td.structural_eq(e.at([0, 0]), a)


def test_at_out_of_range_raises():
    ctx = tender.Context()
    a, _, _ = _abI(ctx)
    e = a * a
    with pytest.raises(IndexError):
        e.at([9])


# ---- addends ---------------------------------------------------------------

def test_addends_enumerates_terms():
    ctx = tender.Context()
    a, b, I = _abI(ctx)
    e = a + b - I.vec()  # (a + b) - vec(I)
    paths = e.addends()
    assert len(paths) == 3
    assert td.structural_eq(e.at(paths[0]), a)
    assert td.structural_eq(e.at(paths[1]), b)
    assert td.structural_eq(e.at(paths[2]), I.vec())


# ---- rewrite_at / replace_at (selective application) -----------------------

def test_replace_at_splices_back():
    ctx = tender.Context()
    a, b, _ = _abI(ctx)
    c = tender.tensor("c", rank=1, ctx=ctx)
    e = a * b
    out = e.replace_at([0], c)
    assert td.structural_eq(out, c * b)


def test_rewrite_at_round_trip_matches_whole_tree():
    """Extract a term, work on it, splice back — equals rewriting in place."""
    ctx = tender.Context()
    a, b, _ = _abI(ctx)
    e = a * b
    p = [1]
    sub = e.at(p)
    out = e.replace_at(p, sub.transpose())
    assert td.structural_eq(out, a * b.transpose())


def test_selective_expand_leaves_neighbours_symbolic():
    """The vibe-54 headline: expand ONLY one I in a × I × b."""
    ctx = tender.Context()
    a, b, I = _abI(ctx)
    frame = tb.wcs(ctx)
    e = (a * I) * b

    p = e.find(kind="Identity")[0]
    out = td.at(
        e, p, lambda s: tb.expand_in_basis(s, frame, tb.Variance.Covariant)
    )
    latex = out.latex()
    # I became Σ_i e_i ⊗ e_i; a and b never acquired coordinate indices.
    assert r"\mathbf{e}_{i}" in latex
    assert "a_" not in latex and "b_" not in latex


def test_at_targets_one_of_several_identities():
    """Only the second I expands; the first stays symbolic."""
    ctx = tender.Context()
    a, b, I = _abI(ctx)
    frame = tb.wcs(ctx)
    e = (a * I) * (b * I)  # two I's

    assert len(e.find(kind="Identity")) == 2
    second = e.find(kind="Identity")[1]
    out = td.at(
        e, second, lambda s: tb.expand_in_basis(s, frame, tb.Variance.Covariant)
    )
    # Exactly one bare I remains (the untouched first occurrence).
    assert len(out.find(kind="Identity")) == 1
