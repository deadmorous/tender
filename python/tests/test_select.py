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


def test_find_other_well_known_kind():
    ctx = tender.Context()
    i = ctx.alloc_index()
    d = tender.delta(
        tender.Realm.Oblique, tender.space_3d,
        tender.Level.Upper, tender.Level.Lower, i, i,
    )
    a = tender.tensor("a", rank=1, ctx=ctx)
    e = a * d
    assert e.find(kind="Delta") == [[1]]
    assert e.find(kind="Identity") == []


def test_find_kind_and_name_together():
    ctx = tender.Context()
    a, b, I = _abI(ctx)
    e = (a * I) * b
    # kind and name both narrow (AND): the identity is not named "a".
    assert e.find(kind="Identity", name="a") == []
    assert e.find(kind="Identity", name="I") == [[0, 1]]


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


# ---- Validation on the chart machinery (vibe 000054 increment 5) -----------
#
# Reassembly of one term is the mechanism vibe 000075 gap D needs (fold one
# e_i-contracted second-derivative term back to an invariant operator without
# disturbing its neighbours).  These exercise extract → transform → splice on
# the real basis machinery.

def _reassemble_term(frame):
    return lambda s: td.canonicalize(tb.reassemble(td.canonicalize(s), frame))


def test_selective_reassembly_round_trip():
    """b + Σ_i a_i e_i → reassemble only the coordinate term → b + a."""
    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    frame = tb.wcs(ctx)

    ax = td.canonicalize(tb.expand_in_basis(a, frame, tb.Variance.Covariant))
    e = td.canonicalize(ax + b)
    coord_p = [p for p in e.addends() if r"\sum" in e.at(p).latex()][0]

    out = td.at(e, coord_p, _reassemble_term(frame))
    assert td.algebraic_eq(out, td.canonicalize(a + b))


def test_selective_reassembly_folds_only_the_targeted_term():
    """Two coordinate terms; fold exactly one — the other stays expanded."""
    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    frame = tb.wcs(ctx)
    cov = tb.Variance.Covariant

    ax = td.canonicalize(tb.expand_in_basis(a, frame, cov))
    bx = td.canonicalize(tb.expand_in_basis(b, frame, cov))
    e = td.canonicalize(ax + bx)
    assert e.latex().count(r"\sum") == 2

    out = td.at(e, e.addends()[0], _reassemble_term(frame))
    # One term reassembled to an invariant; the other is untouched.
    assert out.latex().count(r"\sum") == 1


# ---------------------------------------------------------------------------
# Index-algebra steps under `at` (vibe 000104)
#
# The addressed part is a *subtree*; an index contraction acts on an index
# *cluster* — factors scattered across a product, plus the binder above them.
# A step run under `at` can therefore sum an index away while its Σ stays
# behind, and a leftover binder is not harmless: `Σ_m X` with X free of m is
# dim·X, not X.  Every case below produced a stranded `Σ_?` before the fix.
# ---------------------------------------------------------------------------


def _delta_term(ctx):
    """`Σ_j Σ_k Σ_i δ_kj a_k b_j c_i e_i` — a δ cluster with company."""
    frame = tb.wcs(ctx)
    a, b, c = (tender.tensor(n, rank=1, ctx=ctx) for n in "abc")
    x = tb.expand_in_basis((a @ b) * c, frame, tb.Variance.Covariant)
    return frame, td.canonicalize(tb.simplify_basis_dot(x, frame))


def test_at_drops_the_binder_a_contraction_consumed():
    ctx = tender.Context()
    _, e = _delta_term(ctx)
    whole = td.contract_delta(e)

    # The smallest subtree holding the whole δ cluster; its binder sits above.
    out = td.implicitize(td.at(e, [0, 0], td.contract_delta))
    assert "?" not in out.latex()
    assert td.algebraic_eq(out, whole)


def test_at_agrees_with_the_whole_expression_at_every_enclosing_depth():
    # Addressing further down means more binders above, and each one that the
    # contraction consumed has to go.  The answer must not depend on where the
    # caller happened to point.
    ctx = tender.Context()
    _, e = _delta_term(ctx)
    whole = td.contract_delta(e)
    for path in ([0, 0], [0, 0, 0], [0, 0, 0, 0]):
        out = td.implicitize(td.at(e, path, td.contract_delta))
        assert "?" not in out.latex(), (path, out.latex())
        assert td.algebraic_eq(out, whole), (path, out.latex())


def test_at_leaves_an_already_vacuous_binder_alone():
    # A binder that was vacuous *before* the step is not this step's doing, and
    # `Σ_m X` with X free of m means dim·X — dropping it would change the value.
    ctx = tender.Context()
    i = ctx.alloc_index()
    a = tender.tensor("a", rank=1, ctx=ctx)
    e = tender.explicit_sum(i, a, ctx=ctx)
    out = td.at(e, [0], lambda s: s + s)
    assert out.latex().count(r"\sum") == 1


def test_at_refuses_when_the_index_is_still_carried_outside():
    # Removing an index the rest of the term still uses: the binder is still
    # doing work for that occurrence, and if the step *summed* over the index
    # the result is wrong regardless.  Refuse rather than guess.
    ctx = tender.Context()
    _, e = _delta_term(ctx)
    path = e.find(name="a")[0]
    with pytest.raises(ValueError, match="still carries"):
        td.at(e, path, lambda s: tender.scalar(1, ctx=ctx))


def test_at_is_a_no_op_when_the_step_is():
    ctx = tender.Context()
    _, e = _delta_term(ctx)
    assert td.structural_eq(td.at(e, [0, 0], lambda s: s), e)
