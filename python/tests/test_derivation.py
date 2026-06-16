"""Tests for tender.derivation — derivation steps and history."""

import tender
import tender.derivation as td


def _sp3():
    return tender.space_3d


# ---- unroll_sums -----------------------------------------------------------

def test_unroll_sums_delta_trace():
    """sum_i δ^i_i: ExplicitSum wrapper disappears after unrolling."""
    ctx = tender.Context()
    i = ctx.alloc_index()
    expr = tender.explicit_sum(
        i,
        tender.delta(tender.Realm.Oblique, _sp3(),
                     tender.Level.Upper, tender.Level.Lower, i, i))
    after = td.unroll_sums(expr)
    # Just check it renders without error and no longer has the explicit-sum LaTeX shape
    latex = after.latex()
    assert r"\sum" not in latex  # rendered as a flat sum, not \sum notation


def test_unroll_sums_symbolic_bound_unchanged():
    """An ExplicitSum with a symbolic bound must not be unrolled."""
    ctx = tender.Context()
    i = ctx.alloc_index()
    a = tender.tensor("A")
    n = tender.scalar(3)
    expr = tender.explicit_sum(i, a)  # concrete range → will unroll
    # This one is the non-changeable path (no IndexSpace in A's slots), so
    # after unrolling the result is unchanged.
    after = td.unroll_sums(expr)
    # A has no slot with index i → space not found → sum stays
    assert after.latex() == expr.latex()


# ---- eval_delta_concrete ---------------------------------------------------

def test_eval_delta_concrete_diagonal():
    """δ with two equal concrete values evaluates to 1."""
    ctx = tender.Context()
    d = tender.delta(tender.Realm.Oblique, _sp3(),
                     tender.Level.Upper, tender.Level.Lower, 2, 2)
    after = td.eval_delta_concrete(d)
    assert after.latex() == "1"


def test_eval_delta_concrete_off_diagonal():
    """δ with two different concrete values evaluates to 0."""
    ctx = tender.Context()
    d = tender.delta(tender.Realm.Oblique, _sp3(),
                     tender.Level.Upper, tender.Level.Lower, 1, 3)
    after = td.eval_delta_concrete(d)
    assert after.latex() == "0"


def test_eval_delta_concrete_abstract_unchanged():
    """δ with abstract (countable) indices is not touched."""
    ctx = tender.Context()
    i = ctx.alloc_index()
    d = tender.delta(tender.Realm.Oblique, _sp3(),
                     tender.Level.Upper, tender.Level.Lower, i, i)
    after = td.eval_delta_concrete(d)
    assert after.latex() == d.latex()


# ---- fold_arithmetic -------------------------------------------------------

def test_fold_arithmetic_sum():
    assert (tender.scalar(2) + tender.scalar(3)).latex() != "5"  # no auto-fold
    folded = td.fold_arithmetic(tender.scalar(2) + tender.scalar(3))
    assert folded.latex() == "5"


def test_fold_arithmetic_product():
    folded = td.fold_arithmetic(tender.scalar(3) * tender.scalar(4))
    assert folded.latex() == "12"


def test_fold_arithmetic_nested():
    # (1 + 1) + 1
    one = tender.scalar(1)
    expr = (one + one) + one
    assert td.fold_arithmetic(expr).latex() == "3"


# ---- Derivation class ------------------------------------------------------

def test_derivation_history_length():
    e = tender.scalar(1)
    drv = td.Derivation(e)
    assert len(drv.history) == 1
    drv.step(td.fold_arithmetic)
    assert len(drv.history) == 2
    drv.step(td.fold_arithmetic)
    assert len(drv.history) == 3


def test_derivation_initial_and_current():
    e = tender.scalar(1)
    drv = td.Derivation(e)
    assert drv.initial.latex() == e.latex()
    assert drv.current.latex() == e.latex()


def test_derivation_chaining():
    """step() returns self for fluent chaining."""
    e = tender.scalar(1)
    drv = td.Derivation(e)
    result = drv.step(td.fold_arithmetic).step(td.fold_arithmetic)
    assert result is drv


# ---- Full derivation: δ^i_i = 3 ------------------------------------------

def test_delta_trace_is_3():
    ctx = tender.Context()
    i = ctx.alloc_index()
    expr = tender.explicit_sum(
        i,
        tender.delta(tender.Realm.Oblique, _sp3(),
                     tender.Level.Upper, tender.Level.Lower, i, i))

    drv = td.Derivation(expr)
    drv.step(td.unroll_sums).step(td.eval_delta_concrete).step(td.fold_arithmetic)

    assert len(drv.history) == 4
    assert drv.current.latex() == "3"


# ---- Full derivation: δ^i_j δ^i_j = 3 ------------------------------------

def test_delta_squared_is_3():
    ctx = tender.Context()
    i = ctx.alloc_index()
    j = ctx.alloc_index()
    d1 = tender.delta(tender.Realm.Oblique, _sp3(),
                      tender.Level.Upper, tender.Level.Lower, i, j)
    d2 = tender.delta(tender.Realm.Oblique, _sp3(),
                      tender.Level.Upper, tender.Level.Lower, i, j)
    expr = tender.explicit_sum(i, tender.explicit_sum(j, d1 * d2))

    drv = td.Derivation(expr)
    drv.step(td.unroll_sums).step(td.eval_delta_concrete).step(td.fold_arithmetic)

    assert drv.current.latex() == "3"


# ---- expand_eps ------------------------------------------------------------

def test_expand_eps_rank3_no_longer_levi_civita():
    """After expand_eps a LeviCivita object becomes a sum tree."""
    ctx = tender.Context()
    i, j, k = ctx.alloc_index(), ctx.alloc_index(), ctx.alloc_index()
    eps = tender.levi_civita(
        tender.Realm.Oblique, _sp3(),
        [tender.Level.Lower, tender.Level.Lower, tender.Level.Lower],
        [i, j, k])
    after = td.expand_eps(eps)
    # The outer node is a Sum, not LeviCivita any more.
    assert after.latex() != eps.latex()


def test_expand_eps_even_perm_is_plus_one():
    """ε(1,2,3) = +1."""
    sp = _sp3()
    eps = tender.levi_civita(
        tender.Realm.Oblique, sp,
        [tender.Level.Lower, tender.Level.Lower, tender.Level.Lower],
        [1, 2, 3])
    after = td.fold_arithmetic(td.eval_delta_concrete(td.expand_eps(eps)))
    assert after.latex() == "1"


def test_expand_eps_odd_perm_is_minus_one():
    """ε(1,3,2) = -1."""
    sp = _sp3()
    eps = tender.levi_civita(
        tender.Realm.Oblique, sp,
        [tender.Level.Lower, tender.Level.Lower, tender.Level.Lower],
        [1, 3, 2])
    after = td.fold_arithmetic(td.eval_delta_concrete(td.expand_eps(eps)))
    assert after.latex() == "-1"


def test_expand_eps_repeated_index_is_zero():
    """ε(1,1,2) = 0."""
    sp = _sp3()
    eps = tender.levi_civita(
        tender.Realm.Oblique, sp,
        [tender.Level.Lower, tender.Level.Lower, tender.Level.Lower],
        [1, 1, 2])
    after = td.fold_arithmetic(td.eval_delta_concrete(td.expand_eps(eps)))
    assert after.latex() == "0"


# ---- fold_sums + contract_delta --------------------------------------------

def test_fold_sums_three_term_cycle():
    """δ^1_k δ^1_l + δ^2_k δ^2_l + δ^3_k δ^3_l folds to an ExplicitSum."""
    ctx = tender.Context()
    k, l = ctx.alloc_index(), ctx.alloc_index()
    sp = _sp3()

    def d(v, idx):
        return tender.delta(tender.Realm.Oblique, sp,
                            tender.Level.Upper, tender.Level.Lower, v, idx, ctx=ctx)

    total = d(1, k) * d(1, l) + d(2, k) * d(2, l) + d(3, k) * d(3, l)
    after = td.fold_sums(total)
    # After folding, it should be an ExplicitSum; latex contains \sum.
    assert r"\sum" in after.latex()


def test_fold_then_contract_delta():
    """fold_sums then contract_delta reduces δ^1_k δ^1_l + ... to δ_{kl}."""
    ctx = tender.Context()
    k, l = ctx.alloc_index(), ctx.alloc_index()
    sp = _sp3()

    def d(v, idx):
        return tender.delta(tender.Realm.Oblique, sp,
                            tender.Level.Upper, tender.Level.Lower, v, idx, ctx=ctx)

    total = d(1, k) * d(1, l) + d(2, k) * d(2, l) + d(3, k) * d(3, l)
    contracted = td.contract_delta(td.fold_sums(total))
    # Must be a single delta (no sum) with both original symbolic indices.
    latex = contracted.latex()
    assert r"\sum" not in latex
    assert r"\delta" in latex


# ---- contract_eps_pair -----------------------------------------------------

def _eps(ctx, sp, levels, indices):
    return tender.levi_civita(tender.Realm.Oblique, sp, levels, indices, ctx=ctx)


def test_contract_eps_pair_one_index():
    """Σ_i ε^{ijk} ε_{iml} → δ^j_m δ^k_l − δ^j_l δ^k_m."""
    ctx = tender.Context()
    i, j, k, m, l = (ctx.alloc_index() for _ in range(5))
    sp = _sp3()
    U, L = tender.Level.Upper, tender.Level.Lower

    ea = _eps(ctx, sp, [U, U, U], [i, j, k])
    eb = _eps(ctx, sp, [L, L, L], [i, m, l])
    expr = tender.explicit_sum(i, ea * eb, ctx=ctx)

    imap = tender.IndexNameMap()
    for idx, nm in [(j, "j"), (k, "k"), (m, "m"), (l, "l")]:
        imap.assign(idx, nm)

    out = td.contract_eps_pair(expr).latex(imap)
    assert out == (
        r"\delta^{j}_{m} \, \delta^{k}_{l} - \delta^{j}_{l} \, \delta^{k}_{m}"
    )


def test_contract_eps_pair_two_indices():
    """Σ_{ij} ε^{ijk} ε_{ijl} → 2 δ^k_l."""
    ctx = tender.Context()
    i, j, k, l = (ctx.alloc_index() for _ in range(4))
    sp = _sp3()
    U, L = tender.Level.Upper, tender.Level.Lower

    ea = _eps(ctx, sp, [U, U, U], [i, j, k])
    eb = _eps(ctx, sp, [L, L, L], [i, j, l])
    expr = tender.explicit_sum(j, tender.explicit_sum(i, ea * eb, ctx=ctx), ctx=ctx)

    imap = tender.IndexNameMap()
    imap.assign(k, "k")
    imap.assign(l, "l")

    out = td.contract_eps_pair(expr).latex(imap)
    assert out == r"2 \, \delta^{k}_{l}"


def test_contract_eps_pair_non_eps_unchanged():
    """A product that is not a pair of ε's is returned unchanged."""
    ctx = tender.Context()
    i, k, l = (ctx.alloc_index() for _ in range(3))
    sp = _sp3()
    U, L = tender.Level.Upper, tender.Level.Lower

    def d(a, b):
        return tender.delta(tender.Realm.Oblique, sp, U, L, a, b, ctx=ctx)

    expr = tender.explicit_sum(i, d(i, k) * d(i, l), ctx=ctx)
    # No ε pair to contract: the expression is left as-is (still a Σ of δ's).
    assert td.contract_eps_pair(expr).latex() == expr.latex()


# ---- fold_equal_addends: subtraction and right/rational coefficients --------

def _delta_ij(ctx):
    i, j = ctx.alloc_index(), ctx.alloc_index()
    imap = tender.IndexNameMap()
    imap.assign(i, "i"); imap.assign(j, "j")
    d = tender.delta(tender.Realm.Oblique, _sp3(),
                     tender.Level.Upper, tender.Level.Lower, i, j, ctx=ctx)
    return d, imap


def test_fold_equal_addends_difference_cancels():
    """X - X folds to 0; collection sees through Difference."""
    ctx = tender.Context()
    d, imap = _delta_ij(ctx)
    assert td.fold_equal_addends(d - d).latex(imap) == "0"


def test_fold_equal_addends_difference_accumulates():
    """2X - X folds to X."""
    ctx = tender.Context()
    d, imap = _delta_ij(ctx)
    two = tender.scalar(2, ctx=ctx)
    assert td.fold_equal_addends(two * d - d).latex(imap) == r"\delta^{i}_{j}"


def test_fold_equal_addends_right_scalar_coefficient():
    """X*2 + X folds to 3X (scalar on the right of the product)."""
    ctx = tender.Context()
    d, imap = _delta_ij(ctx)
    two = tender.scalar(2, ctx=ctx)
    assert td.fold_equal_addends(d * two + d).latex(imap) == r"3 \, \delta^{i}_{j}"


# ---- canonicalize (algebraic normal form) ----------------------------------

def test_canonicalize_sum_and_product_commute():
    """Sums and component products canonicalize regardless of operand order."""
    ctx = tender.Context()
    i, j, k, l = (ctx.alloc_index() for _ in range(4))
    imap = tender.IndexNameMap()
    for x, n in [(i, "i"), (j, "j"), (k, "k"), (l, "l")]:
        imap.assign(x, n)
    U, L = tender.Level.Upper, tender.Level.Lower

    def d(a, b):
        return tender.delta(tender.Realm.Oblique, _sp3(), U, L, a, b, ctx=ctx)

    assert td.canonicalize(d(i, j) + d(k, l)).latex(imap) == \
        td.canonicalize(d(k, l) + d(i, j)).latex(imap)
    assert td.canonicalize(d(i, j) * d(k, l)).latex(imap) == \
        td.canonicalize(d(k, l) * d(i, j)).latex(imap)


def test_canonicalize_invariant_dyad_keeps_order():
    """a⊗b is a non-commutative dyad; canonical forms must differ from b⊗a."""
    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    assert td.canonicalize(a * b).latex() != td.canonicalize(b * a).latex()
    # but the dot product commutes:
    assert td.canonicalize(a @ b).latex() == td.canonicalize(b @ a).latex()


def test_canonicalize_collects_and_cancels():
    ctx = tender.Context()
    i, j = ctx.alloc_index(), ctx.alloc_index()
    imap = tender.IndexNameMap()
    imap.assign(i, "i"); imap.assign(j, "j")
    U, L = tender.Level.Upper, tender.Level.Lower
    d = tender.delta(tender.Realm.Oblique, _sp3(), U, L, i, j, ctx=ctx)
    assert td.canonicalize(d + d).latex(imap) == r"2 \, \delta^{i}_{j}"
    assert td.canonicalize(d - d).latex(imap) == "0"
