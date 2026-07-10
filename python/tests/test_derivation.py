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


def test_fold_equal_addends_self_prepares_across_dummy_renaming():
    """x1 - x2 cancels to 0 when x1, x2 are equal only up to dummy renaming.

    This is the I×a playthrough: ``I×a`` and ``a×I`` expand to the same tensor
    written with differently-named summed indices and permuted ε.  The bare
    structural fold cannot merge them; the self-preparing fold canonicalizes
    first and reduces the difference to 0 (vibe 000065).
    """
    import tender.basis as tb

    ctx = tender.Context()
    basis = tb.wcs(ctx)
    co = tb.Variance.Covariant
    I = tender.identity(ctx)
    a = tender.tensor("a", 1, ctx)

    def transform(x):
        x = tb.expand_in_basis(x, basis, co)
        x = tb.simplify_basis_cross(x, basis)
        return x

    x1 = transform(I % a)
    x2 = transform(a % I)
    assert td.algebraic_eq(x1, x2)

    # Structural fold leaves the difference standing (different dummy names).
    dx_structural = td.fold_equal_addends_structural(x1 - x2)
    assert dx_structural.latex() != "0"

    # Self-preparing fold cancels it outright.
    assert td.fold_equal_addends(x1 - x2).latex() == "0"


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


def test_canonicalize_alpha_equivalent_dummies():
    """Σ_i δ^i_a δ^i_b and Σ_p δ^p_a δ^p_b canonicalize identically."""
    ctx = tender.Context()
    a, b, i, p = (ctx.alloc_index() for _ in range(4))
    U, L = tender.Level.Upper, tender.Level.Lower

    def d(x, y):
        return tender.delta(tender.Realm.Oblique, _sp3(), U, L, x, y, ctx=ctx)

    e1 = tender.explicit_sum(i, d(i, a) * d(i, b), ctx=ctx)
    e2 = tender.explicit_sum(p, d(p, a) * d(p, b), ctx=ctx)
    # Fresh maps assign names deterministically by id, so equal canonical forms
    # render identically.
    assert td.canonicalize(e1).latex(tender.IndexNameMap()) == \
        td.canonicalize(e2).latex(tender.IndexNameMap())


# ---- identities (apply_identity) -------------------------------------------

def _delta_contraction(ctx, sp):
    """The identity  Σ_p δ^p_A δ^p_B = δ_{AB}  as a tender.derivation.Identity."""
    U, L = tender.Level.Upper, tender.Level.Lower
    p, a, b = (ctx.alloc_index() for _ in range(3))
    lhs = tender.explicit_sum(
        p,
        tender.delta(tender.Realm.Oblique, sp, U, L, p, a, ctx=ctx)
        * tender.delta(tender.Realm.Oblique, sp, U, L, p, b, ctx=ctx),
        ctx=ctx,
    )
    rhs = tender.delta(tender.Realm.Oblique, sp, L, L, a, b, ctx=ctx)
    return td.Identity("delta-contraction", lhs, rhs)


def test_apply_identity_delta_contraction():
    ctx = tender.Context()
    sp = _sp3()
    U, L = tender.Level.Upper, tender.Level.Lower
    ident = _delta_contraction(ctx, sp)

    q, m, n = (ctx.alloc_index() for _ in range(3))
    target = tender.explicit_sum(
        q,
        tender.delta(tender.Realm.Oblique, sp, U, L, q, m, ctx=ctx)
        * tender.delta(tender.Realm.Oblique, sp, U, L, q, n, ctx=ctx),
        ctx=ctx,
    )
    expected = tender.delta(tender.Realm.Oblique, sp, L, L, m, n, ctx=ctx)

    result = ident(target)
    assert td.algebraic_eq(result, expected)


def test_apply_identity_no_match_returns_canonical():
    ctx = tender.Context()
    sp = _sp3()
    U, L = tender.Level.Upper, tender.Level.Lower
    ident = _delta_contraction(ctx, sp)

    m, n = (ctx.alloc_index() for _ in range(2))
    target = tender.delta(tender.Realm.Oblique, sp, U, L, m, n, ctx=ctx)
    result = ident(target)
    assert td.algebraic_eq(result, target)


def test_apply_identity_as_derivation_step():
    ctx = tender.Context()
    sp = _sp3()
    U, L = tender.Level.Upper, tender.Level.Lower
    ident = _delta_contraction(ctx, sp)

    q, m, n = (ctx.alloc_index() for _ in range(3))
    target = tender.explicit_sum(
        q,
        tender.delta(tender.Realm.Oblique, sp, U, L, q, m, ctx=ctx)
        * tender.delta(tender.Realm.Oblique, sp, U, L, q, n, ctx=ctx),
        ctx=ctx,
    )
    expected = tender.delta(tender.Realm.Oblique, sp, L, L, m, n, ctx=ctx)

    drv = td.Derivation(target)
    drv.step(td.apply_identity(ident))
    assert td.algebraic_eq(drv.current, expected)
    assert len(drv.history) == 2


def test_apply_identity_eps_delta_two_index():
    """Σ_i Σ_j ε^{ijk} ε_{ijl} = 2 δ^k_l, applied as a generic identity."""
    ctx = tender.Context()
    sp = _sp3()
    U, L = tender.Level.Upper, tender.Level.Lower

    i, j, k, l = (ctx.alloc_index() for _ in range(4))
    lhs = tender.explicit_sum(
        i,
        tender.explicit_sum(
            j, _eps(ctx, sp, [U, U, U], [i, j, k]) * _eps(ctx, sp, [L, L, L], [i, j, l]),
            ctx=ctx),
        ctx=ctx,
    )
    rhs = tender.scalar(2) * tender.delta(tender.Realm.Oblique, sp, U, L, k, l, ctx=ctx)
    ident = td.Identity("eps-delta-2", lhs, rhs)

    a, b, c, d = (ctx.alloc_index() for _ in range(4))
    target = tender.explicit_sum(
        a,
        tender.explicit_sum(
            b, _eps(ctx, sp, [U, U, U], [a, b, c]) * _eps(ctx, sp, [L, L, L], [a, b, d]),
            ctx=ctx),
        ctx=ctx,
    )
    expected = tender.scalar(2) * tender.delta(tender.Realm.Oblique, sp, U, L, c, d, ctx=ctx)

    result = ident(target)
    assert td.algebraic_eq(result, expected)


def test_structural_vs_algebraic_eq():
    ctx = tender.Context()
    sp = _sp3()
    U, L = tender.Level.Upper, tender.Level.Lower
    i, j = (ctx.alloc_index() for _ in range(2))
    d = tender.delta(tender.Realm.Oblique, sp, U, L, i, j, ctx=ctx)

    # d + d and 2d are algebraically equal but not structurally equal.
    assert not td.structural_eq(d + d, tender.scalar(2) * d)
    assert td.algebraic_eq(d + d, tender.scalar(2) * d)


# ---- implicit Einstein summation (vibe 000028) -----------------------------

def test_implicit_summation_equals_explicit():
    """An implicitly-contracted index canonicalizes like an explicit sum."""
    import pytest

    ctx = tender.Context()
    sp = _sp3()
    O, N = tender.Realm.Oblique, tender.Realm.Orthonormal
    U, L = tender.Level.Upper, tender.Level.Lower
    r, m, n = (ctx.alloc_index() for _ in range(3))

    # Orthonormal: a doubled index contracts whether or not a sum is written.
    implicit = (
        tender.delta(N, sp, U, L, r, m, ctx=ctx)
        * tender.delta(N, sp, U, L, r, n, ctx=ctx)
    )
    explicit = tender.explicit_sum(r, implicit, ctx=ctx)
    assert td.algebraic_eq(implicit, explicit)

    # The contraction identity fires on the implicit (sum-less) form.
    p, a, b = (ctx.alloc_index() for _ in range(3))
    ident = td.Identity(
        "delta-contraction",
        tender.explicit_sum(
            p,
            tender.delta(N, sp, U, L, p, a, ctx=ctx)
            * tender.delta(N, sp, U, L, p, b, ctx=ctx),
            ctx=ctx,
        ),
        tender.delta(N, sp, L, L, a, b, ctx=ctx),
    )
    assert td.algebraic_eq(ident(implicit), tender.delta(N, sp, L, L, m, n, ctx=ctx))

    # Oblique trace δ^i_i contracts to a sum.
    i = ctx.alloc_index()
    trace = tender.delta(O, sp, U, L, i, i, ctx=ctx)
    assert td.algebraic_eq(trace, tender.explicit_sum(i, trace, ctx=ctx))

    # An ill-formed Oblique same-level pair throws — unless overridden.
    bad = (
        tender.delta(O, sp, U, L, r, m, ctx=ctx)
        * tender.delta(O, sp, U, L, r, n, ctx=ctx)
    )
    with pytest.raises(ValueError):
        td.canonicalize(bad)
    td.canonicalize(tender.explicit_sum(r, bad, ctx=ctx))  # override: no throw


# ---- saturate (e-graph) ----------------------------------------------------

def test_saturate_contracts_delta():
    ctx = tender.Context()
    sp = _sp3()
    U, L = tender.Level.Upper, tender.Level.Lower
    rule = _delta_contraction(ctx, sp)

    q, m, n = (ctx.alloc_index() for _ in range(3))
    target = tender.explicit_sum(
        q,
        tender.delta(tender.Realm.Oblique, sp, U, L, q, m, ctx=ctx)
        * tender.delta(tender.Realm.Oblique, sp, U, L, q, n, ctx=ctx),
        ctx=ctx,
    )
    result = td.saturate(target, [rule])
    expected = tender.delta(tender.Realm.Oblique, sp, L, L, m, n, ctx=ctx)
    assert td.algebraic_eq(result, expected)


def test_saturate_rewrites_nested_subexpression():
    # δ_{rs} + Σ_q δ^q_m δ^q_n  →  δ_{rs} + δ_{mn}, no manual step ordering.
    ctx = tender.Context()
    sp = _sp3()
    U, L = tender.Level.Upper, tender.Level.Lower
    rule = _delta_contraction(ctx, sp)

    q, m, n, r, s = (ctx.alloc_index() for _ in range(5))
    contraction = tender.explicit_sum(
        q,
        tender.delta(tender.Realm.Oblique, sp, U, L, q, m, ctx=ctx)
        * tender.delta(tender.Realm.Oblique, sp, U, L, q, n, ctx=ctx),
        ctx=ctx,
    )
    drs = tender.delta(tender.Realm.Oblique, sp, L, L, r, s, ctx=ctx)
    result = td.saturate(drs + contraction, [rule])

    expected = drs + tender.delta(tender.Realm.Oblique, sp, L, L, m, n, ctx=ctx)
    assert td.algebraic_eq(result, expected)


def test_saturate_no_match_returns_canonical():
    ctx = tender.Context()
    sp = _sp3()
    L = tender.Level.Lower
    rule = _delta_contraction(ctx, sp)

    m, n = (ctx.alloc_index() for _ in range(2))
    target = tender.delta(tender.Realm.Oblique, sp, L, L, m, n, ctx=ctx)
    result = td.saturate(target, [rule])
    assert td.algebraic_eq(result, target)


# ---- contract_identity -----------------------------------------------------


def test_contract_identity_left():
    a = tender.tensor("a", rank=1)
    assert td.structural_eq(td.contract_identity(tender.identity() @ a), a)


def test_contract_identity_right():
    a = tender.tensor("a", rank=1)
    assert td.structural_eq(td.contract_identity(a @ tender.identity()), a)


def test_contract_identity_no_op():
    a = tender.tensor("a", rank=1)
    b = tender.tensor("b", rank=1)
    assert td.structural_eq(td.contract_identity(a @ b), a @ b)


# ---- distribute_contraction ------------------------------------------------


def test_distribute_contraction_cross_over_dyad():
    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    u = tender.tensor("u", rank=1, ctx=ctx)
    v = tender.tensor("v", rank=1, ctx=ctx)
    # a × (u ⊗ v) → (a × u) ⊗ v
    res = td.distribute_contraction(a % (u * v))
    assert td.structural_eq(res, (a % u) * v)


def test_distribute_contraction_noop():
    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    assert td.structural_eq(td.distribute_contraction(a @ b), a @ b)


# ---- expand_double_dot -----------------------------------------------------


def test_expand_double_dot_vertical():
    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    c = tender.tensor("c", rank=1, ctx=ctx)
    d = tender.tensor("d", rank=1, ctx=ctx)
    # (a⊗b):(c⊗d) → (a·c)(b·d)
    res = td.expand_double_dot((a * b).ddot(c * d))
    assert td.algebraic_eq(res, (a @ c) * (b @ d))


def test_expand_double_dot_alternate():
    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    c = tender.tensor("c", rank=1, ctx=ctx)
    d = tender.tensor("d", rank=1, ctx=ctx)
    # (a⊗b)··(c⊗d) → (a·d)(b·c); // is the ddot_alt operator
    res = td.expand_double_dot((a * b) // (c * d))
    assert td.algebraic_eq(res, (a @ d) * (b @ c))


# ---- tr / vec / transpose --------------------------------------------------


def test_expand_dyad_ops():
    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    assert td.algebraic_eq(td.expand_dyad_ops(tender.tr(a * b)), a @ b)
    assert td.algebraic_eq(td.expand_dyad_ops(tender.vec(a * b)), a % b)
    assert td.algebraic_eq(td.expand_dyad_ops(tender.transpose(a * b)), b * a)


def test_transpose_identity_is_self():
    ctx = tender.Context()
    I = tender.identity(ctx=ctx)
    assert td.structural_eq(td.expand_dyad_ops(tender.transpose(I)), I)


def test_trace_of_dimensioned_identity():
    # vibe 000080 Increment 1 (literal-only): a dimensioned identity folds its
    # trace to the space dimension; the bare identity stays symbolic tr(I).
    ctx = tender.Context()
    I3 = tender.identity(ctx=ctx, space=tender.space_3d)
    assert I3.rank == 2
    assert td.algebraic_eq(
        td.expand_dyad_ops(tender.tr(I3)), tender.scalar(3, ctx=ctx)
    )
    I = tender.identity(ctx=ctx)
    assert td.expand_dyad_ops(tender.tr(I)).latex() == r"\operatorname{tr}(\mathbf{I})"


def test_trace_of_scaled_dimensioned_identity():
    # vibe 000080 Increment 2: tr(c·I) = c·n peels the scalar off the identity
    # leg (the Δθ·I / (∇∇··ε)I trace terms).  A bare identity stays symbolic.
    ctx = tender.Context()
    I3 = tender.identity(ctx=ctx, space=tender.space_3d)
    c = tender.tensor("c", rank=0, ctx=ctx)
    assert td.algebraic_eq(
        td.expand_dyad_ops(tender.tr(c * I3)), tender.scalar(3, ctx=ctx) * c
    )
    Ibare = tender.identity(ctx=ctx)
    assert "operatorname{tr}" in td.expand_dyad_ops(tender.tr(c * Ibare)).latex()


def test_unary_op_ranks():
    ctx = tender.Context()
    A = tender.tensor("A", rank=2, ctx=ctx)
    assert tender.tr(A).rank == 0
    assert tender.vec(A).rank == 1
    assert tender.transpose(A).rank == 2


# ---- eval_eps_concrete -----------------------------------------------------


def test_eval_eps_concrete():
    ctx = tender.Context()
    sp = tender.space_3d
    L = [tender.Level.Lower] * 3

    def eps(a, b, c):
        return tender.levi_civita(tender.Realm.Orthonormal, sp, L, [a, b, c], ctx=ctx)

    one = tender.scalar(1, ctx=ctx)
    assert td.algebraic_eq(td.eval_eps_concrete(eps(1, 2, 3)), one)
    assert td.algebraic_eq(td.eval_eps_concrete(eps(2, 1, 3)), -one)
    assert td.algebraic_eq(
        td.eval_eps_concrete(eps(1, 1, 2)), tender.scalar(0, ctx=ctx)
    )


# ---- subtree pattern variables (vibe 000051) -------------------------------


def test_subtree_variable_identity():
    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    c = tender.tensor("c", rank=1, ctx=ctx)
    d = tender.tensor("d", rank=1, ctx=ctx)
    ddot = td.Identity("ddot", (a * b).ddot(c * d), (a @ c) * (b @ d))
    # Fires on a *different* dyad pair via the subtree variables a,b,c,d.
    x = tender.tensor("x", rank=1, ctx=ctx)
    y = tender.tensor("y", rank=1, ctx=ctx)
    u = tender.tensor("u", rank=1, ctx=ctx)
    w = tender.tensor("w", rank=1, ctx=ctx)
    res = td.apply_identity(ddot)((x * y).ddot(u * w))
    assert td.algebraic_eq(res, (x @ u) * (y @ w))


# ---- implicitize / simplify (vibe 000064 #4) -------------------------------


def test_implicitize_strips_einstein_sum():
    """canonicalize materializes Σ; implicitize strips it back to implicit."""
    import tender.basis as tb

    ctx = tender.Context()
    frame = tb.wcs(ctx)
    a = tender.tensor("a", rank=1, ctx=ctx)
    expanded = tb.expand_in_basis(a, frame, tb.Variance.Covariant)

    canon = td.canonicalize(expanded)
    assert "\\sum" in canon.latex()  # binder materialized
    implicit = td.implicitize(canon)
    assert "\\sum" not in implicit.latex()  # and stripped back


def test_simplify_is_canonicalize_then_implicitize():
    import tender.basis as tb

    ctx = tender.Context()
    frame = tb.wcs(ctx)
    a = tender.tensor("a", rank=1, ctx=ctx)
    expanded = tb.expand_in_basis(a, frame, tb.Variance.Covariant)

    assert td.structural_eq(
        td.simplify(expanded), td.implicitize(td.canonicalize(expanded))
    )


# ---- algebraic_eq fraction fallback (vibe 000074) --------------------------

def test_algebraic_eq_folds_fraction_shapes():
    # Canonical forms (theory T0) keep x/r + y/r and (x+y)/r apart; the
    # fallback checks that the difference simplifies to the literal 0, so the
    # two shapes compare equal — no manual simplify_scalars(a - b) needed.
    ctx = tender.Context()
    x = tender.field("x", 0, ctx=ctx)
    y = tender.field("y", 0, ctx=ctx)
    r = tender.coordinate("r", chart_id=1, slot=0, nonneg=True, ctx=ctx)
    assert not td.structural_eq(
        td.canonicalize(x / r + y / r), td.canonicalize((x + y) / r)
    )
    assert td.algebraic_eq(x / r + y / r, (x + y) / r)
    assert not td.algebraic_eq(x / r, y / r)


# ---- fan-in contraction leg topology (vibe 000078 bug 3b) ------------------

def test_right_nested_fan_in_stays_scalar():
    # a·(b·T): b is consumed contracting into T, so a fans onto T's other leg —
    # the term is a scalar a_j b_i T_ij, NOT the rank-2 (a·b)·T the flat chain
    # once mis-produced.  It must equal the fan-in-free b·(T·a), and differ from
    # the transposed b·(Tᵀ·a) (so T's orientation is genuinely tracked).
    a = tender.tensor("a", rank=1)
    b = tender.tensor("b", rank=1)
    T = tender.tensor("T", rank=2)
    canon = td.canonicalize(a @ (b @ T))
    assert canon.rank == 0
    assert td.structural_eq(canon, td.canonicalize(b @ (T @ a)))
    assert not td.structural_eq(canon, td.canonicalize(b @ (T.transpose() @ a)))


def test_rank2_fan_in_inserts_transpose():
    # T·(a·S), rank-2 T,S: a·S is a vector on S's free leg, so T fans onto S's
    # second leg — faithfully T·Sᵀ·a.  Cross-checked against T·(Sᵀ·a).
    a = tender.tensor("a", rank=1)
    T = tender.tensor("T", rank=2)
    S = tender.tensor("S", rank=2)
    canon = td.canonicalize(T @ (a @ S))
    assert canon.rank == 1
    assert td.structural_eq(canon, td.canonicalize(T @ (S.transpose() @ a)))


# ---- symmetric transpose folds (vibe 000078) -------------------------------

def test_symmetric_transpose_folds():
    # A symmetric rank-2 tensor equals its transpose: εᵀ = ε.  A general rank-2
    # keeps an explicit transpose.  Needed so (∂∂ε)ᵀ folds in the strain
    # reduction.
    ws = tender.Workspace()
    E = ws.field("E", 2, symmetric=True)
    S = ws.tensor("S", 2)
    assert td.canonicalize(E.transpose()).latex() == E.latex()
    assert td.algebraic_eq(E.transpose(), E)
    assert "mathsf{T}" in td.canonicalize(S.transpose()).latex()  # Sᵀ stays
    assert not td.algebraic_eq(S.transpose(), S)


# ---- sym/skew constructors (vibe 000080 Increment 7A) ----------------------

def test_sym_skew_constructors():
    # sym(A)=(A+Aᵀ)/2, skew(A)=(A−Aᵀ)/2 — thin builders for the (anti)symmetric
    # part of a rank-2 tensor.
    ws = tender.Workspace()
    A = ws.field("A", 2)
    assert td.structural_eq(td.sym(A), (A + A.transpose()) / 2)
    assert td.structural_eq(td.skew(A), (A - A.transpose()) / 2)


def test_sym_of_symmetric_field_is_the_field():
    # For a symmetric-by-declaration field, sym(E) = (E+Eᵀ)/2 = (E+E)/2 = E:
    # recognised via the symmetric-transpose fold (algebraic_eq), and the
    # ½·2 scalar folds under simplify_scalars.
    ws = tender.Workspace()
    E = ws.field("E", 2, symmetric=True)
    assert td.algebraic_eq(td.sym(E), E)
    assert td.simplify_scalars(td.sym(E)).latex() == E.latex()


def test_sym_part_recognised_symmetric_by_construction():
    # vibe 000080 Increment 7(b1): a symmetric part is recognised symmetric with
    # no declared trait — transpose distributes through the /2 fence and the
    # (Aᵀ)ᵀ→A involution, so ((A+Aᵀ)/2)ᵀ normalises back to (A+Aᵀ)/2.
    ws = tender.Workspace()
    A = ws.field("A", 2)
    assert td.algebraic_eq(td.sym(A), td.sym(A).transpose())
    # sym + skew = A once the shared coefficient is distributed (expand_products;
    # canonicalize keeps the factored 1/2·(…) form).
    assert td.algebraic_eq(td.expand_products(td.sym(A) + td.skew(A)), A)


def test_scalar_div_distributes_over_sum():
    # (A ± B)/c → A/c ± B/c under expand_products (vibe 000080 Increment 7 b1).
    ws = tender.Workspace()
    A = ws.field("A", 2)
    B = ws.field("B", 2)
    assert td.structural_eq(td.expand_products((A + B) / 2), A / 2 + B / 2)
    assert td.structural_eq(td.expand_products((A - B) / 2), A / 2 - B / 2)
    # transpose commutes through the divisor and distributes over the sum:
    # ((A+B)/c)ᵀ = (Aᵀ+Bᵀ)/c under expand_dyad_ops (the /c stays a ScalarDiv —
    # splitting it is expand_products' job).
    assert td.structural_eq(
        td.expand_dyad_ops(((A + B) / 2).transpose()),
        (A.transpose() + B.transpose()) / 2,
    )


# ---- factor_common: reverse of distribution (vibe 000080) ------------------

def test_factor_common_scalar_factor():
    # λ (∇·u) + μ (∇·u) → (λ + μ) (∇·u): a common rank-0 factor (∇·u, itself a
    # scalar) is pulled out — the case collect_terms misses (it folds the whole
    # scalar product into a coefficient).
    ws = tender.Workspace()
    u = ws.field("u", 1)
    nab = tender.nabla(ctx=ws.ctx)
    lam = tender.tensor(r"\lambda", 0, ctx=ws.ctx)
    mu = tender.tensor(r"\mu", 0, ctx=ws.ctx)
    s = lam * (nab @ u) + mu * (nab @ u)
    fc = td.factor_common(s)
    assert fc.latex() == r"(\lambda + \mu) \, \nabla \cdot \mathbf{u}"
    # correctness: distributing it back recovers the original sum.
    assert td.algebraic_eq(td.expand_products(fc), s)


def test_factor_common_nested_in_gradient_and_noop():
    ws = tender.Workspace()
    u = ws.field("u", 1)
    nab = tender.nabla(ctx=ws.ctx)
    lam = tender.tensor(r"\lambda", 0, ctx=ws.ctx)
    mu = tender.tensor(r"\mu", 0, ctx=ws.ctx)
    # reaches a sum nested inside a gradient: ∇(λ∇·u + μ∇·u) → ∇((λ+μ)∇·u).
    g = nab * (lam * (nab @ u) + mu * (nab @ u))
    fc = td.factor_common(g)
    assert fc.latex() == r"\nabla \, (\lambda + \mu) \, \nabla \cdot \mathbf{u}"
    # correctness: both distribute to the same fully-expanded form (a robust
    # check that avoids canonicalising the bare ∇, which is unstable).
    assert td.structural_eq(td.expand_products(fc), td.expand_products(g))
    # no common factor → unchanged.
    s2 = lam * (nab @ u) + mu * u
    assert td.structural_eq(td.factor_common(s2), s2)
