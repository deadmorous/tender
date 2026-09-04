"""Tests for tender.derivation — derivation steps and history."""

import pytest

import tender
import tender.basis as tb
import tender.derivation as td
import tender.steps as ts


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
    drv.step(td.fold_arithmetic, optional=True)
    assert len(drv.history) == 2
    drv.step(td.fold_arithmetic, optional=True)
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
    result = drv.step(td.fold_arithmetic, optional=True).step(
        td.fold_arithmetic, optional=True
    )
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


def test_apply_identity_no_match_returns_input_unchanged():
    # The step no-op contract (vibe 000095): a failed match must not reshape
    # the input, not even by canonicalizing it.
    ctx = tender.Context()
    sp = _sp3()
    U, L = tender.Level.Upper, tender.Level.Lower
    ident = _delta_contraction(ctx, sp)

    m, n = (ctx.alloc_index() for _ in range(2))
    target = tender.delta(tender.Realm.Oblique, sp, U, L, m, n, ctx=ctx)
    result = ident(target)
    assert td.structural_eq(result, target)


def test_derivation_step_warns_on_noop():
    import warnings

    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    drv = td.Derivation(a * b)  # a plain dyad: nothing to expand

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        drv.step(td.expand_products)
    assert len(caught) == 1
    assert issubclass(caught[0].category, td.NoOpStep)
    assert "expand_products" in str(caught[0].message)


def test_derivation_step_optional_silences_noop():
    import warnings

    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    drv = td.Derivation(a * b)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        drv.step(td.expand_products, optional=True)
    assert caught == []


def test_derivation_records_step_names_and_fired_flags():
    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    c = tender.tensor("c", rank=1, ctx=ctx)
    drv = td.Derivation((a + b) * c)
    drv.step(td.expand_products)  # fires: distributes over the sum
    drv.step(td.expand_products, optional=True)  # second pass: no-op
    drv.step(lambda e: td.simplify(e), label="simplify", optional=True)

    names = [name for name, _ in drv.steps]
    fired = [f for _, f in drv.steps]
    assert names == ["expand_products", "expand_products", "simplify"]
    assert fired[0] is True
    assert fired[1] is False


def test_run_applies_a_list_of_steps():
    # A derivation is a list: built once, it runs on another expression too.
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    a, b = (tender.tensor(n, rank=1, ctx=ctx) for n in "ab")
    binder = ts.using(basis=frame)
    steps = [binder.expand_in_basis, binder.reduce_frame, binder.reassemble]

    drv = td.derive(a @ b, steps)
    assert td.structural_eq(drv.current, a @ b)
    assert [name for name, _ in drv.steps] == [
        "expand_in_basis", "reduce_frame", "reassemble",
    ]
    assert len(drv.history) == 4
    # the same list, a different expression
    assert td.structural_eq(td.derive(b @ a, steps).current, a @ b)


def test_a_bound_step_carries_only_what_the_step_declares():
    # One context serves the whole derivation, because each step takes what it
    # asks for and nothing else.
    ctx = tender.Context()
    frame = tb.wcs(ctx)
    a, b = (tender.tensor(n, rank=1, ctx=ctx) for n in "ab")
    binder = ts.using(basis=frame, level=tender.Level.Upper)
    # `level` is along for the ride and expand_in_basis never sees it.
    # (Fresh dummy indices per call, so compare up to α-renaming.)
    assert td.algebraic_eq(
        binder.expand_in_basis(a @ b), tb.expand_in_basis(a @ b, frame)
    )
    assert binder.canonicalize.__name__ == "canonicalize"
    with pytest.raises(AttributeError, match="no step named"):
        binder.no_such_step


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
    drv.step(ident)
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
    # vibe 000080/000082: an identity carries its dimension, so tr(I) folds to n.
    # There is no dimension-agnostic identity — the default is 3-D (000082).
    ctx = tender.Context()
    I3 = tender.identity(ctx=ctx, space=tender.space_3d)
    assert I3.rank == 2
    assert td.algebraic_eq(
        td.expand_dyad_ops(tender.tr(I3)), tender.scalar(3, ctx=ctx)
    )
    I = tender.identity(ctx=ctx)  # default 3-D
    assert td.structural_eq(I, I3)  # bearing, but the default IS 3-D
    assert td.algebraic_eq(
        td.expand_dyad_ops(tender.tr(I)), tender.scalar(3, ctx=ctx)
    )
    # the dimension is not an index slot — it still renders as a clean I.
    assert I3.latex() == r"\mathbf{I}" == I.latex()


def test_trace_of_scaled_dimensioned_identity():
    # vibe 000080 Increment 2 / 000082: tr(c·I) = c·n; the default I is 3-D.
    ctx = tender.Context()
    I3 = tender.identity(ctx=ctx, space=tender.space_3d)
    c = tender.tensor("c", rank=0, ctx=ctx)
    assert td.algebraic_eq(
        td.expand_dyad_ops(tender.tr(c * I3)), tender.scalar(3, ctx=ctx) * c
    )
    assert td.algebraic_eq(
        td.expand_dyad_ops(tender.tr(c * tender.identity(ctx=ctx))),
        tender.scalar(3, ctx=ctx) * c,
    )


def test_trace_commutes_through_laplacian():
    # vibe 000080 Increment 4: tr(∇·(∇⊗ε)) = ∇·(∇⊗ tr ε) = Δ(tr ε); the dyad
    # cases tr(∇⊗v)=∇·v and tr((∇⊗w)ᵀ)=∇·w already fold via split_dyad.
    ctx = tender.Context()
    nab = tender.nabla(ctx=ctx)
    eps = tender.field("e", 2, ctx=ctx, symmetric=True)
    v = tender.field("v", 1, ctx=ctx)
    assert td.algebraic_eq(
        td.expand_dyad_ops(tender.tr(nab @ (nab * eps))),
        nab @ (nab * tender.tr(eps)),
    )
    assert td.algebraic_eq(td.expand_dyad_ops(tender.tr(nab * v)), nab @ v)
    # rank-1 Δv has no trace — stays symbolic.
    assert "operatorname{tr}" in td.expand_dyad_ops(tender.tr(nab @ (nab * v))).latex()


def test_trace_of_scalar_hessian_keeps_operand():
    # vibe 000081, B2: tr(∇⊗∇⊗θ) (scalar Hessian, θ = tr ε) must reduce to the
    # applied Laplacian ∇·(∇θ) = Δθ, not float θ off into θ·(∇·∇) with a bare,
    # un-appliable operator.  Guard: an operator dyad leg goes on the left with
    # the scalar as its operand, kept attached.
    ctx = tender.Context()
    nab = tender.nabla(ctx=ctx)
    eps = tender.field("e", 2, ctx=ctx, symmetric=True)
    theta = tender.tr(eps)
    reduced = td.expand_dyad_ops(tender.tr(nab * (nab * theta)))
    # equals Δθ = ∇·(∇θ), and carries no detached θ·(∇·∇)
    assert td.algebraic_eq(reduced, nab @ (nab * theta))
    assert r"\Delta" in reduced.latex()


def test_trace_of_operator_dyad_is_transpose_independent():
    # vibe 000081, B3: tr(∇⊗v) and its transpose tr((∇⊗v)ᵀ) = tr(v⊗∇) must both
    # reduce to the operator-left form ∇·v (not v·∇), so the two are structurally
    # identical and a structural like-term fold can combine them.
    ctx = tender.Context()
    nab = tender.nabla(ctx=ctx)
    eps = tender.field("e", 2, ctx=ctx, symmetric=True)
    grad_div = nab * (nab @ eps)  # ∇⊗(∇·ε), rank 2
    direct = td.expand_dyad_ops(tender.tr(grad_div))
    transposed = td.expand_dyad_ops(tender.tr(grad_div.transpose()))
    assert td.structural_eq(direct, transposed)


def test_trace_of_strain_incompatibility_reduces():
    # vibe 000081, B2+B3 endpoint: tr(inc ε) of the closed cross-free form
    # reduces to Δ tr(ε) − ∇·(∇·ε) via expand_dyad_ops (tr through operators,
    # dimensioned tr(c·I)=c·n) + a structural like-term fold that keeps every
    # operator attached (no canon float).  Needs a *dimensioned* identity.
    ctx = tender.Context()
    nab = tender.nabla(ctx=ctx)
    eps = tender.field(r"\varepsilon", 2, ctx=ctx, symmetric=True)
    theta = tender.tr(eps)
    ident = tender.identity(ctx, space=tender.space_3d)
    closed = (
        -(nab @ (nab @ eps)) * ident
        + (nab @ (nab * theta)) * ident
        - (nab @ (nab * eps))
        - (nab * (nab * theta))
        + (nab * (nab @ eps))
        + (nab * (nab @ eps)).transpose()
    )
    reduced = td.fold_equal_addends_structural(td.expand_dyad_ops(tender.tr(closed)))
    target = nab @ (nab * theta) - nab @ (nab @ eps)  # Δ tr ε − ∇·(∇·ε)
    # order-independent structural equality: the difference cancels to 0
    assert td.fold_equal_addends_structural(reduced - target).latex() == "0"


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
    res = td.apply_identity((x * y).ddot(u * w), ddot)
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


def test_sym_and_skew_at_rank_zero():
    """A scalar has no slots to swap, so sᵀ = s — and the two differ (vibe 000106).

    `sym` is then the identity and `skew` is zero, which is the asymmetry worth
    knowing: they do not both degenerate the same way.  Before this, both built
    ½(s ± sᵀ) and neither collapsed, which is how the `applicable` probe came to
    list `sym` as an option on a dot product.
    """
    ws = tender.Workspace()
    s = ws.field("s", 0)
    assert td.structural_eq(td.sym(s), s)
    assert td.algebraic_eq(td.skew(s), tender.scalar(0, ctx=ws.ctx))
    # …and the decomposition still holds at rank 0.
    assert td.algebraic_eq(td.sym(s) + td.skew(s), s)


def test_sym_and_skew_refuse_a_rank_they_cannot_mean():
    # Transpose swaps *two* slots: a vector has none to swap, and for rank ≥ 3
    # "which pair?" is exactly the missing information.  Refuse rather than
    # build a formula that reads as if it meant something.
    ws = tender.Workspace()
    v = ws.field("v", 1)
    T = ws.field("T", 3)
    for fn in (td.sym, td.skew):
        with pytest.raises(ValueError, match="rank 2"):
            fn(v)
        with pytest.raises(ValueError, match="which pair"):
            fn(T)


def test_sym_and_skew_still_build_at_rank_two():
    # The case they are for is untouched.
    ws = tender.Workspace()
    A = ws.field("A", 2)
    assert td.structural_eq(td.sym(A), (A + A.transpose()) / 2)
    assert td.structural_eq(td.skew(A), (A - A.transpose()) / 2)


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
    # reaches a sum nested inside a gradient AND hoists the constant coefficient
    # fully out: ∇(λ∇·u + μ∇·u) → (λ+μ)∇(∇·u), valid since ∇(λ+μ)=0.
    g = nab * (lam * (nab @ u) + mu * (nab @ u))
    fc = td.factor_common(g)
    assert fc.latex() == r"(\lambda + \mu) \, \nabla \, \nabla \cdot \mathbf{u}"
    assert td.factor_common(fc).latex() == fc.latex()  # idempotent
    # a *field*-dependent coefficient is NOT hoisted (∇f ≠ 0): the factored sum
    # stays inside the gradient.
    f = ws.field("f", 0)
    h = ws.field("h", 0)
    gf = nab * (f * (nab @ u) + h * (nab @ u))
    assert td.factor_common(gf).latex() == r"\nabla \, (f + h) \, \nabla \cdot \mathbf{u}"
    # no common factor → unchanged.
    s2 = lam * (nab @ u) + mu * u
    assert td.structural_eq(td.factor_common(s2), s2)


# ---- engine verbs (vibe 000096 M2) -----------------------------------------


def _delta(ctx, a, b, levels=None):
    U, L = tender.Level.Upper, tender.Level.Lower
    lv = levels or (U, L)
    return tender.delta(tender.Realm.Oblique, _sp3(), lv[0], lv[1], a, b, ctx=ctx)


def _contraction_setup():
    """Σ_q δ^q_m δ^q_n, its contracted form δ_mn, and the rule between them."""
    ctx = tender.Context()
    L = tender.Level.Lower
    q, m, n = (ctx.alloc_index() for _ in range(3))
    lhs = tender.explicit_sum(q, _delta(ctx, q, m) * _delta(ctx, q, n), ctx=ctx)
    rhs = _delta(ctx, m, n, (L, L))
    p1, p2, p3 = (ctx.alloc_index() for _ in range(3))
    rule = td.Identity(
        "delta_contraction",
        tender.explicit_sum(p1, _delta(ctx, p1, p2) * _delta(ctx, p1, p3), ctx=ctx),
        _delta(ctx, p2, p3, (L, L)),
    )
    return ctx, lhs, rhs, rule


def test_prove_equal_proves_and_attributes_the_rule():
    _, lhs, rhs, rule = _contraction_setup()
    result = td.prove_equal(lhs, rhs, [rule])
    assert result.proved
    assert bool(result) is True
    assert result.status == "proved"
    assert result.fired == {"delta_contraction": 1}


def test_prove_equal_exhausted_is_not_a_disproof():
    ctx, lhs, _, rule = _contraction_setup()
    unrelated = tender.tensor("c", rank=2, ctx=ctx)
    result = td.prove_equal(lhs, unrelated, [rule])
    assert not result.proved
    # Exhausted means "these rules did not suffice", never "they differ".
    assert result.status == "exhausted"


def test_prove_equal_budget_trip_warns_and_is_inconclusive():
    import warnings

    _, lhs, rhs, rule = _contraction_setup()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = td.prove_equal(lhs, rhs, [rule], max_passes=0)
    assert not result.proved
    assert result.status == "budget"  # NOT "exhausted"
    assert any(issubclass(c.category, td.BudgetExceeded) for c in caught)


def test_engine_simplify_contracts_and_reports():
    _, lhs, rhs, rule = _contraction_setup()
    out, report = td.engine_simplify(lhs, [rule])
    assert td.algebraic_eq(out, rhs)
    assert report["complete"] is True
    assert report["fired"] == {"delta_contraction": 1}


def test_engine_simplify_budget_returns_best_so_far_with_warning():
    import warnings

    _, lhs, _, rule = _contraction_setup()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out, report = td.engine_simplify(lhs, [rule], max_passes=0)
    assert report["complete"] is False
    assert td.algebraic_eq(out, lhs)  # unsimplified, but never garbage
    assert any(issubclass(c.category, td.BudgetExceeded) for c in caught)


def test_uncompilable_rule_is_reported_not_silently_inert():
    import warnings

    ctx = tender.Context()
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)
    multi = td.Identity("sum_lhs", a + b, a)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = td.prove_equal(a, b, [multi])
    assert result.skipped == ["sum_lhs"]
    assert any("could not be compiled" in str(c.message) for c in caught)


# ---- rule library groups (vibe 000096 M2 increment 2) ----------------------


def test_rule_groups_are_named_and_populated():
    assert set(td.rule_groups()) == {
        "eps_delta", "cross", "double_dot", "dyadic", "transpose",
        "rotation", "leibniz",
    }
    ctx = tender.Context()
    assert sorted(r.name for r in td.rules("cross", ctx=ctx)) == [
        "bac-cab",
        "cross-identity",
        "cross-removal",
        "cross-self",
        "lagrange",
        "triple-rotate",
    ]
    assert len(td.rules("eps_delta", "cross", "dyadic", ctx=ctx)) == 18


def test_cross_group_proves_bac_cab():
    ctx = tender.Context()
    a, b, c = (tender.tensor(n, rank=1, ctx=ctx) for n in "abc")
    result = td.prove_equal(
        a % (b % c), b * (a @ c) - c * (a @ b), td.rules("cross", ctx=ctx)
    )
    assert result.proved
    assert result.fired.get("bac-cab") == 1


def test_cross_group_proves_the_vibe56_case():
    # a×(b×I) = b⊗a − (a·b)I — the derivation vibe 000056 said no user could
    # discover as a step sequence; here it is one goal-directed call.
    ctx = tender.Context()
    a, b = (tender.tensor(n, rank=1, ctx=ctx) for n in "ab")
    I = tender.identity(ctx=ctx)
    result = td.prove_equal(
        a % (b % I), b * a - (a @ b) * I, td.rules("cross", ctx=ctx)
    )
    assert result.proved


def test_unknown_group_raises():
    import pytest

    with pytest.raises(ValueError):
        td.rules("no_such_group")


# ---- the multi-term-LHS boundary (vibe 000096 increment 3) -----------------


def _bac_cab_case():
    ctx = tender.Context()
    a, b, c = (tender.tensor(n, rank=1, ctx=ctx) for n in "abc")
    u, v, w = (tender.tensor(n, rank=1, ctx=ctx) for n in "uvw")
    forward = td.Identity("bac-cab", u % (v % w), v * (u @ w) - w * (u @ v))
    return ctx, a % (b % c), b * (a @ c) - c * (a @ b), forward


def test_prove_equal_does_not_need_the_factoring_direction():
    """Both sides saturate in one graph, so a forward rule proves either way.

    This is why the engine's inability to compile a multi-term left-hand side
    does not block `prove_equal`: starting from the *expanded* side, the rule
    still fires on the other side and the two meet in the middle.
    """
    _, crossed, expanded, forward = _bac_cab_case()
    assert td.prove_equal(expanded, crossed, [forward]).proved


def test_multi_term_lhs_rule_is_reported_skipped_not_silently_inert():
    _, crossed, expanded, _ = _bac_cab_case()
    ctx = tender.Context()
    u, v, w = (tender.tensor(n, rank=1, ctx=ctx) for n in "uvw")
    reverse = td.Identity(
        "bac-cab-rev", v * (u @ w) - w * (u @ v), u % (v % w)
    )
    result = td.prove_equal(expanded, crossed, [reverse])
    assert result.skipped == ["bac-cab-rev"]


@pytest.mark.xfail(
    strict=True,
    reason="vibe 000096: simplify cannot factor — a multi-term LHS has no Nf "
    "sub-sum matcher, so the compact form is never introduced into the graph",
)
def test_simplify_can_factor_an_expanded_form_back():
    """The one place the missing sub-sum matcher genuinely bites.

    `prove_equal` gets both forms handed to it; `simplify` is given only the
    expanded one, and no compilable rule rewrites *toward* the cross form, so
    the cheaper representative never enters the e-graph to be extracted.
    """
    _, crossed, expanded, forward = _bac_cab_case()
    out, _ = td.engine_simplify(expanded, [forward])
    assert td.algebraic_eq(out, crossed)


# ---- intent-driven extraction cost (vibe 000097) ---------------------------


def _cross_setup():
    ctx = tender.Context()
    a, b, c = (tender.tensor(n, rank=1, ctx=ctx) for n in "abc")
    return ctx, a, b, c, td.rules("cross", ctx=ctx)


def test_intent_changes_the_answer_for_the_same_graph():
    """The point of vibe 000097: "simplest" is the user's goal, not a fact.

    One expression, one rule set — the default keeps the compact crossed form
    because it has fewer nodes, while `prefer="fewest_crosses"` takes the
    *larger* expansion in order to be rid of the ×.
    """
    ctx, a, b, c, rules = _cross_setup()
    crossed = a % (b % c)
    expanded = b * (a @ c) - c * (a @ b)

    default, _ = td.engine_simplify(crossed, rules)
    assert td.algebraic_eq(default, crossed)

    nocross, _ = td.engine_simplify(crossed, rules, prefer="fewest_crosses")
    assert td.algebraic_eq(nocross, expanded)


def test_raw_cost_weights_bypass_the_named_intents():
    ctx, a, b, c, rules = _cross_setup()
    out, _ = td.engine_simplify(a % (b % c), rules, cost={"cross": 1_000_000})
    assert td.algebraic_eq(out, b * (a @ c) - c * (a @ b))


def test_unknown_intent_raises_listing_the_available_ones():
    ctx, a, b, c, rules = _cross_setup()
    with pytest.raises(ValueError, match="fewest_crosses"):
        td.engine_simplify(a % (b % c), rules, prefer="no_such_intent")


def test_named_intents_are_documented_weight_maps():
    assert set(td.PREFER) >= {
        "fewest_eps",
        "smallest",
        "fewest_crosses",
        "fewest_identities",
    }
    # Every intent is a plain dict of weights, so a user can start from one.
    for name, weights in td.PREFER.items():
        assert isinstance(weights, dict), name
        assert all(isinstance(v, int) for v in weights.values()), name


# ---- refutation (vibe 000097) ----------------------------------------------


def test_false_statements_are_refuted():
    """A false claim gets a verdict, not merely "I could not prove it".

    Saturation alone can only exhaust; an independent component expansion
    decides the chart-free algebraic fragment and supplies the negative.
    """
    ctx = tender.Context()
    a, b, c = (tender.tensor(n, rank=1, ctx=ctx) for n in "abc")
    rules = td.rules("cross", ctx=ctx)

    swapped = td.prove_equal(a % b, b % a, rules)  # cross anticommutes
    assert swapped.refuted
    assert swapped.status == "refuted"
    assert not swapped.proved

    wrong = td.prove_equal(  # bac-cab with the terms interchanged
        a % (b % c), c * (a @ b) - b * (a @ c), rules
    )
    assert wrong.refuted


def test_true_statements_are_never_refuted():
    ctx = tender.Context()
    a, b = (tender.tensor(n, rank=1, ctx=ctx) for n in "ab")
    result = td.prove_equal(a % b, -(b % a), [])
    assert result.proved
    assert not result.refuted


def test_true_but_unprovable_points_at_the_rules():
    """Lagrange holds, but with no rules saturation cannot reach it.

    The component check agrees the sides are equal, so the result blames the
    incomplete rule set rather than the claim — a different problem, and the
    useful thing to tell the user.
    """
    ctx = tender.Context()
    a, b, c, d = (tender.tensor(n, rank=1, ctx=ctx) for n in "abcd")
    result = td.prove_equal(
        (a % b) @ (c % d), (a @ c) * (b @ d) - (a @ d) * (b @ c), []
    )
    assert result.status == "exhausted"
    assert not result.refuted
    assert result.components_agree
    assert "rules incomplete" in repr(result)


def test_differential_content_is_undecided_not_refuted():
    """The component procedure decides the algebraic fragment only.

    A ∇ leaves a residue it cannot evaluate, so it stays silent rather than
    returning a wrong verdict.
    """
    import warnings

    ws = tender.Workspace()
    u = ws.field("u", 1)
    nabla = tender.nabla(ctx=ws.ctx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = td.prove_equal(
            nabla @ u, nabla @ u + tender.scalar(1, ctx=ws.ctx), []
        )
    assert result.status == "exhausted"
    assert not result.refuted
    assert not result.components_agree


# ---- budgets in user units (vibe 000097) -----------------------------------


def _bac_cab_rules():
    ctx = tender.Context()
    a, b, c = (tender.tensor(n, rank=1, ctx=ctx) for n in "abc")
    return ctx, a % (b % c), b * (a @ c) - c * (a @ b), td.rules("cross", ctx=ctx)


def test_budget_defaults_are_the_deterministic_pair():
    b = td.Budget()
    assert (b.max_passes, b.max_nodes) == (30, 10_000)
    # No resource cap unless asked for: a default that depended on machine
    # speed would make every result unreproducible.
    assert b.max_seconds is None
    assert b.max_bytes is None


def test_memory_cap_stops_and_is_reported_as_such():
    import warnings

    _, lhs, rhs, rules = _bac_cab_rules()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = td.prove_equal(lhs, rhs, rules, budget=td.Budget(max_bytes=1))
    assert result.status == "budget"
    assert result.stopped_by == "memory"


def test_deterministic_caps_win_over_resource_caps():
    """Which cap is reported matters: a "passes" stop can be reasoned about
    on any machine, a "time" stop cannot."""
    import warnings

    _, lhs, rhs, rules = _bac_cab_rules()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = td.prove_equal(
            lhs,
            rhs,
            rules,
            budget=td.Budget(
                max_passes=0, max_nodes=1, max_seconds=0.001, max_bytes=1
            ),
        )
    assert result.stopped_by == "passes"


def test_default_budget_is_settable_and_per_call_wins():
    import warnings

    _, lhs, rhs, rules = _bac_cab_rules()
    previous = td.set_default_budget(td.Budget(max_passes=0))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The session default applies…
            assert td.prove_equal(lhs, rhs, rules).status == "budget"
            # …and a per-call budget overrides it.
            assert td.prove_equal(
                lhs, rhs, rules, budget=td.Budget()
            ).proved
    finally:
        td.set_default_budget(previous)
    assert td.default_budget().max_passes == 30


def test_reports_carry_resource_usage():
    _, lhs, rhs, rules = _bac_cab_rules()
    result = td.prove_equal(lhs, rhs, rules)
    assert result.proved
    assert result.seconds >= 0.0
    assert result.bytes > 0
    assert result.stopped_by == ""  # not a budget stop


def test_budget_replace_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown budget field"):
        td.Budget().replace(max_wishes=3)


def test_legacy_kwargs_still_work():
    import warnings

    _, lhs, rhs, rules = _bac_cab_rules()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert td.prove_equal(lhs, rhs, rules, max_passes=0).status == "budget"


# ---- unsupported input is reported, not raised (vibe 000098) ---------------


def _unsupported_expr():
    """An expression canon genuinely cannot process.

    Was `T··ε` with a ⊗-product in the operand, until vibe 000101 taught canon
    to distribute that fence.  Now an ill-formed implicit summation: an
    Oblique index repeated at the *same* level contracts nothing, so no
    canonical form exists.
    """
    ws = tender.Workspace()
    i = ws.ctx.alloc_index()
    bad = tender.delta(
        tender.Realm.Oblique,
        tender.space_3d,
        tender.Level.Upper,
        tender.Level.Upper,
        i,
        i,
        ctx=ws.ctx,
    )
    return ws, bad


def test_prove_equal_reports_unsupported_instead_of_raising():
    """A verb must never surface a canonicalization-internal message.

    A shape canon cannot handle is a fact about *tender*, not about the
    claim, so it comes back as a result the caller can inspect.
    """
    ws, bad = _unsupported_expr()
    result = td.prove_equal(bad, tender.scalar(0, ctx=ws.ctx), [])
    assert result.status == "unsupported"
    assert not result.proved and not result.refuted
    assert "canonical form" in result.detail
    assert "unsupported" in repr(result)


def test_engine_simplify_returns_the_input_when_unsupported():
    import warnings

    ws, bad = _unsupported_expr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out, report = td.engine_simplify(bad, [])
    assert report["unsupported"]
    assert report["complete"] is False
    # Returned untouched, never a partial result.  (Identity is by value:
    # the binding wraps the same underlying expression in a fresh object.)
    assert td.structural_eq(out, bad)
    assert any("could not process" in str(c.message) for c in caught)


# ---- notebook display (vibe 000098) ----------------------------------------


def test_derivation_renders_as_a_table_with_fired_marks():
    """A derivation is a narrative; a Python repr loses the part that matters."""
    ctx = tender.Context()
    a, b, c = (tender.tensor(n, rank=1, ctx=ctx) for n in "abc")
    drv = td.Derivation((a + b) * c)
    drv.step(td.expand_products)
    drv.step(td.expand_products, optional=True)  # second pass: no-op

    html = drv._repr_html_()
    assert "initial" in html
    assert "expand_products" in html
    assert "✓" in html and "·" in html  # one fired, one did not
    assert "changed\nnothing" in html or "changed " in html  # the legend


def test_proof_result_renders_its_verdict_and_evidence():
    ctx = tender.Context()
    a, b, c = (tender.tensor(n, rank=1, ctx=ctx) for n in "abc")
    rules = td.rules("cross", ctx=ctx)

    proved = td.prove_equal(a % (b % c), b * (a @ c) - c * (a @ b), rules)
    html = proved._repr_html_()
    assert "proved" in html
    assert "bac-cab" in html  # the identity that did the work

    refuted = td.prove_equal(a % b, b % a, rules)
    assert "false" in refuted._repr_html_()


def test_proof_result_display_explains_an_incomplete_rule_set():
    ctx = tender.Context()
    a, b, c, d = (tender.tensor(n, rank=1, ctx=ctx) for n in "abcd")
    result = td.prove_equal(
        (a % b) @ (c % d), (a @ c) * (b @ d) - (a @ d) * (b @ c), []
    )
    html = result._repr_html_()
    assert "looks" in html and "incomplete" in html


# ---------------------------------------------------------------------------
# Naming which factor an index move applies to (vibe 000104)
#
# A metric carries two indices and either may be spent, so `contract_metric`
# alone takes whichever binder it reaches first.  "Raise the other one" is an
# ordinary thing to want, and a *path* cannot express it: the factors sharing
# an index are scattered across a product, and no subtree holds just the ones
# meant.  A name can, and survives canonicalization where a path does not.
# ---------------------------------------------------------------------------


def _oblique_dots(ctx):
    """a·b expanded on one oblique frame, contravariant and covariant."""
    import tender.basis as tb

    frame = tb.make_oblique_basis(
        [tender.tensor(n, rank=1, ctx=ctx) for n in ("p", "q", "s")],
        tender.space_3d,
    )
    a = tender.tensor("a", rank=1, ctx=ctx)
    b = tender.tensor("b", rank=1, ctx=ctx)

    def on(variance):
        x = tb.expand_in_basis(a @ b, frame, variance)
        return td.canonicalize(tb.simplify_basis_dot(x, frame))

    return on(tb.Variance.Contravariant), on(tb.Variance.Covariant)


def test_contract_metric_target_picks_which_index_moves():
    ctx = tender.Context()
    x, _ = _oblique_dots(ctx)  # g^ij a_i b_j

    raised_a = td.contract_metric(x, target="a")
    raised_b = td.contract_metric(x, target="b")
    assert "a^{i}" in raised_a.latex() and "b_{i}" in raised_a.latex()
    assert "a_{i}" in raised_b.latex() and "b^{i}" in raised_b.latex()
    # Different expressions — that is the whole point — but the same scalar.
    assert not td.structural_eq(raised_a, raised_b)


def test_contract_metric_untargeted_still_moves_something():
    ctx = tender.Context()
    x, _ = _oblique_dots(ctx)
    out = td.contract_metric(x)
    assert "g" not in out.latex(), out.latex()


def test_contract_metric_target_that_is_not_there_is_a_no_op():
    ctx = tender.Context()
    x, _ = _oblique_dots(ctx)
    assert td.structural_eq(td.contract_metric(x, target="c"), x)


def test_insert_metric_target_picks_which_index_moves_back():
    ctx = tender.Context()
    x, covariant = _oblique_dots(ctx)

    mixed = td.contract_metric(x, target="b")  # a_i b^i
    # Moving `a` up pays a covariant metric and lands on the covariant form.
    assert td.algebraic_eq(
        td.insert_metric(mixed, tender.Level.Upper, target="a"), covariant
    )
    # `b` is already upper, so naming it is a no-op rather than a wrong move.
    assert td.structural_eq(
        td.insert_metric(mixed, tender.Level.Upper, target="b"), mixed
    )
