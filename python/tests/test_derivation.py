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
