"""Integration tests for the tender Python bindings."""

import pytest
import tender
from tender import (
    Context,
    CountableIndex,
    IndexNameMap,
    IndexSpace,
    Level,
    Rational,
    Realm,
)


# ---------------------------------------------------------------------------
# Rational
# ---------------------------------------------------------------------------


class TestRational:
    def test_integer(self):
        r = Rational(3)
        assert r.num == 3
        assert r.den == 1
        assert r.is_integer
        assert not r.is_zero

    def test_fraction(self):
        r = Rational(1, 2)
        assert r.num == 1
        assert r.den == 2
        assert not r.is_integer

    def test_normalisation(self):
        r = Rational(4, 6)
        assert r.num == 2
        assert r.den == 3

    def test_arithmetic(self):
        assert Rational(1, 2) + Rational(1, 3) == Rational(5, 6)
        assert Rational(3, 4) - Rational(1, 4) == Rational(1, 2)
        assert Rational(2, 3) * Rational(3, 4) == Rational(1, 2)
        assert Rational(1, 2) / Rational(1, 4) == Rational(2)

    def test_negation(self):
        assert -Rational(3) == Rational(-3)

    def test_repr(self):
        assert repr(Rational(3)) == "3"
        assert repr(Rational(1, 2)) == "1/2"

    def test_zero(self):
        assert Rational(0).is_zero


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TestEnums:
    def test_realm_values(self):
        assert Realm.Oblique != Realm.Orthonormal
        assert Realm.Collection != Realm.Label

    def test_level_values(self):
        assert Level.Upper != Level.Lower


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


class TestContext:
    def test_default_context_used_implicitly(self):
        # tensor() with no ctx= uses the module-level default context
        a = tender.tensor("a")
        assert a is not None

    def test_explicit_context(self):
        ctx = Context()
        a = tender.tensor("a", ctx=ctx)
        b = tender.tensor("b", ctx=ctx)
        result = a + b
        assert "mathbf{a}" in result.latex()
        assert "mathbf{b}" in result.latex()

    def test_alloc_index_via_context(self):
        ctx = Context()
        i = ctx.alloc_index()
        j = ctx.alloc_index()
        assert isinstance(i, CountableIndex)
        assert i.id != j.id

    def test_new_context_shares_ids(self):
        ctx = Context()
        i = ctx.alloc_index()
        child = ctx.new_context()
        j = child.alloc_index()
        # ids must be unique across parent and child
        assert i.id != j.id


# ---------------------------------------------------------------------------
# Index spaces
# ---------------------------------------------------------------------------


class TestIndexSpaces:
    def test_singletons_are_index_space(self):
        assert isinstance(tender.space_2d, IndexSpace)
        assert isinstance(tender.space_3d, IndexSpace)
        assert isinstance(tender.space_4d, IndexSpace)

    def test_singletons_are_distinct(self):
        # pointer identity is preserved across calls to the module-level attrs
        assert tender.space_2d is not tender.space_3d
        assert tender.space_3d is not tender.space_4d


# ---------------------------------------------------------------------------
# CountableIndex / alloc_index
# ---------------------------------------------------------------------------


class TestAllocIndex:
    def test_returns_countable_index(self):
        i = tender.alloc_index()
        assert isinstance(i, CountableIndex)

    def test_ids_are_unique(self):
        ids = {tender.alloc_index().id for _ in range(10)}
        assert len(ids) == 10

    def test_eq_and_hash(self):
        i = tender.alloc_index()
        j = CountableIndex.__new__(CountableIndex)
        # Build a second CountableIndex with the same id via a fresh alloc
        # and check inequality
        k = tender.alloc_index()
        assert i == i
        assert i != k
        assert hash(i) == hash(i)


# ---------------------------------------------------------------------------
# Tensor factory
# ---------------------------------------------------------------------------


class TestTensorFactory:
    def test_scalar_rank(self):
        s = tender.tensor("s", rank=0)
        latex = s.latex()
        # rank-0 → plain, no \mathbf
        assert "mathbf" not in latex
        assert "boldsymbol" not in latex
        assert "s" in latex

    def test_vector_rank(self):
        v = tender.tensor("v", rank=1)
        assert "mathbf{v}" in v.latex()

    def test_tensor_rank(self):
        A = tender.tensor("A", rank=2)
        assert "mathbf{A}" in A.latex()

    def test_unknown_rank_bold(self):
        # rank=None (default) is treated as ≥1, rendered bold
        x = tender.tensor("x")
        assert "mathbf{x}" in x.latex()

    def test_latex_command_name(self):
        sig = tender.tensor(r"\sigma", rank=1)
        assert r"boldsymbol{\sigma}" in sig.latex()


# ---------------------------------------------------------------------------
# Scalar factory
# ---------------------------------------------------------------------------


class TestScalarFactory:
    def test_integer(self):
        s = tender.scalar(5)
        assert s.latex() == "5"

    def test_rational(self):
        s = tender.scalar(Rational(1, 3))
        assert r"\frac{1}{3}" in s.latex()

    def test_negative(self):
        s = tender.scalar(-2)
        assert s.latex() == "-2"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_renders(self):
        I = tender.identity()
        assert "mathbf{I}" in I.latex()


# ---------------------------------------------------------------------------
# Kronecker delta
# ---------------------------------------------------------------------------


class TestDelta:
    def test_oblique_3d(self):
        sp = tender.space_3d
        i = tender.alloc_index()
        j = tender.alloc_index()
        d = tender.delta(Realm.Oblique, sp, Level.Upper, Level.Lower, i, j)
        latex = d.latex()
        # rank=0 → plain (no bold)
        assert r"\delta" in latex
        assert r"\boldsymbol" not in latex
        # positional interleaving: upper band has \cdot placeholder, lower band has \cdot placeholder
        assert r"^{" in latex
        assert r"_{" in latex
        assert r"\cdot" in latex

    def test_orthonormal_2d(self):
        sp = tender.space_2d
        i = tender.alloc_index()
        j = tender.alloc_index()
        d = tender.delta(Realm.Orthonormal, sp, Level.Lower, Level.Lower, i, j)
        latex = d.latex()
        assert r"\delta" in latex
        assert r"\boldsymbol" not in latex

    def test_concrete_index(self):
        sp = tender.space_3d
        i = tender.alloc_index()
        d = tender.delta(Realm.Oblique, sp, Level.Upper, Level.Lower, i, 1)
        latex = d.latex()
        assert "1" in latex

    def test_index_names_allocated(self):
        sp = tender.space_3d
        i = tender.alloc_index()
        j = tender.alloc_index()
        d = tender.delta(Realm.Oblique, sp, Level.Upper, Level.Lower, i, j)
        m = IndexNameMap()
        d.latex(m)
        assert m.lookup(i) is not None
        assert m.lookup(j) is not None
        assert m.lookup(i) != m.lookup(j)


# ---------------------------------------------------------------------------
# Levi-Civita
# ---------------------------------------------------------------------------


class TestLeviCivita:
    def test_3d_orthonormal(self):
        sp = tender.space_3d
        i, j, k = tender.alloc_index(), tender.alloc_index(), tender.alloc_index()
        lc = tender.levi_civita(
            Realm.Orthonormal, sp,
            [Level.Lower, Level.Lower, Level.Lower],
            [i, j, k],
        )
        latex = lc.latex()
        # rank=0 → plain (no bold), pure lower → flat grouping, no \cdot
        assert r"\varepsilon" in latex
        assert r"\boldsymbol" not in latex
        assert "_{" in latex
        assert r"\cdot" not in latex

    def test_wrong_size_raises(self):
        sp = tender.space_3d
        i, j = tender.alloc_index(), tender.alloc_index()
        with pytest.raises(Exception):
            tender.levi_civita(
                Realm.Orthonormal, sp,
                [Level.Lower, Level.Lower, Level.Lower],
                [i, j],  # size mismatch
            )


# ---------------------------------------------------------------------------
# Operator overloads on Expr
# ---------------------------------------------------------------------------


class TestOperators:
    def setup_method(self):
        self.a = tender.tensor("a", rank=1)
        self.b = tender.tensor("b", rank=1)
        self.A = tender.tensor("A", rank=2)

    def test_add(self):
        assert r"\mathbf{a} + \mathbf{b}" in (self.a + self.b).latex()

    def test_sub(self):
        assert r"\mathbf{a} - \mathbf{b}" in (self.a - self.b).latex()

    def test_neg(self):
        assert "-" in (-self.a).latex()
        assert "mathbf{a}" in (-self.a).latex()

    def test_scalar_mul_right(self):
        latex = (self.a * 2).latex()
        assert "mathbf{a}" in latex
        assert "2" in latex

    def test_scalar_mul_left(self):
        latex = (2 * self.a).latex()
        assert "mathbf{a}" in latex
        assert "2" in latex

    def test_tensor_product(self):
        latex = (self.a * self.b).latex()
        assert r"\," in latex

    def test_dot(self):
        assert r"\cdot" in (self.a @ self.b).latex()

    def test_cross(self):
        assert r"\times" in (self.a % self.b).latex()

    def test_scalar_div(self):
        s = tender.scalar(2)
        assert r"\frac" in (self.a / s).latex()

    def test_div_by_int(self):
        assert r"\frac" in (self.a / 3).latex()

    def test_ddot(self):
        assert ":" in self.A.ddot(self.A).latex()

    def test_ddot_alt(self):
        assert r"\cdot" in self.A.ddot_alt(self.A).latex()

    def test_add_int_right(self):
        s = tender.scalar(1)
        assert "1" in (s + 2).latex()

    def test_radd_int(self):
        s = tender.scalar(1)
        assert "1" in (2 + s).latex()

    def test_rsub_int(self):
        s = tender.scalar(1)
        result = (3 - s).latex()
        assert "3" in result and "1" in result


# ---------------------------------------------------------------------------
# Parenthesisation
# ---------------------------------------------------------------------------


class TestParens:
    def test_sum_inside_difference_rhs(self):
        a = tender.tensor("a", rank=1)
        b = tender.tensor("b", rank=1)
        c = tender.tensor("c", rank=1)
        # a - (b + c) must parenthesise b+c
        result = (a - (b + c)).latex()
        assert "(" in result

    def test_no_parens_for_same_prec_add(self):
        a = tender.tensor("a", rank=1)
        b = tender.tensor("b", rank=1)
        c = tender.tensor("c", rank=1)
        # (a + b) + c needs no parens
        result = ((a + b) + c).latex()
        assert result.count("(") == 0


# ---------------------------------------------------------------------------
# ExplicitSum / NoSum
# ---------------------------------------------------------------------------


class TestSumAnnotations:
    def test_explicit_sum(self):
        sp = tender.space_3d
        i = tender.alloc_index()
        j = tender.alloc_index()
        d = tender.delta(Realm.Oblique, sp, Level.Upper, Level.Lower, i, j)
        es = tender.explicit_sum(i, d)
        latex = es.latex()
        assert r"\sum" in latex

    def test_no_sum(self):
        sp = tender.space_3d
        i = tender.alloc_index()
        j = tender.alloc_index()
        d = tender.delta(Realm.Oblique, sp, Level.Upper, Level.Lower, i, j)
        ns = tender.no_sum(i, d)
        latex = ns.latex()
        assert r"\cancel" in latex
        assert r"\sum" in latex


# ---------------------------------------------------------------------------
# IndexNameMap
# ---------------------------------------------------------------------------


class TestIndexNameMap:
    def test_allocates_names(self):
        sp = tender.space_3d
        i = tender.alloc_index()
        j = tender.alloc_index()
        d = tender.delta(Realm.Oblique, sp, Level.Upper, Level.Lower, i, j)
        m = IndexNameMap()
        d.latex(m)
        assert m.lookup(i) == "i"
        assert m.lookup(j) == "j"

    def test_names_stable_across_calls(self):
        sp = tender.space_3d
        i = tender.alloc_index()
        j = tender.alloc_index()
        d = tender.delta(Realm.Oblique, sp, Level.Upper, Level.Lower, i, j)
        m = IndexNameMap()
        d.latex(m)
        name_i_first = m.lookup(i)
        d.latex(m)
        assert m.lookup(i) == name_i_first

    def test_assign_and_reverse_lookup(self):
        sp = tender.space_3d
        i = tender.alloc_index()
        m = IndexNameMap()
        m.assign(i, "k")
        assert m.lookup(i) == "k"
        found = m.index_for("k")
        assert found is not None
        assert found == i

    def test_name_for(self):
        sp = tender.space_3d
        i = tender.alloc_index()
        m = IndexNameMap()
        name = m.name_for(i, sp)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_lookup_unknown_returns_none(self):
        i = tender.alloc_index()
        m = IndexNameMap()
        assert m.lookup(i) is None

    def test_index_for_unknown_returns_none(self):
        m = IndexNameMap()
        assert m.index_for("z") is None


# ---------------------------------------------------------------------------
# _repr_latex_ for Jupyter
# ---------------------------------------------------------------------------


class TestReprLatex:
    def test_wrapped_in_dollar(self):
        a = tender.tensor("a", rank=1)
        s = a._repr_latex_()
        assert s.startswith("$") and s.endswith("$")

    def test_content_is_valid_latex(self):
        a = tender.tensor("a", rank=1)
        inner = a._repr_latex_()[1:-1]  # strip $...$
        assert r"\mathbf{a}" == inner
