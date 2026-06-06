#include <tender/expr.hpp>

#include <gtest/gtest.h>

using namespace tender;

// ===========================================================================
// Helpers
// ===========================================================================

static ResourceList make_rl()
{
    return ResourceList{};
}

// ===========================================================================
// RationalConst
// ===========================================================================

TEST(RationalConst, RankIsZero)
{
    auto rl = make_rl();
    auto* e = make_rational(rl, Rational{3, 4});
    EXPECT_EQ(e->rank(), 0);
}

TEST(RationalConst, LatexInteger)
{
    auto rl = make_rl();
    EXPECT_EQ(make_rational(rl, Rational{5})->latex(), "5");
}

TEST(RationalConst, LatexFraction)
{
    auto rl = make_rl();
    EXPECT_EQ(make_rational(rl, Rational(3, 4))->latex(), "\\frac{3}{4}");
}

TEST(RationalConst, LatexNegativeFraction)
{
    auto rl = make_rl();
    EXPECT_EQ(make_rational(rl, Rational(-3, 4))->latex(), "-\\frac{3}{4}");
}

TEST(RationalConst, PythonOutput)
{
    auto rl = make_rl();
    EXPECT_EQ(make_rational(rl, Rational(2, 3))->python(), "Rational(2/3)");
}

TEST(RationalConst, NamedLatex)
{
    auto rl = make_rl();
    auto* e = named("alpha", make_rational(rl, Rational{1}));
    EXPECT_EQ(e->latex(), "\\text{alpha}");
    EXPECT_EQ(e->python(), "alpha");
}

// ===========================================================================
// NamedConst
// ===========================================================================

TEST(NamedConst, SingleCharLatex)
{
    auto rl = make_rl();
    EXPECT_EQ(make_named_const(rl, "e")->latex(), "e");
}

TEST(NamedConst, MultiCharLatex)
{
    auto rl = make_rl();
    EXPECT_EQ(make_named_const(rl, "pi")->latex(), "\\text{pi}");
}

TEST(NamedConst, Python)
{
    auto rl = make_rl();
    EXPECT_EQ(make_named_const(rl, "pi")->python(), "named_const('pi')");
}

// ===========================================================================
// SymbolicVar
// ===========================================================================

TEST(SymbolicVar, RankIsZero)
{
    auto rl = make_rl();
    EXPECT_EQ(make_symbolic_var(rl, "x")->rank(), 0);
}

TEST(SymbolicVar, LatexAndPython)
{
    auto rl = make_rl();
    auto* v = make_symbolic_var(rl, "x");
    EXPECT_EQ(v->latex(), "x");
    EXPECT_EQ(v->python(), "symbolic_var('x')");
}

// ===========================================================================
// Scale
// ===========================================================================

TEST(Scale, ZeroCoeffYieldsZero)
{
    auto rl = make_rl();
    auto* v = make_symbolic_var(rl, "x");
    auto* e = make_scale(rl, Rational{0}, v);
    auto* rc = dynamic_cast<RationalConst*>(e);
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{0});
}

TEST(Scale, UnitCoeffPassthrough)
{
    auto rl = make_rl();
    auto* v = make_symbolic_var(rl, "x");
    auto* e = make_scale(rl, Rational{1}, v);
    EXPECT_EQ(e, v);
}

TEST(Scale, ScalesRationalConst)
{
    auto rl = make_rl();
    auto* rc = make_rational(rl, Rational(1, 2));
    auto* e = make_scale(rl, Rational(3, 4), rc);
    auto* result = dynamic_cast<RationalConst*>(e);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->value(), Rational(3, 8));
}

TEST(Scale, NestedScaleCollapses)
{
    auto rl = make_rl();
    auto* v = make_symbolic_var(rl, "x");
    auto* inner = make_scale(rl, Rational(1, 2), v);
    auto* outer = make_scale(rl, Rational(2), inner);
    // 2 * (1/2 * x) → 1 * x → x
    EXPECT_EQ(outer, v);
}

TEST(Scale, LatexNegativeOne)
{
    auto rl = make_rl();
    auto* v = make_symbolic_var(rl, "x");
    auto* e = make_scale(rl, Rational{-1}, v);
    EXPECT_EQ(e->latex(), "-x");
}

TEST(Scale, LatexFractional)
{
    auto rl = make_rl();
    auto* v = make_symbolic_var(rl, "x");
    auto* e = make_scale(rl, Rational(2, 3), v);
    EXPECT_EQ(e->latex(), "\\frac{2}{3} x");
}

TEST(Scale, RankMatchesInner)
{
    auto rl = make_rl();
    auto* v = make_symbolic_var(rl, "x");
    auto* e = make_scale(rl, Rational{2}, v);
    EXPECT_EQ(e->rank(), 0);
}

// ===========================================================================
// Sum
// ===========================================================================

TEST(Sum, EmptySumIsZero)
{
    auto rl = make_rl();
    auto* e = make_sum(rl, {});
    auto* rc = dynamic_cast<RationalConst*>(e);
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{0});
}

TEST(Sum, SingleTermPassthrough)
{
    auto rl = make_rl();
    auto* v = make_symbolic_var(rl, "x");
    auto* e = make_sum(rl, {v});
    EXPECT_EQ(e, v);
}

TEST(Sum, CancellationYieldsZero)
{
    auto rl = make_rl();
    auto* v = make_symbolic_var(rl, "x");
    auto* pos = make_scale(rl, Rational{1}, v);
    auto* neg = make_scale(rl, Rational{-1}, v);
    auto* e = make_sum(rl, {pos, neg});
    auto* rc = dynamic_cast<RationalConst*>(e);
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{0});
}

TEST(Sum, LikeTermsCollected)
{
    auto rl = make_rl();
    auto* v = make_symbolic_var(rl, "x");
    auto* a = make_scale(rl, Rational{2}, v);
    auto* b = make_scale(rl, Rational{3}, v);
    auto* e = make_sum(rl, {a, b});
    // 2x + 3x = 5x
    auto* sc = dynamic_cast<Scale*>(e);
    ASSERT_NE(sc, nullptr);
    EXPECT_EQ(sc->coeff(), Rational{5});
    EXPECT_EQ(sc->expr(), v);
}

TEST(Sum, ConstantsAccumulated)
{
    auto rl = make_rl();
    auto* a = make_rational(rl, Rational{2});
    auto* b = make_rational(rl, Rational{3});
    auto* e = make_sum(rl, {a, b});
    auto* rc = dynamic_cast<RationalConst*>(e);
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{5});
}

TEST(Sum, FlattenNestedSum)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    auto* y = make_symbolic_var(rl, "y");
    auto* z = make_symbolic_var(rl, "z");
    auto* inner = make_sum(rl, {x, y});
    auto* outer = make_sum(rl, {inner, z});
    // Result must be a Sum with 3 terms, not nested Sums.
    auto* s = dynamic_cast<Sum*>(outer);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->terms().size(), 3u);
}

TEST(Sum, LatexWithMixedSigns)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    auto* y = make_symbolic_var(rl, "y");
    // 3x + (-2)y  → "3 x - 2 y"
    auto* pos = make_scale(rl, Rational{3}, x);
    auto* neg = make_scale(rl, Rational{-2}, y);
    auto* e = make_sum(rl, {pos, neg});
    EXPECT_EQ(e->latex(), "3 x - 2 y");
}

TEST(Sum, ScaleDistributesOverSum)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    auto* y = make_symbolic_var(rl, "y");
    auto* s = make_sum(rl, {x, y});
    auto* e = make_scale(rl, Rational{2}, s);
    // 2*(x+y) = 2x+2y, a Sum
    auto* sum = dynamic_cast<Sum*>(e);
    ASSERT_NE(sum, nullptr);
    EXPECT_EQ(sum->terms().size(), 2u);
}

// ===========================================================================
// TensorProduct
// ===========================================================================

TEST(TensorProduct, RankIsSum)
{
    auto rl = make_rl();
    // Rank-0 scalars — tensor product is rank 0.
    auto* x = make_symbolic_var(rl, "x");
    auto* y = make_symbolic_var(rl, "y");
    auto* tp = make_tensor_product(rl, x, y);
    EXPECT_EQ(tp->rank(), 0);
}

TEST(TensorProduct, Latex)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    auto* y = make_symbolic_var(rl, "y");
    EXPECT_EQ(make_tensor_product(rl, x, y)->latex(), "x \\otimes y");
}

TEST(TensorProduct, Python)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    auto* y = make_symbolic_var(rl, "y");
    EXPECT_EQ(
        make_tensor_product(rl, x, y)->python(),
        "tp(symbolic_var('x'), symbolic_var('y'))");
}

// ===========================================================================
// named()
// ===========================================================================

TEST(Named, SetNameIdempotent)
{
    auto rl = make_rl();
    auto* v = make_symbolic_var(rl, "x");
    named("A", v);
    named("A", v); // idempotent — no throw
    EXPECT_EQ(v->name(), "A");
}

TEST(Named, ConflictingNameThrows)
{
    auto rl = make_rl();
    auto* v = make_symbolic_var(rl, "x");
    named("A", v);
    EXPECT_THROW(named("B", v), std::logic_error);
}

// ===========================================================================
// Coverage gap tests
// ===========================================================================

TEST(NamedConst, Rank)
{
    auto rl = make_rl();
    EXPECT_EQ(make_named_const(rl, "e")->rank(), 0);
}

TEST(Scale, Python)
{
    auto rl = make_rl();
    auto* v = make_symbolic_var(rl, "x");
    auto* e = make_scale(rl, Rational(2, 3), v);
    EXPECT_EQ(e->python(), "scale(2/3, symbolic_var('x'))");
}

TEST(Sum, Rank)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    auto* y = make_symbolic_var(rl, "y");
    auto* e = make_sum(rl, {x, y});
    EXPECT_EQ(e->rank(), 0);
}

TEST(Sum, Python)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    auto* y = make_symbolic_var(rl, "y");
    auto* e = make_sum(rl, {x, y});
    EXPECT_EQ(e->python(), "sum([symbolic_var('x'), symbolic_var('y')])");
}

TEST(Sum, LatexPositiveSecondTerm)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    auto* y = make_symbolic_var(rl, "y");
    // -x + y: first term negative, second positive → exercises " + " branch
    auto* e = make_sum(rl, {make_scale(rl, Rational{-1}, x), y});
    EXPECT_EQ(e->latex(), "-x + y");
}

TEST(Sum, LatexNegativeUnitCoeff)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    auto* y = make_symbolic_var(rl, "y");
    // x - y: second term has coeff -1, exercises abs_coeff==1 branch
    auto* e = make_sum(rl, {x, make_scale(rl, Rational{-1}, y)});
    EXPECT_EQ(e->latex(), "x - y");
}

TEST(Sum, RankMismatchThrows)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x"); // rank 0
    auto* v = make_named_tensor(rl, "v", 1, {{SlotLevel::Upper, "i"}});
    EXPECT_THROW(make_sum(rl, {x, v}), std::invalid_argument);
}
