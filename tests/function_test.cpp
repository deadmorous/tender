#include <tender/expr.hpp>

#include <gtest/gtest.h>

using namespace tender;

static ResourceList make_rl()
{
    return ResourceList{};
}

static auto scalar(ResourceList& rl, char const* sym) -> Expr*
{
    return make_symbolic_var(rl, sym);
}

// ===========================================================================
// FunctionApply — basics
// ===========================================================================

TEST(FunctionApply, RankIsZero)
{
    auto rl = make_rl();
    EXPECT_EQ(make_sin(rl, scalar(rl, "x"))->rank(), 0);
}

TEST(FunctionApply, NonScalarArgThrows)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    EXPECT_THROW(make_sin(rl, v), std::invalid_argument);
    EXPECT_THROW(make_exp(rl, v), std::invalid_argument);
}

// ===========================================================================
// FunctionApply — LaTeX
// ===========================================================================

TEST(FunctionApply, LatexSin)
{
    auto rl = make_rl();
    EXPECT_EQ(make_sin(rl, scalar(rl, "x"))->latex(), "\\sin(x)");
}

TEST(FunctionApply, LatexCos)
{
    auto rl = make_rl();
    EXPECT_EQ(make_cos(rl, scalar(rl, "x"))->latex(), "\\cos(x)");
}

TEST(FunctionApply, LatexTan)
{
    auto rl = make_rl();
    EXPECT_EQ(make_tan(rl, scalar(rl, "x"))->latex(), "\\tan(x)");
}

TEST(FunctionApply, LatexExp)
{
    auto rl = make_rl();
    EXPECT_EQ(make_exp(rl, scalar(rl, "x"))->latex(), "\\exp(x)");
}

TEST(FunctionApply, LatexLog)
{
    auto rl = make_rl();
    EXPECT_EQ(make_log(rl, scalar(rl, "x"))->latex(), "\\ln(x)");
}

TEST(FunctionApply, LatexSqrt)
{
    // sqrt uses \sqrt{arg} (no parens)
    auto rl = make_rl();
    EXPECT_EQ(make_sqrt(rl, scalar(rl, "x"))->latex(), "\\sqrt{x}");
}

TEST(FunctionApply, LatexASin)
{
    auto rl = make_rl();
    EXPECT_EQ(make_asin(rl, scalar(rl, "x"))->latex(), "\\arcsin(x)");
}

TEST(FunctionApply, LatexACos)
{
    auto rl = make_rl();
    EXPECT_EQ(make_acos(rl, scalar(rl, "x"))->latex(), "\\arccos(x)");
}

TEST(FunctionApply, LatexATan)
{
    auto rl = make_rl();
    EXPECT_EQ(make_atan(rl, scalar(rl, "x"))->latex(), "\\arctan(x)");
}

TEST(FunctionApply, LatexSinh)
{
    auto rl = make_rl();
    EXPECT_EQ(make_sinh(rl, scalar(rl, "x"))->latex(), "\\sinh(x)");
}

TEST(FunctionApply, LatexCosh)
{
    auto rl = make_rl();
    EXPECT_EQ(make_cosh(rl, scalar(rl, "x"))->latex(), "\\cosh(x)");
}

TEST(FunctionApply, LatexTanh)
{
    auto rl = make_rl();
    EXPECT_EQ(make_tanh(rl, scalar(rl, "x"))->latex(), "\\tanh(x)");
}

// ===========================================================================
// FunctionApply — Python
// ===========================================================================

TEST(FunctionApply, PythonSin)
{
    auto rl = make_rl();
    EXPECT_EQ(make_sin(rl, scalar(rl, "x"))->python(), "sin(symbolic_var('x'))");
}

TEST(FunctionApply, PythonExp)
{
    auto rl = make_rl();
    EXPECT_EQ(make_exp(rl, scalar(rl, "x"))->python(), "exp(symbolic_var('x'))");
}

TEST(FunctionApply, PythonSqrt)
{
    auto rl = make_rl();
    EXPECT_EQ(
        make_sqrt(rl, scalar(rl, "x"))->python(), "sqrt(symbolic_var('x'))");
}

// ===========================================================================
// FunctionApply — kind accessor
// ===========================================================================

TEST(FunctionApply, KindAccessor)
{
    auto rl = make_rl();
    auto* x = scalar(rl, "x");
    auto* e = make_function(rl, FunctionKind::Cos, x);
    auto* fa = dynamic_cast<FunctionApply*>(e);
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(fa->kind(), FunctionKind::Cos);
    EXPECT_EQ(fa->arg(), x);
}

// ===========================================================================
// Pow
// ===========================================================================

TEST(Pow, RankIsZero)
{
    auto rl = make_rl();
    EXPECT_EQ(make_pow(rl, scalar(rl, "x"), Rational{2})->rank(), 0);
}

TEST(Pow, ZeroExponentYieldsOne)
{
    auto rl = make_rl();
    auto* e = make_pow(rl, scalar(rl, "x"), Rational{0});
    auto* rc = dynamic_cast<RationalConst*>(e);
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{1});
}

TEST(Pow, UnitExponentPassthrough)
{
    auto rl = make_rl();
    auto* x = scalar(rl, "x");
    EXPECT_EQ(make_pow(rl, x, Rational{1}), x);
}

TEST(Pow, NonScalarBaseThrows)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    EXPECT_THROW(make_pow(rl, v, Rational{2}), std::invalid_argument);
}

TEST(Pow, LatexIntegerExponent)
{
    auto rl = make_rl();
    EXPECT_EQ(make_pow(rl, scalar(rl, "x"), Rational{3})->latex(), "x^{3}");
}

TEST(Pow, LatexNegativeExponent)
{
    auto rl = make_rl();
    EXPECT_EQ(make_pow(rl, scalar(rl, "x"), Rational{-2})->latex(), "x^{-2}");
}

TEST(Pow, LatexFractionalExponent)
{
    auto rl = make_rl();
    EXPECT_EQ(
        make_pow(rl, scalar(rl, "x"), Rational{1, 2})->latex(),
        "x^{\\frac{1}{2}}");
}

TEST(Pow, Python)
{
    auto rl = make_rl();
    EXPECT_EQ(
        make_pow(rl, scalar(rl, "x"), Rational{2})->python(),
        "pow(symbolic_var('x'), 2)");
}

TEST(Pow, PythonFractional)
{
    auto rl = make_rl();
    EXPECT_EQ(
        make_pow(rl, scalar(rl, "x"), Rational{2, 3})->python(),
        "pow(symbolic_var('x'), 2/3)");
}

// ===========================================================================
// ATan2
// ===========================================================================

TEST(ATan2, RankIsZero)
{
    auto rl = make_rl();
    auto* e = make_atan2(rl, scalar(rl, "y"), scalar(rl, "x"));
    EXPECT_EQ(e->rank(), 0);
}

TEST(ATan2, Latex)
{
    auto rl = make_rl();
    auto* e = make_atan2(rl, scalar(rl, "y"), scalar(rl, "x"));
    EXPECT_EQ(e->latex(), "\\operatorname{atan2}(y, x)");
}

TEST(ATan2, Python)
{
    auto rl = make_rl();
    auto* e = make_atan2(rl, scalar(rl, "y"), scalar(rl, "x"));
    EXPECT_EQ(e->python(), "atan2(symbolic_var('y'), symbolic_var('x'))");
}

TEST(ATan2, NonScalarArgThrows)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    auto* s = scalar(rl, "x");
    EXPECT_THROW(make_atan2(rl, v, s), std::invalid_argument);
    EXPECT_THROW(make_atan2(rl, s, v), std::invalid_argument);
}

// ===========================================================================
// derivative_of — exit criterion: derivative rules correct
// ===========================================================================

TEST(DerivativeOf, SinIsCosFlatExpr)
{
    auto rl = make_rl();
    auto* x = scalar(rl, "x");
    auto* d = derivative_of(rl, FunctionKind::Sin, x);
    // Must be cos(x)
    auto* fa = dynamic_cast<FunctionApply*>(d);
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(fa->kind(), FunctionKind::Cos);
    EXPECT_EQ(fa->arg(), x);
}

TEST(DerivativeOf, CosIsNegSin)
{
    auto rl = make_rl();
    auto* x = scalar(rl, "x");
    auto* d = derivative_of(rl, FunctionKind::Cos, x);
    // Must be -sin(x) = Scale(-1, sin(x))
    auto* sc = dynamic_cast<Scale*>(d);
    ASSERT_NE(sc, nullptr);
    EXPECT_EQ(sc->coeff(), Rational{-1});
    auto* fa = dynamic_cast<FunctionApply*>(sc->expr());
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(fa->kind(), FunctionKind::Sin);
}

TEST(DerivativeOf, ExpIsExp)
{
    auto rl = make_rl();
    auto* x = scalar(rl, "x");
    auto* d = derivative_of(rl, FunctionKind::Exp, x);
    auto* fa = dynamic_cast<FunctionApply*>(d);
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(fa->kind(), FunctionKind::Exp);
    EXPECT_EQ(fa->arg(), x);
}

TEST(DerivativeOf, LogIsInverse)
{
    auto rl = make_rl();
    auto* x = scalar(rl, "x");
    auto* d = derivative_of(rl, FunctionKind::Log, x);
    // Must be x^(-1)
    auto* pw = dynamic_cast<Pow*>(d);
    ASSERT_NE(pw, nullptr);
    EXPECT_EQ(pw->base(), x);
    EXPECT_EQ(pw->exponent(), Rational{-1});
}

TEST(DerivativeOf, SqrtIsHalfInverseSqrt)
{
    auto rl = make_rl();
    auto* x = scalar(rl, "x");
    auto* d = derivative_of(rl, FunctionKind::Sqrt, x);
    // Must be (1/2) * x^(-1/2)
    auto* sc = dynamic_cast<Scale*>(d);
    ASSERT_NE(sc, nullptr);
    Rational const half{1, 2};
    EXPECT_EQ(sc->coeff(), half);
    auto* pw = dynamic_cast<Pow*>(sc->expr());
    ASSERT_NE(pw, nullptr);
    Rational const neg_half{-1, 2};
    EXPECT_EQ(pw->exponent(), neg_half);
}

TEST(DerivativeOf, TanIsSecSquared)
{
    auto rl = make_rl();
    auto* x = scalar(rl, "x");
    auto* d = derivative_of(rl, FunctionKind::Tan, x);
    // Must be cos(x)^(-2)
    auto* pw = dynamic_cast<Pow*>(d);
    ASSERT_NE(pw, nullptr);
    EXPECT_EQ(pw->exponent(), Rational{-2});
    auto* fa = dynamic_cast<FunctionApply*>(pw->base());
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(fa->kind(), FunctionKind::Cos);
}

TEST(DerivativeOf, SinhIsCosh)
{
    auto rl = make_rl();
    auto* x = scalar(rl, "x");
    auto* d = derivative_of(rl, FunctionKind::Sinh, x);
    auto* fa = dynamic_cast<FunctionApply*>(d);
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(fa->kind(), FunctionKind::Cosh);
}

TEST(DerivativeOf, CoshIsSinh)
{
    auto rl = make_rl();
    auto* x = scalar(rl, "x");
    auto* d = derivative_of(rl, FunctionKind::Cosh, x);
    auto* fa = dynamic_cast<FunctionApply*>(d);
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(fa->kind(), FunctionKind::Sinh);
}

TEST(DerivativeOf, ATanIsOneOverOnePlusXSquared)
{
    auto rl = make_rl();
    auto* x = scalar(rl, "x");
    auto* d = derivative_of(rl, FunctionKind::ATan, x);
    // (1 + x^2)^(-1)
    auto* pw = dynamic_cast<Pow*>(d);
    ASSERT_NE(pw, nullptr);
    EXPECT_EQ(pw->exponent(), Rational{-1});
}

TEST(DerivativeOf, ASinResultIsAPow)
{
    auto rl = make_rl();
    auto* x = scalar(rl, "x");
    auto* d = derivative_of(rl, FunctionKind::ASin, x);
    // (1 - x^2)^(-1/2)
    auto* pw = dynamic_cast<Pow*>(d);
    ASSERT_NE(pw, nullptr);
    Rational const neg_half{-1, 2};
    EXPECT_EQ(pw->exponent(), neg_half);
}

TEST(DerivativeOf, ACosIsNegASinDeriv)
{
    auto rl = make_rl();
    auto* x = scalar(rl, "x");
    auto* d = derivative_of(rl, FunctionKind::ACos, x);
    auto* sc = dynamic_cast<Scale*>(d);
    ASSERT_NE(sc, nullptr);
    EXPECT_EQ(sc->coeff(), Rational{-1});
}

TEST(DerivativeOf, TanhIsSechSquared)
{
    auto rl = make_rl();
    auto* x = scalar(rl, "x");
    auto* d = derivative_of(rl, FunctionKind::Tanh, x);
    // cosh(x)^(-2)
    auto* pw = dynamic_cast<Pow*>(d);
    ASSERT_NE(pw, nullptr);
    EXPECT_EQ(pw->exponent(), Rational{-2});
    auto* fa = dynamic_cast<FunctionApply*>(pw->base());
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(fa->kind(), FunctionKind::Cosh);
}

// ===========================================================================
// derivative_of_pow
// ===========================================================================

TEST(DerivativeOfPow, IntegerExponent)
{
    auto rl = make_rl();
    auto* x = scalar(rl, "x");
    auto* d = derivative_of_pow(rl, x, Rational{3});
    // 3 * x^2
    auto* sc = dynamic_cast<Scale*>(d);
    ASSERT_NE(sc, nullptr);
    EXPECT_EQ(sc->coeff(), Rational{3});
    auto* pw = dynamic_cast<Pow*>(sc->expr());
    ASSERT_NE(pw, nullptr);
    EXPECT_EQ(pw->exponent(), Rational{2});
}

TEST(DerivativeOfPow, UnitExponentYieldsCoeff)
{
    auto rl = make_rl();
    auto* x = scalar(rl, "x");
    auto* d = derivative_of_pow(rl, x, Rational{1});
    // 1 * x^0 = 1 * 1 = 1
    auto* rc = dynamic_cast<RationalConst*>(d);
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{1});
}
