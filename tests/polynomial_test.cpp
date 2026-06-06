#include <tender/polynomial.hpp>

#include <gtest/gtest.h>

using namespace tender;

// ===========================================================================
// Construction and accessors
// ===========================================================================

TEST(Polynomial, ZeroPolynomial)
{
    Polynomial p;
    EXPECT_TRUE(p.is_zero());
    EXPECT_EQ(p.degree(), -1);
    EXPECT_EQ(p.coeff(0), Rational{0});
    EXPECT_EQ(p.coeff(5), Rational{0});
}

TEST(Polynomial, ConstantPolynomial)
{
    Polynomial p{{{Rational{7}, 0}}};
    EXPECT_EQ(p.degree(), 0);
    EXPECT_EQ(p.coeff(0), Rational{7});
    EXPECT_FALSE(p.is_zero());
}

TEST(Polynomial, LinearPolynomial)
{
    // 2x + 3
    Polynomial p{{{Rational{2}, 1}, {Rational{3}, 0}}};
    EXPECT_EQ(p.degree(), 1);
    EXPECT_EQ(p.coeff(1), Rational{2});
    EXPECT_EQ(p.coeff(0), Rational{3});
}

TEST(Polynomial, DuplicateExponentsAccumulated)
{
    // (x + x) = 2x
    Polynomial p{{{Rational{1}, 1}, {Rational{1}, 1}}};
    EXPECT_EQ(p.degree(), 1);
    EXPECT_EQ(p.coeff(1), Rational{2});
}

TEST(Polynomial, ZeroTermsDropped)
{
    // 2x + 0 = 2x
    Polynomial p{{{Rational{2}, 1}, {Rational{0}, 0}}};
    EXPECT_EQ(p.degree(), 1);
    EXPECT_EQ(p.coeff(0), Rational{0});
}

TEST(Polynomial, NegativeExponentThrows)
{
    EXPECT_THROW(Polynomial({{Rational{1}, -1}}), std::invalid_argument);
}

// ===========================================================================
// Arithmetic
// ===========================================================================

TEST(Polynomial, Addition)
{
    // (3x^2 - 2x + 1) + (x^2 + x) = 4x^2 - x + 1
    Polynomial p{{{Rational{3}, 2}, {Rational{-2}, 1}, {Rational{1}, 0}}};
    Polynomial q{{{Rational{1}, 2}, {Rational{1}, 1}}};
    auto r = p + q;
    EXPECT_EQ(r.coeff(2), Rational{4});
    EXPECT_EQ(r.coeff(1), Rational{-1});
    EXPECT_EQ(r.coeff(0), Rational{1});
}

TEST(Polynomial, Subtraction)
{
    // (3x^2) - (x^2) = 2x^2
    Polynomial p{{{Rational{3}, 2}}};
    Polynomial q{{{Rational{1}, 2}}};
    auto r = p - q;
    EXPECT_EQ(r.degree(), 2);
    EXPECT_EQ(r.coeff(2), Rational{2});
}

TEST(Polynomial, SubtractionYieldsZero)
{
    Polynomial p{{{Rational{5}, 1}}};
    auto r = p - p;
    EXPECT_TRUE(r.is_zero());
}

TEST(Polynomial, Multiplication)
{
    // (x + 1) * (x - 1) = x^2 - 1
    Polynomial p{{{Rational{1}, 1}, {Rational{1}, 0}}};
    Polynomial q{{{Rational{1}, 1}, {Rational{-1}, 0}}};
    auto r = p * q;
    EXPECT_EQ(r.degree(), 2);
    EXPECT_EQ(r.coeff(2), Rational{1});
    EXPECT_EQ(r.coeff(1), Rational{0});
    EXPECT_EQ(r.coeff(0), Rational{-1});
}

TEST(Polynomial, UnaryNegation)
{
    Polynomial p{{{Rational{2}, 1}, {Rational{-3}, 0}}};
    auto r = -p;
    EXPECT_EQ(r.coeff(1), Rational{-2});
    EXPECT_EQ(r.coeff(0), Rational{3});
}

TEST(Polynomial, CompoundAssignmentAdd)
{
    Polynomial p{{{Rational{1}, 1}}};
    Polynomial q{{{Rational{2}, 0}}};
    p += q;
    EXPECT_EQ(p.coeff(1), Rational{1});
    EXPECT_EQ(p.coeff(0), Rational{2});
}

TEST(Polynomial, Equality)
{
    Polynomial p{{{Rational{1}, 2}, {Rational{3}, 0}}};
    Polynomial q{{{Rational{3}, 0}, {Rational{1}, 2}}}; // same, different order
    EXPECT_EQ(p, q);
}

TEST(Polynomial, Inequality)
{
    Polynomial p{{{Rational{1}, 2}}};
    Polynomial q{{{Rational{2}, 2}}};
    EXPECT_NE(p, q);
}

// ===========================================================================
// Differentiation
// ===========================================================================

TEST(Polynomial, DiffConstantIsZero)
{
    Polynomial p{{{Rational{5}, 0}}};
    EXPECT_TRUE(p.diff().is_zero());
}

TEST(Polynomial, DiffLinear)
{
    // d/dx (2x + 3) = 2
    Polynomial p{{{Rational{2}, 1}, {Rational{3}, 0}}};
    auto dp = p.diff();
    EXPECT_EQ(dp.degree(), 0);
    EXPECT_EQ(dp.coeff(0), Rational{2});
}

TEST(Polynomial, DiffQuadratic)
{
    // d/dx (3x^2 - 2x + 1) = 6x - 2
    Polynomial p{{{Rational{3}, 2}, {Rational{-2}, 1}, {Rational{1}, 0}}};
    auto dp = p.diff();
    EXPECT_EQ(dp.degree(), 1);
    EXPECT_EQ(dp.coeff(1), Rational{6});
    EXPECT_EQ(dp.coeff(0), Rational{-2});
}

TEST(Polynomial, DiffCubic)
{
    // d/dx x^3 = 3x^2
    Polynomial p{{{Rational{1}, 3}}};
    auto dp = p.diff();
    EXPECT_EQ(dp.degree(), 2);
    EXPECT_EQ(dp.coeff(2), Rational{3});
}

// ===========================================================================
// Evaluation
// ===========================================================================

TEST(Polynomial, EvalAtZero)
{
    Polynomial p{{{Rational{5}, 2}, {Rational{3}, 1}, {Rational{7}, 0}}};
    EXPECT_EQ(p.eval(Rational{0}), Rational{7});
}

TEST(Polynomial, EvalAtOne)
{
    // 3x^2 - 2x + 1 at x=1: 3 - 2 + 1 = 2
    Polynomial p{{{Rational{3}, 2}, {Rational{-2}, 1}, {Rational{1}, 0}}};
    EXPECT_EQ(p.eval(Rational{1}), Rational{2});
}

TEST(Polynomial, EvalAtTwo)
{
    // 3x^2 - 2x + 1 at x=2: 12 - 4 + 1 = 9
    Polynomial p{{{Rational{3}, 2}, {Rational{-2}, 1}, {Rational{1}, 0}}};
    EXPECT_EQ(p.eval(Rational{2}), Rational{9});
}

TEST(Polynomial, EvalZeroPolynomial)
{
    Polynomial p;
    EXPECT_EQ(p.eval(Rational{42}), Rational{0});
}

// ===========================================================================
// Output
// ===========================================================================

TEST(Polynomial, ToLatexQuadratic)
{
    // 3x^2 - 2x + 1
    Polynomial p{{{Rational{3}, 2}, {Rational{-2}, 1}, {Rational{1}, 0}}};
    EXPECT_EQ(p.to_latex(), "3 x^{2} - 2 x + 1");
}

TEST(Polynomial, ToLatexMonic)
{
    // x^2 + x (coefficient 1 omitted before variable)
    Polynomial p{{{Rational{1}, 2}, {Rational{1}, 1}}};
    EXPECT_EQ(p.to_latex(), "x^{2} + x");
}

TEST(Polynomial, ToLatexNegativeLead)
{
    // -x^2 + 1
    Polynomial p{{{Rational{-1}, 2}, {Rational{1}, 0}}};
    EXPECT_EQ(p.to_latex(), "-x^{2} + 1");
}

TEST(Polynomial, ToLatexZero)
{
    Polynomial p;
    EXPECT_EQ(p.to_latex(), "0");
}

TEST(Polynomial, ToLatexCustomVar)
{
    Polynomial p{{{Rational{2}, 1}}};
    EXPECT_EQ(p.to_latex("t"), "2 t");
}

TEST(Polynomial, ToPythonQuadratic)
{
    // 3x**2 - 2x + 1
    Polynomial p{{{Rational{3}, 2}, {Rational{-2}, 1}, {Rational{1}, 0}}};
    EXPECT_EQ(p.to_python(), "3*x**2 - 2*x + 1");
}

TEST(Polynomial, ToPythonMonic)
{
    Polynomial p{{{Rational{1}, 2}, {Rational{1}, 1}}};
    EXPECT_EQ(p.to_python(), "x**2 + x");
}

TEST(Polynomial, ToPythonZero)
{
    Polynomial p;
    EXPECT_EQ(p.to_python(), "0");
}
