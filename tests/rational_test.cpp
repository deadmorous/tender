#include <gtest/gtest.h>

#include <tender/rational.hpp>

using tender::Rational;

// --- Construction and normalisation ---

TEST(Rational, DefaultIsZero)
{
    Rational const r;
    EXPECT_EQ(r.num(), 0);
    EXPECT_EQ(r.den(), 1);
}

TEST(Rational, FromInteger)
{
    Rational const r{7};
    EXPECT_EQ(r.num(), 7);
    EXPECT_EQ(r.den(), 1);
}

TEST(Rational, NormalisesGCD)
{
    Rational const r(4, 6);
    EXPECT_EQ(r.num(), 2);
    EXPECT_EQ(r.den(), 3);
}

TEST(Rational, NegativeNumerator)
{
    Rational const r(-3, 4);
    EXPECT_EQ(r.num(), -3);
    EXPECT_EQ(r.den(), 4);
}

TEST(Rational, NegativeDenominatorMovedToNumerator)
{
    Rational const r(3, -4);
    EXPECT_EQ(r.num(), -3);
    EXPECT_EQ(r.den(), 4);
}

TEST(Rational, BothNegativeBecomesPositive)
{
    Rational const r(-3, -4);
    EXPECT_EQ(r.num(), 3);
    EXPECT_EQ(r.den(), 4);
}

TEST(Rational, ZeroNumeratorNormalisesToZeroOverOne)
{
    Rational const r(0, 7);
    EXPECT_EQ(r.num(), 0);
    EXPECT_EQ(r.den(), 1);
}

TEST(Rational, LargeGCD)
{
    Rational const r(100, 75);
    EXPECT_EQ(r.num(), 4);
    EXPECT_EQ(r.den(), 3);
}

// --- Accessors ---

TEST(Rational, IsZero)
{
    EXPECT_TRUE(Rational{}.is_zero());
    EXPECT_TRUE(Rational{0}.is_zero());
    EXPECT_FALSE(Rational{1}.is_zero());
    Rational const r(1, 3);
    EXPECT_FALSE(r.is_zero());
}

TEST(Rational, IsInteger)
{
    EXPECT_TRUE(Rational{}.is_integer());
    EXPECT_TRUE(Rational{5}.is_integer());
    Rational const third(1, 3);
    EXPECT_FALSE(third.is_integer());
    Rational const two_thirds(4, 6); // normalises to 2/3
    EXPECT_FALSE(two_thirds.is_integer());
}

// --- Conversions ---

TEST(Rational, ToDouble)
{
    EXPECT_DOUBLE_EQ(Rational(1, 2).to_double(), 0.5);
    EXPECT_DOUBLE_EQ(Rational(1, 3).to_double(), 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(Rational(-3, 4).to_double(), -0.75);
    EXPECT_DOUBLE_EQ(Rational{0}.to_double(), 0.0);
}

TEST(Rational, ToString)
{
    EXPECT_EQ(Rational{3}.to_string(), "3");
    EXPECT_EQ(Rational{-5}.to_string(), "-5");
    EXPECT_EQ(Rational(1, 3).to_string(), "1/3");
    EXPECT_EQ(Rational(-2, 5).to_string(), "-2/5");
    EXPECT_EQ(Rational(4, 6).to_string(), "2/3"); // normalised
}

// --- Unary negation ---

TEST(Rational, UnaryNegation)
{
    EXPECT_EQ(-Rational{3}, Rational{-3});
    EXPECT_EQ(-Rational{-3}, Rational{3});
    EXPECT_EQ(-Rational(1, 3), Rational(-1, 3));
    EXPECT_EQ(-Rational{0}, Rational{0});
}

// --- Addition ---

TEST(Rational, AdditionSameDenominator)
{
    EXPECT_EQ(Rational(1, 5) + Rational(2, 5), Rational(3, 5));
}

TEST(Rational, AdditionDifferentDenominators)
{
    EXPECT_EQ(Rational(1, 2) + Rational(1, 3), Rational(5, 6));
}

TEST(Rational, AdditionReduces)
{
    EXPECT_EQ(Rational(1, 6) + Rational(1, 6), Rational(1, 3));
}

TEST(Rational, AdditionWithInteger)
{
    EXPECT_EQ(Rational(1, 3) + Rational{2}, Rational(7, 3));
}

TEST(Rational, AdditionNegative)
{
    EXPECT_EQ(Rational(1, 2) + Rational(-1, 2), Rational{0});
}

// --- Subtraction ---

TEST(Rational, Subtraction)
{
    EXPECT_EQ(Rational(3, 4) - Rational(1, 4), Rational(1, 2));
    EXPECT_EQ(Rational(1, 2) - Rational(1, 3), Rational(1, 6));
    EXPECT_EQ(Rational{1} - Rational{1}, Rational{0});
}

// --- Multiplication ---

TEST(Rational, Multiplication)
{
    EXPECT_EQ(Rational(2, 3) * Rational(3, 4), Rational(1, 2));
    EXPECT_EQ(Rational(1, 2) * Rational{2}, Rational{1});
    EXPECT_EQ(Rational(-1, 3) * Rational{3}, Rational{-1});
    EXPECT_EQ(Rational{0} * Rational(5, 7), Rational{0});
}

// --- Division ---

TEST(Rational, Division)
{
    EXPECT_EQ(Rational(1, 2) / Rational(1, 3), Rational(3, 2));
    EXPECT_EQ(Rational(2, 3) / Rational{4}, Rational(1, 6));
    EXPECT_EQ(Rational(-1, 2) / Rational(1, 2), Rational{-1});
}

// --- Compound assignment ---

TEST(Rational, CompoundAssignment)
{
    Rational r(1, 2);
    r += Rational(1, 2);
    EXPECT_EQ(r, Rational{1});
    r -= Rational(1, 3);
    EXPECT_EQ(r, Rational(2, 3));
    r *= Rational{3};
    EXPECT_EQ(r, Rational{2});
    r /= Rational{4};
    EXPECT_EQ(r, Rational(1, 2));
}

// --- Comparison ---

TEST(Rational, Equality)
{
    EXPECT_EQ(Rational(1, 2), Rational(2, 4));
    EXPECT_EQ(Rational{3}, Rational(6, 2));
    EXPECT_NE(Rational(1, 2), Rational(1, 3));
}

TEST(Rational, Ordering)
{
    EXPECT_LT(Rational(1, 3), Rational(1, 2));
    EXPECT_GT(Rational(2, 3), Rational(1, 2));
    EXPECT_LE(Rational(1, 2), Rational(1, 2));
    EXPECT_GE(Rational(3, 4), Rational(1, 2));
    EXPECT_LT(Rational{-1}, Rational{0});
    EXPECT_GT(Rational{0}, Rational(-1, 3));
}

TEST(Rational, OrderingAcrossDenominators)
{
    // 5/7 vs 3/4: cross-products 5*4=20 vs 3*7=21  →  5/7 < 3/4
    EXPECT_LT(Rational(5, 7), Rational(3, 4));
}

// --- GCD edge cases ---

TEST(Rational, GcdWithOne)
{
    Rational const r(7, 1);
    EXPECT_EQ(r.num(), 7);
    EXPECT_EQ(r.den(), 1);
}

TEST(Rational, GcdLargeValues)
{
    // 1000000000 / 999999999 is already in lowest terms
    Rational const r(1'000'000'000LL, 999'999'999LL);
    EXPECT_EQ(r.num(), 1'000'000'000LL);
    EXPECT_EQ(r.den(), 999'999'999LL);
}

// --- Fatal errors (overflow / bad input) ---

TEST(Rational, ZeroDenominatorAborts)
{
    EXPECT_DEATH(Rational(1, 0), "zero denominator");
}

TEST(Rational, DivisionByZeroAborts)
{
    EXPECT_DEATH(
        {
            auto r = Rational{1} / Rational{0};
            (void)r;
        },
        "division by zero");
}

TEST(Rational, NegationOfMinInt64Aborts)
{
    EXPECT_DEATH(
        {
            auto r = -Rational{INT64_MIN};
            (void)r;
        },
        "overflow");
}

TEST(Rational, AdditionOverflowAborts)
{
    EXPECT_DEATH(
        {
            auto r = Rational{INT64_MAX} + Rational{1};
            (void)r;
        },
        "overflow");
}

TEST(Rational, MultiplicationOverflowAborts)
{
    EXPECT_DEATH(
        {
            auto r = Rational{INT64_MAX / 2 + 1} * Rational{2};
            (void)r;
        },
        "overflow");
}
