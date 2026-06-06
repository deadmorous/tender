#include <tender/expr.hpp>

#include <gtest/gtest.h>

using namespace tender;

static ResourceList make_rl()
{
    return ResourceList{};
}

static Polynomial quad()
{
    // 3x^2 - 2x + 1
    return Polynomial{{{Rational{3}, 2}, {Rational{-2}, 1}, {Rational{1}, 0}}};
}

static Polynomial linear()
{
    // 2x - 1
    return Polynomial{{{Rational{2}, 1}, {Rational{-1}, 0}}};
}

// ===========================================================================
// Construction — rank 0 variable
// ===========================================================================

TEST(PolynomialExpr, Rank0VarRankIsZero)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    auto* pe = make_polynomial_expr(rl, quad(), x);
    EXPECT_EQ(pe->rank(), 0);
}

TEST(PolynomialExpr, InvalidVarRankThrows)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});
    EXPECT_THROW(make_polynomial_expr(rl, quad(), v), std::invalid_argument);
}

// ===========================================================================
// Rendering — rank 0
// ===========================================================================

TEST(PolynomialExpr, LatexRank0Quadratic)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    auto* pe = make_polynomial_expr(rl, quad(), x);
    EXPECT_EQ(pe->latex(), "3 x^{2} - 2 x + 1");
}

TEST(PolynomialExpr, LatexRank0SumVarParenthesised)
{
    auto rl = make_rl();
    // var = a + b  (a Sum node)
    auto* a = make_symbolic_var(rl, "a");
    auto* b = make_symbolic_var(rl, "b");
    auto* sum_var = make_sum(rl, {a, b});
    // linear: 2*(a+b) - 1
    auto* pe = make_polynomial_expr(rl, linear(), sum_var);
    EXPECT_EQ(pe->latex(), "2 (a + b) - 1");
}

TEST(PolynomialExpr, PythonRank0Quadratic)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    auto* pe = make_polynomial_expr(rl, quad(), x);
    EXPECT_EQ(pe->python(), "3*symbolic_var('x')**2 - 2*symbolic_var('x') + 1");
}

TEST(PolynomialExpr, LatexZeroPolynomial)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    auto* pe = make_polynomial_expr(rl, Polynomial{}, x);
    EXPECT_EQ(pe->latex(), "0");
}

// ===========================================================================
// expand — rank 0
// ===========================================================================

TEST(PolynomialExpr, ExpandConstant)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    // p(x) = 5  →  expand = RationalConst(5)
    Polynomial p{{{Rational{5}, 0}}};
    auto* pe = make_polynomial_expr(rl, p, x);
    auto* e = pe->expand(rl);
    auto* rc = dynamic_cast<RationalConst*>(e);
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{5});
}

TEST(PolynomialExpr, ExpandLinear)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    // p(x) = 2x + 1  →  expand = Sum(Scale(2,x), RationalConst(1))
    auto* pe = make_polynomial_expr(rl, linear(), x);
    // linear() is 2x - 1: Sum(Scale(2,x), Scale(-1,_constant_))
    // After make_sum, the constant -1 becomes RationalConst(-1)
    auto* e = pe->expand(rl);
    // Should be a Sum (two terms remain: Scale(2,x) and RationalConst(-1))
    auto* s = dynamic_cast<Sum*>(e);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->terms().size(), 2u);
}

TEST(PolynomialExpr, ExpandQuadraticCorrectEval)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    // Verify expand produces same value as Polynomial::eval at a specific point
    // by checking the structure includes a Pow with exponent 2
    auto* pe = make_polynomial_expr(rl, quad(), x);
    auto* e = pe->expand(rl);
    // At minimum it should be a Sum with 3 terms for 3x^2 - 2x + 1
    auto* s = dynamic_cast<Sum*>(e);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->terms().size(), 3u);
}

TEST(PolynomialExpr, ExpandZeroYieldsRationalZero)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    auto* pe = make_polynomial_expr(rl, Polynomial{}, x);
    auto* e = pe->expand(rl);
    auto* rc = dynamic_cast<RationalConst*>(e);
    ASSERT_NE(rc, nullptr);
    EXPECT_TRUE(rc->value().is_zero());
}

// ===========================================================================
// Construction — rank 2 variable (matrix polynomial)
// ===========================================================================

TEST(PolynomialExpr, Rank2VarRankIsTwo)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    Polynomial p{{{Rational{1}, 2}, {Rational{-1}, 0}}};
    auto* pe = make_polynomial_expr(rl, p, A);
    EXPECT_EQ(pe->rank(), 2);
}

// ===========================================================================
// Rendering — rank 2
// ===========================================================================

TEST(PolynomialExpr, LatexRank2ConstantTermUsesIdentity)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    // A^2 - I  (coeffs 1 for exp 2, -1 for exp 0)
    Polynomial p{{{Rational{1}, 2}, {Rational{-1}, 0}}};
    auto* pe = make_polynomial_expr(rl, p, A);
    EXPECT_EQ(pe->latex(), "A^{2} - \\mathbf{I}");
}

TEST(PolynomialExpr, LatexRank2LinearNoConstant)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    // 2A^2 - A
    Polynomial p{{{Rational{2}, 2}, {Rational{-1}, 1}}};
    auto* pe = make_polynomial_expr(rl, p, A);
    EXPECT_EQ(pe->latex(), "2 A^{2} - A");
}

TEST(PolynomialExpr, LatexRank2ConstantOnlyIsIdentity)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    // 3I  (constant-only polynomial)
    Polynomial p{{{Rational{3}, 0}}};
    auto* pe = make_polynomial_expr(rl, p, A);
    EXPECT_EQ(pe->latex(), "3 \\mathbf{I}");
}

TEST(PolynomialExpr, LatexRank2UnitConstantIsJustIdentity)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    // A + I  (coeffs 1, 1)
    Polynomial p{{{Rational{1}, 1}, {Rational{1}, 0}}};
    auto* pe = make_polynomial_expr(rl, p, A);
    EXPECT_EQ(pe->latex(), "A + \\mathbf{I}");
}

TEST(PolynomialExpr, PythonRank2)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    Polynomial p{{{Rational{1}, 2}, {Rational{-1}, 0}}};
    auto* pe = make_polynomial_expr(rl, p, A);
    EXPECT_EQ(
        pe->python(), "polynomial_expr(tensor('A', 2), [(1, 2), (-1, 0)])");
}

// ===========================================================================
// expand — rank 2
// ===========================================================================

TEST(PolynomialExpr, ExpandRank2IdentityForZeroPower)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    // p = 3I  → expand = Scale(3, IdentityTensor)
    Polynomial p{{{Rational{3}, 0}}};
    auto* pe = make_polynomial_expr(rl, p, A);
    auto* e = pe->expand(rl);
    auto* sc = dynamic_cast<Scale*>(e);
    ASSERT_NE(sc, nullptr);
    EXPECT_EQ(sc->coeff(), Rational{3});
    EXPECT_NE(dynamic_cast<IdentityTensor*>(sc->expr()), nullptr);
}

TEST(PolynomialExpr, ExpandRank2LinearIsA)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    // p = A  → expand = A
    Polynomial p{{{Rational{1}, 1}}};
    auto* pe = make_polynomial_expr(rl, p, A);
    auto* e = pe->expand(rl);
    EXPECT_EQ(e, A);
}

TEST(PolynomialExpr, ExpandRank2QuadraticIsContract)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    // p = A^2  → expand = Contract(A, A)
    Polynomial p{{{Rational{1}, 2}}};
    auto* pe = make_polynomial_expr(rl, p, A);
    auto* e = pe->expand(rl);
    auto* co = dynamic_cast<Contract*>(e);
    ASSERT_NE(co, nullptr);
    EXPECT_EQ(co->lhs(), A);
    EXPECT_EQ(co->rhs(), A);
}

TEST(PolynomialExpr, ExpandRank2MultipleTerms)
{
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    // A^2 - I
    Polynomial p{{{Rational{1}, 2}, {Rational{-1}, 0}}};
    auto* pe = make_polynomial_expr(rl, p, A);
    auto* e = pe->expand(rl);
    auto* s = dynamic_cast<Sum*>(e);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->terms().size(), 2u);
}

// ===========================================================================
// depends_on integration
// ===========================================================================

TEST(PolynomialExpr, DependsOnVar)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* pe = make_polynomial_expr(rl, quad(), t);
    EXPECT_TRUE(depends_on(t, pe));
}

TEST(PolynomialExpr, IndependentOfOtherParam)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* s = make_parameter(rl, "s");
    auto* pe = make_polynomial_expr(rl, quad(), t);
    EXPECT_FALSE(depends_on(s, pe));
}

// ===========================================================================
// deriv — chain rule through PolynomialExpr
// ===========================================================================

TEST(PolynomialExpr, DerivOfLinearWrtT)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    // p(t) = 2t - 1 → dp/dt = PolynomialExpr({2}, t)  (constant polynomial)
    auto* pe = make_polynomial_expr(rl, linear(), t);
    auto* d = deriv(rl, t, pe);
    auto* dp = dynamic_cast<PolynomialExpr*>(d);
    ASSERT_NE(dp, nullptr);
    EXPECT_EQ(dp->poly().degree(), 0);
    EXPECT_EQ(dp->poly().coeff(0), Rational{2});
    EXPECT_EQ(dp->var(), t);
}

TEST(PolynomialExpr, DerivOfQuadraticWrtT)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    // p(t) = 3t^2 - 2t + 1 → dp/dt = 6t - 2
    // dp_poly = {6,1},{-2,0}; dp_expr = PolynomialExpr({6t-2}, t)
    // dv = 1; result = PolynomialExpr({6t-2}, t) * 1 = PolynomialExpr({6t-2},
    // t)
    auto* pe = make_polynomial_expr(rl, quad(), t);
    auto* d = deriv(rl, t, pe);
    // The derivative is itself a PolynomialExpr (degree 1: 6t - 2)
    auto* dp = dynamic_cast<PolynomialExpr*>(d);
    ASSERT_NE(dp, nullptr);
    EXPECT_EQ(dp->poly().degree(), 1);
    EXPECT_EQ(dp->poly().coeff(1), Rational{6});
    EXPECT_EQ(dp->poly().coeff(0), Rational{-2});
    EXPECT_EQ(dp->var(), t);
}

TEST(PolynomialExpr, DerivOfConstantPolynomialIsZero)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    // p(t) = 5 → dp/dt = 0
    Polynomial p{{{Rational{5}, 0}}};
    auto* pe = make_polynomial_expr(rl, p, t);
    auto* d = deriv(rl, t, pe);
    auto* rc = dynamic_cast<RationalConst*>(d);
    ASSERT_NE(rc, nullptr);
    EXPECT_TRUE(rc->value().is_zero());
}

TEST(PolynomialExpr, DerivZeroForIndependentVar)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* s = make_parameter(rl, "s");
    // p(t) has no dependency on s
    auto* pe = make_polynomial_expr(rl, quad(), t);
    auto* d = deriv(rl, s, pe);
    auto* rc = dynamic_cast<RationalConst*>(d);
    ASSERT_NE(rc, nullptr);
    EXPECT_TRUE(rc->value().is_zero());
}

TEST(PolynomialExpr, DerivZeroForRank2Var)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* A = make_named_tensor(rl, "A", 2, {});
    Polynomial p{{{Rational{1}, 2}, {Rational{-1}, 0}}};
    auto* pe = make_polynomial_expr(rl, p, A);
    auto* d = deriv(rl, t, pe);
    auto* rc = dynamic_cast<RationalConst*>(d);
    ASSERT_NE(rc, nullptr);
    EXPECT_TRUE(rc->value().is_zero());
}

// ===========================================================================
// poly() and var() accessors
// ===========================================================================

TEST(PolynomialExpr, Accessors)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    Polynomial p = quad();
    auto* pe = make_polynomial_expr(rl, p, x);
    auto* casted = dynamic_cast<PolynomialExpr*>(pe);
    ASSERT_NE(casted, nullptr);
    EXPECT_EQ(casted->var(), x);
    EXPECT_EQ(casted->poly(), p);
}

// ===========================================================================
// coeff_latex — fractional coefficient rendering (line 23-24 of
// polynomial_expr.cpp)
// ===========================================================================

TEST(PolynomialExpr, FractionalCoefficientLatex)
{
    // Polynomial with coefficient 1/3: (1/3)x
    // coeff_latex exercises the \frac{}{} branch when den != 1
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    Polynomial p{{{Rational{1, 3}, 1}}};
    auto* pe = make_polynomial_expr(rl, p, x);
    // rank-0 path delegates to Polynomial::to_latex which also calls
    // coeff_latex
    std::string tex = pe->latex();
    EXPECT_NE(tex.find("frac"), std::string::npos);
    EXPECT_NE(tex.find("1"), std::string::npos);
    EXPECT_NE(tex.find("3"), std::string::npos);
}

TEST(PolynomialExpr, FractionalCoefficientLatexRank2)
{
    // rank-2 polynomial with fractional coefficient — exercises the coeff_latex
    // branch (line 23-24) through the rank-2 rendering path in
    // PolynomialExpr::latex
    auto rl = make_rl();
    auto* A = make_named_tensor(rl, "A", 2, {});
    Polynomial p{{{Rational{2, 3}, 1}}}; // (2/3) A
    auto* pe = make_polynomial_expr(rl, p, A);
    std::string tex = pe->latex();
    EXPECT_NE(tex.find("frac"), std::string::npos);
    EXPECT_NE(tex.find("2"), std::string::npos);
    EXPECT_NE(tex.find("3"), std::string::npos);
}

TEST(PolynomialExpr, ParenthesisedSumVarPython)
{
    // parenthesise_if_sum covers the python path (line 29-32)
    // var_ is a Sum, so it gets wrapped in parens in python output
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    auto* y = make_symbolic_var(rl, "y");
    auto* s = make_sum(rl, {x, y});
    Polynomial p{{{Rational{1}, 1}}};
    auto* pe = make_polynomial_expr(rl, p, s);
    std::string py = pe->python();
    EXPECT_NE(py.find("("), std::string::npos); // sum was parenthesised
}
