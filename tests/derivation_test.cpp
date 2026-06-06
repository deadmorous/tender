#include <tender/derivation.hpp>

#include <gtest/gtest.h>

using namespace tender;

static ResourceList make_rl()
{
    return ResourceList{};
}

// ===========================================================================
// State
// ===========================================================================

TEST(State, LatexDelegates)
{
    auto rl = make_rl();
    auto* x = make_symbolic_var(rl, "x");
    State s{x};
    EXPECT_EQ(s.latex(), "x");
}

TEST(State, LabelDefaultEmpty)
{
    auto rl = make_rl();
    State s{make_rational(rl, Rational{1})};
    EXPECT_TRUE(s.label().empty());
}

TEST(State, LabelSet)
{
    auto rl = make_rl();
    State s{make_rational(rl, Rational{1}), "my step"};
    EXPECT_EQ(s.label(), "my step");
}

TEST(State, ExprAccessor)
{
    auto rl = make_rl();
    auto* e = make_rational(rl, Rational{3});
    State s{e};
    EXPECT_EQ(s.expr(), e);
}

// ===========================================================================
// DerivationStep
// ===========================================================================

TEST(DerivationStep, NameAccessor)
{
    auto* step = new DerivationStep{
        "double",
        [](ResourceList& rl, Expr* e) -> Expr*
        {
            return make_scale(rl, Rational{2}, e);
        }};
    EXPECT_EQ(step->name(), "double");
    delete step;
}

TEST(DerivationStep, Apply)
{
    auto rl = make_rl();
    auto* two = make_rational(rl, Rational{2});
    State s{two};

    DerivationStep step{
        "negate",
        [](ResourceList& rl2, Expr* e) -> Expr*
        {
            return make_scale(rl2, Rational{-1}, e);
        }};

    auto result = step.apply(rl, s);
    EXPECT_EQ(result.label(), "negate");
    auto* rc = dynamic_cast<RationalConst*>(result.expr());
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{-2});
}

// ===========================================================================
// Derivation — apply
// ===========================================================================

TEST(Derivation, ApplyEmptyReturnsJustInitial)
{
    auto rl = make_rl();
    auto* e = make_rational(rl, Rational{1});
    State initial{e};
    Derivation d{{}};
    auto history = d.apply(rl, initial);
    ASSERT_EQ(history.size(), 1u);
    EXPECT_EQ(history[0].expr(), e);
}

TEST(Derivation, ApplyOneStep)
{
    auto rl = make_rl();
    auto* one = make_rational(rl, Rational{1});
    State initial{one};
    Derivation d{{DerivationStep{
        "times3",
        [](ResourceList& rl2, Expr* e) -> Expr*
        {
            return make_scale(rl2, Rational{3}, e);
        }}}};
    auto history = d.apply(rl, initial);
    ASSERT_EQ(history.size(), 2u);
    EXPECT_EQ(history[0].expr(), one);
    auto* rc = dynamic_cast<RationalConst*>(history[1].expr());
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{3});
    EXPECT_EQ(history[1].label(), "times3");
}

TEST(Derivation, ApplyTwoStepsChained)
{
    // Exit criterion: two-step derivation, correct State list,
    // renders step-by-step.
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    // p(t) = t^2
    auto* pe = make_polynomial_expr(rl, Polynomial{{{Rational{1}, 2}}}, t);
    State initial{pe};

    Derivation d{{diff_step(t), diff_step(t)}};
    auto history = d.apply(rl, initial);

    ASSERT_EQ(history.size(), 3u);

    // Initial: t^2
    EXPECT_TRUE(history[0].label().empty());
    EXPECT_NE(dynamic_cast<PolynomialExpr*>(history[0].expr()), nullptr);

    // After first diff(t): 2t — degree-1 polynomial
    auto* dp1 = dynamic_cast<PolynomialExpr*>(history[1].expr());
    ASSERT_NE(dp1, nullptr);
    EXPECT_EQ(dp1->poly().degree(), 1);
    EXPECT_EQ(dp1->poly().coeff(1), Rational{2});
    EXPECT_EQ(history[1].label(), "diff(t)");

    // After second diff(t): constant 2 — degree-0 polynomial
    auto* dp2 = dynamic_cast<PolynomialExpr*>(history[2].expr());
    ASSERT_NE(dp2, nullptr);
    EXPECT_EQ(dp2->poly().degree(), 0);
    EXPECT_EQ(dp2->poly().coeff(0), Rational{2});
    EXPECT_EQ(history[2].label(), "diff(t)");
}

TEST(Derivation, ApplyDiffSinT)
{
    // d/dt sin(t) = cos(t), one step
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* e = make_sin(rl, t);
    State initial{e};
    Derivation d{{diff_step(t)}};
    auto history = d.apply(rl, initial);
    ASSERT_EQ(history.size(), 2u);
    auto* fa = dynamic_cast<FunctionApply*>(history[1].expr());
    ASSERT_NE(fa, nullptr);
    EXPECT_EQ(fa->kind(), FunctionKind::Cos);
}

// ===========================================================================
// Derivation — concatenation
// ===========================================================================

TEST(Derivation, Concatenation)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* pe = make_polynomial_expr(rl, Polynomial{{{Rational{1}, 3}}}, t);
    State initial{pe};

    // Split diff(t) + diff(t) + diff(t) across two Derivations
    Derivation d1{{diff_step(t)}};
    Derivation d2{{diff_step(t), diff_step(t)}};
    auto d3 = d1 + d2;

    EXPECT_EQ(d3.steps().size(), 3u);
    auto history = d3.apply(rl, initial);
    ASSERT_EQ(history.size(), 4u);

    // d/dt t^3 = 3t^2, d²/dt² = 6t, d³/dt³ = 6
    auto* dp3 = dynamic_cast<PolynomialExpr*>(history[3].expr());
    ASSERT_NE(dp3, nullptr);
    EXPECT_EQ(dp3->poly().degree(), 0);
    EXPECT_EQ(dp3->poly().coeff(0), Rational{6});
}

// ===========================================================================
// show
// ===========================================================================

TEST(Show, FullHistoryContainsInitialTag)
{
    auto rl = make_rl();
    auto* e = make_symbolic_var(rl, "x");
    State initial{e};
    Derivation d{{diff_step(time_parameter())}};
    auto history = d.apply(rl, initial);
    auto s = show(history);
    EXPECT_NE(s.find("[initial]"), std::string::npos);
}

TEST(Show, FullHistoryContainsStepLabel)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* e = make_sin(rl, t);
    State initial{e};
    Derivation d{{diff_step(t)}};
    auto history = d.apply(rl, initial);
    auto s = show(history);
    EXPECT_NE(s.find("[diff(t)]"), std::string::npos);
}

TEST(Show, FullHistoryContainsLatex)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* e = make_sin(rl, t);
    State initial{e};
    auto history = Derivation{{diff_step(t)}}.apply(rl, initial);
    auto s = show(history);
    EXPECT_NE(s.find("\\sin"), std::string::npos);
    EXPECT_NE(s.find("\\cos"), std::string::npos);
}

TEST(Show, FinalOnlyReturnsFinalLatex)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* e = make_cos(rl, t);
    State initial{e};
    auto history = Derivation{{diff_step(t)}}.apply(rl, initial);
    auto final_s = show_final(history);
    // d/dt cos(t) = -sin(t)  → Scale(-1, sin(t))
    EXPECT_NE(final_s.find("\\sin"), std::string::npos);
    // Should not contain the step label
    EXPECT_EQ(final_s.find("["), std::string::npos);
}

TEST(Show, EmptyHistoryShowFinalEmpty)
{
    std::vector<State> history;
    EXPECT_EQ(show_final(history), "");
}

TEST(Show, TwoLinesForOneStep)
{
    auto rl = make_rl();
    auto* e = make_rational(rl, Rational{1});
    auto history =
        Derivation{{DerivationStep{
            "id",
            [](ResourceList&, Expr* x) -> Expr*
            {
                return x;
            }}}}
            .apply(rl, State{e});
    auto s = show(history);
    // Two lines separated by '\n'
    EXPECT_NE(s.find('\n'), std::string::npos);
}

// ===========================================================================
// expand_poly_step
// ===========================================================================

TEST(ExpandPolyStep, ExpandsPolynomialExpr)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    // p(t) = t^2 + 1
    Polynomial p{{{Rational{1}, 2}, {Rational{1}, 0}}};
    auto* pe = make_polynomial_expr(rl, p, t);
    State initial{pe};
    auto history = Derivation{{expand_poly_step()}}.apply(rl, initial);
    ASSERT_EQ(history.size(), 2u);
    // After expand: should be a Sum, not a PolynomialExpr
    EXPECT_EQ(dynamic_cast<PolynomialExpr*>(history[1].expr()), nullptr);
    EXPECT_NE(dynamic_cast<Sum*>(history[1].expr()), nullptr);
}

TEST(ExpandPolyStep, LeafExprPassthrough)
{
    auto rl = make_rl();
    auto* e = make_rational(rl, Rational{5});
    State initial{e};
    auto history = Derivation{{expand_poly_step()}}.apply(rl, initial);
    ASSERT_EQ(history.size(), 2u);
    EXPECT_EQ(history[1].expr(), e);
}

TEST(ExpandPolyStep, ExpandInsideScale)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    Polynomial p{{{Rational{1}, 2}}};
    auto* pe = make_polynomial_expr(rl, p, t);
    // 3 * pe
    auto* scaled = make_scale(rl, Rational{3}, pe);
    State initial{scaled};
    auto history = Derivation{{expand_poly_step()}}.apply(rl, initial);
    ASSERT_EQ(history.size(), 2u);
    // Should no longer contain PolynomialExpr
    auto* sc = dynamic_cast<Scale*>(history[1].expr());
    ASSERT_NE(sc, nullptr);
    EXPECT_EQ(dynamic_cast<PolynomialExpr*>(sc->expr()), nullptr);
}

// ===========================================================================
// named_step
// ===========================================================================

TEST(NamedStep, NameAndFunctionWork)
{
    auto rl = make_rl();
    auto step = named_step(
        "add_one",
        [](ResourceList& rl2, Expr* e) -> Expr*
        {
            return make_sum(rl2, {e, make_rational(rl2, Rational{1})});
        });
    EXPECT_EQ(step.name(), "add_one");
    auto* c = make_rational(rl, Rational{4});
    auto result = step.apply(rl, State{c});
    auto* rc = dynamic_cast<RationalConst*>(result.expr());
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{5});
}

// ===========================================================================
// expand_poly_step — additional node types (Sum, TensorProduct, Contract, Product)

// ===========================================================================
// expand_poly_step — additional node types
// ===========================================================================

TEST(ExpandPolyStep, ExpandInsideSum)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    Polynomial p{{{Rational{1}, 2}}};
    auto* pe = make_polynomial_expr(rl, p, t);
    auto* c = make_rational(rl, Rational{3});
    auto* expr = make_sum(rl, {pe, c});
    auto history = Derivation{{expand_poly_step()}}.apply(rl, State{expr});
    ASSERT_EQ(history.size(), 2u);
    // Sum is rebuilt; the PolynomialExpr term is expanded
    auto* s = dynamic_cast<Sum*>(history[1].expr());
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(dynamic_cast<PolynomialExpr*>(s->terms()[0]), nullptr);
}

TEST(ExpandPolyStep, ExpandInsideTensorProduct)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* v = make_named_tensor(rl, "v", 1, {});
    Polynomial p{{{Rational{1}, 1}}};  // p(t) = t
    auto* pe = make_polynomial_expr(rl, p, t);
    // tp(pe, v): pe is rank-0, v is rank-1 → rank-1 TensorProduct
    auto* expr = make_tensor_product(rl, pe, v);
    auto history = Derivation{{expand_poly_step()}}.apply(rl, State{expr});
    ASSERT_EQ(history.size(), 2u);
    auto* tp = dynamic_cast<TensorProduct*>(history[1].expr());
    ASSERT_NE(tp, nullptr);
    EXPECT_EQ(dynamic_cast<PolynomialExpr*>(tp->lhs()), nullptr);
}

TEST(ExpandPolyStep, ExpandInsideContract)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    auto* v = make_named_tensor(rl, "v", 1, {});
    Polynomial p{{{Rational{1}, 1}}};
    auto* pe = make_polynomial_expr(rl, p, t);
    // Contract(tp(pe, v), v): tp(pe,v) is rank-1, v is rank-1 → rank-0
    auto* tv = make_tensor_product(rl, pe, v);
    auto* expr = make_contract(rl, tv, v);
    auto history = Derivation{{expand_poly_step()}}.apply(rl, State{expr});
    ASSERT_EQ(history.size(), 2u);
    // Contract rebuilt; PolynomialExpr inside TensorProduct is expanded
    auto* co = dynamic_cast<Contract*>(history[1].expr());
    ASSERT_NE(co, nullptr);
}

TEST(ExpandPolyStep, ExpandInsideProduct)
{
    auto rl = make_rl();
    auto* t = make_parameter(rl, "t");
    Polynomial p{{{Rational{1}, 1}}};
    auto* pe = make_polynomial_expr(rl, p, t);
    auto* x = make_symbolic_var(rl, "x");
    // Product(pe, x): both rank-0, pe is not RationalConst → Product node
    auto* expr = make_product(rl, pe, x);
    auto history = Derivation{{expand_poly_step()}}.apply(rl, State{expr});
    ASSERT_EQ(history.size(), 2u);
    auto* pr = dynamic_cast<Product*>(history[1].expr());
    ASSERT_NE(pr, nullptr);
    EXPECT_EQ(dynamic_cast<PolynomialExpr*>(pr->lhs()), nullptr);
}
