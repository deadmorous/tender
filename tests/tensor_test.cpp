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

static auto vec(ResourceList& rl, char const* sym) -> Expr*
{
    return make_named_tensor(rl, sym, 1, {});
}

static auto mat(ResourceList& rl, char const* sym) -> Expr*
{
    return make_named_tensor(rl, sym, 2, {});
}

// ===========================================================================
// IdentityTensor
// ===========================================================================

TEST(IdentityTensor, Rank)
{
    auto rl = make_rl();
    EXPECT_EQ(make_identity(rl)->rank(), 2);
}

TEST(IdentityTensor, Latex)
{
    auto rl = make_rl();
    EXPECT_EQ(make_identity(rl)->latex(), "\\mathbf{I}");
}

TEST(IdentityTensor, Python)
{
    auto rl = make_rl();
    EXPECT_EQ(make_identity(rl)->python(), "I");
}

// ===========================================================================
// LeviCivitaTensor
// ===========================================================================

TEST(LeviCivitaTensor, Rank)
{
    auto rl = make_rl();
    EXPECT_EQ(make_levi_civita(rl)->rank(), 3);
}

TEST(LeviCivitaTensor, Latex)
{
    auto rl = make_rl();
    EXPECT_EQ(make_levi_civita(rl)->latex(), "\\boldsymbol{\\varepsilon}");
}

TEST(LeviCivitaTensor, Python)
{
    auto rl = make_rl();
    EXPECT_EQ(make_levi_civita(rl)->python(), "eps");
}

// ===========================================================================
// Trace
// ===========================================================================

TEST(Trace, RankIsZero)
{
    auto rl = make_rl();
    EXPECT_EQ(make_trace(rl, mat(rl, "A"))->rank(), 0);
}

TEST(Trace, Latex)
{
    auto rl = make_rl();
    EXPECT_EQ(make_trace(rl, mat(rl, "A"))->latex(), "\\mathrm{tr}(A)");
}

TEST(Trace, Python)
{
    auto rl = make_rl();
    EXPECT_EQ(make_trace(rl, mat(rl, "A"))->python(), "trace(tensor('A', 2))");
}

TEST(Trace, NonRank2Throws)
{
    auto rl = make_rl();
    EXPECT_THROW(make_trace(rl, vec(rl, "v")), std::invalid_argument);
}

// ===========================================================================
// Contract
// ===========================================================================

TEST(Contract, RankFormula)
{
    auto rl = make_rl();
    // rank-3 contracted with rank-2 → rank 3
    auto* T = make_named_tensor(rl, "T", 3, {});
    auto* e = make_contract(rl, T, mat(rl, "A"));
    EXPECT_EQ(e->rank(), 3);
}

TEST(Contract, IdentityLeftSimplification)
{
    // Exit criterion: I · v → v
    auto rl = make_rl();
    auto* v = vec(rl, "v");
    auto* e = make_contract(rl, make_identity(rl), v);
    EXPECT_EQ(e, v);
}

TEST(Contract, IdentityRightSimplification)
{
    auto rl = make_rl();
    auto* v = vec(rl, "v");
    auto* e = make_contract(rl, v, make_identity(rl));
    EXPECT_EQ(e, v);
}

TEST(Contract, IdentityLeftMatrixSimplification)
{
    auto rl = make_rl();
    auto* A = mat(rl, "A");
    auto* e = make_contract(rl, make_identity(rl), A);
    EXPECT_EQ(e, A);
}

TEST(Contract, RankTooLowThrows)
{
    auto rl = make_rl();
    auto* s = make_symbolic_var(rl, "s"); // rank 0
    auto* v = vec(rl, "v");
    EXPECT_THROW(make_contract(rl, s, v), std::invalid_argument);
    EXPECT_THROW(make_contract(rl, v, s), std::invalid_argument);
}

TEST(Contract, Latex)
{
    auto rl = make_rl();
    auto* e = make_contract(rl, mat(rl, "A"), vec(rl, "v"));
    EXPECT_EQ(e->latex(), "A \\cdot v");
}

TEST(Contract, LatexSumOperandGetsParens)
{
    auto rl = make_rl();
    auto* v = vec(rl, "v");
    auto* w = vec(rl, "w");
    auto* u = vec(rl, "u");
    auto* s = make_sum(rl, {v, w});
    auto* e = make_contract(rl, s, u);
    EXPECT_EQ(e->latex(), "(v + w) \\cdot u");
}

TEST(Contract, Python)
{
    auto rl = make_rl();
    auto* e = make_contract(rl, vec(rl, "v"), vec(rl, "w"));
    EXPECT_EQ(e->python(), "dot(tensor('v', 1), tensor('w', 1))");
}

TEST(Contract, EpsContractedWithVectorRankTwo)
{
    // Exit criterion: eps (rank 3) contractable.
    auto rl = make_rl();
    auto* e = make_contract(rl, make_levi_civita(rl), vec(rl, "v"));
    EXPECT_EQ(e->rank(), 2); // 3 + 1 - 2 = 2
}

// ===========================================================================
// DoubleContract
// ===========================================================================

TEST(DoubleContract, RankFormula)
{
    auto rl = make_rl();
    auto* e = make_double_contract(rl, mat(rl, "A"), mat(rl, "B"));
    EXPECT_EQ(e->rank(), 0); // 2 + 2 - 4 = 0
}

TEST(DoubleContract, IdentitySimplifiesToTrace)
{
    auto rl = make_rl();
    auto* A = mat(rl, "A");
    auto* e = make_double_contract(rl, make_identity(rl), A);
    auto* tr = dynamic_cast<Trace*>(e);
    ASSERT_NE(tr, nullptr);
    EXPECT_EQ(tr->arg(), A);
}

TEST(DoubleContract, IdentityHigherRankNoSimplification)
{
    // I : T4 (rank 4) → DoubleContract node, not Trace (rank 2 result)
    auto rl = make_rl();
    auto* T4 = make_named_tensor(rl, "T", 4, {});
    auto* e = make_double_contract(rl, make_identity(rl), T4);
    EXPECT_NE(dynamic_cast<DoubleContract*>(e), nullptr);
    EXPECT_EQ(e->rank(), 2);
}

TEST(DoubleContract, RankTooLowThrows)
{
    auto rl = make_rl();
    EXPECT_THROW(
        make_double_contract(rl, vec(rl, "v"), mat(rl, "A")),
        std::invalid_argument);
    EXPECT_THROW(
        make_double_contract(rl, mat(rl, "A"), vec(rl, "v")),
        std::invalid_argument);
}

TEST(DoubleContract, Latex)
{
    auto rl = make_rl();
    auto* e = make_double_contract(rl, mat(rl, "A"), mat(rl, "B"));
    EXPECT_EQ(e->latex(), "A : B");
}

TEST(DoubleContract, Python)
{
    auto rl = make_rl();
    auto* e = make_double_contract(rl, mat(rl, "A"), mat(rl, "B"));
    EXPECT_EQ(e->python(), "ddot(tensor('A', 2), tensor('B', 2))");
}

// ===========================================================================
// DoubleContractReversed
// ===========================================================================

TEST(DoubleContractReversed, RankFormula)
{
    auto rl = make_rl();
    auto* e = make_double_contract_reversed(rl, mat(rl, "A"), mat(rl, "B"));
    EXPECT_EQ(e->rank(), 0);
}

TEST(DoubleContractReversed, RankTooLowThrows)
{
    auto rl = make_rl();
    EXPECT_THROW(
        make_double_contract_reversed(rl, vec(rl, "v"), mat(rl, "A")),
        std::invalid_argument);
}

TEST(DoubleContractReversed, Latex)
{
    auto rl = make_rl();
    auto* e = make_double_contract_reversed(rl, mat(rl, "A"), mat(rl, "B"));
    EXPECT_EQ(e->latex(), "A \\cdot\\!\\cdot B");
}

TEST(DoubleContractReversed, Python)
{
    auto rl = make_rl();
    auto* e = make_double_contract_reversed(rl, mat(rl, "A"), mat(rl, "B"));
    EXPECT_EQ(e->python(), "ddot_rev(tensor('A', 2), tensor('B', 2))");
}

// ===========================================================================
// CrossProduct
// ===========================================================================

TEST(CrossProduct, RankVectorTimesVector)
{
    auto rl = make_rl();
    auto* e = make_cross_product(rl, vec(rl, "v"), vec(rl, "w"));
    EXPECT_EQ(e->rank(), 1); // 1 + 1 - 1 = 1
}

TEST(CrossProduct, RankMatrixTimesVector)
{
    auto rl = make_rl();
    auto* e = make_cross_product(rl, mat(rl, "A"), vec(rl, "v"));
    EXPECT_EQ(e->rank(), 2); // 2 + 1 - 1 = 2
}

TEST(CrossProduct, Latex)
{
    auto rl = make_rl();
    auto* e = make_cross_product(rl, vec(rl, "v"), vec(rl, "w"));
    EXPECT_EQ(e->latex(), "v \\times w");
}

TEST(CrossProduct, Python)
{
    auto rl = make_rl();
    auto* e = make_cross_product(rl, vec(rl, "v"), vec(rl, "w"));
    EXPECT_EQ(e->python(), "cross(tensor('v', 1), tensor('w', 1))");
}

TEST(CrossProduct, ChainingLhsThrows)
{
    // Exit criterion: cross product raises error on chaining.
    auto rl = make_rl();
    auto* a = vec(rl, "a");
    auto* b = vec(rl, "b");
    auto* c = vec(rl, "c");
    auto* ab = make_cross_product(rl, a, b);
    EXPECT_THROW(make_cross_product(rl, ab, c), std::invalid_argument);
}

TEST(CrossProduct, ChainingRhsThrows)
{
    auto rl = make_rl();
    auto* a = vec(rl, "a");
    auto* b = vec(rl, "b");
    auto* c = vec(rl, "c");
    auto* bc = make_cross_product(rl, b, c);
    EXPECT_THROW(make_cross_product(rl, a, bc), std::invalid_argument);
}

TEST(CrossProduct, RankTooLowThrows)
{
    auto rl = make_rl();
    auto* s = make_symbolic_var(rl, "s"); // rank 0
    auto* v = vec(rl, "v");
    EXPECT_THROW(make_cross_product(rl, s, v), std::invalid_argument);
    EXPECT_THROW(make_cross_product(rl, v, s), std::invalid_argument);
}

TEST(CrossProduct, LatexNestedRhsParenthesised)
{
    // a×(b×c) must render as "a \times (b \times c)", not "a \times b \times
    // c". Nested CrossProduct is valid in patterns; bypass the guard to build
    // it.
    auto rl = make_rl();
    auto* a = vec(rl, "a");
    auto* b = vec(rl, "b");
    auto* c = vec(rl, "c");
    auto* bc = rl.make<CrossProduct>(b, c);
    auto* abc = rl.make<CrossProduct>(a, bc);
    EXPECT_EQ(abc->latex(), "a \\times (b \\times c)");
}

TEST(CrossProduct, LatexNestedLhsParenthesised)
{
    auto rl = make_rl();
    auto* a = vec(rl, "a");
    auto* b = vec(rl, "b");
    auto* c = vec(rl, "c");
    auto* ab = rl.make<CrossProduct>(a, b);
    auto* abc = rl.make<CrossProduct>(ab, c);
    EXPECT_EQ(abc->latex(), "(a \\times b) \\times c");
}

// ===========================================================================
// TensorProduct zero propagation (Phase 4 extension to existing factory)
// ===========================================================================

TEST(ZeroPropagation, TensorProductLhsZero)
{
    auto rl = make_rl();
    auto* z = make_rational(rl, Rational{0});
    auto* v = vec(rl, "v");
    auto* e = make_tensor_product(rl, z, v);
    auto* rc = dynamic_cast<RationalConst*>(e);
    ASSERT_NE(rc, nullptr);
    EXPECT_TRUE(rc->value().is_zero());
}

TEST(ZeroPropagation, TensorProductRhsZero)
{
    auto rl = make_rl();
    auto* v = vec(rl, "v");
    auto* z = make_rational(rl, Rational{0});
    auto* e = make_tensor_product(rl, v, z);
    auto* rc = dynamic_cast<RationalConst*>(e);
    ASSERT_NE(rc, nullptr);
    EXPECT_TRUE(rc->value().is_zero());
}
