#include <tender/basis.hpp>
#include <tender/coord_system.hpp>
#include <tender/derivation.hpp>
#include <tender/expr.hpp>

#include <gtest/gtest.h>

using namespace tender;

static ResourceList make_rl()
{
    return ResourceList{};
}

// ===========================================================================
// simplify_basis_dot_step
// ===========================================================================

TEST(SimplifyBasisDot, BasisSelfDot)
{
    // e_i · e_i (for WCS where cobasis == basis) → 1
    auto rl = make_rl();
    auto const& cs = wcs();
    for (int i = 0; i < cs.dim(); ++i)
    {
        auto* e_i = cs.basis(i);
        auto* expr = make_contract(rl, e_i, e_i);
        auto step = simplify_basis_dot_step(cs);
        auto* result = step.apply(rl, State{expr}).expr();
        auto* rc = dynamic_cast<RationalConst*>(result);
        ASSERT_NE(rc, nullptr);
        EXPECT_EQ(rc->value(), Rational{1});
    }
}

TEST(SimplifyBasisDot, BasisCrossDot)
{
    // e_i · e_j for i != j → 0
    auto rl = make_rl();
    auto const& cs = wcs();
    for (int i = 0; i < cs.dim(); ++i)
    {
        for (int j = 0; j < cs.dim(); ++j)
        {
            if (i == j)
                continue;
            auto* expr = make_contract(rl, cs.basis(i), cs.basis(j));
            auto step = simplify_basis_dot_step(cs);
            auto* result = step.apply(rl, State{expr}).expr();
            auto* rc = dynamic_cast<RationalConst*>(result);
            ASSERT_NE(rc, nullptr) << "i=" << i << " j=" << j;
            EXPECT_EQ(rc->value(), Rational{0}) << "i=" << i << " j=" << j;
        }
    }
}

TEST(SimplifyBasisDot, ScalarTimesVectorDot)
{
    // (s * e_i) · (t * e_i) → s * t
    auto rl = make_rl();
    auto const& cs = wcs();
    auto* s = make_named_tensor(rl, "s", 0, {});
    auto* t = make_named_tensor(rl, "t", 0, {});
    auto* e0 = cs.basis(0);

    auto* lhs = make_tensor_product(rl, s, e0); // s ⊗ e_1 (rank 1)
    auto* rhs = make_tensor_product(rl, t, e0); // t ⊗ e_1 (rank 1)
    auto* expr = make_contract(rl, lhs, rhs);   // (s e_1)·(t e_1)

    auto step = simplify_basis_dot_step(cs);
    auto* result = step.apply(rl, State{expr}).expr();

    // Result should be Product(s, t) (since 1 folds away)
    auto* pr = dynamic_cast<Product*>(result);
    ASSERT_NE(pr, nullptr);
    EXPECT_EQ(pr->lhs(), s);
    EXPECT_EQ(pr->rhs(), t);
}

TEST(SimplifyBasisDot, ScalarTimesVectorDotCross)
{
    // (s * e_1) · (t * e_2) → 0
    auto rl = make_rl();
    auto const& cs = wcs();
    auto* s = make_named_tensor(rl, "s", 0, {});
    auto* t = make_named_tensor(rl, "t", 0, {});

    auto* lhs = make_tensor_product(rl, s, cs.basis(0));
    auto* rhs = make_tensor_product(rl, t, cs.basis(1));
    auto* expr = make_contract(rl, lhs, rhs);

    auto step = simplify_basis_dot_step(cs);
    auto* result = step.apply(rl, State{expr}).expr();

    auto* rc = dynamic_cast<RationalConst*>(result);
    ASSERT_NE(rc, nullptr);
    EXPECT_EQ(rc->value(), Rational{0});
}

TEST(SimplifyBasisDot, InsideSum)
{
    // Sum of 3 basis dot products (one non-zero diagonal)
    auto rl = make_rl();
    auto const& cs = wcs();

    auto* a = make_named_tensor(rl, "a", 0, {});
    auto* b = make_named_tensor(rl, "b", 0, {});

    // a*e_1·b*e_1 + a*e_1·b*e_2 + a*e_2·b*e_1  →  a·b + 0 + 0 = a·b
    auto* t0 = make_contract(
        rl,
        make_tensor_product(rl, a, cs.basis(0)),
        make_tensor_product(rl, b, cs.basis(0)));
    auto* t1 = make_contract(
        rl,
        make_tensor_product(rl, a, cs.basis(0)),
        make_tensor_product(rl, b, cs.basis(1)));
    auto* t2 = make_contract(
        rl,
        make_tensor_product(rl, a, cs.basis(1)),
        make_tensor_product(rl, b, cs.basis(0)));

    auto* expr = make_sum(rl, {t0, t1, t2});

    auto step = simplify_basis_dot_step(cs);
    auto* result = step.apply(rl, State{expr}).expr();

    // Only the diagonal term survives: Product(a, b).
    auto* pr = dynamic_cast<Product*>(result);
    ASSERT_NE(pr, nullptr) << "Expected Product, got: " << result->latex();
    EXPECT_EQ(pr->lhs(), a);
    EXPECT_EQ(pr->rhs(), b);
}

TEST(SimplifyBasisDot, LeafNodePassThrough)
{
    // NamedTensor leaves are not modified.
    auto rl = make_rl();
    auto const& cs = wcs();
    auto* v = make_named_tensor(rl, "v", 1, {});

    auto step = simplify_basis_dot_step(cs);
    auto* result = step.apply(rl, State{v}).expr();
    EXPECT_EQ(result, v);
}

// ===========================================================================
// collect_zero_terms_step
// ===========================================================================

TEST(CollectZeroTerms, RemovesZeroFromSum)
{
    auto rl = make_rl();
    auto* a = make_named_tensor(rl, "a", 0, {});
    auto* b = make_named_tensor(rl, "b", 0, {});
    auto* zero = make_rational(rl, Rational{0});

    auto* expr = make_sum(rl, {a, zero, b, zero});

    auto step = collect_zero_terms_step();
    auto* result = step.apply(rl, State{expr}).expr();

    // Zero terms are removed; result is Sum[a, b] or a+b.
    auto* s = dynamic_cast<Sum*>(result);
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(static_cast<int>(s->terms().size()), 2);
}

TEST(CollectZeroTerms, SingleNonZeroUnwrapped)
{
    auto rl = make_rl();
    auto* a = make_named_tensor(rl, "a", 1, {});
    auto* zero = make_rational(rl, Rational{0});

    // make_sum with only one non-zero term should unwrap to just a.
    auto* expr = make_sum(rl, {zero, a, zero});

    auto step = collect_zero_terms_step();
    auto* result = step.apply(rl, State{expr}).expr();

    EXPECT_EQ(result, a);
}

TEST(CollectZeroTerms, NonSumPassThrough)
{
    auto rl = make_rl();
    auto* v = make_named_tensor(rl, "v", 1, {});

    auto step = collect_zero_terms_step();
    auto* result = step.apply(rl, State{v}).expr();
    EXPECT_EQ(result, v);
}

// ===========================================================================
// reassemble_from_components_step
// ===========================================================================

TEST(ReassembleFromComponents, RecognisesIdentityTensor)
{
    auto rl = make_rl();
    auto const& cs = wcs();

    // Build e_1⊗e^1 + e_2⊗e^2 + e_3⊗e^3 (for WCS cobasis == basis)
    auto* t0 = make_tensor_product(rl, cs.basis(0), cs.cobasis(0));
    auto* t1 = make_tensor_product(rl, cs.basis(1), cs.cobasis(1));
    auto* t2 = make_tensor_product(rl, cs.basis(2), cs.cobasis(2));
    auto* expr = make_sum(rl, {t0, t1, t2});

    auto step = reassemble_from_components_step(cs);
    auto* result = step.apply(rl, State{expr}).expr();

    EXPECT_NE(dynamic_cast<IdentityTensor*>(result), nullptr)
        << "Expected IdentityTensor, got: " << result->latex();
}

TEST(ReassembleFromComponents, IdentityTensorPermuted)
{
    // Terms in different order should still be recognised.
    auto rl = make_rl();
    auto const& cs = wcs();

    auto* t2 = make_tensor_product(rl, cs.basis(2), cs.cobasis(2));
    auto* t0 = make_tensor_product(rl, cs.basis(0), cs.cobasis(0));
    auto* t1 = make_tensor_product(rl, cs.basis(1), cs.cobasis(1));
    auto* expr = make_sum(rl, {t2, t0, t1});

    auto step = reassemble_from_components_step(cs);
    auto* result = step.apply(rl, State{expr}).expr();

    EXPECT_NE(dynamic_cast<IdentityTensor*>(result), nullptr)
        << "Expected IdentityTensor, got: " << result->latex();
}

// Pass-through:
TEST(ReassembleFromComponents, Rank1SumPassThrough)
{
    auto rl = make_rl();
    auto const& cs = wcs();
    auto* a = make_named_tensor(rl, "a", 1, {});
    auto* b = make_named_tensor(rl, "b", 1, {});
    auto* expr = make_sum(rl, {a, b});

    auto step = reassemble_from_components_step(cs);
    auto* result = step.apply(rl, State{expr}).expr();

    // Not recognised as IdentityTensor.
    EXPECT_EQ(dynamic_cast<IdentityTensor*>(result), nullptr);
}
