#include <tender/nf_lower.hpp>

#include <gtest/gtest.h>

using namespace tender;
using namespace tender::nf;

namespace
{

auto atom(Context& ctx, std::string_view name) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, 1);
}

// Assert the flattened layer equals the given (sign, body) pairs, in order.
void expect_terms(
    std::vector<SignedExpr> const& got,
    std::vector<std::pair<int, Expr const*>> const& want)
{
    ASSERT_EQ(got.size(), want.size());
    for (std::size_t i = 0; i < got.size(); ++i)
    {
        EXPECT_EQ(got[i].sign, want[i].first) << "term " << i;
        EXPECT_EQ(got[i].body, want[i].second) << "term " << i;
    }
}

} // namespace

// ---- additive flatten --------------------------------------------------

TEST(AdditiveFlatten, SingleLeaf)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    expect_terms(additive_flatten(a), {{+1, a}});
}

TEST(AdditiveFlatten, Sum)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    expect_terms(additive_flatten(make_sum(ctx, a, b)), {{+1, a}, {+1, b}});
}

TEST(AdditiveFlatten, Difference)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    expect_terms(
        additive_flatten(make_difference(ctx, a, b)), {{+1, a}, {-1, b}});
}

TEST(AdditiveFlatten, Negate)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    expect_terms(additive_flatten(make_negate(ctx, a)), {{-1, a}});
}

TEST(AdditiveFlatten, DoubleNegateCancels)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    expect_terms(
        additive_flatten(make_negate(ctx, make_negate(ctx, a))), {{+1, a}});
}

TEST(AdditiveFlatten, NestedSignPropagation)
{
    // (a + b) - (c - d)  ->  +a +b -c +d
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* d = atom(ctx, "d");
    auto const* e =
        make_difference(ctx, make_sum(ctx, a, b), make_difference(ctx, c, d));
    expect_terms(additive_flatten(e), {{+1, a}, {+1, b}, {-1, c}, {+1, d}});
}

TEST(AdditiveFlatten, SubtractedSumFlipsBoth)
{
    // a - (b + c)  ->  +a -b -c
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* e = make_difference(ctx, a, make_sum(ctx, b, c));
    expect_terms(additive_flatten(e), {{+1, a}, {-1, b}, {-1, c}});
}

TEST(AdditiveFlatten, ProductIsOpaqueLeaf)
{
    // (a ⊗ b) + c  ->  the product is one leaf, NOT split.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* prod = make_tensor_product(ctx, a, b);
    expect_terms(
        additive_flatten(make_sum(ctx, prod, c)), {{+1, prod}, {+1, c}});
}

TEST(AdditiveFlatten, NoDistributionOverInnerSum)
{
    // (a + b) ⊗ c  ->  one leaf (the product); the inner sum is untouched.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* prod = make_tensor_product(ctx, make_sum(ctx, a, b), c);
    expect_terms(additive_flatten(prod), {{+1, prod}});
}

// ---- multiplicative flatten --------------------------------------------

namespace
{
auto flat(int sign, Expr const* body) -> ProductParts
{
    return multiplicative_flatten(SignedExpr{sign, body});
}
} // namespace

TEST(MultiplicativeFlatten, BareAtomCarriesSign)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto pos = flat(+1, a);
    EXPECT_EQ(pos.coeff, Rational{1});
    ASSERT_EQ(pos.factors.size(), 1u);
    EXPECT_EQ(pos.factors[0], a);

    auto neg = flat(-1, a);
    EXPECT_EQ(neg.coeff, Rational{-1});
    EXPECT_EQ(neg.factors, (std::vector<Expr const*>{a}));
}

TEST(MultiplicativeFlatten, ScalarLiteralFolds)
{
    Context ctx;
    auto const* three = make_scalar(ctx, Rational{3});
    auto pp = flat(-1, three);
    EXPECT_EQ(pp.coeff, Rational{-3});
    EXPECT_TRUE(pp.factors.empty());
}

TEST(MultiplicativeFlatten, ProductChainPreservesOrder)
{
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* prod =
        make_tensor_product(ctx, make_tensor_product(ctx, a, b), c);
    auto pp = flat(+1, prod);
    EXPECT_EQ(pp.coeff, Rational{1});
    EXPECT_EQ(pp.factors, (std::vector<Expr const*>{a, b, c}));
}

TEST(MultiplicativeFlatten, NumericFactorsMultiplyIntoCoeff)
{
    // 2 ⊗ (3 ⊗ a)  ->  coeff 6, factor a.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* two = make_scalar(ctx, Rational{2});
    auto const* three = make_scalar(ctx, Rational{3});
    auto const* e =
        make_tensor_product(ctx, two, make_tensor_product(ctx, three, a));
    auto pp = flat(+1, e);
    EXPECT_EQ(pp.coeff, Rational{6});
    EXPECT_EQ(pp.factors, (std::vector<Expr const*>{a}));
}

TEST(MultiplicativeFlatten, NestedNegateFoldsSign)
{
    // a ⊗ (-b)  ->  coeff -1, factors [a, b].
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* e = make_tensor_product(ctx, a, make_negate(ctx, b));
    auto pp = flat(+1, e);
    EXPECT_EQ(pp.coeff, Rational{-1});
    EXPECT_EQ(pp.factors, (std::vector<Expr const*>{a, b}));
}

TEST(MultiplicativeFlatten, NumericDivisionFolds)
{
    // a / 2  ->  coeff 1/2, factor a.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* e = make_scalar_div(ctx, a, make_scalar(ctx, Rational{2}));
    auto pp = flat(+1, e);
    EXPECT_EQ(pp.coeff, (Rational{1, 2}));
    EXPECT_EQ(pp.factors, (std::vector<Expr const*>{a}));
}

TEST(MultiplicativeFlatten, NonNumericDivisionStaysOpaque)
{
    // a / (b·c)  ->  coeff 1, one opaque ScalarDiv factor (no reciprocal yet).
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* div = make_scalar_div(ctx, a, make_dot(ctx, b, c));
    auto pp = flat(+1, div);
    EXPECT_EQ(pp.coeff, Rational{1});
    ASSERT_EQ(pp.factors.size(), 1u);
    EXPECT_EQ(pp.factors[0], div);
}

TEST(MultiplicativeFlatten, ContractionIsOneFactorNotFlattened)
{
    // a ⊗ (b·c)  ->  two factors [a, (b·c)]; only ⊗ flattens, not the dot.
    Context ctx;
    auto const* a = atom(ctx, "a");
    auto const* b = atom(ctx, "b");
    auto const* c = atom(ctx, "c");
    auto const* dot = make_dot(ctx, b, c);
    auto const* e = make_tensor_product(ctx, a, dot);
    auto pp = flat(+1, e);
    EXPECT_EQ(pp.factors, (std::vector<Expr const*>{a, dot}));
}
