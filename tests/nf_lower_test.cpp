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
