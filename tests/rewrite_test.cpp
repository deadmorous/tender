#include <tender/derivation.hpp>
#include <tender/rewrite.hpp>

#include <gtest/gtest.h>

#include <stdexcept>

using namespace tender;

namespace
{

auto vec(Context& ctx, char const* name) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, 1);
}

// Predicate: a well-known identity tensor I.
auto is_identity(Expr const* e) -> bool
{
    if (auto* t = std::get_if<TensorObject>(&e->node))
        return t->traits && t->traits->well_known == WellKnownKind::Identity;
    return false;
}

} // namespace

// ---- children ----------------------------------------------------------

TEST(Children, LeavesHaveNone)
{
    Context ctx;
    EXPECT_TRUE(children(make_scalar(ctx, Rational{2})).size() == 0u);
    EXPECT_TRUE(children(vec(ctx, "a")).size() == 0u);
    EXPECT_TRUE(children(make_nabla(ctx)).size() == 0u);
}

TEST(Children, UnaryOne)
{
    Context ctx;
    auto* a = vec(ctx, "a");
    auto* t = make_transpose(ctx, a);
    auto kids = children(t);
    ASSERT_EQ(kids.size(), 1u);
    EXPECT_EQ(kids[0], a);
}

TEST(Children, BinaryLeftRight)
{
    Context ctx;
    auto* a = vec(ctx, "a");
    auto* b = vec(ctx, "b");
    auto kids = children(make_tensor_product(ctx, a, b));
    ASSERT_EQ(kids.size(), 2u);
    EXPECT_EQ(kids[0], a); // left = 0
    EXPECT_EQ(kids[1], b); // right = 1
}

TEST(Children, ExplicitSumBoundIsChildOnlyWhenPresent)
{
    Context ctx;
    auto* body = vec(ctx, "a");
    CountableIndex i{ctx.alloc_index_id()};

    auto kids_no_bound = children(make_explicit_sum(ctx, i, body));
    ASSERT_EQ(kids_no_bound.size(), 1u);
    EXPECT_EQ(kids_no_bound[0], body);

    auto* bound = make_scalar(ctx, Rational{3});
    auto kids_bound = children(make_explicit_sum(ctx, i, body, bound));
    ASSERT_EQ(kids_bound.size(), 2u);
    EXPECT_EQ(kids_bound[0], body); // body stays at selector 0
    EXPECT_EQ(kids_bound[1], bound);
}

// ---- with_children -----------------------------------------------------

TEST(WithChildren, IdentityRewriteReusesPointer)
{
    Context ctx;
    auto* a = vec(ctx, "a");
    auto* b = vec(ctx, "b");
    auto* e = make_tensor_product(ctx, a, b);
    EXPECT_EQ(with_children(ctx, e, children(e)), e); // no allocation
}

TEST(WithChildren, RebuildsWithNewChild)
{
    Context ctx;
    auto* a = vec(ctx, "a");
    auto* b = vec(ctx, "b");
    auto* c = vec(ctx, "c");
    auto* e = make_tensor_product(ctx, a, b);

    auto kids = children(e);
    kids[1] = c;
    auto* rebuilt = with_children(ctx, e, kids);
    EXPECT_NE(rebuilt, e);
    EXPECT_TRUE(structural_eq(rebuilt, make_tensor_product(ctx, a, c)));
}

TEST(WithChildren, PreservesNonExprFields)
{
    Context ctx;
    auto* body = vec(ctx, "a");
    auto* body2 = vec(ctx, "b");
    CountableIndex i{ctx.alloc_index_id()};
    auto* e = make_explicit_sum(ctx, i, body);

    auto kids = children(e);
    kids[0] = body2;
    auto* rebuilt = with_children(ctx, e, kids);
    ASSERT_TRUE(std::holds_alternative<ExplicitSum>(rebuilt->node));
    EXPECT_EQ(std::get<ExplicitSum>(rebuilt->node).index.id, i.id);
}

TEST(WithChildren, ArityMismatchThrows)
{
    Context ctx;
    auto* a = vec(ctx, "a");
    auto* e = make_tensor_product(ctx, a, a);
    ExprChildren one;
    one.push_back(a);
    EXPECT_THROW(with_children(ctx, e, one), std::invalid_argument);
}

// ---- subexpr_at --------------------------------------------------------

TEST(SubexprAt, NavigatesToTarget)
{
    Context ctx;
    auto* a = vec(ctx, "a");
    auto* b = vec(ctx, "b");
    auto* I = make_identity(ctx);
    // (a × I) × b  — I sits at [0, 1]
    auto* e = make_cross(ctx, make_cross(ctx, a, I), b);
    EXPECT_EQ(subexpr_at(e, {}), e);     // empty path = root
    EXPECT_EQ(subexpr_at(e, {0, 1}), I); // the I
    EXPECT_EQ(subexpr_at(e, {1}), b);
    EXPECT_EQ(subexpr_at(e, {0, 0}), a);
}

TEST(SubexprAt, OutOfRangeThrows)
{
    Context ctx;
    auto* a = vec(ctx, "a");
    auto* e = make_tensor_product(ctx, a, a);
    EXPECT_THROW(subexpr_at(e, {2}), std::out_of_range);
    EXPECT_THROW(subexpr_at(e, {0, 0}), std::out_of_range); // a is a leaf
}

// ---- rewrite_at --------------------------------------------------------

TEST(RewriteAt, RewritesTargetAndSharesOffPath)
{
    Context ctx;
    auto* a = vec(ctx, "a");
    auto* b = vec(ctx, "b");
    auto* c = vec(ctx, "c");
    auto* I = make_identity(ctx);
    auto* inner = make_cross(ctx, a, I);
    auto* e = make_cross(ctx, inner, b); // (a × I) × b

    auto* out =
        rewrite_at(ctx, e, {0, 1}, [&](Context&, Expr const*) { return c; });

    EXPECT_EQ(subexpr_at(out, {0, 1}), c); // I replaced by c
    // Off-path nodes keep their pointers (only the spine above [0,1] rebuilt).
    EXPECT_EQ(subexpr_at(out, {1}), b);
    EXPECT_EQ(subexpr_at(out, {0, 0}), a);
}

TEST(RewriteAt, IdentityFunctionIsNoOp)
{
    Context ctx;
    auto* a = vec(ctx, "a");
    auto* b = vec(ctx, "b");
    auto* e = make_cross(ctx, a, b);
    auto* out =
        rewrite_at(ctx, e, {1}, [](Context&, Expr const* n) { return n; });
    EXPECT_EQ(out, e); // nothing changed, pointer reused
}

TEST(RewriteAt, OutOfRangeThrows)
{
    Context ctx;
    auto* a = vec(ctx, "a");
    auto* e = make_tensor_product(ctx, a, a);
    EXPECT_THROW(
        rewrite_at(ctx, e, {5}, [](Context&, Expr const* n) { return n; }),
        std::out_of_range);
}

// ---- replace_at --------------------------------------------------------

TEST(ReplaceAt, SplicesSubBackIn)
{
    Context ctx;
    auto* a = vec(ctx, "a");
    auto* b = vec(ctx, "b");
    auto* c = vec(ctx, "c");
    auto* e = make_tensor_product(ctx, a, b);
    auto* out = replace_at(ctx, e, {0}, c);
    EXPECT_TRUE(structural_eq(out, make_tensor_product(ctx, c, b)));
}

TEST(ReplaceAt, RoundTripThroughExtraction)
{
    Context ctx;
    auto* a = vec(ctx, "a");
    auto* b = vec(ctx, "b");
    auto* e = make_tensor_product(ctx, a, b);
    // Extract a term, "work" on it (transpose), splice back.
    Path p = {1};
    auto* sub = subexpr_at(e, p);
    auto* sub2 = make_transpose(ctx, sub);
    auto* out = replace_at(ctx, e, p, sub2);
    EXPECT_TRUE(
        structural_eq(out, make_tensor_product(ctx, a, make_transpose(ctx, b))));
}

// ---- find_occurrences --------------------------------------------------

TEST(FindOccurrences, PreOrderAllIdentities)
{
    Context ctx;
    auto* a = vec(ctx, "a");
    auto* b = vec(ctx, "b");
    auto* I = make_identity(ctx);
    // a ⊗ I ⊗ b ⊗ I  (left-nested): I's at [0,0,1] and [1]
    auto* e = make_tensor_product(
        ctx, make_tensor_product(ctx, make_tensor_product(ctx, a, I), b), I);

    auto paths = find_occurrences(e, is_identity);
    ASSERT_EQ(paths.size(), 2u);
    EXPECT_EQ(paths[0], (Path{0, 0, 1})); // pre-order: deeper-left first
    EXPECT_EQ(paths[1], (Path{1}));
    // Nth sugar: the k-th identity.
    EXPECT_EQ(subexpr_at(e, paths[1]), I);
}

TEST(FindOccurrences, NoneWhenAbsent)
{
    Context ctx;
    auto* a = vec(ctx, "a");
    auto* e = make_tensor_product(ctx, a, a);
    EXPECT_TRUE(find_occurrences(e, is_identity).empty());
}

// ---- addend_paths ------------------------------------------------------

TEST(AddendPaths, EnumeratesSumDifferenceSpine)
{
    Context ctx;
    auto* a = vec(ctx, "a");
    auto* b = vec(ctx, "b");
    auto* c = vec(ctx, "c");
    // (a + b) - c
    auto* e = make_difference(ctx, make_sum(ctx, a, b), c);
    auto paths = addend_paths(e);
    ASSERT_EQ(paths.size(), 3u);
    EXPECT_EQ(subexpr_at(e, paths[0]), a);
    EXPECT_EQ(subexpr_at(e, paths[1]), b);
    EXPECT_EQ(subexpr_at(e, paths[2]), c);
}

TEST(AddendPaths, NonAdditiveYieldsSelf)
{
    Context ctx;
    auto* a = vec(ctx, "a");
    auto* b = vec(ctx, "b");
    auto* e = make_tensor_product(ctx, a, b);
    auto paths = addend_paths(e);
    ASSERT_EQ(paths.size(), 1u);
    EXPECT_TRUE(paths[0].empty());
}
