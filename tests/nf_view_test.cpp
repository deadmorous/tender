#include <tender/nf_view.hpp>

#include <tender/derivation.hpp>
#include <tender/render.hpp>

#include <gtest/gtest.h>

using namespace tender;

namespace
{

// An abstract rank-1 tensor named `name` (additive structure only needs
// shapes, not slots).
auto vec(Context& ctx, std::string_view name) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, 1);
}

// A rank-0 scalar symbol.
auto sym(Context& ctx, std::string_view name) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, 0);
}

} // namespace

// ---- signed_addends / sum_of -------------------------------------------

TEST(SignedAddends, FlattensThroughDifferenceAndNegate)
{
    Context ctx;
    auto const* a = vec(ctx, "a");
    auto const* b = vec(ctx, "b");
    auto const* c = vec(ctx, "c");
    // (a + b) − (−c)  →  (+a) (+b) (−(−c) leaf? no: Negate flips) (+c)
    auto const* e =
        make_difference(ctx, make_sum(ctx, a, b), make_negate(ctx, c));

    auto const addends = view::signed_addends(e);
    ASSERT_EQ(addends.size(), 3u);
    EXPECT_EQ(addends[0].sign, +1);
    EXPECT_EQ(addends[0].body, a);
    EXPECT_EQ(addends[1].sign, +1);
    EXPECT_EQ(addends[1].body, b);
    EXPECT_EQ(addends[2].sign, +1); // − of − is +
    EXPECT_EQ(addends[2].body, c);
}

TEST(SignedAddends, DoesNotDistributeIntoProducts)
{
    Context ctx;
    auto const* s = sym(ctx, "s");
    auto const* inner = make_sum(ctx, vec(ctx, "a"), vec(ctx, "b"));
    auto const* e = make_tensor_product(ctx, s, inner);

    auto const addends = view::signed_addends(e);
    ASSERT_EQ(addends.size(), 1u);
    EXPECT_EQ(addends[0].body, e); // the product is one opaque leaf
}

TEST(SumOf, RoundTripsSignsExactly)
{
    Context ctx;
    auto const* a = vec(ctx, "a");
    auto const* b = vec(ctx, "b");
    auto const* c = vec(ctx, "c");
    std::vector<nf::SignedExpr> const addends{{-1, a}, {+1, b}, {-1, c}};

    auto const* rebuilt = view::sum_of(ctx, addends);
    // −a + b − c: leading Negate, then Sum, then Difference.
    auto const round = view::signed_addends(rebuilt);
    ASSERT_EQ(round.size(), 3u);
    EXPECT_EQ(round[0].sign, -1);
    EXPECT_EQ(round[0].body, a);
    EXPECT_EQ(round[1].sign, +1);
    EXPECT_EQ(round[1].body, b);
    EXPECT_EQ(round[2].sign, -1);
    EXPECT_EQ(round[2].body, c);
}

TEST(SumOf, EmptyListIsZero)
{
    Context ctx;
    auto const* zero = view::sum_of(ctx, {});
    auto const* lit = std::get_if<ScalarLiteral>(&zero->node);
    ASSERT_NE(lit, nullptr);
    EXPECT_TRUE(lit->value.is_zero());
}

// ---- map_additive_leaves -----------------------------------------------

TEST(MapAdditiveLeaves, PreservesSkeletonExactly)
{
    Context ctx;
    auto const* a = vec(ctx, "a");
    auto const* b = vec(ctx, "b");
    auto const* d = vec(ctx, "d");
    auto const* two = make_scalar(ctx, Rational{2});
    // (a + b)/2 − d, leaves negated: ((−a) + (−b))/2 − (−d)
    auto const* e =
        make_difference(ctx, make_scalar_div(ctx, make_sum(ctx, a, b), two), d);

    auto const* out = view::map_additive_leaves(
        ctx, e, [&](Expr const* leaf) { return make_negate(ctx, leaf); });

    auto const* expected = make_difference(
        ctx,
        make_scalar_div(
            ctx, make_sum(ctx, make_negate(ctx, a), make_negate(ctx, b)), two),
        make_negate(ctx, d));
    EXPECT_TRUE(structural_eq(out, expected));
}

TEST(MapAdditiveLeaves, IdentityMapReturnsInputPointer)
{
    Context ctx;
    auto const* e = make_difference(
        ctx,
        make_sum(ctx, vec(ctx, "a"), vec(ctx, "b")),
        make_negate(ctx, vec(ctx, "c")));

    auto const* out =
        view::map_additive_leaves(ctx, e, [](Expr const* leaf) { return leaf; });
    EXPECT_EQ(out, e); // no allocation, no rebuild — the same tree
}

TEST(MapAdditiveLeaves, ScalarDivDescentIsOptional)
{
    Context ctx;
    auto const* a = vec(ctx, "a");
    auto const* two = make_scalar(ctx, Rational{2});
    auto const* e = make_scalar_div(ctx, a, two);

    int leaves_seen = 0;
    auto count = [&](Expr const* leaf)
    {
        ++leaves_seen;
        return leaf;
    };
    (void)view::map_additive_leaves(ctx, e, count);
    EXPECT_EQ(leaves_seen, 1); // descended: the leaf is `a`

    leaves_seen = 0;
    (void)view::map_additive_leaves(
        ctx, e, count, view::AdditiveOptions{.descend_scalar_div = false});
    EXPECT_EQ(leaves_seen, 1); // not descended: the whole `a/2` is the leaf
}

// ---- map_nf_terms ------------------------------------------------------

TEST(MapNfTerms, IdentityTransformReturnsInputPointer)
{
    Context ctx;
    auto const* a = vec(ctx, "a");
    auto const* e = make_sum(ctx, a, a); // non-canonical on purpose

    auto const* out = view::map_nf_terms(ctx, e, [](std::vector<nf::Term>&) {});
    EXPECT_EQ(out, e); // identity transform is a no-op, input kept verbatim
}

TEST(MapNfTerms, EditsCoefficientsAndRecanonicalizes)
{
    Context ctx;
    auto const* a = vec(ctx, "a");
    auto const* b = vec(ctx, "b");
    auto const* e = make_sum(ctx, a, b);

    auto const* out = view::map_nf_terms(
        ctx,
        e,
        [](std::vector<nf::Term>& terms)
        {
            for (auto& t: terms)
                t.coeff = t.coeff * Rational{3};
        });

    auto const* expected = steps::canonicalize(
        ctx,
        make_sum(
            ctx,
            make_tensor_product(ctx, make_scalar(ctx, Rational{3}), a),
            make_tensor_product(ctx, make_scalar(ctx, Rational{3}), b)));
    EXPECT_TRUE(structural_eq(steps::canonicalize(ctx, out), expected));
}

TEST(MapNfTerms, DroppingEveryTermYieldsZero)
{
    Context ctx;
    auto const* e = make_sum(ctx, vec(ctx, "a"), vec(ctx, "b"));
    auto const* out = view::map_nf_terms(
        ctx, e, [](std::vector<nf::Term>& terms) { terms.clear(); });
    auto const* lit = std::get_if<ScalarLiteral>(&out->node);
    ASSERT_NE(lit, nullptr);
    EXPECT_TRUE(lit->value.is_zero());
}

// ---- fixpoint ----------------------------------------------------------

TEST(Fixpoint, IteratesUntilPointerStable)
{
    Context ctx;
    auto const* a = vec(ctx, "a");
    auto const* e =
        make_negate(ctx, make_negate(ctx, make_negate(ctx, a))); // −−−a

    // One step peels a single −− pair per call.
    auto peel = [](Context& /*ctx*/, Expr const* x) -> Expr const*
    {
        if (auto const* n = std::get_if<Negate>(&x->node))
            if (auto const* nn = std::get_if<Negate>(&n->operand->node))
                return nn->operand;
        return x;
    };
    auto const* out = view::fixpoint(ctx, e, peel);
    // −−−a → −a, then stable.
    auto const* n = std::get_if<Negate>(&out->node);
    ASSERT_NE(n, nullptr);
    EXPECT_EQ(n->operand, a);
}

TEST(Fixpoint, ThrowsWhenStepNeverConverges)
{
    Context ctx;
    auto const* a = vec(ctx, "a");
    auto flip = [](Context& c, Expr const* x) -> Expr const*
    { return make_negate(c, x); };
    EXPECT_THROW((void)view::fixpoint(ctx, a, flip, 8), std::runtime_error);
}
