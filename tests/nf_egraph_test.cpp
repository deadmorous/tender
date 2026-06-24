#include <tender/nf_egraph.hpp>

#include <tender/expr.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>

#include <gtest/gtest.h>

using namespace tender;
using namespace tender::nf;

namespace
{

auto vrank1(Context& ctx, char const* n) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n), {}, 1);
}

auto delta_ul(Context& ctx, IndexSpace const* sp, IndexAssoc a, IndexAssoc b)
    -> Expr const*
{
    return make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, a, b);
}

} // namespace

TEST(NfEGraphCore, EqualExpressionsShareClass)
{
    Context ctx;
    auto const* a = vrank1(ctx, "a");
    auto const* b = vrank1(ctx, "b");
    NfEGraph g(ctx);
    // a + b and b + a canonicalize to the same Nf.
    auto c1 = g.add(make_sum(ctx, a, b));
    auto c2 = g.add(make_sum(ctx, b, a));
    EXPECT_EQ(g.find(c1), g.find(c2));
}

TEST(NfEGraphCore, DistinctExpressionsDifferentClass)
{
    Context ctx;
    auto const* a = vrank1(ctx, "a");
    auto const* b = vrank1(ctx, "b");
    NfEGraph g(ctx);
    EXPECT_NE(g.find(g.add(a)), g.find(g.add(b)));
}

TEST(NfEGraphCore, ReaddingIsIdempotent)
{
    Context ctx;
    auto const* a = vrank1(ctx, "a");
    NfEGraph g(ctx);
    auto c1 = g.add(a);
    auto const n1 = g.node_count();
    auto c2 = g.add(a);
    EXPECT_EQ(g.find(c1), g.find(c2));
    EXPECT_EQ(g.node_count(), n1); // hash-consed, no new nodes
}

TEST(NfEGraphCore, SharedSubexpressionNotDuplicated)
{
    Context ctx;
    auto const* a = vrank1(ctx, "a");
    auto const* b = vrank1(ctx, "b");
    NfEGraph g(ctx);
    (void)g.add(a); // creates the Atom(a) node
    auto const n_after_a = g.node_count();
    (void)g.add(make_tensor_product(ctx, a, b)); // reuses Atom(a)
    // New nodes: Atom(b), the Term, the Sum — three.  Atom(a) is shared.
    EXPECT_EQ(g.node_count(), n_after_a + 3);
}

TEST(NfEGraphCore, MergeUnifiesClasses)
{
    Context ctx;
    auto const* a = vrank1(ctx, "a");
    auto const* b = vrank1(ctx, "b");
    NfEGraph g(ctx);
    auto c1 = g.add(a);
    auto c2 = g.add(b);
    ASSERT_NE(g.find(c1), g.find(c2));
    g.merge(c1, c2);
    EXPECT_EQ(g.find(c1), g.find(c2));
}

TEST(NfEGraphCore, ContractionStructureDedups)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex p{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    NfEGraph g(ctx);
    // The same contraction added twice shares all nodes.
    auto mk = [&]
    {
        return make_explicit_sum(
            ctx,
            p,
            make_tensor_product(
                ctx, delta_ul(ctx, sp, p, m), delta_ul(ctx, sp, p, n)));
    };
    auto c1 = g.add(mk());
    auto const n1 = g.node_count();
    auto c2 = g.add(mk());
    EXPECT_EQ(g.find(c1), g.find(c2));
    EXPECT_EQ(g.node_count(), n1);
}
