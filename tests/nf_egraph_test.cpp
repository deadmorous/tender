#include <tender/nf_egraph.hpp>

#include <tender/derivation.hpp> // steps::canonicalize
#include <tender/expr.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>
#include <tender/nf_lower.hpp> // canonicalize_nf

#include <gtest/gtest.h>

using namespace tender;
using namespace tender::nf;

namespace
{

auto vrank1(Context& ctx, char const* n) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n), {}, 1);
}

auto scalar0(Context& ctx, char const* n) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n), {}, 0);
}

auto canon_nf(Context& ctx, Expr const* e) -> Nf const*
{
    return canonicalize_nf(ctx, steps::canonicalize(ctx, e));
}

auto delta_ul(Context& ctx, IndexSpace const* sp, IndexAssoc a, IndexAssoc b)
    -> Expr const*
{
    return make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, a, b);
}

auto delta_ll(Context& ctx, IndexSpace const* sp, IndexAssoc a, IndexAssoc b)
    -> Expr const*
{
    return make_delta(ctx, Realm::Oblique, sp, Level::Lower, Level::Lower, a, b);
}

// Σ_p δ^p_a δ^p_b
auto contraction(
    Context& ctx,
    IndexSpace const* sp,
    CountableIndex p,
    CountableIndex a,
    CountableIndex b) -> Expr const*
{
    return make_explicit_sum(
        ctx,
        p,
        make_tensor_product(
            ctx, delta_ul(ctx, sp, p, a), delta_ul(ctx, sp, p, b)));
}

// The delta-contraction identity  Σ_p δ^p_a δ^p_b = δ_{ab}, with fresh ids.
auto contraction_rule(Context& ctx, IndexSpace const* sp) -> Identity
{
    CountableIndex p{ctx.alloc_index_id()};
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    return Identity{
        "delta-contraction",
        contraction(ctx, sp, p, a, b),
        delta_ll(ctx, sp, a, b)};
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

TEST(NfEGraphExtract, RoundTripsTheCanonicalNf)
{
    Context ctx;
    auto const* a = vrank1(ctx, "a");
    auto const* b = vrank1(ctx, "b");
    NfEGraph g(ctx);
    auto const* e = make_tensor_product(ctx, a, b);
    auto c = g.add(e);
    EXPECT_TRUE(equal(g.extract(c), canon_nf(ctx, e)));
}

TEST(NfEGraphExtract, PicksCheaperFormAfterMerge)
{
    Context ctx;
    auto const* a = vrank1(ctx, "a");
    auto const* b = vrank1(ctx, "b");
    auto const* c = vrank1(ctx, "c");
    NfEGraph g(ctx);
    auto big = g.add(
        make_tensor_product(ctx, make_tensor_product(ctx, a, b), c)); // a⊗b⊗c
    auto small = g.add(a);
    g.merge(big, small);
    g.rebuild();
    // The cheapest representative of the merged class is `a`.
    EXPECT_TRUE(equal(g.extract(big), canon_nf(ctx, a)));
}

TEST(NfEGraphRebuild, CongruenceMergesParents)
{
    // x / (a + b) and x / (a + c) hold a `Div` whose denominator is the Sum
    // class of the bracket.  Merging the two bracket classes must, after
    // rebuild, merge the two divisions (congruence: equal children → equal
    // node).
    Context ctx;
    auto const* x = scalar0(ctx, "x");
    auto const* a = scalar0(ctx, "a");
    auto const* b = scalar0(ctx, "b");
    auto const* c = scalar0(ctx, "c");

    NfEGraph g(ctx);
    auto e1 = g.add(make_scalar_div(ctx, x, make_sum(ctx, a, b)));
    auto e2 = g.add(make_scalar_div(ctx, x, make_sum(ctx, a, c)));
    ASSERT_NE(g.find(e1), g.find(e2));

    auto ab = g.add(make_sum(ctx, a, b));
    auto ac = g.add(make_sum(ctx, a, c));
    g.merge(ab, ac);
    g.rebuild();

    EXPECT_EQ(g.find(e1), g.find(e2));
}

TEST(NfEGraphSaturate, ContractsDelta)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    NfEGraph g(ctx);
    auto root = g.add(contraction(ctx, sp, q, m, n));
    int passes = g.saturate({contraction_rule(ctx, sp)});

    EXPECT_TRUE(
        equal(g.extract(g.find(root)), canon_nf(ctx, delta_ll(ctx, sp, m, n))));
    EXPECT_LT(passes, 30); // converged below the cap
}

TEST(NfEGraphSaturate, FiresOnSubProductOfLargerTerm)
{
    // The C14 payoff the Expr e-graph could not reach: the contraction sits
    // among an extra factor of the term — δ^p_m δ^p_n ⊗ c — yet the rule still
    // fires on the sub-product, leaving δ_{mn} ⊗ c.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const* c = vrank1(ctx, "c");

    NfEGraph g(ctx);
    auto root =
        g.add(make_tensor_product(ctx, contraction(ctx, sp, q, m, n), c));
    (void)g.saturate({contraction_rule(ctx, sp)});

    auto const* expected = make_tensor_product(ctx, delta_ll(ctx, sp, m, n), c);
    EXPECT_TRUE(equal(g.extract(g.find(root)), canon_nf(ctx, expected)));
}

TEST(NfEGraphSaturate, RewritesTermInsideSum)
{
    // δ_{rs} + Σ_q δ^q_m δ^q_n  saturates to  δ_{rs} + δ_{mn}: the rule fires
    // on the second additive term only, the first carried through.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    CountableIndex r{ctx.alloc_index_id()};
    CountableIndex s{ctx.alloc_index_id()};

    NfEGraph g(ctx);
    auto root = g.add(
        make_sum(ctx, delta_ll(ctx, sp, r, s), contraction(ctx, sp, q, m, n)));
    (void)g.saturate({contraction_rule(ctx, sp)});

    auto const* expected =
        make_sum(ctx, delta_ll(ctx, sp, r, s), delta_ll(ctx, sp, m, n));
    EXPECT_TRUE(equal(g.extract(g.find(root)), canon_nf(ctx, expected)));
}

TEST(NfEGraphSaturate, NoMatchLeavesGraphUnchanged)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    NfEGraph g(ctx);
    auto root = g.add(delta_ll(ctx, sp, m, n)); // bare delta, no contraction
    auto before = g.class_count();

    EXPECT_EQ(g.saturate({contraction_rule(ctx, sp)}), 1); // one no-op pass
    EXPECT_EQ(g.class_count(), before);
    EXPECT_TRUE(
        equal(g.extract(g.find(root)), canon_nf(ctx, delta_ll(ctx, sp, m, n))));
}

TEST(NfEGraphSaturate, ReachesFixedPointAndIsIdempotent)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    NfEGraph g(ctx);
    (void)g.add(contraction(ctx, sp, q, m, n));
    (void)g.saturate({contraction_rule(ctx, sp)});
    // A second saturation merges nothing new: one pass, then convergence.
    EXPECT_EQ(g.saturate({contraction_rule(ctx, sp)}), 1);
}

TEST(NfEGraphSaturate, RespectsIterationCap)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    NfEGraph g(ctx);
    (void)g.add(contraction(ctx, sp, q, m, n));
    EXPECT_EQ(g.saturate({contraction_rule(ctx, sp)}, 1), 1); // hard stop
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
