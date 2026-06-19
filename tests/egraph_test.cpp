#include <gtest/gtest.h>
#include <tender/derivation.hpp>
#include <tender/egraph.hpp>
#include <tender/expr.hpp>
#include <tender/identity.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>

#include <optional>
#include <utility>
#include <vector>

using namespace tender;

namespace
{
auto obj(Context& ctx, char const* n) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(n));
}

auto delta_ul(Context& ctx, CountableIndex a, CountableIndex b) -> Expr const*
{
    return make_delta(
        ctx, Realm::Oblique, space_3d(), Level::Upper, Level::Lower, a, b);
}

auto delta_ll(Context& ctx, CountableIndex a, CountableIndex b) -> Expr const*
{
    return make_delta(
        ctx, Realm::Oblique, space_3d(), Level::Lower, Level::Lower, a, b);
}

// Σ_p δ^p_a δ^p_b
auto contraction(
    Context& ctx,
    CountableIndex p,
    CountableIndex a,
    CountableIndex b) -> Expr const*
{
    return make_explicit_sum(
        ctx,
        p,
        make_tensor_product(ctx, delta_ul(ctx, p, a), delta_ul(ctx, p, b)));
}

// Is δ_{m n} the instantiation of the contraction RHS under some returned
// match?
auto binds_to_delta_mn(
    Context& ctx,
    std::vector<std::pair<EClassId, MatchBinding>> const& ms,
    CountableIndex a,
    CountableIndex b,
    CountableIndex m,
    CountableIndex n) -> bool
{
    auto const* rhs = delta_ll(ctx, a, b);
    auto const* expected = delta_ll(ctx, m, n);
    for (auto const& [cls, bnd]: ms)
        if (algebraic_eq(ctx, instantiate(ctx, rhs, bnd), expected))
            return true;
    return false;
}
} // namespace

// ---- hash-consing and round-trip -------------------------------------------

TEST(EGraph, RoundTripsCanonicalExpression)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");
    auto const* e = make_sum(ctx, make_tensor_product(ctx, A, B), A);

    EGraph eg{ctx};
    auto root = eg.add(e);
    EXPECT_TRUE(algebraic_eq(ctx, eg.extract(root), e));
}

TEST(EGraph, StructurallyEqualInputsShareAClass)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");

    EGraph eg{ctx};
    auto c1 = eg.add(make_sum(ctx, A, B));
    auto c2 = eg.add(make_sum(ctx, A, B));
    EXPECT_EQ(eg.find(c1), eg.find(c2));
}

TEST(EGraph, ACEquivalentInputsShareAClass)
{
    // Canonicalization on insertion means A + B and B + A hash-cons together.
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");

    EGraph eg{ctx};
    EXPECT_EQ(
        eg.find(eg.add(make_sum(ctx, A, B))),
        eg.find(eg.add(make_sum(ctx, B, A))));
}

TEST(EGraph, AlphaEquivalentSumsShareAClass)
{
    // Σ_i δ^i_i and Σ_j δ^j_j are α-equivalent; canonicalization aligns them.
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    auto const* ei = make_explicit_sum(ctx, i, delta_ul(ctx, i, i));
    auto const* ej = make_explicit_sum(ctx, j, delta_ul(ctx, j, j));

    EGraph eg{ctx};
    EXPECT_EQ(eg.find(eg.add(ei)), eg.find(eg.add(ej)));
}

TEST(EGraph, SharedSubtermIsNotDuplicated)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");
    auto const* C = obj(ctx, "C");

    EGraph eg{ctx};
    (void)eg.add(make_dot(ctx, A, B));                   // e-nodes: A, B, Dot
    (void)eg.add(make_sum(ctx, make_dot(ctx, A, B), C)); // reuse Dot; add C,
                                                         // Sum
    // A, B, C, Dot(A,B), Sum — five distinct e-nodes (Dot hash-cons'd once).
    EXPECT_EQ(eg.node_count(), 5u);
}

// ---- merge and congruence --------------------------------------------------

TEST(EGraph, MergePropagatesByCongruence)
{
    // Merge A == B; then f(A) and f(B) must become equal after rebuild.
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");
    auto const* C = obj(ctx, "C");

    EGraph eg{ctx};
    auto ca = eg.add(A);
    auto cb = eg.add(B);
    auto fa = eg.add(make_dot(ctx, A, C));
    auto fb = eg.add(make_dot(ctx, B, C));
    EXPECT_NE(eg.find(fa), eg.find(fb));

    eg.merge(ca, cb);
    eg.rebuild();
    EXPECT_EQ(eg.find(fa), eg.find(fb)); // congruence closure
}

TEST(EGraph, ExtractChoosesCheapestRepresentative)
{
    // After proving a big expression equals a small one, extraction yields the
    // small one.
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");
    auto const* big = make_sum(ctx, make_dot(ctx, A, B), make_dot(ctx, B, A));
    auto const* small = obj(ctx, "C");

    EGraph eg{ctx};
    auto cbig = eg.add(big);
    auto csmall = eg.add(small);
    eg.merge(cbig, csmall);
    eg.rebuild();

    EXPECT_TRUE(algebraic_eq(ctx, eg.extract(eg.find(cbig)), small));
}

TEST(EGraph, ClassCountDropsAfterMerge)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");

    EGraph eg{ctx};
    auto ca = eg.add(A);
    auto cb = eg.add(B);
    auto before = eg.class_count();
    eg.merge(ca, cb);
    eg.rebuild();
    EXPECT_EQ(eg.class_count(), before - 1);
}

// ---- round-trips across node kinds (add + extract) -------------------------

TEST(EGraph, RoundTripsEveryReachableNodeKind)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");

    EGraph eg{ctx};
    for (auto const* e: {
             make_negate(ctx, A),
             make_scalar_div(ctx, A, B),
             make_dot(ctx, A, B),
             make_ddot(ctx, A, B),
             make_ddot_alt(ctx, A, B),
             make_cross(ctx, A, B),
             make_sum(ctx, A, A), // → 2·A, exercises a ScalarLiteral leaf
         })
        EXPECT_TRUE(algebraic_eq(ctx, eg.extract(eg.add(e)), e));
}

TEST(EGraph, RoundTripsBoundNodes)
{
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    auto const* body = delta_ul(ctx, i, j);

    EGraph eg{ctx};
    // ExplicitSum with a symbolic bound (two children), and NoSum.
    auto const* es =
        make_explicit_sum(ctx, i, body, make_scalar(ctx, Rational{3}));
    auto const* ns = make_no_sum(ctx, i, body);
    EXPECT_TRUE(algebraic_eq(ctx, eg.extract(eg.add(es)), es));
    EXPECT_TRUE(algebraic_eq(ctx, eg.extract(eg.add(ns)), ns));
}

TEST(EGraph, IsMovable)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    EGraph eg{ctx};
    auto c = eg.add(A);

    EGraph moved{std::move(eg)}; // move-construct
    EXPECT_TRUE(algebraic_eq(ctx, moved.extract(moved.find(c)), A));

    EGraph eg2{ctx};
    eg2 = std::move(moved); // move-assign
    EXPECT_TRUE(algebraic_eq(ctx, eg2.extract(eg2.find(c)), A));
}

TEST(EGraph, MergingAClassWithItselfIsANoOp)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    EGraph eg{ctx};
    auto c = eg.add(A);
    EXPECT_EQ(eg.merge(c, c), eg.find(c));
}

TEST(EGraph, RoundTripsTensorWithVoidSlot)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex i{ctx.alloc_index_id()};
    auto const* t = make_tensor_object(
        ctx,
        make_tensor_name("T"),
        {SlotBinding{IndexSlot{Level::Upper, Realm::Oblique, sp}, std::nullopt},
         SlotBinding{IndexSlot{Level::Lower, Realm::Oblique, sp}, i}},
        2);
    EGraph eg{ctx};
    EXPECT_TRUE(algebraic_eq(ctx, eg.extract(eg.add(t)), t));
}

TEST(EGraph, ConcreteAndLabelLeaves)
{
    Context ctx;
    auto const* sp = space_3d();
    auto const* tc = make_tensor_object(
        ctx,
        make_tensor_name("v"),
        {SlotBinding{
            IndexSlot{Level::Upper, Realm::Orthonormal, sp}, ConcreteIndex{2}}},
        1);
    auto const* tl = make_tensor_object(
        ctx,
        make_tensor_name("A"),
        {SlotBinding{
            IndexSlot{Level::Lower, Realm::Label, nullptr},
            LabelIndex{make_index_name("vol")}}});

    EGraph eg{ctx};
    EXPECT_TRUE(algebraic_eq(ctx, eg.extract(eg.add(tc)), tc));
    EXPECT_TRUE(algebraic_eq(ctx, eg.extract(eg.add(tl)), tl));
    // A different concrete value is a distinct leaf (distinct e-class).
    auto const* tc3 = make_tensor_object(
        ctx,
        make_tensor_name("v"),
        {SlotBinding{
            IndexSlot{Level::Upper, Realm::Orthonormal, sp}, ConcreteIndex{3}}},
        1);
    EXPECT_NE(eg.find(eg.add(tc)), eg.find(eg.add(tc3)));
}

// ---- e-matching ------------------------------------------------------------

TEST(EMatch, FindsContractionAndBindsFreeIndices)
{
    Context ctx;
    CountableIndex p{ctx.alloc_index_id()};
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    EGraph eg{ctx};
    auto root = eg.add(contraction(ctx, q, m, n));

    auto ms = eg.ematch(contraction(ctx, p, a, b));
    ASSERT_FALSE(ms.empty());
    // The match lands on the contraction's class and binds a→m, b→n.
    bool at_root = false;
    for (auto const& [cls, bnd]: ms)
        if (cls == eg.find(root))
            at_root = true;
    EXPECT_TRUE(at_root);
    EXPECT_TRUE(binds_to_delta_mn(ctx, ms, a, b, m, n));
}

TEST(EMatch, MatchesModuloFactorOrder)
{
    // The target writes the two delta factors in the opposite order; the
    // component product is AC, so the pattern still matches.
    Context ctx;
    CountableIndex p{ctx.alloc_index_id()};
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    EGraph eg{ctx};
    (void)eg.add(make_explicit_sum(
        ctx,
        q,
        make_tensor_product(ctx, delta_ul(ctx, q, n), delta_ul(ctx, q, m))));

    auto ms = eg.ematch(contraction(ctx, p, a, b));
    EXPECT_TRUE(binds_to_delta_mn(ctx, ms, a, b, m, n));
}

TEST(EMatch, FindsPatternInSubexpression)
{
    Context ctx;
    CountableIndex p{ctx.alloc_index_id()};
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    CountableIndex r{ctx.alloc_index_id()};
    CountableIndex s{ctx.alloc_index_id()};

    EGraph eg{ctx};
    // δ_{rs} + Σ_q δ^q_m δ^q_n  — the contraction is a sub-term.
    (void)eg.add(make_sum(ctx, delta_ll(ctx, r, s), contraction(ctx, q, m, n)));

    auto ms = eg.ematch(contraction(ctx, p, a, b));
    EXPECT_TRUE(binds_to_delta_mn(ctx, ms, a, b, m, n));
}

TEST(EMatch, NoMatchYieldsNoBindings)
{
    Context ctx;
    CountableIndex p{ctx.alloc_index_id()};
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    EGraph eg{ctx};
    (void)eg.add(delta_ll(ctx, m, n)); // a bare delta, not a contraction

    EXPECT_TRUE(eg.ematch(contraction(ctx, p, a, b)).empty());
}

TEST(EMatch, SearchesNonRepresentativeNodes)
{
    // Prove the contraction equals the bare δ_{mn}, then extraction prefers the
    // smaller δ.  The pattern must still match via the contraction e-node that
    // is no longer the representative — the core e-graph payoff.
    Context ctx;
    CountableIndex p{ctx.alloc_index_id()};
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    EGraph eg{ctx};
    auto cx = eg.add(contraction(ctx, q, m, n));
    auto cy = eg.add(delta_ll(ctx, m, n));
    eg.merge(cx, cy);
    eg.rebuild();

    // Extraction picks the cheaper bare delta.
    EXPECT_TRUE(
        algebraic_eq(ctx, eg.extract(eg.find(cx)), delta_ll(ctx, m, n)));
    // But the pattern is still found in the class via the contraction e-node.
    auto ms = eg.ematch(contraction(ctx, p, a, b));
    bool at_class = false;
    for (auto const& [cls, bnd]: ms)
        if (cls == eg.find(cx))
            at_class = true;
    EXPECT_TRUE(at_class);
    EXPECT_TRUE(binds_to_delta_mn(ctx, ms, a, b, m, n));
}

TEST(EMatch, EpsDeltaTwoIndexThroughTheEGraph)
{
    // Σ_i Σ_j ε^{ijk} ε_{ijl}: nested binders + component-product AC + rank-3
    // Levi-Civita leaves, all through the generic e-matcher.
    Context ctx;
    auto const* sp = space_3d();
    auto eps =
        [&](Level lvl, CountableIndex x, CountableIndex y, CountableIndex z)
    {
        return make_levi_civita(
            ctx,
            Realm::Oblique,
            sp,
            {lvl, lvl, lvl},
            {IndexAssoc{x}, IndexAssoc{y}, IndexAssoc{z}});
    };
    auto eps_pair = [&](CountableIndex i,
                        CountableIndex j,
                        CountableIndex k,
                        CountableIndex l)
    {
        return make_explicit_sum(
            ctx,
            i,
            make_explicit_sum(
                ctx,
                j,
                make_tensor_product(
                    ctx,
                    eps(Level::Upper, i, j, k),
                    eps(Level::Lower, i, j, l))));
    };

    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};
    CountableIndex l{ctx.alloc_index_id()};
    auto const* lhs = eps_pair(i, j, k, l);

    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex bb{ctx.alloc_index_id()};
    CountableIndex c{ctx.alloc_index_id()};
    CountableIndex d{ctx.alloc_index_id()};

    EGraph eg{ctx};
    (void)eg.add(eps_pair(a, bb, c, d));

    auto ms = eg.ematch(lhs);
    ASSERT_FALSE(ms.empty());
    // RHS 2 δ^k_l instantiates to 2 δ^c_d under some match.
    auto const* rhs = make_tensor_product(
        ctx, make_scalar(ctx, Rational{2}), delta_ul(ctx, k, l));
    auto const* expected = make_tensor_product(
        ctx, make_scalar(ctx, Rational{2}), delta_ul(ctx, c, d));
    bool ok = false;
    for (auto const& [cls, bnd]: ms)
        if (algebraic_eq(ctx, instantiate(ctx, rhs, bnd), expected))
            ok = true;
    EXPECT_TRUE(ok);
}

TEST(EMatch, MatchesAcrossNodeKinds)
{
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");

    EGraph eg{ctx};
    (void)eg.add(make_negate(ctx, A));
    (void)eg.add(make_dot(ctx, A, B));
    (void)eg.add(make_ddot(ctx, A, B));
    (void)eg.add(make_ddot_alt(ctx, A, B));
    (void)eg.add(make_cross(ctx, A, B));
    (void)eg.add(make_scalar_div(ctx, A, B));
    (void)eg.add(make_sum(ctx, A, B));
    (void)eg.add(make_tensor_product(ctx, A, B)); // invariant → positional

    EXPECT_FALSE(eg.ematch(make_negate(ctx, A)).empty());
    EXPECT_FALSE(eg.ematch(make_dot(ctx, A, B)).empty());
    EXPECT_FALSE(eg.ematch(make_ddot(ctx, A, B)).empty());
    EXPECT_FALSE(eg.ematch(make_ddot_alt(ctx, A, B)).empty());
    EXPECT_FALSE(eg.ematch(make_cross(ctx, A, B)).empty());
    EXPECT_FALSE(eg.ematch(make_scalar_div(ctx, A, B)).empty());
    EXPECT_FALSE(eg.ematch(make_sum(ctx, A, B)).empty());
    EXPECT_FALSE(eg.ematch(make_tensor_product(ctx, A, B)).empty());

    // A leaf pattern is tried against every class, including non-leaf ones.
    EXPECT_FALSE(eg.ematch(A).empty());

    // A Sum pattern of the wrong cardinality does not match (size guard).
    auto const* C = obj(ctx, "C");
    EXPECT_TRUE(eg.ematch(make_sum(ctx, A, make_sum(ctx, B, C))).empty());
}

TEST(EMatch, MatchesSumFlattenedAcrossENodes)
{
    // A 3-addend sum canonicalizes to a nested binary Sum; the target is
    // flattened across e-nodes so the 3-term pattern lines up.
    Context ctx;
    auto const* A = obj(ctx, "A");
    auto const* B = obj(ctx, "B");
    auto const* C = obj(ctx, "C");

    EGraph eg{ctx};
    (void)eg.add(make_sum(ctx, A, make_sum(ctx, B, C)));
    EXPECT_FALSE(eg.ematch(make_sum(ctx, A, make_sum(ctx, B, C))).empty());
}

TEST(EMatch, MatchesNoSumAndBoundExplicitSum)
{
    Context ctx;
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    CountableIndex p{ctx.alloc_index_id()};
    CountableIndex a{ctx.alloc_index_id()};

    EGraph eg{ctx};
    (void)eg.add(make_no_sum(ctx, i, delta_ul(ctx, i, j)));
    (void)eg.add(make_explicit_sum(
        ctx, i, delta_ul(ctx, i, j), make_scalar(ctx, Rational{3})));

    // NoSum pattern binds its index and matches the body.
    EXPECT_FALSE(eg.ematch(make_no_sum(ctx, p, delta_ul(ctx, p, a))).empty());
    // ExplicitSum with a symbolic bound: the bound child is matched too.
    EXPECT_FALSE(
        eg
            .ematch(make_explicit_sum(
                ctx, p, delta_ul(ctx, p, a), make_scalar(ctx, Rational{3})))
            .empty());
    // A bound-presence mismatch does not match.
    EXPECT_TRUE(
        eg.ematch(make_explicit_sum(ctx, p, delta_ul(ctx, p, a))).empty());
}

// ---- saturation ------------------------------------------------------------

namespace
{
// The delta-contraction identity  Σ_p δ^p_a δ^p_b = δ_{ab}, with fresh ids.
auto contraction_rule(Context& ctx) -> Identity
{
    CountableIndex p{ctx.alloc_index_id()};
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    return Identity{
        "delta-contraction", contraction(ctx, p, a, b), delta_ll(ctx, a, b)};
}

// The two-index eps-delta identity  Σ_i Σ_j ε^{ijk} ε_{ijl} = 2 δ^k_l.
auto eps_delta_rule(Context& ctx) -> Identity
{
    auto eps =
        [&](Level lvl, CountableIndex x, CountableIndex y, CountableIndex z)
    {
        return make_levi_civita(
            ctx,
            Realm::Oblique,
            space_3d(),
            {lvl, lvl, lvl},
            {IndexAssoc{x}, IndexAssoc{y}, IndexAssoc{z}});
    };
    CountableIndex i{ctx.alloc_index_id()};
    CountableIndex j{ctx.alloc_index_id()};
    CountableIndex k{ctx.alloc_index_id()};
    CountableIndex l{ctx.alloc_index_id()};
    auto const* lhs = make_explicit_sum(
        ctx,
        i,
        make_explicit_sum(
            ctx,
            j,
            make_tensor_product(
                ctx, eps(Level::Upper, i, j, k), eps(Level::Lower, i, j, l))));
    auto const* rhs = make_tensor_product(
        ctx, make_scalar(ctx, Rational{2}), delta_ul(ctx, k, l));
    return Identity{"eps-delta-2", lhs, rhs};
}
} // namespace

TEST(Saturate, ContractsDelta)
{
    Context ctx;
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    EGraph eg{ctx};
    auto root = eg.add(contraction(ctx, q, m, n));
    int passes = eg.saturate({contraction_rule(ctx)});

    EXPECT_TRUE(
        algebraic_eq(ctx, eg.extract(eg.find(root)), delta_ll(ctx, m, n)));
    EXPECT_LT(passes, 30); // converged below the cap
}

TEST(Saturate, ContractsEpsDelta)
{
    Context ctx;
    auto eps =
        [&](Level lvl, CountableIndex x, CountableIndex y, CountableIndex z)
    {
        return make_levi_civita(
            ctx,
            Realm::Oblique,
            space_3d(),
            {lvl, lvl, lvl},
            {IndexAssoc{x}, IndexAssoc{y}, IndexAssoc{z}});
    };
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    CountableIndex c{ctx.alloc_index_id()};
    CountableIndex d{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(
        ctx,
        a,
        make_explicit_sum(
            ctx,
            b,
            make_tensor_product(
                ctx, eps(Level::Upper, a, b, c), eps(Level::Lower, a, b, d))));

    EGraph eg{ctx};
    auto root = eg.add(target);
    (void)eg.saturate({eps_delta_rule(ctx)});

    auto const* expected = make_tensor_product(
        ctx, make_scalar(ctx, Rational{2}), delta_ul(ctx, c, d));
    EXPECT_TRUE(algebraic_eq(ctx, eg.extract(eg.find(root)), expected));
}

TEST(Saturate, RewritesNestedSubexpression)
{
    // δ_{rs} + Σ_q δ^q_m δ^q_n  saturates to  δ_{rs} + δ_{mn} via congruence.
    Context ctx;
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    CountableIndex r{ctx.alloc_index_id()};
    CountableIndex s{ctx.alloc_index_id()};

    EGraph eg{ctx};
    auto root =
        eg.add(make_sum(ctx, delta_ll(ctx, r, s), contraction(ctx, q, m, n)));
    (void)eg.saturate({contraction_rule(ctx)});

    auto const* expected =
        make_sum(ctx, delta_ll(ctx, r, s), delta_ll(ctx, m, n));
    EXPECT_TRUE(algebraic_eq(ctx, eg.extract(eg.find(root)), expected));
}

TEST(Saturate, ReachesFixedPointAndIsIdempotent)
{
    Context ctx;
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    EGraph eg{ctx};
    (void)eg.add(contraction(ctx, q, m, n));
    (void)eg.saturate({contraction_rule(ctx)});
    // A second saturation changes nothing: one pass, then convergence.
    EXPECT_EQ(eg.saturate({contraction_rule(ctx)}), 1);
}

TEST(Saturate, NoMatchLeavesGraphUnchanged)
{
    Context ctx;
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    EGraph eg{ctx};
    auto root = eg.add(delta_ll(ctx, m, n)); // a bare delta, no contraction
    auto before = eg.class_count();

    EXPECT_EQ(eg.saturate({contraction_rule(ctx)}), 1); // one no-op pass
    EXPECT_EQ(eg.class_count(), before);
    EXPECT_TRUE(
        algebraic_eq(ctx, eg.extract(eg.find(root)), delta_ll(ctx, m, n)));
}

TEST(Saturate, RespectsIterationCap)
{
    Context ctx;
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    EGraph eg{ctx};
    (void)eg.add(contraction(ctx, q, m, n));
    EXPECT_EQ(eg.saturate({contraction_rule(ctx)}, 1), 1); // hard stop after 1
}

// ---- built-in distributivity (vibe 000048) ---------------------------------

TEST(Distribute, RightSum)
{
    // X·(A+B) ≡ X·A + X·B
    Context ctx;
    auto* X = obj(ctx, "X");
    auto* A = obj(ctx, "A");
    auto* B = obj(ctx, "B");

    EGraph eg{ctx};
    auto root = eg.add(make_tensor_product(ctx, X, make_sum(ctx, A, B)));
    (void)eg.saturate({}); // distribution is built-in, no data rules needed

    auto expected = eg.add(make_sum(
        ctx, make_tensor_product(ctx, X, A), make_tensor_product(ctx, X, B)));
    EXPECT_EQ(eg.find(root), eg.find(expected));
}

TEST(Distribute, LeftSum)
{
    // (A+B)·X ≡ A·X + B·X
    Context ctx;
    auto* X = obj(ctx, "X");
    auto* A = obj(ctx, "A");
    auto* B = obj(ctx, "B");

    EGraph eg{ctx};
    auto root = eg.add(make_tensor_product(ctx, make_sum(ctx, A, B), X));
    (void)eg.saturate({});

    auto expected = eg.add(make_sum(
        ctx, make_tensor_product(ctx, A, X), make_tensor_product(ctx, B, X)));
    EXPECT_EQ(eg.find(root), eg.find(expected));
}

TEST(Distribute, OverDifference)
{
    // X·(A−B) ≡ X·A − X·B.  Canonical forms carry the sign as a coefficient
    // (A−B is A + (−1)·B), so the engine distributes the Sum and the −1 rides
    // along.
    Context ctx;
    auto* X = obj(ctx, "X");
    auto* A = obj(ctx, "A");
    auto* B = obj(ctx, "B");

    EGraph eg{ctx};
    auto root = eg.add(make_tensor_product(ctx, X, make_difference(ctx, A, B)));
    (void)eg.saturate({});

    auto expected = eg.add(make_difference(
        ctx, make_tensor_product(ctx, X, A), make_tensor_product(ctx, X, B)));
    EXPECT_EQ(eg.find(root), eg.find(expected));
}

TEST(Distribute, MultiFactor)
{
    // X·Y·(A+B) ≡ X·Y·A + X·Y·B — the binary rule plus iteration handles the
    // extra factor (the "rest of the product" is just the other binary child).
    Context ctx;
    auto* X = obj(ctx, "X");
    auto* Y = obj(ctx, "Y");
    auto* A = obj(ctx, "A");
    auto* B = obj(ctx, "B");

    auto xy = [&](Expr const* t)
    { return make_tensor_product(ctx, make_tensor_product(ctx, X, Y), t); };

    EGraph eg{ctx};
    auto root = eg.add(xy(make_sum(ctx, A, B)));
    (void)eg.saturate({});

    auto expected = eg.add(make_sum(ctx, xy(A), xy(B)));
    EXPECT_EQ(eg.find(root), eg.find(expected));
}

TEST(Distribute, DotDistributes)
{
    // (A+B)·C ≡ A·C + B·C for the Dot product too.
    Context ctx;
    auto* A = obj(ctx, "A");
    auto* B = obj(ctx, "B");
    auto* C = obj(ctx, "C");

    EGraph eg{ctx};
    auto root = eg.add(make_dot(ctx, make_sum(ctx, A, B), C));
    (void)eg.saturate({});

    auto expected =
        eg.add(make_sum(ctx, make_dot(ctx, A, C), make_dot(ctx, B, C)));
    EXPECT_EQ(eg.find(root), eg.find(expected));
}

TEST(Distribute, AllMultiplicativeOps)
{
    // The remaining distributive ops — DDot, DDotAlt, Cross — each distribute:
    // (A+B) ∘ C ≡ A∘C + B∘C.
    Context ctx;
    auto* A = obj(ctx, "A");
    auto* B = obj(ctx, "B");
    auto* C = obj(ctx, "C");

    auto check = [&](auto make_op)
    {
        EGraph eg{ctx};
        auto root = eg.add(make_op(make_sum(ctx, A, B), C));
        (void)eg.saturate({});
        auto expected = eg.add(make_sum(ctx, make_op(A, C), make_op(B, C)));
        EXPECT_EQ(eg.find(root), eg.find(expected));
    };

    check([&](Expr const* l, Expr const* r) { return make_ddot(ctx, l, r); });
    check([&](Expr const* l, Expr const* r)
          { return make_ddot_alt(ctx, l, r); });
    check([&](Expr const* l, Expr const* r) { return make_cross(ctx, l, r); });
}

TEST(Distribute, UnlocksAMatch)
{
    // Distribution exposes a product the rule could not see before: with
    // A·B → C, the target A·(B+D) distributes to A·B + A·D, then A·B rewrites
    // to C, giving C + A·D.
    Context ctx;
    auto* A = obj(ctx, "A");
    auto* B = obj(ctx, "B");
    auto* C = obj(ctx, "C");
    auto* D = obj(ctx, "D");
    Identity rule{"ab->c", make_tensor_product(ctx, A, B), C};

    EGraph eg{ctx};
    auto root = eg.add(make_tensor_product(ctx, A, make_sum(ctx, B, D)));
    (void)eg.saturate({rule});

    auto expected = eg.add(make_sum(ctx, C, make_tensor_product(ctx, A, D)));
    EXPECT_EQ(eg.find(root), eg.find(expected));
}

TEST(Distribute, NoSumFactorIsNoOp)
{
    // A plain product has nothing to distribute: saturation settles immediately
    // and adds no classes.
    Context ctx;
    auto* A = obj(ctx, "A");
    auto* B = obj(ctx, "B");

    EGraph eg{ctx};
    (void)eg.add(make_tensor_product(ctx, A, B));
    auto before = eg.class_count();
    EXPECT_EQ(eg.saturate({}), 1); // one no-op pass
    EXPECT_EQ(eg.class_count(), before);
}
