#include <gtest/gtest.h>
#include <tender/derivation.hpp>
#include <tender/expr.hpp>
#include <tender/identity.hpp>
#include <tender/index_space.hpp>

#include <vector>

using namespace tender;

// ---- helpers ---------------------------------------------------------------

namespace
{

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

// The delta-contraction identity  Σ_p δ^p_A δ^p_B  =  δ_{AB}.
auto delta_contraction(Context& ctx, IndexSpace const* sp) -> Identity
{
    CountableIndex p{ctx.alloc_index_id()};
    CountableIndex a{ctx.alloc_index_id()};
    CountableIndex b{ctx.alloc_index_id()};
    auto const* lhs = make_explicit_sum(
        ctx,
        p,
        make_tensor_product(
            ctx, delta_ul(ctx, sp, p, a), delta_ul(ctx, sp, p, b)));
    auto const* rhs = delta_ll(ctx, sp, a, b);
    return Identity{"delta-contraction", lhs, rhs};
}

} // namespace

// ---- match -----------------------------------------------------------------

TEST(Match, DeltaContractionBindsFreeIndices)
{
    Context ctx;
    auto const* sp = space_3d();
    auto id = delta_contraction(ctx, sp);

    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(
        ctx,
        q,
        make_tensor_product(
            ctx, delta_ul(ctx, sp, q, m), delta_ul(ctx, sp, q, n)));

    // match expects canonical forms (apply_identity arranges this).
    auto bnd = match(
        steps::canonicalize(ctx, id.lhs), steps::canonicalize(ctx, target));
    ASSERT_TRUE(bnd.has_value());
}

TEST(Match, DistinctNodeKindsDoNotMatch)
{
    Context ctx;
    auto const* sp = space_3d();
    auto id = delta_contraction(ctx, sp);

    // A bare delta is not an ExplicitSum, so the LHS pattern must not match.
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const* target = delta_ul(ctx, sp, m, n);

    auto bnd = match(
        steps::canonicalize(ctx, id.lhs), steps::canonicalize(ctx, target));
    EXPECT_FALSE(bnd.has_value());
}

// ---- apply_identity --------------------------------------------------------

TEST(ApplyIdentity, DeltaContractionRewritesToDelta)
{
    Context ctx;
    auto const* sp = space_3d();
    auto id = delta_contraction(ctx, sp);

    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(
        ctx,
        q,
        make_tensor_product(
            ctx, delta_ul(ctx, sp, q, m), delta_ul(ctx, sp, q, n)));

    auto const* result = apply_identity(ctx, target, id);
    EXPECT_TRUE(algebraic_eq(ctx, result, delta_ll(ctx, sp, m, n)));
}

TEST(ApplyIdentity, FactorOrderIndependent)
{
    Context ctx;
    auto const* sp = space_3d();
    auto id = delta_contraction(ctx, sp);

    // Same contraction with the two delta factors written in the other order;
    // AC matching must still fire.
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(
        ctx,
        q,
        make_tensor_product(
            ctx, delta_ul(ctx, sp, q, n), delta_ul(ctx, sp, q, m)));

    auto const* result = apply_identity(ctx, target, id);
    EXPECT_TRUE(algebraic_eq(ctx, result, delta_ll(ctx, sp, m, n)));
}

TEST(ApplyIdentity, NoMatchReturnsCanonicalUnchanged)
{
    Context ctx;
    auto const* sp = space_3d();
    auto id = delta_contraction(ctx, sp);

    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const* target = delta_ul(ctx, sp, m, n);

    auto const* result = apply_identity(ctx, target, id);
    EXPECT_TRUE(algebraic_eq(ctx, result, target));
}

TEST(ApplyIdentity, MatchesNestedSubexpression)
{
    Context ctx;
    auto const* sp = space_3d();
    auto id = delta_contraction(ctx, sp);

    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    CountableIndex r{ctx.alloc_index_id()};
    CountableIndex s{ctx.alloc_index_id()};
    auto const* contraction = make_explicit_sum(
        ctx,
        q,
        make_tensor_product(
            ctx, delta_ul(ctx, sp, q, m), delta_ul(ctx, sp, q, n)));
    // δ_{rs} + Σ_q δ^q_m δ^q_n  →  δ_{rs} + δ_{mn}
    auto const* target = make_sum(ctx, delta_ll(ctx, sp, r, s), contraction);

    auto const* result = apply_identity(ctx, target, id);
    auto const* expected =
        make_sum(ctx, delta_ll(ctx, sp, r, s), delta_ll(ctx, sp, m, n));
    EXPECT_TRUE(algebraic_eq(ctx, result, expected));
}

TEST(ApplyIdentity, AsDerivationStep)
{
    Context ctx;
    auto const* sp = space_3d();
    auto id = delta_contraction(ctx, sp);

    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(
        ctx,
        q,
        make_tensor_product(
            ctx, delta_ul(ctx, sp, q, m), delta_ul(ctx, sp, q, n)));

    Derivation drv{ctx, target};
    drv.step(steps::apply_identity(id));
    EXPECT_TRUE(algebraic_eq(ctx, drv.current(), delta_ll(ctx, sp, m, n)));
    EXPECT_EQ(drv.history().size(), 2u);
}

// ---- a real index identity: two-index eps-delta ----------------------------

TEST(ApplyIdentity, EpsDeltaTwoIndex)
{
    Context ctx;
    auto const* sp = space_3d();

    auto eps = [&](Level lvl, IndexAssoc x, IndexAssoc y, IndexAssoc z)
    {
        return make_levi_civita(
            ctx, Realm::Oblique, sp, {lvl, lvl, lvl}, {x, y, z});
    };

    // Identity:  Σ_i Σ_j ε^{ijk} ε_{ijl}  =  2 δ^k_l.
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
        ctx, make_scalar(ctx, Rational{2}), delta_ul(ctx, sp, k, l));
    Identity id{"eps-delta-2", lhs, rhs};

    // Target written with fresh indices.
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

    auto const* result = apply_identity(ctx, target, id);
    auto const* expected = make_tensor_product(
        ctx, make_scalar(ctx, Rational{2}), delta_ul(ctx, sp, c, d));
    EXPECT_TRUE(algebraic_eq(ctx, result, expected));
}
