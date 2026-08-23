#include <tender/engine.hpp>

#include <tender/derivation.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>

#include <gtest/gtest.h>

using namespace tender;
using namespace tender::engine;

namespace
{

auto delta_ul(
    Context& ctx,
    IndexSpace const* sp,
    CountableIndex a,
    CountableIndex b) -> Expr const*
{
    return make_delta(ctx, Realm::Oblique, sp, Level::Upper, Level::Lower, a, b);
}

auto delta_ll(
    Context& ctx,
    IndexSpace const* sp,
    CountableIndex a,
    CountableIndex b) -> Expr const*
{
    return make_delta(ctx, Realm::Oblique, sp, Level::Lower, Level::Lower, a, b);
}

// Σ_q δ^q_m δ^q_n — the contractible form the delta_contraction rule collapses.
auto contraction(
    Context& ctx,
    IndexSpace const* sp,
    CountableIndex q,
    CountableIndex m,
    CountableIndex n) -> Expr const*
{
    return make_explicit_sum(
        ctx,
        q,
        make_tensor_product(
            ctx, delta_ul(ctx, sp, q, m), delta_ul(ctx, sp, q, n)));
}

// The δ-contraction rule, built locally: the shipped identity library lives
// in Python (tender/identities.py) so it can be extended without a rebuild,
// so the C++ engine tests carry the small rules they need themselves.
auto rules_eps_delta(Context& ctx) -> std::vector<Identity>
{
    auto const* sp = space_3d();
    CountableIndex const p{ctx.alloc_index_id()};
    CountableIndex const a{ctx.alloc_index_id()};
    CountableIndex const b{ctx.alloc_index_id()};
    return {Identity{
        "delta-contraction",
        make_explicit_sum(
            ctx,
            p,
            make_tensor_product(
                ctx, delta_ul(ctx, sp, p, a), delta_ul(ctx, sp, p, b))),
        delta_ll(ctx, sp, a, b)}};
}

} // namespace

// ---- prove_equal --------------------------------------------------------

TEST(ProveEqual, ProvesByFiringARule)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    auto const res = prove_equal(
        ctx,
        contraction(ctx, sp, q, m, n),
        delta_ll(ctx, sp, m, n),
        rules_eps_delta(ctx));

    EXPECT_TRUE(res.proved());
    EXPECT_EQ(res.status, ProofStatus::Proved);
    EXPECT_EQ(res.report.fired.at(0), 1); // the trace attributes the work
}

TEST(ProveEqual, AlreadyEqualUnderCanonicalizationNeedsNoRules)
{
    // a ⊗ b vs a ⊗ b written differently is theory-T0 equal: proved with zero
    // passes and no rule firing.
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* b = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    auto const* two = make_scalar(ctx, Rational{2});

    auto const res = prove_equal(
        ctx,
        make_tensor_product(ctx, two, make_tensor_product(ctx, a, b)),
        make_tensor_product(ctx, make_tensor_product(ctx, a, two), b),
        {});

    EXPECT_TRUE(res.proved());
    EXPECT_EQ(res.report.passes, 0);
}

TEST(ProveEqual, UnrelatedExpressionsExhaustWithoutProof)
{
    // Genuinely different expressions: the rules run to a fixed point and no
    // proof appears.  `Exhausted` — NOT a claim that they are unequal.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};
    auto const* c = make_tensor_object(ctx, make_tensor_name("c"), {}, 2);

    auto const res =
        prove_equal(ctx, delta_ll(ctx, sp, m, n), c, rules_eps_delta(ctx));

    EXPECT_FALSE(res.proved());
    EXPECT_EQ(res.status, ProofStatus::Exhausted);
    EXPECT_EQ(res.report.outcome, SaturateOutcome::Saturated);
}

TEST(ProveEqual, BudgetTripIsInconclusiveNotDisproof)
{
    // The safety property: a proof that exists but is cut short by the budget
    // must report `Budget`, never `Exhausted` — a caller reading "not proved"
    // as "not equal" would be wrong.  Zero passes allowed, so the rule that
    // would prove it never fires.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    auto const res = prove_equal(
        ctx,
        contraction(ctx, sp, q, m, n),
        delta_ll(ctx, sp, m, n),
        rules_eps_delta(ctx),
        SaturateBudget{.max_passes = 0});

    EXPECT_FALSE(res.proved());
    EXPECT_EQ(res.status, ProofStatus::Budget);
    EXPECT_NE(res.status, ProofStatus::Exhausted);
}

TEST(ProveEqual, NodeBudgetAlsoReportsInconclusive)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    auto const res = prove_equal(
        ctx,
        contraction(ctx, sp, q, m, n),
        delta_ll(ctx, sp, m, n),
        rules_eps_delta(ctx),
        SaturateBudget{.max_passes = 30, .max_nodes = 1});

    EXPECT_EQ(res.status, ProofStatus::Budget);
    EXPECT_EQ(res.report.outcome, SaturateOutcome::NodeBudget);
}

TEST(ProveEqual, StopsEarlyOnceTheGoalIsReached)
{
    // The goal check ends saturation as soon as the sides join, rather than
    // running the rule set to its fixed point.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    auto const res = prove_equal(
        ctx,
        contraction(ctx, sp, q, m, n),
        delta_ll(ctx, sp, m, n),
        rules_eps_delta(ctx));

    EXPECT_TRUE(res.proved());
    EXPECT_EQ(res.report.outcome, SaturateOutcome::EarlyStop);
    EXPECT_EQ(res.report.passes, 1); // proved on the first pass, then stopped
}

// ---- simplify -----------------------------------------------------------

TEST(Simplify, ContractsAndReportsCompletion)
{
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    auto const res =
        simplify(ctx, contraction(ctx, sp, q, m, n), rules_eps_delta(ctx));

    EXPECT_TRUE(res.complete());
    EXPECT_TRUE(algebraic_eq(ctx, res.expr, delta_ll(ctx, sp, m, n)));
}

TEST(Simplify, BudgetTripStillReturnsTheBestFormFound)
{
    // The loud-fallback contract: on a budget trip the caller still gets a
    // usable expression, flagged incomplete rather than silently presented as
    // a fixed point.
    Context ctx;
    auto const* sp = space_3d();
    CountableIndex q{ctx.alloc_index_id()};
    CountableIndex m{ctx.alloc_index_id()};
    CountableIndex n{ctx.alloc_index_id()};

    auto const res = simplify(
        ctx,
        contraction(ctx, sp, q, m, n),
        rules_eps_delta(ctx),
        SaturateBudget{.max_passes = 0});

    EXPECT_FALSE(res.complete());
    ASSERT_NE(res.expr, nullptr);
    // Unsimplified, but algebraically the input — never garbage.
    EXPECT_TRUE(algebraic_eq(ctx, res.expr, contraction(ctx, sp, q, m, n)));
}

TEST(Simplify, MultiTermLhsRuleIsReportedAsSkipped)
{
    // A rule the engine cannot compile must be *visible* in the trace, not
    // silently inert (vibe 000096).
    Context ctx;
    auto const* a = make_tensor_object(ctx, make_tensor_name("a"), {}, 1);
    auto const* b = make_tensor_object(ctx, make_tensor_name("b"), {}, 1);
    Identity multi{"sum-lhs", make_sum(ctx, a, b), a};

    auto const res = simplify(ctx, a, {multi});
    ASSERT_EQ(res.report.skipped.size(), 1u);
    EXPECT_EQ(res.report.skipped.at(0), 0u);
}
