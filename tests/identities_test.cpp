#include <gtest/gtest.h>
#include <tender/context.hpp>
#include <tender/derivation.hpp> // contract_eps_pair / contract_delta oracles
#include <tender/egraph.hpp>
#include <tender/expr.hpp>
#include <tender/identities.hpp>
#include <tender/index_space.hpp>

using namespace tender;

namespace
{
constexpr Level U = Level::Upper;
constexpr Level L = Level::Lower;

auto eps(
    Context& ctx,
    Level lvl,
    CountableIndex x,
    CountableIndex y,
    CountableIndex z) -> Expr const*
{
    return make_levi_civita(
        ctx,
        Realm::Oblique,
        space_3d(),
        {lvl, lvl, lvl},
        {IndexAssoc{x}, IndexAssoc{y}, IndexAssoc{z}});
}

auto d(Context& ctx, Level la, Level lb, CountableIndex a, CountableIndex b)
    -> Expr const*
{
    return make_delta(ctx, Realm::Oblique, space_3d(), la, lb, a, b);
}

// Saturate `target` with one identity and return the extracted result.
auto run(Context& ctx, Expr const* target, Identity rule) -> Expr const*
{
    EGraph eg{ctx};
    auto const root = eg.add(target);
    eg.saturate({std::move(rule)});
    return eg.extract(eg.find(root));
}
} // namespace

TEST(Identities, DeltaContraction)
{
    Context ctx;
    auto const* sp = space_3d();
    auto const q = CountableIndex{ctx.alloc_index_id()};
    auto const m = CountableIndex{ctx.alloc_index_id()};
    auto const n = CountableIndex{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(
        ctx,
        q,
        make_tensor_product(ctx, d(ctx, U, L, q, m), d(ctx, U, L, q, n)));

    auto const* result =
        run(ctx, target, identities::delta_contraction(ctx, sp));
    EXPECT_TRUE(algebraic_eq(ctx, result, d(ctx, L, L, m, n)));
}

TEST(Identities, DeltaTrace)
{
    Context ctx;
    auto const* sp = space_3d();
    auto const q = CountableIndex{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(ctx, q, d(ctx, U, L, q, q)); // Σ_q
                                                                        // δ^q_q

    auto const* result = run(ctx, target, identities::delta_trace(ctx, sp));
    EXPECT_TRUE(algebraic_eq(ctx, result, make_scalar(ctx, Rational{3})));
}

TEST(Identities, EpsDelta1MatchesOracle)
{
    // The δ-expansion is larger than the ε-form, so this only extracts
    // correctly because the cost function weights Levi-Civita symbols heavily.
    Context ctx;
    auto const a = CountableIndex{ctx.alloc_index_id()};
    auto const b = CountableIndex{ctx.alloc_index_id()};
    auto const c = CountableIndex{ctx.alloc_index_id()};
    auto const dd = CountableIndex{ctx.alloc_index_id()};
    auto const e = CountableIndex{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(
        ctx,
        a,
        make_tensor_product(ctx, eps(ctx, U, a, b, c), eps(ctx, L, a, dd, e)));

    auto const* oracle = steps::contract_eps_pair(ctx, target);
    auto const* result = run(ctx, target, identities::eps_delta_1(ctx));
    EXPECT_TRUE(algebraic_eq(ctx, result, oracle));
}

TEST(Identities, EpsDelta2MatchesOracle)
{
    Context ctx;
    auto const a = CountableIndex{ctx.alloc_index_id()};
    auto const b = CountableIndex{ctx.alloc_index_id()};
    auto const c = CountableIndex{ctx.alloc_index_id()};
    auto const dd = CountableIndex{ctx.alloc_index_id()};
    auto const* target = make_explicit_sum(
        ctx,
        a,
        make_explicit_sum(
            ctx,
            b,
            make_tensor_product(
                ctx, eps(ctx, U, a, b, c), eps(ctx, L, a, b, dd))));

    auto const* oracle = steps::contract_eps_pair(ctx, target);
    auto const* result = run(ctx, target, identities::eps_delta_2(ctx));
    EXPECT_TRUE(algebraic_eq(ctx, result, oracle));
}
