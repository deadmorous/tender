#include <tender/identities.hpp>

#include <tender/context.hpp>
#include <tender/expr.hpp>
#include <tender/index_space.hpp>

#include <vector>

namespace tender::identities
{

namespace
{
constexpr Level U = Level::Upper;
constexpr Level L = Level::Lower;

auto fresh(Context& ctx) -> CountableIndex
{
    return CountableIndex{ctx.alloc_index_id()};
}
} // namespace

auto delta_contraction(Context& ctx, IndexSpace const* space) -> Identity
{
    auto d = [&](Level la, Level lb, CountableIndex a, CountableIndex b)
    { return make_delta(ctx, Realm::Oblique, space, la, lb, a, b); };

    auto const p = fresh(ctx);
    auto const a = fresh(ctx);
    auto const b = fresh(ctx);
    auto const* lhs = make_explicit_sum(
        ctx, p, make_tensor_product(ctx, d(U, L, p, a), d(U, L, p, b)));
    return Identity{"delta-contraction", lhs, d(L, L, a, b)};
}

auto delta_trace(Context& ctx, IndexSpace const* space) -> Identity
{
    auto const p = fresh(ctx);
    auto const* lhs = make_explicit_sum(
        ctx, p, make_delta(ctx, Realm::Oblique, space, U, L, p, p));
    auto const dim = static_cast<std::int64_t>(space->values().size());
    return Identity{"delta-trace", lhs, make_scalar(ctx, Rational{dim})};
}

auto eps_delta_1(Context& ctx) -> Identity
{
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
    auto d = [&](CountableIndex a, CountableIndex b)
    { return make_delta(ctx, Realm::Oblique, sp, U, L, a, b); };

    auto const i = fresh(ctx);
    auto const j = fresh(ctx);
    auto const k = fresh(ctx);
    auto const l = fresh(ctx);
    auto const m = fresh(ctx);
    auto const* lhs = make_explicit_sum(
        ctx, i, make_tensor_product(ctx, eps(U, i, j, k), eps(L, i, l, m)));
    auto const* rhs = make_difference(
        ctx,
        make_tensor_product(ctx, d(j, l), d(k, m)),
        make_tensor_product(ctx, d(j, m), d(k, l)));
    return Identity{"eps-delta-1", lhs, rhs};
}

auto eps_delta_2(Context& ctx) -> Identity
{
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

    auto const i = fresh(ctx);
    auto const j = fresh(ctx);
    auto const k = fresh(ctx);
    auto const l = fresh(ctx);
    auto const* lhs = make_explicit_sum(
        ctx,
        i,
        make_explicit_sum(
            ctx, j, make_tensor_product(ctx, eps(U, i, j, k), eps(L, i, j, l))));
    auto const* rhs = make_tensor_product(
        ctx,
        make_scalar(ctx, Rational{2}),
        make_delta(ctx, Realm::Oblique, sp, U, L, k, l));
    return Identity{"eps-delta-2", lhs, rhs};
}

} // namespace tender::identities
