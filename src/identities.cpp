#include <tender/identities.hpp>

#include <tender/context.hpp>
#include <tender/expr.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
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

// Orthonormal indices carry no upper/lower distinction, so the library spells
// every Orthonormal index lower by convention (vibe 000047); other realms keep
// the requested level.  Applied wherever the level is a default choice, so a
// rule and the targets it matches are spelled identically (match_slot is
// level-exact, and canonicalize deliberately does not coerce levels).
auto level_for(Realm realm, Level requested) -> Level
{
    return realm == Realm::Orthonormal ? Level::Lower : requested;
}

// A rank-`r` subtree pattern variable (vibe 000051): a slot-less,
// non-well-known named tensor, which the matcher binds to a whole target
// factor.
//
// The *name* matters (see the header's warning): canon sorts symmetric
// contraction chains by name, so a rule spelled with an ill-chosen variable
// name silently matches only part of the alphabet.  The names here are the ones
// the fire-tests in identities_test verify across a spread of target names.
auto var(Context& ctx, char const* name, int rank) -> Expr const*
{
    return make_tensor_object(ctx, make_tensor_name(name), {}, rank);
}
} // namespace

auto delta_contraction(Context& ctx, IndexSpace const* space, Realm realm)
    -> Identity
{
    auto d = [&](Level la, Level lb, CountableIndex a, CountableIndex b)
    {
        return make_delta(
            ctx, realm, space, level_for(realm, la), level_for(realm, lb), a, b);
    };

    auto const p = fresh(ctx);
    auto const a = fresh(ctx);
    auto const b = fresh(ctx);
    auto const* lhs = make_explicit_sum(
        ctx, p, make_tensor_product(ctx, d(U, L, p, a), d(U, L, p, b)));
    return Identity{"delta-contraction", lhs, d(L, L, a, b)};
}

auto delta_trace(Context& ctx, IndexSpace const* space, Realm realm) -> Identity
{
    auto const p = fresh(ctx);
    auto const* lhs = make_explicit_sum(
        ctx,
        p,
        make_delta(
            ctx, realm, space, level_for(realm, U), level_for(realm, L), p, p));
    auto const dim = static_cast<std::int64_t>(space->values().size());
    return Identity{"delta-trace", lhs, make_scalar(ctx, Rational{dim})};
}

auto eps_delta_1(Context& ctx, Realm realm) -> Identity
{
    auto const* sp = space_3d();
    auto eps =
        [&](Level lvl, CountableIndex x, CountableIndex y, CountableIndex z)
    {
        auto const el = level_for(realm, lvl);
        return make_levi_civita(
            ctx,
            realm,
            sp,
            {el, el, el},
            {IndexAssoc{x}, IndexAssoc{y}, IndexAssoc{z}});
    };
    auto d = [&](CountableIndex a, CountableIndex b)
    {
        return make_delta(
            ctx, realm, sp, level_for(realm, U), level_for(realm, L), a, b);
    };

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

auto eps_delta_2(Context& ctx, Realm realm) -> Identity
{
    auto const* sp = space_3d();
    auto eps =
        [&](Level lvl, CountableIndex x, CountableIndex y, CountableIndex z)
    {
        auto const el = level_for(realm, lvl);
        return make_levi_civita(
            ctx,
            realm,
            sp,
            {el, el, el},
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
        make_delta(
            ctx, realm, sp, level_for(realm, U), level_for(realm, L), k, l));
    return Identity{"eps-delta-2", lhs, rhs};
}

// ---- cross group --------------------------------------------------------

auto bac_cab(Context& ctx) -> Identity
{
    auto const* u = var(ctx, "u", 1);
    auto const* v = var(ctx, "v", 1);
    auto const* w = var(ctx, "w", 1);
    // u × (v × w) = v (u·w) − w (u·v)
    auto const* lhs = make_cross(ctx, u, make_cross(ctx, v, w));
    auto const* rhs = make_difference(
        ctx,
        make_tensor_product(ctx, v, make_dot(ctx, u, w)),
        make_tensor_product(ctx, w, make_dot(ctx, u, v)));
    return Identity{"bac-cab", lhs, rhs};
}

auto cross_identity(Context& ctx) -> Identity
{
    auto const* u = var(ctx, "u", 1);
    auto const* id = make_identity(ctx);
    return Identity{
        "cross-identity", make_cross(ctx, u, id), make_cross(ctx, id, u)};
}

auto cross_removal(Context& ctx) -> Identity
{
    auto const* u = var(ctx, "u", 1);
    auto const* v = var(ctx, "v", 1);
    auto const* id = make_identity(ctx);
    // u × (v × I) = v ⊗ u − (u·v) I
    auto const* lhs = make_cross(ctx, u, make_cross(ctx, v, id));
    auto const* rhs = make_difference(
        ctx,
        make_tensor_product(ctx, v, u),
        make_tensor_product(ctx, make_dot(ctx, u, v), id));
    return Identity{"cross-removal", lhs, rhs};
}

auto lagrange(Context& ctx) -> Identity
{
    auto const* p = var(ctx, "p", 1);
    auto const* q = var(ctx, "q", 1);
    auto const* r = var(ctx, "r", 1);
    auto const* s = var(ctx, "s", 1);
    // (p × q) · (r × s) = (p·r)(q·s) − (p·s)(q·r)
    auto const* lhs =
        make_dot(ctx, make_cross(ctx, p, q), make_cross(ctx, r, s));
    auto const* rhs = make_difference(
        ctx,
        make_tensor_product(ctx, make_dot(ctx, p, r), make_dot(ctx, q, s)),
        make_tensor_product(ctx, make_dot(ctx, p, s), make_dot(ctx, q, r)));
    return Identity{"lagrange", lhs, rhs};
}

// ---- dyadic group -------------------------------------------------------

auto trace_cyclic(Context& ctx) -> Identity
{
    auto const* a = var(ctx, "U", 2);
    auto const* b = var(ctx, "W", 2);
    return Identity{
        "trace-cyclic",
        make_trace(ctx, make_dot(ctx, a, b)),
        make_trace(ctx, make_dot(ctx, b, a))};
}

auto identity_dot(Context& ctx) -> Identity
{
    auto const* u = var(ctx, "u", 1);
    return Identity{"identity-dot", make_dot(ctx, make_identity(ctx), u), u};
}

// ---- groups -------------------------------------------------------------

auto group_names() -> std::vector<std::string_view>
{
    return {"eps_delta", "cross", "dyadic"};
}

auto group(
    Context& ctx,
    std::string_view name,
    Realm realm,
    IndexSpace const* space) -> std::vector<Identity>
{
    auto const* sp = space ? space : space_3d();
    if (name == "eps_delta")
        return {
            delta_contraction(ctx, sp, realm),
            delta_trace(ctx, sp, realm),
            eps_delta_1(ctx, realm),
            eps_delta_2(ctx, realm)};
    if (name == "cross")
        return {
            bac_cab(ctx),
            cross_identity(ctx),
            cross_removal(ctx),
            lagrange(ctx)};
    if (name == "dyadic")
        return {trace_cyclic(ctx), identity_dot(ctx)};
    throw std::invalid_argument(
        "identities::group: unknown group name \'" + std::string{name} + "\'");
}

auto all_rules(Context& ctx, Realm realm, IndexSpace const* space)
    -> std::vector<Identity>
{
    std::vector<Identity> out;
    for (auto n: group_names())
    {
        auto g = group(ctx, n, realm, space);
        out.insert(out.end(), g.begin(), g.end());
    }
    return out;
}

} // namespace tender::identities
