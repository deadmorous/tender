#include <tender/expr.hpp>

#include <stdexcept>

namespace tender
{

// ---- Generic factories -------------------------------------------------

auto make_tensor_object(
    Context& ctx,
    TensorName name,
    std::vector<SlotBinding> slots,
    std::optional<int> rank) -> Expr const*
{
    return ctx.make<Expr>(TensorObject{
        .name = std::move(name),
        .rank = rank,
        .label = std::nullopt,
        .slots = std::move(slots)});
}

auto make_scalar(Context& ctx, Rational value) -> Expr const*
{
    return ctx.make<Expr>(ScalarLiteral{value});
}

// ---- Unary factory -----------------------------------------------------

auto make_negate(Context& ctx, Expr const* operand) -> Expr const*
{
    return ctx.make<Expr>(Negate{operand});
}

// ---- Binary factories --------------------------------------------------

auto make_sum(Context& ctx, Expr const* left, Expr const* right) -> Expr const*
{
    return ctx.make<Expr>(Sum{left, right});
}

auto make_difference(Context& ctx, Expr const* left, Expr const* right)
    -> Expr const*
{
    return ctx.make<Expr>(Difference{left, right});
}

auto make_tensor_product(Context& ctx, Expr const* left, Expr const* right)
    -> Expr const*
{
    return ctx.make<Expr>(TensorProduct{left, right});
}

auto make_scalar_div(Context& ctx, Expr const* left, Expr const* right)
    -> Expr const*
{
    return ctx.make<Expr>(ScalarDiv{left, right});
}

auto make_dot(Context& ctx, Expr const* left, Expr const* right) -> Expr const*
{
    return ctx.make<Expr>(Dot{left, right});
}

auto make_ddot(Context& ctx, Expr const* left, Expr const* right) -> Expr const*
{
    return ctx.make<Expr>(DDot{left, right});
}

auto make_ddot_alt(Context& ctx, Expr const* left, Expr const* right)
    -> Expr const*
{
    return ctx.make<Expr>(DDotAlt{left, right});
}

auto make_cross(Context& ctx, Expr const* left, Expr const* right) -> Expr const*
{
    return ctx.make<Expr>(Cross{left, right});
}

// ---- Summation annotation factories ------------------------------------

auto make_explicit_sum(
    Context& ctx,
    CountableIndex index,
    Expr const* body,
    Expr const* bound) -> Expr const*
{
    return ctx.make<Expr>(ExplicitSum{index, body, bound});
}

auto make_no_sum(Context& ctx, CountableIndex index, Expr const* body)
    -> Expr const*
{
    return ctx.make<Expr>(NoSum{index, body});
}

// ---- Well-known tensor factories ---------------------------------------

auto make_identity(Context& ctx) -> Expr const*
{
    return ctx.make<Expr>(TensorObject{
        .name = make_tensor_name("I"),
        .rank = 2,
        .label = TensorLabel::Identity,
        .slots = {}});
}

auto make_delta(
    Context& ctx,
    Realm realm,
    IndexSpace const* space,
    Level level0,
    Level level1,
    IndexAssoc index0,
    IndexAssoc index1) -> Expr const*
{
    return ctx.make<Expr>(TensorObject{
        .name = make_tensor_name("\\delta"),
        .rank = 0,
        .label = TensorLabel::Delta,
        .slots = {
            SlotBinding{IndexSlot{level0, realm, space}, std::move(index0)},
            SlotBinding{IndexSlot{level1, realm, space}, std::move(index1)}}});
}

auto make_levi_civita(
    Context& ctx,
    Realm realm,
    IndexSpace const* space,
    std::vector<Level> levels,
    std::vector<IndexAssoc> indices) -> Expr const*
{
    auto const n = space->values().size();
    if (levels.size() != n)
        throw std::invalid_argument(
            "make_levi_civita: levels.size() must equal space dimension");
    if (indices.size() != n)
        throw std::invalid_argument(
            "make_levi_civita: indices.size() must equal space dimension");

    std::vector<SlotBinding> slots;
    slots.reserve(n);
    for (std::size_t k = 0; k < n; ++k)
        slots.push_back(SlotBinding{
            IndexSlot{levels[k], realm, space}, std::move(indices[k])});

    return ctx.make<Expr>(TensorObject{
        .name = make_tensor_name("\\varepsilon"),
        .rank = 0,
        .label = TensorLabel::LeviCivita,
        .slots = std::move(slots)});
}

} // namespace tender
