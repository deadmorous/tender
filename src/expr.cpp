#include <tender/expr.hpp>

#include <stdexcept>

namespace tender
{

// ---- internal helpers --------------------------------------------------

static auto count_index_slots(std::vector<Slot> const& slots) -> std::size_t
{
    std::size_t n = 0;
    for (auto const& s: slots)
        if (std::holds_alternative<IndexSlot>(s))
            ++n;
    return n;
}

// Build a two-slot TensorObject under a given name. Used by make_delta and
// make_identity which share identical slot layouts.
static auto make_two_slot_tensor(
    Context& ctx,
    TensorName name,
    Realm realm,
    IndexSpace const* space,
    Level level0,
    Level level1,
    IndexAssoc index0,
    IndexAssoc index1) -> Expr const*
{
    return ctx.make<Expr>(TensorObject{
        std::move(name),
        {IndexSlot{level0, realm, space}, IndexSlot{level1, realm, space}},
        {std::move(index0), std::move(index1)}});
}

// ---- Generic factories -------------------------------------------------

auto make_tensor_object(
    Context& ctx,
    TensorName name,
    std::vector<Slot> slots,
    std::vector<IndexAssoc> indices) -> Expr const*
{
    if (indices.size() != count_index_slots(slots))
        throw std::invalid_argument(
            "make_tensor_object: indices.size() must equal number of IndexSlots in slots");
    return ctx.make<Expr>(
        TensorObject{std::move(name), std::move(slots), std::move(indices)});
}

auto make_scalar_object(Context& ctx, TensorName name) -> Expr const*
{
    return ctx.make<Expr>(TensorObject{std::move(name), {}, {}});
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

auto make_delta(
    Context& ctx,
    Realm realm,
    IndexSpace const* space,
    Level level0,
    Level level1,
    IndexAssoc index0,
    IndexAssoc index1) -> Expr const*
{
    return make_two_slot_tensor(
        ctx,
        make_tensor_name("\\delta"),
        realm,
        space,
        level0,
        level1,
        std::move(index0),
        std::move(index1));
}

auto make_identity(
    Context& ctx,
    Realm realm,
    IndexSpace const* space,
    Level level0,
    Level level1,
    IndexAssoc index0,
    IndexAssoc index1) -> Expr const*
{
    return make_two_slot_tensor(
        ctx,
        make_tensor_name("I"),
        realm,
        space,
        level0,
        level1,
        std::move(index0),
        std::move(index1));
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

    std::vector<Slot> slots;
    slots.reserve(n);
    for (auto level: levels)
        slots.push_back(IndexSlot{level, realm, space});

    return ctx.make<Expr>(TensorObject{
        make_tensor_name("\\varepsilon"), std::move(slots), std::move(indices)});
}

} // namespace tender
