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
        .traits = std::nullopt,
        .slots = std::move(slots)});
}

auto make_scalar(Context& ctx, Rational value) -> Expr const*
{
    return ctx.make<Expr>(ScalarLiteral{value});
}

auto make_coordinate(
    Context& ctx, TensorName name, int chart_id, int slot, bool nonneg)
    -> Expr const*
{
    return ctx.make<Expr>(TensorObject{
        .name = std::move(name),
        .rank = 0,
        .traits =
            TensorTraits{.coordinate = CoordinateRef{chart_id, slot, nonneg}},
        .slots = {}});
}

auto make_scalar_fn(Context& ctx, ScalarFnKind kind, Expr const* operand)
    -> Expr const*
{
    return ctx.make<Expr>(ScalarFn{kind, operand});
}

auto make_pow(Context& ctx, Expr const* base, Expr const* exponent)
    -> Expr const*
{
    return ctx.make<Expr>(Pow{base, exponent});
}

// ---- Unary factory -----------------------------------------------------

auto make_negate(Context& ctx, Expr const* operand) -> Expr const*
{
    return ctx.make<Expr>(Negate{operand});
}

auto make_trace(Context& ctx, Expr const* operand) -> Expr const*
{
    return ctx.make<Expr>(Trace{operand});
}

auto make_vector_invariant(Context& ctx, Expr const* operand) -> Expr const*
{
    return ctx.make<Expr>(VectorInvariant{operand});
}

auto make_transpose(Context& ctx, Expr const* operand) -> Expr const*
{
    return ctx.make<Expr>(Transpose{operand});
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
        .traits = TensorTraits{.well_known = WellKnownKind::Identity},
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
    // δ is symmetric in its two slots (δ^a_b = δ^b_a in value), so the swap is
    // a value-preserving generator (vibe 000047).  same_level_only is false:
    // the symmetry holds across the upper/lower level difference.
    return ctx.make<Expr>(TensorObject{
        .name = make_tensor_name("\\delta"),
        .rank = 0,
        .traits =
            TensorTraits{
                .well_known = WellKnownKind::Delta,
                .symmetry =
                    SymmetrySpec{PermutationSpec{false, Permutation<2>{{1, 0}}}},
                .render_hints = RenderHint::OmitVoidIndexPlaceholders},
        .slots = {
            SlotBinding{IndexSlot{level0, realm, space}, std::move(index0)},
            SlotBinding{IndexSlot{level1, realm, space}, std::move(index1)}}});
}

auto make_metric(
    Context& ctx,
    Realm realm,
    IndexSpace const* space,
    Level level0,
    Level level1,
    IndexAssoc index0,
    IndexAssoc index1) -> Expr const*
{
    // g is symmetric (g_ij = g_ji, g^ij = g^ji), so the slot swap is a
    // value-preserving generator, like δ (vibe 000047).
    return ctx.make<Expr>(TensorObject{
        .name = make_tensor_name("g"),
        .rank = 0,
        .traits =
            TensorTraits{
                .well_known = WellKnownKind::Metric,
                .symmetry =
                    SymmetrySpec{PermutationSpec{false, Permutation<2>{{1, 0}}}},
                .render_hints = RenderHint::OmitVoidIndexPlaceholders},
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

    // ε is totally antisymmetric: the adjacent transpositions are sign-flipping
    // generators of the full S_n permutation symmetry (vibe 000047).  The
    // generated even permutations (sign +1) — e.g. the rank-3 cyclic shift —
    // fall out of the closure, so no separate symmetry generators are needed.
    // Only ranks 2 and 3 (the practically relevant dimensions) are realised;
    // any other rank is rejected rather than left with a silently-empty
    // antisymmetry, which would be a latent bug.
    TensorTraits traits{.well_known = WellKnownKind::LeviCivita};
    switch (n)
    {
        case 2:
            traits.antisymmetry = AntisymmetrySpec{
                PermutationSpec{false, Permutation<2>{{1, 0}}}};
            break;
        case 3:
            traits.antisymmetry = AntisymmetrySpec{PermutationSpec{
                false, Permutation<3>{{1, 0, 2}}, Permutation<3>{{0, 2, 1}}}};
            break;
        default:
            throw std::invalid_argument(
                "make_levi_civita: only ranks 2 and 3 are supported (antisymmetry "
                "generators are undefined for other ranks)");
    }

    std::vector<SlotBinding> slots;
    slots.reserve(n);
    for (std::size_t k = 0; k < n; ++k)
        slots.push_back(SlotBinding{
            IndexSlot{levels[k], realm, space}, std::move(indices[k])});

    return ctx.make<Expr>(TensorObject{
        .name = make_tensor_name("\\varepsilon"),
        .rank = 0,
        .traits = traits,
        .slots = std::move(slots)});
}

} // namespace tender
