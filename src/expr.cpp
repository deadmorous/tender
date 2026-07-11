#include <tender/expr.hpp>

#include <tender/index_space.hpp> // space_3d (default identity dimension)

#include <algorithm>
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

auto make_field(
    Context& ctx,
    TensorName name,
    int rank,
    std::vector<CoordinateRef> deps,
    bool symmetric) -> Expr const*
{
    FieldDeps fd;
    if (deps.empty())
        fd.all = true;
    else
    {
        fd.all = false;
        fd.only = std::move(deps);
    }
    TensorTraits traits{.field = std::move(fd)};
    if (symmetric)
    {
        if (rank != 2)
            throw std::invalid_argument(
                "make_field: symmetric is only supported for a rank-2 field");
        // The slot swap is a value-preserving generator (T_ij = T_ji), like the
        // metric g (vibe 000073).  Expanded components inherit it, so T_θr
        // canonicalizes to T_rθ.
        traits.symmetry =
            SymmetrySpec{PermutationSpec{false, Permutation<2>{{1, 0}}}};
    }
    return ctx.make<Expr>(TensorObject{
        .name = std::move(name),
        .rank = rank,
        .traits = std::move(traits),
        .slots = {}});
}

// Keep a field's applied-derivative marks in a canonical order so mixed
// partials coincide and hash-cons to one node: by (chart_id, slot) for concrete
// marks (∂_x∂_y T = ∂_y∂_x T), with the free-index `link` as the final
// tiebreaker for abstract-direction ∂_i marks (vibe 000078).
static void sort_deriv_marks(std::vector<DerivMark>& marks)
{
    std::sort(
        marks.begin(),
        marks.end(),
        [](DerivMark const& a, DerivMark const& b)
        {
            if (a.wrt.chart_id != b.wrt.chart_id)
                return a.wrt.chart_id < b.wrt.chart_id;
            if (a.wrt.slot != b.wrt.slot)
                return a.wrt.slot < b.wrt.slot;
            return a.link < b.link;
        });
}

auto make_field_derivative(
    Context& ctx,
    Expr const* base,
    TensorName coord_name,
    CoordinateRef coord) -> Expr const*
{
    auto const* t = std::get_if<TensorObject>(&base->node);
    if (!t || !t->traits || !t->traits->field)
        throw std::invalid_argument(
            "make_field_derivative: base must be a field (make_field)");
    TensorObject d = *t;
    d.deriv_marks.push_back(DerivMark{std::move(coord_name), coord});
    sort_deriv_marks(d.deriv_marks);
    return ctx.make<Expr>(std::move(d));
}

auto make_field_derivative_free(
    Context& ctx,
    Expr const* base,
    CountableIndex dir,
    IndexSlot free_slot) -> Expr const*
{
    auto const* t = std::get_if<TensorObject>(&base->node);
    if (!t || !t->traits || !t->traits->field)
        throw std::invalid_argument(
            "make_field_derivative_free: base must be a field (make_field)");
    TensorObject d = *t;
    // The direction is a summation index, tied to a frame vector e_i via link.
    // (coord_name is unused for a free mark — render uses the index letter —
    // but must be a valid TensorName.)
    d.deriv_marks.push_back(DerivMark{
        make_tensor_name("q"), CoordinateRef{}, dir.id, free_slot, true});
    sort_deriv_marks(d.deriv_marks);
    return ctx.make<Expr>(std::move(d));
}

auto make_coordinate_direction(
    Context& ctx,
    TensorName name,
    int chart_id,
    CountableIndex dir,
    IndexSlot slot) -> Expr const*
{
    // A coordinate atom carrying a countable slot: the countable slot on a
    // coordinate is the signal (read by as_diff_coord) that ∂ is a free-index
    // frame direction rather than a fixed coordinate.  slot = -1 in the
    // CoordinateRef marks "no fixed coordinate slot".
    return ctx.make<Expr>(TensorObject{
        .name = std::move(name),
        .rank = 0,
        .traits =
            TensorTraits{.coordinate = CoordinateRef{chart_id, -1, false}},
        .slots = {SlotBinding{slot, IndexAssoc{dir}}}});
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

auto make_deriv(Context& ctx, Expr const* wrt) -> Expr const*
{
    return ctx.make<Expr>(Deriv{wrt});
}

auto make_nabla(Context& ctx) -> Expr const*
{
    return ctx.make<Expr>(Nabla{});
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

auto infer_rank(Expr const* e) -> std::optional<int>
{
    // Rank arithmetic for a contraction that removes `removed` indices from the
    // outer-product rank of its two operands; nullopt if either is unknown or
    // the result is negative (ill-formed).
    auto contracted = [](std::optional<int> l,
                         std::optional<int> r,
                         int removed) -> std::optional<int>
    {
        if (!l || !r || *l + *r - removed < 0)
            return std::nullopt;
        return *l + *r - removed;
    };
    return visit(
        mpk::mix::Overloads{
            [](TensorObject const& t) -> std::optional<int> { return t.rank; },
            [](ScalarLiteral const&) -> std::optional<int> { return 0; },
            [](Negate const& n) -> std::optional<int>
            { return infer_rank(n.operand); },
            // tr(A) is a scalar; vec(A) is a vector; transpose keeps the rank.
            [](Trace const&) -> std::optional<int> { return 0; },
            [](VectorInvariant const&) -> std::optional<int> { return 1; },
            [](Transpose const& u) -> std::optional<int>
            { return infer_rank(u.operand); },
            // A sum keeps the shared rank of its operands; trust the known
            // side.
            [](Sum const& s) -> std::optional<int>
            {
                auto const l = infer_rank(s.left);
                return l ? l : infer_rank(s.right);
            },
            [](Difference const& s) -> std::optional<int>
            {
                auto const l = infer_rank(s.left);
                return l ? l : infer_rank(s.right);
            },
            // Outer product adds ranks; scalar division keeps the left rank.
            [&](TensorProduct const& s) -> std::optional<int>
            { return contracted(infer_rank(s.left), infer_rank(s.right), 0); },
            [](ScalarDiv const& s) -> std::optional<int>
            { return infer_rank(s.left); },
            [&](Dot const& s) -> std::optional<int>
            { return contracted(infer_rank(s.left), infer_rank(s.right), 2); },
            [&](DDot const& s) -> std::optional<int>
            { return contracted(infer_rank(s.left), infer_rank(s.right), 4); },
            [&](DDotAlt const& s) -> std::optional<int>
            { return contracted(infer_rank(s.left), infer_rank(s.right), 4); },
            [&](Cross const& s) -> std::optional<int>
            { return contracted(infer_rank(s.left), infer_rank(s.right), 1); },
            [](ExplicitSum const& s) -> std::optional<int>
            { return infer_rank(s.body); },
            [](NoSum const& s) -> std::optional<int>
            { return infer_rank(s.body); },
            // Scalar fields are rank 0.
            [](ScalarFn const&) -> std::optional<int> { return 0; },
            [](Pow const&) -> std::optional<int> { return 0; },
            // A differential operator's rank is that of the object it
            // differentiates with respect to (vibe 000077): a coordinate ⇒ 0.
            [](Deriv const& d) -> std::optional<int>
            { return infer_rank(d.wrt); },
            // ∇ is a rank-1 vector operator (vibe 000078).
            [](Nabla const&) -> std::optional<int> { return 1; },
        },
        *e);
}

namespace
{

// A contraction with a scalar (rank-0) operand is not a contraction — a scalar
// has no leg to contract — it is scalar multiplication.  Every `·` / `:` / `··`
// / `×` factory funnels through here so such a node is *never* built: a scalar
// operand redirects to `⊗` (which the layer above reads as multiplication),
// keeping a bare scalar out of a contraction slot where the reductions and the
// nf lowering would choke on it.  Only redirects when an operand is *known*
// rank 0; an unknown (abstract, rankless) operand keeps the contraction.
auto scalar_mult_or(
    Context& ctx,
    Expr const* left,
    Expr const* right,
    Expr const* (*contraction)(Context&, Expr const*, Expr const*))
    -> Expr const*
{
    if (infer_rank(left) == std::optional<int>{0}
        || infer_rank(right) == std::optional<int>{0})
        return make_tensor_product(ctx, left, right);
    return contraction(ctx, left, right);
}

auto raw_dot(Context& ctx, Expr const* l, Expr const* r) -> Expr const*
{
    return ctx.make<Expr>(Dot{l, r});
}
auto raw_ddot(Context& ctx, Expr const* l, Expr const* r) -> Expr const*
{
    return ctx.make<Expr>(DDot{l, r});
}
auto raw_ddot_alt(Context& ctx, Expr const* l, Expr const* r) -> Expr const*
{
    return ctx.make<Expr>(DDotAlt{l, r});
}
auto raw_cross(Context& ctx, Expr const* l, Expr const* r) -> Expr const*
{
    return ctx.make<Expr>(Cross{l, r});
}

} // namespace

auto make_dot(Context& ctx, Expr const* left, Expr const* right) -> Expr const*
{
    return scalar_mult_or(ctx, left, right, &raw_dot);
}

auto make_ddot(Context& ctx, Expr const* left, Expr const* right) -> Expr const*
{
    return scalar_mult_or(ctx, left, right, &raw_ddot);
}

auto make_ddot_alt(Context& ctx, Expr const* left, Expr const* right)
    -> Expr const*
{
    return scalar_mult_or(ctx, left, right, &raw_ddot_alt);
}

auto make_cross(Context& ctx, Expr const* left, Expr const* right) -> Expr const*
{
    return scalar_mult_or(ctx, left, right, &raw_cross);
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
    // Default to 3-D: there is no dimension-agnostic invariant (vibe 000082).
    return make_identity(ctx, space_3d());
}

auto make_identity(Context& ctx, IndexSpace const* space) -> Expr const*
{
    // A dimension-aware identity: the abstract I with its `dim` attribute set
    // (vibe 000081/000082).  Dimension-awareness is orthogonal to the index
    // slots — the object stays slotless, so it behaves exactly like a plain I
    // in basis expansion / contraction / rendering, and only `tr(I) → n` reads
    // the dimension.  `dim` IS part of structural identity (a 2-D I ≠ a 3-D I),
    // so congruence holds; a shared default means a library I and a user I are
    // both 3-D and still cancel.  (Earlier this fabricated two fake unbound
    // slots to carry the space, which broke
    // `expand_in_basis`/`simplify_basis_cross` and printed `I^{•·}_{·•}` — a
    // design defect.)
    return ctx.make<Expr>(TensorObject{
        .name = make_tensor_name("I"),
        .rank = 2,
        .traits = TensorTraits{.well_known = WellKnownKind::Identity},
        .slots = {},
        .dim = space});
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
