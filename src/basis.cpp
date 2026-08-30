#include <tender/basis.hpp>

#include <tender/derivation.hpp>
#include <tender/rewrite.hpp>

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace tender
{

Basis::Basis(
    Realm realm,
    IndexSpace const* space,
    TensorName symbol,
    std::vector<Expr const*> vectors,
    std::vector<Expr const*> covectors,
    Expr const* volume,
    BasisNaming naming) :
  realm_(realm),
  space_(space),
  symbol_(symbol),
  vectors_(std::move(vectors)),
  covectors_(std::move(covectors)),
  volume_(volume),
  naming_(std::move(naming))
{
}

namespace
{
// Position of `value` in the space's value list, or -1 if absent.
auto value_pos(IndexSpace const* space, int value) -> int
{
    auto const vals = space->values();
    for (std::size_t k = 0; k < vals.size(); ++k)
        if (vals[k] == value)
            return static_cast<int>(k);
    return -1;
}
} // namespace

auto Basis::value_name(int value) const -> std::optional<IndexName>
{
    int const k = value_pos(space_, value);
    if (k < 0 || static_cast<std::size_t>(k) >= naming_.value_names.size())
        return std::nullopt;
    return naming_.value_names[static_cast<std::size_t>(k)];
}

auto Basis::vector_symbol_for(int value) const -> std::optional<TensorName>
{
    int const k = value_pos(space_, value);
    if (k < 0 || static_cast<std::size_t>(k) >= naming_.vector_symbols.size())
        return std::nullopt;
    return naming_.vector_symbols[static_cast<std::size_t>(k)];
}

auto Basis::basis(int i) const -> Expr const*
{
    return vectors_.at(static_cast<std::size_t>(i));
}

auto Basis::cobasis(int i) const -> Expr const*
{
    return covectors_.at(static_cast<std::size_t>(i));
}

auto Basis::covariant_vector(Context& ctx, CountableIndex index) const
    -> Expr const*
{
    return make_tensor_object(
        ctx,
        symbol_,
        {SlotBinding{
            IndexSlot{Level::Lower, realm_, space_, id_}, IndexAssoc{index}}},
        1);
}

auto Basis::direction(Context& ctx, int i) const -> Expr const*
{
    auto const vals = space_->values();
    if (i < 0 || static_cast<std::size_t>(i) >= vals.size())
        throw std::out_of_range("Basis::direction: index out of range");
    return make_tensor_object(
        ctx,
        symbol_,
        {SlotBinding{
            IndexSlot{Level::Lower, realm_, space_, id_},
            IndexAssoc{ConcreteIndex{vals[static_cast<std::size_t>(i)]}}}},
        1);
}

auto Basis::contravariant_vector(Context& ctx, CountableIndex index) const
    -> Expr const*
{
    auto const level = is_orthonormal() ? Level::Lower : Level::Upper;
    return make_tensor_object(
        ctx,
        symbol_,
        {SlotBinding{IndexSlot{level, realm_, space_, id_}, IndexAssoc{index}}},
        1);
}

namespace
{

// A provided basis vector must be rank 1 where its rank is known.  Composite
// vectors (not a bare TensorObject) and rank-unknown objects pass; the check
// only rejects an object that is explicitly some other rank.
auto rank_ok(Expr const* v) -> bool
{
    auto const* t = std::get_if<TensorObject>(&v->node);
    if (!t || !t->rank)
        return true;
    return *t->rank == 1;
}

// Shared precondition checks for the basis factories.  `who` names the caller
// for the error message.
void validate_basis_vectors(
    char const* who,
    IndexSpace const* space,
    std::vector<Expr const*> const& vectors)
{
    auto fail = [who](char const* msg)
    { throw std::invalid_argument(std::string{who} + ": " + msg); };
    if (!space)
        fail("null space");
    if (vectors.empty())
        fail("at least one vector is required");
    if (vectors.size() != space->values().size())
        fail("number of vectors must equal the index space cardinality");
    for (auto const* v: vectors)
    {
        if (!v)
            fail("null basis vector");
        if (!rank_ok(v))
            fail("basis vectors must be rank 1");
    }
}

} // namespace

auto intern_basis(Context& ctx, Basis b) -> Basis
{
    // Keep a context-owned copy so a slot's basis_id always resolves to a live
    // Basis (vibe 000067), and stamp the same id on the value we hand back.
    Basis* owned = ctx.make<Basis>(std::move(b));
    owned->id_ = ctx.register_basis(owned);
    return *owned;
}

auto make_orthonormal_basis(
    Context& ctx,
    IndexSpace const* space,
    std::vector<Expr const*> vectors,
    TensorName vector_symbol,
    Handedness handedness,
    BasisNaming naming) -> Basis
{
    validate_basis_vectors("make_orthonormal_basis", space, vectors);

    // Orthonormal: the cobasis coincides with the basis, and √g = ±1 by
    // handedness (+1 right-handed, -1 left-handed).
    auto covectors = vectors;
    Expr const* const vol =
        make_scalar(ctx, Rational{handedness == Handedness::Right ? 1 : -1});
    Basis b{
        Realm::Orthonormal,
        space,
        vector_symbol,
        std::move(vectors),
        std::move(covectors),
        vol,
        std::move(naming)};
    return intern_basis(ctx, std::move(b));
}

auto make_oblique_basis(
    Context& ctx,
    IndexSpace const* space,
    std::vector<Expr const*> vectors,
    TensorName vector_symbol,
    BasisNaming naming) -> Basis
{
    validate_basis_vectors("make_oblique_basis", space, vectors);
    if (vectors.size() != 3)
        throw std::invalid_argument(
            "make_oblique_basis: only 3D oblique bases are supported (cobasis "
            "derived via the cross-product formula)");

    // Reciprocal basis: e^0 = (e_1×e_2)/V, e^1 = (e_2×e_0)/V, e^2 = (e_0×e_1)/V
    // with the cell volume V = √g = e_0·(e_1×e_2).
    Expr const* const vol =
        make_dot(ctx, vectors[0], make_cross(ctx, vectors[1], vectors[2]));
    auto cob = [&](Expr const* a, Expr const* b) -> Expr const*
    { return make_scalar_div(ctx, make_cross(ctx, a, b), vol); };
    std::vector<Expr const*> covectors{
        cob(vectors[1], vectors[2]),
        cob(vectors[2], vectors[0]),
        cob(vectors[0], vectors[1])};

    Basis b{
        Realm::Oblique,
        space,
        vector_symbol,
        std::move(vectors),
        std::move(covectors),
        vol,
        std::move(naming)};
    return intern_basis(ctx, std::move(b));
}

namespace
{

// A generic invariant tensor worth expanding: slot-less, rank >= 1, and not a
// well-known tensor (whose coordinates are special, e.g. I -> δ).
auto is_expandable_invariant(TensorObject const& t) -> bool
{
    return t.slots.empty() && t.rank && *t.rank >= 1
           && !(t.traits && t.traits->well_known);
}

// Coordinate index level for one slot, chosen so the shared index
// Einstein-contracts against its basis vector: orthonormal pairs two lower
// indices; oblique pairs one upper coordinate with one lower (covariant) basis
// vector, or one lower coordinate with one upper (contravariant) cobasis
// vector.
auto coord_level_for(Variance v, bool ortho) -> Level
{
    if (ortho)
        return Level::Lower;
    return v == Variance::Covariant ? Level::Upper : Level::Lower;
}

// A frame is "moving" when its connection is non-trivial — some ∂_{q^j} e_i ≠
// 0, as for a curvilinear physical frame (vibe 000073).  Only a chart registers
// a connection (via physical_frame); a bare or Cartesian frame has none, or an
// all-zero one.  Used to refuse expanding a field derivative where doing so
// would silently drop the connection terms.
auto has_moving_connection(Context const& ctx, int basis_id) -> bool
{
    auto const* conn = ctx.connection(basis_id);
    if (!conn)
        return false;
    for (auto const& row: conn->deriv)
        for (auto const* d: row)
        {
            auto const* s = std::get_if<ScalarLiteral>(&d->node);
            if (!s || !s->value.is_zero())
                return true;
        }
    return false;
}

} // namespace

auto expand_in_basis(
    Context& ctx,
    Expr const* e,
    Basis const& basis,
    std::vector<Variance> variances) -> Expr const*
{
    if (variances.empty())
        throw std::invalid_argument(
            "expand_in_basis: at least one variance is required");
    bool const ortho = basis.is_orthonormal();

    return rewrite_tree(
        ctx,
        e,
        [&](Context& c, Expr const* node) -> Expr const*
        {
            auto const* t = std::get_if<TensorObject>(&node->node);
            if (!t)
                return node;

            // The identity is well-known (its coordinate is δ/g, not generic),
            // so it gets the resolution of identity I = Σ_i e_i ⊗ e^i directly
            // — the coordinate δ^i_j of I^i_j = e^i·I·e_j = e^i·e_j contracted
            // away (vibe 000049).  Intrinsically mixed, so `variances` is not
            // consulted; the pure-variance metric forms await the oblique
            // flavor.
            if (t->slots.empty() && t->traits
                && t->traits->well_known == WellKnownKind::Identity)
            {
                // A dimension-aware identity (vibe 000081) must match the basis
                // it is expanded in: I of ℝ³ makes no sense on a 2-D frame.
                if (t->dim && t->dim != basis.space())
                    throw std::invalid_argument(
                        "expand_in_basis: identity dimension does not match the "
                        "basis dimension");
                CountableIndex const idx{c.alloc_index_id()};
                return make_tensor_product(
                    c,
                    basis.covariant_vector(c, idx),
                    basis.contravariant_vector(c, idx));
            }

            if (!is_expandable_invariant(*t))
                return node;

            // A dimension-aware invariant must match the basis it is expanded
            // in (vibe 000082): a 3-D vector makes no sense on a 2-D frame.  A
            // dimension-agnostic invariant (dim == null) expands in any basis.
            if (t->dim && t->dim != basis.space())
                throw std::invalid_argument(
                    "expand_in_basis: tensor dimension does not match the basis "
                    "dimension");

            // A field derivative ∂T cannot be expanded correctly in a moving
            // frame: the connection ∂e_i (the spin Ω, ∂e_i = Ω×e_i) lives on
            // the chart, not the basis, so expanding here would silently drop
            // the connection terms and yield only the coordinate part d*T (vibe
            // 000073).  Refuse loudly and point at the chart operator, which
            // expands the field *then* differentiates.
            if (!t->deriv_marks.empty()
                && has_moving_connection(c, basis.basis_id()))
                throw std::invalid_argument(
                    "expand_in_basis: cannot expand a field derivative (∂T) in "
                    "a moving frame — the connection ∂e_i belongs to the chart, "
                    "not the basis, so expansion here would drop the connection "
                    "terms and give only the coordinate part d*T. Differentiate "
                    "an already-expanded field, or use the chart's differential "
                    "operator (div / grad / rot / laplacian).");

            int const r = *t->rank;
            // One variance broadcasts to every slot; otherwise the count must
            // match the tensor rank exactly (no silent misapplication).
            if (variances.size() != 1
                && variances.size() != static_cast<std::size_t>(r))
                throw std::invalid_argument(
                    "expand_in_basis: variance count must be 1 or the tensor "
                    "rank");
            auto const slot_variance = [&](int k) -> Variance
            {
                return variances.size() == 1 ?
                           variances[0] :
                           variances[static_cast<std::size_t>(k)];
            };

            std::vector<SlotBinding> coord_slots;
            coord_slots.reserve(static_cast<std::size_t>(r));
            Expr const* polyad = nullptr;
            for (int k = 0; k < r; ++k)
            {
                Variance const v = slot_variance(k);
                CountableIndex const idx{c.alloc_index_id()};
                coord_slots.push_back(SlotBinding{
                    IndexSlot{
                        coord_level_for(v, ortho),
                        basis.realm(),
                        basis.space(),
                        basis.basis_id()},
                    IndexAssoc{idx}});
                Expr const* const vec = v == Variance::Covariant ?
                                            basis.covariant_vector(c, idx) :
                                            basis.contravariant_vector(c, idx);
                polyad = polyad ? make_tensor_product(c, polyad, vec) : vec;
            }
            // A component of a field is itself a field with the same
            // coordinate dependence (vibe 000073): carry the source's FieldDeps
            // and any accumulated ∂ directions onto the indexed component, so
            // ∂_q T_ij is nonzero and div / grad can differentiate the
            // components rather than treating them as constants.  Carry the
            // (anti)symmetry too, so a symmetric field's T_θr canonicalizes to
            // T_rθ — but not `well_known` or `coordinate`, which name the whole
            // tensor, not a component.
            TensorObject comp{
                .name = t->name,
                .rank = 0,
                .traits = std::nullopt,
                .slots = std::move(coord_slots),
                .deriv_marks = t->deriv_marks};
            if (t->traits)
                comp.traits = TensorTraits{
                    .symmetry = t->traits->symmetry,
                    .antisymmetry = t->traits->antisymmetry,
                    .field = t->traits->field};
            Expr const* const coord = c.make<Expr>(std::move(comp));
            return make_tensor_product(c, coord, polyad);
        });
}

auto expand_in_basis(
    Context& ctx,
    Expr const* e,
    Basis const& basis,
    Variance variance) -> Expr const*
{
    return expand_in_basis(ctx, e, basis, std::vector<Variance>{variance});
}

namespace
{

// One side of a basis-vector dot: a (possibly coordinate-scaled) basis vector,
// reduced to its component factor (null when bare) and its index + level.
struct VecSide final
{
    Expr const* scalar; // component-valued factor, or nullptr
    IndexAssoc index;   // dummy or concrete (vibe 000068)
    Level level;
};

// Is e a symbolic basis vector of b — its vector symbol, rank 1, one slot
// carrying a CountableIndex *tagged with b's basis id*?  A foreign basis's
// vector (same symbol "e", different basis_id) is rejected, so a step keyed to
// b acts only on b's own vectors (vibe 000067).  Returns the index and level.
auto as_basis_vector(Expr const* e, Basis const& b)
    -> std::optional<std::pair<CountableIndex, Level>>
{
    auto const* t = std::get_if<TensorObject>(&e->node);
    if (!t || t->name.v.view() != b.vector_symbol().v.view()
        || t->slots.size() != 1 || !t->slots[0].index
        || t->slots[0].slot.basis_id != b.basis_id())
        return std::nullopt;
    auto const* ci = std::get_if<CountableIndex>(&*t->slots[0].index);
    if (!ci)
        return std::nullopt;
    return std::pair{*ci, t->slots[0].slot.level};
}

// Like as_basis_vector but for the contraction steps (vibe 000068): recognises
// the three forms of a basis direction and returns the index *association*
// (dummy or concrete) so a dot/cross can emit a δ/ε with a concrete leg.
//   (1)/(2) the `e`-form — `b`'s vector symbol, one slot tagged with `b`,
//           carrying a CountableIndex (dummy) OR a ConcreteIndex;
//   (3)     a frame vector `b.basis(k)` / `b.cobasis(k)` matched structurally
//           (so `cs.basis(0)` is read as the concrete direction-0 vector).
// Reverse-lookup is by structural identity with *this* basis's vectors, so it
// is inherently single-frame.
auto as_basis_dir(Expr const* e, Basis const& b)
    -> std::optional<std::pair<IndexAssoc, Level>>
{
    if (auto const* t = std::get_if<TensorObject>(&e->node);
        t && t->name.v.view() == b.vector_symbol().v.view()
        && t->slots.size() == 1 && t->slots[0].index
        && t->slots[0].slot.basis_id == b.basis_id())
    {
        auto const& a = *t->slots[0].index;
        if (std::holds_alternative<CountableIndex>(a)
            || std::holds_alternative<ConcreteIndex>(a))
            return std::pair{a, t->slots[0].slot.level};
    }
    auto const vals = b.space()->values();
    for (int k = 0; k < b.dim(); ++k)
    {
        if (structural_eq(e, b.basis(k)))
            return std::pair{IndexAssoc{ConcreteIndex{vals[k]}}, Level::Lower};
        if (structural_eq(e, b.cobasis(k)))
            return std::pair{
                IndexAssoc{ConcreteIndex{vals[k]}},
                b.is_orthonormal() ? Level::Lower : Level::Upper};
    }
    return std::nullopt;
}

auto is_scalar_one(Expr const* e) -> bool
{
    auto const* s = std::get_if<ScalarLiteral>(&e->node);
    return s && s->value == Rational{1};
}

// A true scalar coefficient: component-valued AND rank 0 — so a (fully-indexed)
// basis vector, which is component-valued but rank 1, is NOT mistaken for one.
auto is_scalar_coefficient(Expr const* e) -> bool
{
    return is_component_valued(e) && infer_rank(e) == std::optional<int>{0};
}

// Split one dot operand into an optional scalar factor and a basis vector.
// Accepts a bare basis vector or a product of a scalar coefficient with one;
// a product of two basis vectors (a dyad) is rejected — that needs the
// contraction distributed over the ⊗ first (steps::distribute_contraction).
auto as_vec_side(Expr const* e, Basis const& b) -> std::optional<VecSide>
{
    if (auto bv = as_basis_dir(e, b))
        return VecSide{nullptr, bv->first, bv->second};
    auto const* tp = std::get_if<TensorProduct>(&e->node);
    if (!tp)
        return std::nullopt;
    if (auto bv = as_basis_dir(tp->right, b);
        bv && is_scalar_coefficient(tp->left))
        return VecSide{tp->left, bv->first, bv->second};
    if (auto bv = as_basis_dir(tp->left, b);
        bv && is_scalar_coefficient(tp->right))
        return VecSide{tp->right, bv->first, bv->second};
    return std::nullopt;
}

} // namespace

auto simplify_basis_dot(Context& ctx, Expr const* e, Basis const& basis)
    -> Expr const*
{
    // Distribute first, so a dot with a polyad (e.g. against the identity dyad)
    // becomes dots of single basis vectors that the rule below can reduce.
    e = steps::distribute_contraction(ctx, e);
    return rewrite_tree(
        ctx,
        e,
        [&](Context& c, Expr const* node) -> Expr const*
        {
            auto const* d = std::get_if<Dot>(&node->node);
            if (!d)
                return node;
            auto const l = as_vec_side(d->left, basis);
            auto const r = as_vec_side(d->right, basis);
            if (!l || !r)
                return node;

            // e_i·e^j (mixed level) is the Kronecker δ; two same-variance basis
            // vectors give the metric — but for an orthonormal basis the metric
            // is δ, so only an oblique same-level pair yields g (vibe 000049).
            auto const make_dot_tensor =
                (!basis.is_orthonormal() && l->level == r->level) ?
                    &make_metric :
                    &make_delta;
            Expr const* result = make_dot_tensor(
                c,
                basis.realm(),
                basis.space(),
                l->level,
                r->level,
                l->index,
                r->index);
            if (r->scalar)
                result = make_tensor_product(c, r->scalar, result);
            if (l->scalar)
                result = make_tensor_product(c, l->scalar, result);
            return result;
        });
}

auto simplify_basis_cross(Context& ctx, Expr const* e, Basis const& basis)
    -> Expr const*
{
    // Distribute first, so a cross with a polyad (e.g. against the identity
    // dyad) becomes crosses of single basis vectors that the rule below
    // reduces.
    e = steps::distribute_contraction(ctx, e);
    return rewrite_tree(
        ctx,
        e,
        [&](Context& c, Expr const* node) -> Expr const*
        {
            auto const* x = std::get_if<Cross>(&node->node);
            if (!x || basis.space()->values().size() != 3)
                return node; // the cross-product formula is 3D
            auto const l = as_vec_side(x->left, basis);
            auto const r = as_vec_side(x->right, basis);
            if (!l || !r)
                return node;
            // Only the covariant case e_i × e_j = √g ε_{ijk} e^k (both inputs
            // lower); contravariant / mixed inputs are left for later.
            if (l->level != Level::Lower || r->level != Level::Lower)
                return node;

            CountableIndex const k{c.alloc_index_id()};
            Expr const* const eps = make_levi_civita(
                c,
                basis.realm(),
                basis.space(),
                {Level::Lower, Level::Lower, Level::Lower},
                {l->index, r->index, IndexAssoc{k}});
            Expr const* result =
                make_tensor_product(c, eps, basis.contravariant_vector(c, k));
            // √g weight; the right-handed orthonormal case (√g = 1) needs no
            // factor, so it is omitted for a clean ε_{ijk} e_k.
            if (!is_scalar_one(basis.volume()))
                result = make_tensor_product(c, basis.volume(), result);
            if (r->scalar)
                result = make_tensor_product(c, r->scalar, result);
            if (l->scalar)
                result = make_tensor_product(c, l->scalar, result);
            return result;
        });
}

namespace
{

// Flatten a TensorProduct tree into its leaf factors.
void flatten_product(Expr const* e, std::vector<Expr const*>& out)
{
    if (auto const* tp = std::get_if<TensorProduct>(&e->node))
    {
        flatten_product(tp->left, out);
        flatten_product(tp->right, out);
    }
    else
        out.push_back(e);
}

// The slot indices of a tensor as CountableIndex ids; nullopt if any slot is
// missing or not a CountableIndex.
auto countable_slot_ids(TensorObject const& t) -> std::optional<std::vector<int>>
{
    std::vector<int> ids;
    ids.reserve(t.slots.size());
    for (auto const& sb: t.slots)
    {
        if (!sb.index)
            return std::nullopt;
        auto const* ci = std::get_if<CountableIndex>(&*sb.index);
        if (!ci)
            return std::nullopt;
        ids.push_back(ci->id);
    }
    return ids;
}

auto sorted(std::vector<int> v) -> std::vector<int>
{
    std::sort(v.begin(), v.end());
    return v;
}

// Does index id appear in any tensor slot within e?  (Binders other than the
// summed indices being folded are not expected on the terms passed here.)
auto mentions_index(Context& ctx, Expr const* e, int id) -> bool
{
    bool found = false;
    rewrite_tree(
        ctx,
        e,
        [&](Context&, Expr const* n) -> Expr const*
        {
            if (auto const* t = std::get_if<TensorObject>(&n->node))
                for (auto const& sb: t->slots)
                    if (sb.index)
                        if (auto const* ci =
                                std::get_if<CountableIndex>(&*sb.index);
                            ci && ci->id == id)
                            found = true;
            return n;
        });
    return found;
}

// One half of the resolution of identity Σ_i e_i ⊗ e^i = I: a Dot in which one
// operand is a bare basis vector e_i and the other is an invariant X of rank
// ≥ 1.  Paired with a bare e_i leg over the same summed index, the term
// Σ_i (X·e_i) ⊗ e_i folds to X·I = X — the "X·I = X" route of completeness
// reassembly (the alternative "X·e_i = X_i, then Σ X_i e_i = X" route would
// instead need component materialization and a symbolic δ-substitution, which
// this avoids).  `basis_on_right` records which slot of X the dot contracts:
// X·e_i (true) contracts X's last slot, e_i·X (false) its first — this matters
// for rank ≥ 2, where the leg must sit on the contracted side or the term would
// reassemble to Xᵀ rather than X.
struct CompletenessDot final
{
    int index;
    Expr const* other;   // X
    bool basis_on_right; // true: X·e_i (last slot); false: e_i·X (first slot)
};
auto as_completeness_dot(Expr const* e, Basis const& b)
    -> std::optional<CompletenessDot>
{
    auto const* d = std::get_if<Dot>(&e->node);
    if (!d)
        return std::nullopt;
    if (auto bv = as_basis_vector(d->right, b);
        bv && infer_rank(d->left).value_or(0) >= 1)
        return CompletenessDot{bv->first.id, d->left, true};
    if (auto bv = as_basis_vector(d->left, b);
        bv && infer_rank(d->right).value_or(0) >= 1)
        return CompletenessDot{bv->first.id, d->right, false};
    return std::nullopt;
}

auto product_of(Context& ctx, std::vector<Expr const*> const& fs) -> Expr const*
{
    if (fs.empty())
        return make_scalar(ctx, Rational{1});
    Expr const* p = fs.front();
    for (std::size_t i = 1; i < fs.size(); ++i)
        p = make_tensor_product(ctx, p, fs[i]);
    return p;
}

auto wrap_sums(Context& ctx, std::vector<int> const& ids, Expr const* e)
    -> Expr const*
{
    for (int id: ids)
        e = make_explicit_sum(ctx, CountableIndex{id}, e, nullptr);
    return e;
}

// Fold a resolution of identity in a product term Σ_summed (factors) over one
// summed index i that occurs nowhere else.  Two shapes (nullptr if neither):
//   A. one bare leg e_i + one completeness dot (X·e_i) → X at the leg's place
//      (Σ_i (X·e_i) ⊗ e_i = X·I = X).  X may be any rank ≥ 1, provided it can
//      stay atomic: every factor strictly between the dot and the leg must be a
//      scalar (so the dot slides to the leg), and for non-scalar X the leg must
//      sit on the contracted side (else the term would reassemble to Xᵀ); and
//   B. two bare legs e_i and only rank-0 (scalar) other factors → the two legs
//      become I (Σ_i (scalars) e_i⊗e_i = (scalars) I); the scalars commute out
//      so they need not be adjacent to the legs.
auto fold_completeness_term(
    Context& ctx,
    std::vector<int> const& summed,
    std::vector<Expr const*> const& factors,
    Basis const& basis) -> Expr const*
{
    for (int id: summed)
    {
        std::vector<int> legs;
        int dot = -1, dots = 0;
        Expr const* X = nullptr;
        bool dot_basis_right = false;
        bool other = false, nonscalar_other = false;
        for (std::size_t p = 0; p < factors.size(); ++p)
        {
            auto const* f = factors[p];
            if (auto bv = as_basis_vector(f, basis); bv && bv->first.id == id)
            {
                legs.push_back(static_cast<int>(p));
                continue;
            }
            if (auto cd = as_completeness_dot(f, basis); cd && cd->index == id)
            {
                ++dots;
                dot = static_cast<int>(p);
                X = cd->other;
                dot_basis_right = cd->basis_on_right;
                continue;
            }
            if (mentions_index(ctx, f, id))
                other = true;
            if (infer_rank(f) != std::optional<int>{0})
                nonscalar_other = true;
        }
        if (other)
            continue;

        // A. completeness contraction: Σ_i (X·e_i) ⊗ e_i → X.
        if (legs.size() == 1 && dots == 1)
        {
            int const leg = legs[0];
            int const xr = infer_rank(X).value_or(1);
            bool ok = xr <= 1; // scalar dot: X commutes freely to the leg
            if (!ok)
            {
                // Non-scalar X stays atomic only if it can slide to the leg
                // (scalars only strictly between) and the leg is on X's
                // contracted side (right of X·e_i, left of e_i·X), so the
                // reassembled legs spell X, not Xᵀ.
                bool scalars_between = true;
                for (int q = std::min(dot, leg) + 1; q < std::max(dot, leg);
                     ++q)
                    if (infer_rank(factors[static_cast<std::size_t>(q)])
                        != std::optional<int>{0})
                        scalars_between = false;
                bool const side_ok =
                    dot_basis_right ? (leg > dot) : (leg < dot);
                ok = scalars_between && side_ok;
            }
            if (ok)
            {
                std::vector<Expr const*> out;
                for (std::size_t p = 0; p < factors.size(); ++p)
                {
                    if (static_cast<int>(p) == dot)
                        continue;
                    out.push_back(static_cast<int>(p) == leg ? X : factors[p]);
                }
                std::vector<int> rest;
                for (int s: summed)
                    if (s != id)
                        rest.push_back(s);
                return wrap_sums(ctx, rest, product_of(ctx, out));
            }
        }

        // B. resolution of identity: Σ_i (scalars) e_i⊗e_i → (scalars) I.  The
        // scalars are emitted first (the conventional coefficient·tensor
        // order), then I; both are invariants whose order the canonicalizer
        // preserves.
        if (legs.size() == 2 && dots == 0 && !nonscalar_other)
        {
            std::vector<Expr const*> out;
            for (std::size_t p = 0; p < factors.size(); ++p)
                if (static_cast<int>(p) != legs[0]
                    && static_cast<int>(p) != legs[1])
                    out.push_back(factors[p]);
            out.push_back(make_identity(ctx, basis.space())); // vibe 000082
            std::vector<int> rest;
            for (int s: summed)
                if (s != id)
                    rest.push_back(s);
            return wrap_sums(ctx, rest, product_of(ctx, out));
        }
    }
    return nullptr;
}

// Recursive driver: peel the summed binders, distribute over Sum/Negate by
// linearity (only when a fold actually fires below, so the step is a no-op
// otherwise), and fold each product term.  Returns nullptr when nothing folds.
auto fold_completeness(Context& ctx, Expr const* node, Basis const& basis)
    -> Expr const*
{
    std::vector<int> summed;
    Expr const* body = node;
    while (auto const* es = std::get_if<ExplicitSum>(&body->node))
    {
        if (es->bound)
            return nullptr; // symbolic bound: not a basis expansion
        summed.push_back(es->index.id);
        body = es->body;
    }
    if (auto const* s = std::get_if<Sum>(&body->node))
    {
        auto* lf =
            fold_completeness(ctx, wrap_sums(ctx, summed, s->left), basis);
        auto* rf =
            fold_completeness(ctx, wrap_sums(ctx, summed, s->right), basis);
        if (!lf && !rf)
            return nullptr;
        return make_sum(
            ctx,
            lf ? lf : wrap_sums(ctx, summed, s->left),
            rf ? rf : wrap_sums(ctx, summed, s->right));
    }
    if (auto const* n = std::get_if<Negate>(&body->node))
    {
        auto* f =
            fold_completeness(ctx, wrap_sums(ctx, summed, n->operand), basis);
        return f ? make_negate(ctx, f) : nullptr;
    }
    if (summed.empty())
        return nullptr;
    std::vector<Expr const*> factors;
    flatten_product(body, factors);
    auto* folded = fold_completeness_term(ctx, summed, factors, basis);
    if (!folded)
        return nullptr;
    auto* more = fold_completeness(ctx, folded, basis);
    return more ? more : folded;
}

// ---- concrete resolution of identity (vibe 000070 Phase 0) -------------
//
// The differential operators emit their results in the chart's constant
// reference frame, *fully expanded* — Σ_k u_k⊗u_k as separate concrete addends
// `u_0⊗u_0 + u_1⊗u_1 + …`, never the symbolic bound Σ that fold_completeness
// above recognises.  This pass folds that concrete shape back to the identity
// tensor I, and the forward direction expands I to it (used by the contraction
// engine, vibe 000070 Phase 2).

// True for the well-known identity tensor I.
auto is_identity_tensor(Expr const* e) -> bool
{
    auto const* t = std::get_if<TensorObject>(&e->node);
    return t && t->traits
           && t->traits->well_known == std::optional{WellKnownKind::Identity};
}

// The direction k for which `e` is `b`'s k-th basis vector, or -1 otherwise.
// Recognises both forms: (a) the concrete (Cartesian) frame vector b.basis(k),
// and (b) the symbolic e-atom — b's vector symbol, one slot tagged with b,
// carrying a ConcreteIndex for b's k-th direction value (vibe 000071's
// b.direction(k)), so the resolution of identity Σ_k e_k⊗e_k folds to I in the
// curvilinear frame too.
auto concrete_basis_dir(Expr const* e, Basis const& b) -> int
{
    for (int k = 0; k < b.dim(); ++k)
        if (structural_eq(e, b.basis(k)))
            return k;
    auto const* t = std::get_if<TensorObject>(&e->node);
    if (t && t->name.v.view() == b.vector_symbol().v.view()
        && t->slots.size() == 1 && t->slots[0].index
        && t->slots[0].slot.basis_id == b.basis_id())
        if (auto const* ci = std::get_if<ConcreteIndex>(&*t->slots[0].index))
        {
            auto const vals = b.space()->values();
            for (int k = 0; k < b.dim(); ++k)
                if (vals[static_cast<std::size_t>(k)] == ci->value)
                    return k;
        }
    return -1;
}

// Gather the signed addends of a Sum/Difference/Negate tree (so the concrete
// resolution of identity, spread across the addends, can be matched as a set).
void collect_addends(
    Expr const* e, bool neg, std::vector<std::pair<Expr const*, bool>>& out)
{
    if (auto const* s = std::get_if<Sum>(&e->node))
    {
        collect_addends(s->left, neg, out);
        collect_addends(s->right, neg, out);
    }
    else if (auto const* d = std::get_if<Difference>(&e->node))
    {
        collect_addends(d->left, neg, out);
        collect_addends(d->right, !neg, out);
    }
    else if (auto const* n = std::get_if<Negate>(&e->node))
        collect_addends(n->operand, !neg, out);
    else
        out.push_back({e, neg});
}

// A dyad addend c·u_k⊗u_k of the resolution of identity: the same concrete
// basis vector on both legs (dir), times a rank-0 scalar coefficient (null =
// 1).
struct DyadAddend final
{
    int dir;
    Expr const* coeff; // null = 1
};
auto as_identity_dyad(Context& ctx, Expr const* e, Basis const& b)
    -> std::optional<DyadAddend>
{
    std::vector<Expr const*> facs;
    flatten_product(e, facs);
    int dir = -1, legs = 0;
    std::vector<Expr const*> scalars;
    for (auto const* f: facs)
    {
        int const k = concrete_basis_dir(f, b);
        if (k >= 0)
        {
            if (legs == 0)
                dir = k;
            else if (k != dir)
                return std::nullopt; // two different directions — not a u_k⊗u_k
            ++legs;
        }
        else if (infer_rank(f) == std::optional<int>{0})
            scalars.push_back(f);
        else
            return std::nullopt; // a non-scalar, non-leg factor
    }
    if (legs != 2)
        return std::nullopt;
    return DyadAddend{dir, scalars.empty() ? nullptr : product_of(ctx, scalars)};
}

// Rebuild a signed-addend list into a Sum (empty → scalar 0).
auto rebuild_sum(
    Context& ctx,
    std::vector<std::pair<Expr const*, bool>> const& xs) -> Expr const*
{
    Expr const* acc = nullptr;
    for (auto const& [e, neg]: xs)
    {
        Expr const* term = neg ? make_negate(ctx, e) : e;
        acc = acc ? make_sum(ctx, acc, term) : term;
    }
    return acc ? acc : make_scalar(ctx, Rational{0});
}

// Fold complete groups of resolution-of-identity dyads in an addend list:
// addends c·u_k⊗u_k present for every k = 0…dim−1 with one common sign and
// coefficient collapse to one c·I.  nullopt when no group is complete.
auto fold_identity_dyads(
    Context& ctx,
    std::vector<std::pair<Expr const*, bool>> const& addends,
    Basis const& b) -> std::optional<std::vector<std::pair<Expr const*, bool>>>
{
    struct Item final
    {
        std::optional<DyadAddend> dyad;
        Expr const* coeff; // dyad coeff materialised to scalar 1 when null
        Expr const* raw;
        bool neg;
    };
    std::vector<Item> items;
    items.reserve(addends.size());
    for (auto const& [e, neg]: addends)
    {
        auto d = as_identity_dyad(ctx, e, b);
        Expr const* coeff =
            d ? (d->coeff ? d->coeff : make_scalar(ctx, Rational{1})) : nullptr;
        items.push_back(Item{d, coeff, e, neg});
    }

    std::vector<bool> consumed(items.size(), false);
    std::vector<std::pair<Expr const*, bool>> folded;
    bool any = false;
    for (std::size_t i = 0; i < items.size(); ++i)
    {
        if (consumed[i] || !items[i].dyad)
            continue;
        std::vector<int> member(static_cast<std::size_t>(b.dim()), -1);
        member[static_cast<std::size_t>(items[i].dyad->dir)] =
            static_cast<int>(i);
        for (std::size_t j = i + 1; j < items.size(); ++j)
        {
            if (consumed[j] || !items[j].dyad)
                continue;
            if (items[j].neg != items[i].neg)
                continue;
            if (!structural_eq(items[j].coeff, items[i].coeff))
                continue;
            auto& slot = member[static_cast<std::size_t>(items[j].dyad->dir)];
            if (slot == -1)
                slot = static_cast<int>(j);
        }
        if (std::any_of(
                member.begin(), member.end(), [](int x) { return x < 0; }))
            continue;
        for (int x: member)
            consumed[static_cast<std::size_t>(x)] = true;
        Expr const* c = items[i].dyad->coeff;
        Expr const* const id = make_identity(ctx, b.space()); // vibe 000082
        Expr const* term = c ? make_tensor_product(ctx, c, id) : id;
        folded.push_back({term, items[i].neg});
        any = true;
    }
    if (!any)
        return std::nullopt;
    std::vector<std::pair<Expr const*, bool>> out = std::move(folded);
    for (std::size_t i = 0; i < items.size(); ++i)
        if (!consumed[i])
            out.push_back({items[i].raw, items[i].neg});
    return out;
}

// A coordinate component of any rank: a non-basis, non-well-known tensor whose
// slots all carry CountableIndex ids and which expand_in_basis emitted at rank
// 0 (so a rank-1 basis vector — including a *foreign* basis's vector — is
// excluded by the rank-0 test).  Returns the tensor and its slot ids, in slot
// order.
auto as_coord_component(Expr const* e, Basis const& basis)
    -> std::optional<std::pair<TensorObject const*, std::vector<int>>>
{
    if (as_basis_vector(e, basis))
        return std::nullopt;
    auto const* t = std::get_if<TensorObject>(&e->node);
    if (!t || (t->traits && t->traits->well_known) || t->rank != 0
        || t->slots.empty())
        return std::nullopt;
    std::vector<int> ids;
    ids.reserve(t->slots.size());
    for (auto const& sb: t->slots)
    {
        if (!sb.index)
            return std::nullopt;
        // Every slot must belong to this basis (vibe 000067): a coordinate of a
        // foreign basis, or a two-point coordinate F_{iJ} whose slots straddle
        // two bases, is not a clean reassembly target in `basis`.
        if (sb.slot.basis_id != basis.basis_id())
            return std::nullopt;
        auto const* ci = std::get_if<CountableIndex>(&*sb.index);
        if (!ci)
            return std::nullopt;
        ids.push_back(ci->id);
    }
    return std::pair{t, std::move(ids)};
}

// The summed-index ids of a Levi-Civita factor, in slot order; nullopt if this
// is not an ε or any slot carries something other than a dummy index.
auto as_levi_civita_ids(Expr const* e) -> std::optional<std::vector<int>>
{
    auto const* t = std::get_if<TensorObject>(&e->node);
    if (!t || !t->traits || t->traits->well_known != WellKnownKind::LeviCivita)
        return std::nullopt;
    std::vector<int> ids;
    ids.reserve(t->slots.size());
    for (auto const& sb: t->slots)
    {
        if (!sb.index)
            return std::nullopt;
        auto const* ci = std::get_if<CountableIndex>(&*sb.index);
        if (!ci)
            return std::nullopt;
        ids.push_back(ci->id);
    }
    return ids;
}

// A coordinate carrier: an invariant value (the named tensor itself, or a
// contraction/trace of several) and the summed-index id riding on each of its
// slots, in slot order.  `origins` remembers which coordinate-component factor
// positions were absorbed, so they can be dropped once the carrier is realized.
// A rank-0 carrier (`legs` empty) is a scalar invariant (a dot, a trace, a
// bilinear form); rank ≥ 1 carriers are realized against the basis vectors that
// share their leg ids.
struct Carrier final
{
    Expr const* value;
    std::vector<int> legs;
    std::vector<int> origins;
    // Summed ids this carrier has already absorbed and that no later step will
    // see — the two vector indices an ε fold consumed, say.  They are only
    // released (their Σ binders dropped) if the carrier is realized, so a blob
    // that fails still leaves them bound.
    std::vector<int> folds;
};

auto slot_of(std::vector<int> const& legs, int id) -> int
{
    for (int s = 0; s < static_cast<int>(legs.size()); ++s)
        if (legs[s] == id)
            return s;
    return -1;
}

// Re-orient a carrier (rank ≤ 2) so that the slot carrying `id` is its *last*
// slot, transposing a rank-2 value when needed.  Returns false if it cannot
// (rank ≥ 3): the contraction is then left unfolded.
auto expose_last(Context& ctx, Carrier& c, int id) -> bool
{
    int const r = static_cast<int>(c.legs.size());
    int const s = slot_of(c.legs, id);
    if (s == r - 1)
        return true;
    if (r == 2 && s == 0)
    {
        c.value = make_transpose(ctx, c.value);
        std::swap(c.legs[0], c.legs[1]);
        return true;
    }
    return false;
}

// Symmetric to expose_last: bring the slot carrying `id` to the *first* slot.
auto expose_first(Context& ctx, Carrier& c, int id) -> bool
{
    int const r = static_cast<int>(c.legs.size());
    int const s = slot_of(c.legs, id);
    if (s == 0)
        return true;
    if (r == 2 && s == r - 1)
    {
        c.value = make_transpose(ctx, c.value);
        std::swap(c.legs[0], c.legs[1]);
        return true;
    }
    return false;
}

// Contract two carriers over the shared summed `id`, exposing it on X's last
// slot and Y's first so the result is the dot X·Y (X·Y always contracts X's
// last with Y's first).  The surviving legs are X's (minus last) then Y's
// (minus first).  nullopt when either carrier is rank ≥ 3.
auto contract_carriers(Context& ctx, Carrier X, Carrier Y, int id)
    -> std::optional<Carrier>
{
    if (X.legs.size() > 2 || Y.legs.size() > 2)
        return std::nullopt;
    if (!expose_last(ctx, X, id) || !expose_first(ctx, Y, id))
        return std::nullopt;
    Carrier r;
    r.value = make_dot(ctx, X.value, Y.value);
    for (int s = 0; s + 1 < static_cast<int>(X.legs.size()); ++s)
        r.legs.push_back(X.legs[s]);
    for (int s = 1; s < static_cast<int>(Y.legs.size()); ++s)
        r.legs.push_back(Y.legs[s]);
    r.origins = std::move(X.origins);
    r.origins.insert(r.origins.end(), Y.origins.begin(), Y.origins.end());
    r.folds = std::move(X.folds);
    r.folds.insert(r.folds.end(), Y.folds.begin(), Y.folds.end());
    return r;
}

// Contract two carriers over *several* shared summed ids at once — the
// "counted" fold of vibe 000103.  Which contraction results is not a stored
// pattern but a computation from two facts the index structure already holds:
// how many indices the pair shares, and in what order they sit.
//
//   n = 1                        →  X·Y            (handled by
//   contract_carriers) n = 2, Y's first two = (p,q) →  X : Y          [DDot,
//   (a⊗b):(c⊗d) = (a·c)(b·d)] n = 2, Y's first two = (q,p) →  X ·· Y [DDotAlt,
//   (a⊗b)··(c⊗d) = (a·d)(b·c)]
//
// where (p,q) are the ids on X's last two slots.  So C_{ijkl} e_{kl} folds to
// C:e while C_{ijkl} e_{lk} folds to C··e — both well-formed, and *different
// tensors*, which is why the order is read rather than assumed.
//
// The shared ids must occupy exactly X's trailing slots and Y's leading ones:
// no transpose can re-orient a rank ≥ 3 carrier, and a middle-slot contraction
// has no direct notation.  Anything else returns nullopt and the blob is left
// unfolded — a wrong pairing would be silent, so refusing is the safe failure
// (vibe 000103).  n ≥ 3 has no surface operator and is refused likewise.
auto contract_carriers_n(
    Context& ctx,
    Carrier X,
    Carrier Y,
    std::set<int> const& ids) -> std::optional<Carrier>
{
    auto const n = ids.size();
    if (n != 2)
        return std::nullopt; // n == 1 goes to contract_carriers; n ≥ 3
                             // unwritable
    auto const rx = X.legs.size();
    auto const ry = Y.legs.size();
    if (rx < n || ry < n)
        return std::nullopt;
    // The shared ids must be exactly X's last n and Y's first n.
    for (std::size_t k = 0; k < n; ++k)
        if (!ids.count(X.legs[rx - n + k]) || !ids.count(Y.legs[k]))
            return std::nullopt;
    int const p = X.legs[rx - 2];
    int const q = X.legs[rx - 1];
    if (p == q)
        return std::nullopt; // a self-trace, not a pair contraction
    Expr const* value = nullptr;
    if (Y.legs[0] == p && Y.legs[1] == q)
        value = make_ddot(ctx, X.value, Y.value);
    else if (Y.legs[0] == q && Y.legs[1] == p)
        value = make_ddot_alt(ctx, X.value, Y.value);
    else
        return std::nullopt;
    Carrier r;
    r.value = value;
    for (std::size_t k = 0; k + n < rx; ++k)
        r.legs.push_back(X.legs[k]);
    for (std::size_t k = n; k < ry; ++k)
        r.legs.push_back(Y.legs[k]);
    r.origins = std::move(X.origins);
    r.origins.insert(r.origins.end(), Y.origins.begin(), Y.origins.end());
    r.folds = std::move(X.folds);
    r.folds.insert(r.folds.end(), Y.folds.begin(), Y.folds.end());
    return r;
}

// Self-contract a carrier over a summed `id` appearing on two of its slots (a
// trace).  Only the full rank-2 trace tr(B) is expressible here; a partial
// trace of a rank ≥ 3 tensor is left unfolded (nullopt).
auto trace_carrier(Context& ctx, Carrier c, int id) -> std::optional<Carrier>
{
    if (c.legs.size() != 2 || c.legs[0] != id || c.legs[1] != id)
        return std::nullopt;
    Carrier r;
    r.value = make_trace(ctx, c.value);
    r.origins = std::move(c.origins);
    r.folds = std::move(c.folds);
    return r;
}

// A minimal union-find over carrier indices, to group carriers connected by
// shared (carrier-to-carrier) summed indices into independent contraction
// blobs.
struct UnionFind final
{
    std::vector<int> parent;
    explicit UnionFind(int n) : parent(n)
    {
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }
    auto find(int x) -> int
    {
        while (parent[x] != x)
            x = parent[x] = parent[parent[x]];
        return x;
    }
    void unite(int a, int b)
    {
        parent[find(a)] = find(b);
    }
};

// A basis-vector occurrence within a term's factor list: which factor, and
// where inside it (an empty path means the factor *is* the basis vector).
//
// The nested case is what lets a fold survive "pollution" (vibe 000103).  In
// Σ_i a_i (e_i·b) the partner of a_i sits inside a contraction operand, so a
// site named by factor position alone cannot address it and the fold stalls —
// the measured failure of `reassemble` on (a_i e^i)·b.  A path names it.
struct Site final
{
    std::size_t factor;
    Path path;

    [[nodiscard]] auto nested() const -> bool
    {
        return !path.empty();
    }
    auto operator<(Site const& o) const -> bool
    {
        return factor != o.factor ? factor < o.factor : path < o.path;
    }
};

// How many slots in `e` carry the summed index `id`.  Paired with the number of
// basis sites found for `id` in the same factor, this answers the only question
// the classifier needs: does anything *besides* those basis vectors mention the
// index?  (mentions_index gives a weaker yes/no that cannot tell "the e_i we
// are about to fold" from "a second, foreign carrier of i".)
auto count_index(Context& ctx, Expr const* e, int id) -> int
{
    int n = 0;
    rewrite_tree(
        ctx,
        e,
        [&](Context&, Expr const* node) -> Expr const*
        {
            if (auto const* t = std::get_if<TensorObject>(&node->node))
                for (auto const& sb: t->slots)
                    if (sb.index)
                        if (auto const* ci =
                                std::get_if<CountableIndex>(&*sb.index);
                            ci && ci->id == id)
                            ++n;
            return node;
        });
    return n;
}

// Collect the basis vectors of `basis` reachable from `e` through contraction
// and tensor-product structure alone, recording the path to each along with the
// summed id it carries.  Paths are built from `children`, the same accessor
// `replace_at` navigates, so a collected path is directly spliceable.
//
// Every other node kind stops the walk: descending through an ExplicitSum would
// cross a binding boundary, and a basis vector under (say) a Deriv is not a
// free-standing leg the fold may replace.
void collect_basis_sites(
    Expr const* e,
    Basis const& basis,
    std::size_t factor,
    Path& path,
    std::vector<std::pair<Site, int>>& out)
{
    if (auto bv = as_basis_vector(e, basis))
    {
        out.push_back({Site{factor, path}, bv->first.id});
        return;
    }
    bool const descend = std::holds_alternative<Dot>(e->node)
                         || std::holds_alternative<Cross>(e->node)
                         || std::holds_alternative<DDot>(e->node)
                         || std::holds_alternative<DDotAlt>(e->node)
                         || std::holds_alternative<TensorProduct>(e->node);
    if (!descend)
        return;
    auto const kids = children(e);
    for (std::size_t k = 0; k < kids.size(); ++k)
    {
        path.push_back(static_cast<int>(k));
        collect_basis_sites(kids[k], basis, factor, path, out);
        path.pop_back();
    }
}

// Reassemble the recognizable invariants buried in one basis-expanded product
// term, folding each *independently* and leaving every unrelated factor in
// place — so the folds apply even as parts of a larger term.  Coordinate
// components (rank-0 scalars, freely commuting) become carriers; basis vectors
// carry the non-commuting tensor order, so a realized invariant lands at the
// position of the basis vector(s) it pairs with.  Per summed index:
//   • carrier–basis        → leg realization (c_i e_i → c, B_ij e_i e_j → B,
//                            B_ij e_j e_i → Bᵀ);
//   • carrier–carrier      → contraction (u_i v_i → u·v, B_ij a_j → B·a,
//                            B_ij D_jk → B·D), chained within a blob so
//                            B_ki a_i c_k → c·B·a (a bilinear scalar);
//   • carrier self (twice) → trace (B_ii → tr B);
//   • basis–basis          → resolution of identity e_i e_i → I.
// A blob that cannot be fully expressed (rank ≥ 3 ordering/partial trace, a
// middle-slot contraction, or an index also carried by a foreign factor) is
// left entirely untouched, its indices still bound.  nullptr if nothing folds.
auto fold_reassembly_groups(
    Context& ctx,
    std::vector<int> const& summed,
    std::vector<Expr const*> const& factors,
    Basis const& basis,
    std::optional<TensorName> const& target) -> Expr const*
{
    auto coord_invariant = [&](TensorObject const* c, int rank)
    {
        // Preserve the source invariant's dimension (vibe 000082) so the
        // reassembled blob still equals the original user tensor.
        return make_tensor_object(ctx, c->name, {}, rank, c->dim);
    };

    // Classify the factors: basis vectors (by summed id), coordinate carriers,
    // and the summed ids blocked by some other (foreign) factor.
    std::vector<Carrier> carriers;
    std::map<int, std::vector<std::pair<int, int>>> in_carrier; // id→[(car,slot)]
    std::map<int, std::vector<Site>> in_basis; // id→[basis sites]
    std::set<int> const summed_set(summed.begin(), summed.end());
    std::set<int> blocked;
    std::vector<std::pair<std::size_t, std::vector<int>>> eps_factors;
    for (std::size_t p = 0; p < factors.size(); ++p)
    {
        if (auto bv = as_basis_vector(factors[p], basis);
            bv && summed_set.count(bv->first.id))
        {
            in_basis[bv->first.id].push_back(Site{p, {}});
            continue;
        }
        // `target` names the one invariant to rebuild (vibe 000106).  A
        // coordinate of any other name is not made into a carrier, so it falls
        // through to the foreign-factor branch below and blocks its own
        // indices — which is exactly right: its structure must be left alone.
        if (auto cc = as_coord_component(factors[p], basis);
            cc && (!target || cc->first->name == *target))
        {
            int const ci = static_cast<int>(carriers.size());
            Carrier c;
            c.value =
                coord_invariant(cc->first, static_cast<int>(cc->second.size()));
            c.legs = cc->second;
            c.origins = {static_cast<int>(p)};
            for (int s = 0; s < static_cast<int>(cc->second.size()); ++s)
                in_carrier[cc->second[s]].push_back({ci, s});
            carriers.push_back(std::move(c));
            continue;
        }
        // An ε is well-known, so it is not a coordinate carrier; set it aside
        // for the fold below rather than letting it block its own indices.
        if (auto eids = as_levi_civita_ids(factors[p]))
        {
            eps_factors.push_back({p, std::move(*eids)});
            continue;
        }
        // Neither a bare basis vector nor a coordinate — but it may still
        // *contain* basis vectors, reachable through contraction structure
        // (vibe 000103).  Those are foldable legs; anything else carrying a
        // summed id still blocks it.
        std::vector<std::pair<Site, int>> sites;
        Path path;
        collect_basis_sites(factors[p], basis, p, path, sites);
        std::map<int, int> found;
        for (auto const& [site, id]: sites)
            ++found[id];
        for (int id: summed)
        {
            int const occ = count_index(ctx, factors[p], id);
            if (occ == 0)
                continue;
            // Foldable only when *every* occurrence of the id in this factor is
            // one of the basis vectors we just collected; a second carrier of
            // the same index (e.g. c_i inside the same dot) blocks it, exactly
            // as the blanket mentions_index test used to.
            auto const it = found.find(id);
            if (it == found.end() || it->second != occ)
                blocked.insert(id);
        }
        for (auto const& [site, id]: sites)
            if (summed_set.count(id))
                in_basis[id].push_back(site);
    }

    // ---- ε folds: read a cross, or a triple product, off the indices ------
    //
    // ε is the fold table's third row (vibe 000103).  Its three indices say
    // exactly what it is doing, so nothing needs to match the term's shape:
    //
    //   ε_{ikj} a_k b_j e_i  →  (a×b) realized at e_i   two carriers + a leg
    //   ε_{ijk} a_i b_j c_k  →  a·(b×c)                 three carriers
    //
    // The slot *order* fixes the result, as it did for the double dots.  ε is
    // totally antisymmetric, so rotating the leg index to the front is
    // sign-free (ε_{abc} = ε_{bca} = ε_{cab}); the remaining two, read in the
    // rotated order, are the cross's operands.  Getting that order wrong would
    // silently flip a sign, so it is computed, never assumed.
    //
    // Only an orthonormal right-handed frame qualifies: there ε_{ijk} is the
    // plain permutation symbol and √g = 1, which is what `simplify_basis_cross`
    // emitted on the way in.  Elsewhere the weight would have to come back too.
    if (!eps_factors.empty() && basis.is_orthonormal()
        && is_scalar_one(basis.volume()))
    {
        // An index shared by two ε's is the ε-pair contraction's business, not
        // this fold's; count them so such an index is left alone.
        std::map<int, int> in_eps;
        for (auto const& [pos, ids]: eps_factors)
            for (int id: ids)
                ++in_eps[id];

        std::vector<bool> consumed(carriers.size(), false);
        std::vector<Carrier> made;
        for (auto const& [pos, ids]: eps_factors)
        {
            std::set<int> const distinct(ids.begin(), ids.end());
            if (ids.size() != 3 || distinct.size() != 3)
                continue; // a repeated index makes ε vanish — not our fold

            std::vector<int> carrier_of(3, -1);
            int leg_slot = -1;
            bool usable = true;
            for (int k = 0; k < 3 && usable; ++k)
            {
                int const id = ids[k];
                if (blocked.count(id) || !summed_set.count(id)
                    || in_eps[id] != 1)
                {
                    usable = false;
                    break;
                }
                auto const nc = in_carrier[id].size();
                auto const nb = in_basis[id].size();
                if (nc == 1 && nb == 0)
                {
                    auto const [ci, slot] = in_carrier[id][0];
                    // Only a rank-1 carrier is a cross operand; a higher-rank
                    // one would need a slot chosen, which ε does not say.
                    if (consumed[ci] || carriers[ci].legs.size() != 1)
                        usable = false;
                    else
                        carrier_of[k] = ci;
                }
                else if (nc == 0 && nb == 1 && leg_slot < 0)
                    leg_slot = k;
                else
                    usable = false;
            }
            if (!usable)
                continue;

            Carrier c;
            c.origins = {static_cast<int>(pos)};
            if (leg_slot < 0)
            {
                // Three carriers, nothing left over: the scalar triple product,
                // read straight off the slot order.
                for (int k = 0; k < 3; ++k)
                {
                    Carrier const& m = carriers[carrier_of[k]];
                    c.origins.insert(
                        c.origins.end(), m.origins.begin(), m.origins.end());
                }
                c.value = make_dot(
                    ctx,
                    carriers[carrier_of[0]].value,
                    make_cross(
                        ctx,
                        carriers[carrier_of[1]].value,
                        carriers[carrier_of[2]].value));
                c.folds = ids;
            }
            else
            {
                // Rotate the leg index to the front — cyclic, so no sign — and
                // the other two, in the rotated order, are the cross.
                int const u = (leg_slot + 1) % 3;
                int const v = (leg_slot + 2) % 3;
                for (int k: {u, v})
                {
                    Carrier const& m = carriers[carrier_of[k]];
                    c.origins.insert(
                        c.origins.end(), m.origins.begin(), m.origins.end());
                }
                c.value = make_cross(
                    ctx,
                    carriers[carrier_of[u]].value,
                    carriers[carrier_of[v]].value);
                c.legs = {ids[leg_slot]};
                c.folds = {ids[u], ids[v]};
            }
            for (int k = 0; k < 3; ++k)
                if (carrier_of[k] >= 0)
                    consumed[carrier_of[k]] = true;
            made.push_back(std::move(c));
        }

        if (!made.empty())
        {
            std::vector<Carrier> kept;
            for (std::size_t c = 0; c < carriers.size(); ++c)
                if (!consumed[c])
                    kept.push_back(std::move(carriers[c]));
            for (auto& c: made)
                kept.push_back(std::move(c));
            carriers = std::move(kept);
            // The carrier indices moved, so the id→carrier map is rebuilt.
            in_carrier.clear();
            for (int c = 0; c < static_cast<int>(carriers.size()); ++c)
                for (int s = 0; s < static_cast<int>(carriers[c].legs.size());
                     ++s)
                    in_carrier[carriers[c].legs[s]].push_back({c, s});
        }
    }

    // Per summed id: internal (carrier↔carrier or self), leg (carrier↔basis),
    // identity (basis↔basis), or unfoldable.
    enum class Kind
    {
        None,
        Internal,
        Leg,
        Identity
    };
    std::map<int, Kind> kind;
    for (int id: summed)
    {
        int const nc = static_cast<int>(in_carrier[id].size());
        int const nb = static_cast<int>(in_basis[id].size());
        if (blocked.count(id))
            kind[id] = Kind::None;
        else if (nc == 2 && nb == 0)
            kind[id] = Kind::Internal;
        else if (nc == 1 && nb == 1)
            kind[id] = Kind::Leg;
        else if (nc == 0 && nb == 2)
            kind[id] = Kind::Identity;
        else
            kind[id] = Kind::None;
    }

    // Group carriers into blobs joined by Internal (carrier-to-carrier) ids.
    UnionFind uf(static_cast<int>(carriers.size()));
    for (int id: summed)
        if (kind[id] == Kind::Internal)
        {
            auto const& occ = in_carrier[id];
            if (occ[0].first != occ[1].first)
                uf.unite(occ[0].first, occ[1].first);
        }

    std::set<Site> drop;                 // sites removed
    std::map<Site, Expr const*> replace; // basis site → realized invariant
    std::vector<Expr const*> scalars;    // scalar folds, emitted first
    std::set<int> folded;

    // ---- carrier blobs: contract internally, then realize remaining legs ----
    std::map<int, std::vector<int>> blob; // root → carrier indices
    for (int c = 0; c < static_cast<int>(carriers.size()); ++c)
        blob[uf.find(c)].push_back(c);

    for (auto const& [root, members]: blob)
    {
        // Internal ids whose both occurrences lie in this blob.
        std::set<int> internal;
        std::set<int> mem(members.begin(), members.end());
        for (int id: summed)
            if (kind[id] == Kind::Internal
                && mem.count(in_carrier[id][0].first))
                internal.insert(id);

        std::vector<Carrier> active;
        for (int c: members)
            active.push_back(carriers[c]);

        bool ok = true;
        while (ok && !internal.empty())
        {
            int const id = *internal.begin();
            internal.erase(internal.begin());
            std::vector<std::pair<int, int>> occ; // (active idx, slot)
            for (int a = 0; a < static_cast<int>(active.size()); ++a)
                for (int s = 0; s < static_cast<int>(active[a].legs.size());
                     ++s)
                    if (active[a].legs[s] == id)
                        occ.push_back({a, s});
            if (occ.size() != 2)
            {
                ok = false;
                break;
            }
            if (occ[0].first == occ[1].first)
            {
                auto t = trace_carrier(ctx, active[occ[0].first], id);
                if (!t)
                {
                    ok = false;
                    break;
                }
                active[occ[0].first] = std::move(*t);
            }
            else
            {
                int const a = occ[0].first;
                int const b = occ[1].first;
                // Every id this pair shares must be contracted in one move: a
                // pair sharing two indices is a double dot, and taking them one
                // at a time would ask for an intermediate rank the notation
                // cannot express (vibe 000103).
                std::set<int> shared{id};
                for (int other: internal)
                {
                    auto const& oc = in_carrier[other];
                    if (oc.size() != 2)
                        continue;
                    bool const on_a = active[a].legs.size()
                                      && slot_of(active[a].legs, other) >= 0;
                    bool const on_b = active[b].legs.size()
                                      && slot_of(active[b].legs, other) >= 0;
                    if (on_a && on_b)
                        shared.insert(other);
                }
                // Either carrier may be the left operand; only the ordering
                // whose shared ids sit on X's trailing and Y's leading slots is
                // expressible, so try both and let the slot test pick.
                auto m =
                    shared.size() == 1 ?
                        contract_carriers(ctx, active[a], active[b], id) :
                        contract_carriers_n(ctx, active[a], active[b], shared);
                if (!m && shared.size() > 1)
                    m = contract_carriers_n(ctx, active[b], active[a], shared);
                if (!m)
                {
                    ok = false;
                    break;
                }
                for (int done: shared)
                    internal.erase(done);
                int const hi = std::max(a, b);
                int const lo = std::min(a, b);
                active.erase(active.begin() + hi);
                active.erase(active.begin() + lo);
                active.push_back(std::move(*m));
            }
        }

        // Realize each surviving carrier; accumulate into blob-local changes so
        // a failure leaves the whole blob untouched.
        std::set<Site> ldrop;
        std::map<Site, Expr const*> lreplace;
        std::vector<Expr const*> lscalars;
        std::set<int> lfolded;
        for (Carrier& c: active)
        {
            if (!ok)
                break;
            lfolded.insert(c.folds.begin(), c.folds.end());
            if (c.legs.empty()) // scalar invariant (dot / trace / bilinear)
            {
                lscalars.push_back(c.value);
                for (int o: c.origins)
                    ldrop.insert(Site{static_cast<std::size_t>(o), {}});
            }
            else if (c.legs.size() == 1) // vector leg → place at its basis vec
            {
                // Honour the classifier's verdict.  It is the only place that
                // knows an id carries a *second* carrier or a foreign factor;
                // realizing such an id anyway would place this carrier at the
                // basis site and silently drop the other one (Σ_i a_i c_i
                // (e_i·y) folding to c·y).  Before nested sites existed this
                // was masked by in_basis being empty for anything but a bare
                // basis vector — masked, not decided.
                if (kind[c.legs[0]] != Kind::Leg)
                {
                    ok = false;
                    break;
                }
                auto const& bp = in_basis[c.legs[0]];
                if (bp.size() != 1)
                {
                    ok = false;
                    break;
                }
                // A rank-1 invariant may be spliced in at a *nested* site
                // safely: a vector occupies exactly the one slot the basis
                // vector did, so every enclosing contraction keeps its meaning
                // (a_i (e_i·b) → a·b, a_i (e_i×b) → a×b, a_i (B·e_i) → B·a).
                lreplace[bp[0]] = c.value;
                lfolded.insert(c.legs[0]);
                for (int o: c.origins)
                    ldrop.insert(Site{static_cast<std::size_t>(o), {}});
            }
            else if (c.legs.size() == 2) // tensor → place at the leftmost basis
            {
                if (kind[c.legs[0]] != Kind::Leg
                    || kind[c.legs[1]] != Kind::Leg)
                {
                    ok = false; // see the rank-1 branch
                    break;
                }
                auto const& b0 = in_basis[c.legs[0]];
                auto const& b1 = in_basis[c.legs[1]];
                if (b0.size() != 1 || b1.size() != 1)
                {
                    ok = false;
                    break;
                }
                Site const p0 = b0[0];
                Site const p1 = b1[0];
                if (p0.nested() || p1.nested())
                {
                    // Rank ≥ 2 realization drops one basis site and places the
                    // invariant at the other, and inside a contraction neither
                    // move is safe: an operand cannot be dropped, and a rank-2
                    // value spliced at a nested site would silently take the
                    // wrong slot orientation (e_i's position says "first slot",
                    // but a Dot contracts the last).  A wrong pairing is not
                    // detectable downstream, so refuse (vibe 000103).
                    ok = false;
                    break;
                }
                // The leftmost basis vector fixes the first tensor slot: in
                // slot order → value; reversed → its transpose.
                Expr const* tens =
                    (p0 < p1) ? c.value : make_transpose(ctx, c.value);
                lreplace[std::min(p0, p1)] = tens;
                ldrop.insert(std::max(p0, p1));
                lfolded.insert(c.legs[0]);
                lfolded.insert(c.legs[1]);
                for (int o: c.origins)
                    ldrop.insert(Site{static_cast<std::size_t>(o), {}});
            }
            else // rank ≥ 3 leg realization: ordering not expressible here
            {
                ok = false;
                break;
            }
        }
        if (!ok)
            continue; // leave this blob and its indices alone
        drop.insert(ldrop.begin(), ldrop.end());
        for (auto const& [pos, e]: lreplace)
            replace[pos] = e;
        scalars.insert(scalars.end(), lscalars.begin(), lscalars.end());
        // The internal ids of a fully-realized blob are consumed too.
        for (int id: summed)
            if (kind[id] == Kind::Internal
                && mem.count(in_carrier[id][0].first))
                folded.insert(id);
        folded.insert(lfolded.begin(), lfolded.end());
    }

    // ---- basis↔basis: resolution of identity e_i e_i → I ----
    for (int id: summed)
        if (kind[id] == Kind::Identity)
        {
            auto const& bp = in_basis[id];
            if (bp[0].nested() || bp[1].nested())
                continue; // would drop a contraction operand — see above
            replace[bp[0]] = make_identity(ctx, basis.space()); // vibe 000082
            drop.insert(bp[1]);
            folded.insert(id);
        }

    if (folded.empty())
        return nullptr;

    std::vector<Expr const*> out = scalars;
    for (std::size_t p = 0; p < factors.size(); ++p)
    {
        Site const whole{p, {}};
        if (drop.count(whole))
            continue;
        if (auto it = replace.find(whole); it != replace.end())
        {
            out.push_back(it->second);
            continue;
        }
        // Splice each realized invariant into this factor at its own path,
        // leaving the rest of the contraction untouched.  Substituting a rank-1
        // value for a basis vector preserves the tree's shape, so the remaining
        // paths stay valid across the updates.
        Expr const* f = factors[p];
        for (auto const& [site, value]: replace)
            if (site.factor == p && site.nested())
                f = replace_at(ctx, f, site.path, value);
        out.push_back(f);
    }
    std::vector<int> rest;
    for (int id: summed)
        if (!folded.count(id))
            rest.push_back(id);
    return wrap_sums(ctx, rest, product_of(ctx, out));
}

} // namespace

namespace
{

// Run `moves` until the expression stops changing.  The cap is a safety net for
// a non-convergent move, not a tuning knob: convergence is detected, so a
// well-behaved pipeline never reaches it.
template <typename F>
auto to_fixpoint(Context& ctx, Expr const* e, F const& moves, int cap = 16)
    -> Expr const*
{
    for (int pass = 0; pass < cap; ++pass)
    {
        Expr const* const before = e;
        e = moves(ctx, e);
        if (structural_eq(e, before))
            return e;
    }
    return e;
}

} // namespace

auto reduce_frame(
    Context& ctx,
    Expr const* e,
    Basis const& basis,
    StepReport* report) -> Expr const*
{
    auto const* out = to_fixpoint(
        ctx,
        e,
        [&basis](Context& c, Expr const* x) -> Expr const*
        {
            x = simplify_basis_cross(c, x, basis);
            x = simplify_basis_dot(c, x, basis);
            // canonicalize materializes the implicit sums the contraction needs
            // to see; contract_delta puts the expression back in implicit form
            // itself, so the pair composes.
            try
            {
                x = steps::canonicalize(c, x);
            }
            catch (std::invalid_argument const&)
            {
                // An ill-formed implicit summation is not this step's business.
                return x;
            }
            return steps::contract_delta(c, x);
        });
    // "Did it do work?", not "did it change?" — the fixpoint canonicalizes, so
    // a pass that reduced nothing still returns a reordered expression.
    // Comparing fingerprints is what tells the two apart (the same distinction
    // that `applicable` rests on, and that composing self-preparing folds
    // needed). The *return value* is the normalized result, as it always was;
    // the report carries whether any work was done.  Separating those is the
    // point of StepReport: before it, "did it fire?" had to be inferred from
    // the return, which forced a step either to lie about its output or to lie
    // about its effect.  Compare against the finished form — the fixpoint
    // materializes Σ binders internally, so an un-implicitized result always
    // looks changed.
    Expr const* const done =
        structural_eq(out, e) ? e : steps::implicitize(ctx, out);
    auto const before = expression_shape(ctx, e);
    if (before == expression_shape(ctx, done))
    {
        // Nothing reduced.  Say which of the two reasons it was, since they
        // point at different next moves: with no frame structure at all the
        // expression has not been expanded yet, while a frame-bearing term that
        // will not reduce is one the *frame* cannot say more about — an ε-pair
        // to contract, a metric to spend, an identity to use.
        auto const& sh = before;
        report_no_op(
            report,
            sh.basis_vectors == 0 ?
                "there are no frame vectors here to reduce — expand_in_basis "
                "puts an invariant into components first" :
                "the frame has nothing further to say about this term; what "
                "remains needs a step the frame cannot justify (an ε-pair "
                "contraction, a metric move, an identity)");
        return done;
    }
    report_fired(report);
    return done;
}

auto to_concrete(Context& ctx, Expr const* e, Basis const& basis) -> Expr const*
{
    (void)basis; // the directions come from the expression's own index spaces
    auto const* out = to_fixpoint(
        ctx,
        e,
        [](Context& c, Expr const* x) -> Expr const*
        {
            try
            {
                x = steps::canonicalize(c, x);
            }
            catch (std::invalid_argument const&)
            {
                return x;
            }
            x = steps::unroll_sums(c, x);
            x = steps::eval_eps_concrete(c, x);
            x = steps::eval_delta_concrete(c, x);
            return steps::fold_arithmetic(c, x);
        });
    return structural_eq(out, e) ? e : steps::implicitize(ctx, out);
}

auto reassemble_pass(
    Context& ctx,
    Expr const* e,
    Basis const& basis,
    std::optional<TensorName> const& target) -> Expr const*
{
    // Self-prepare: the fold reads the summation binders off explicit
    // ExplicitSum nodes, so materialize the implicit Einstein sums first (via
    // canonicalize) — the caller never has to.  canonicalize throws on an
    // ill-formed implicit sum; that just means "nothing to reassemble".  A
    // genuine no-op returns the original expression untouched.
    Expr const* prepped = e;
    try
    {
        prepped = steps::canonicalize(ctx, e);
    }
    catch (std::invalid_argument const&)
    {
        prepped = e;
    }
    auto const* out = rewrite_tree(
        ctx,
        prepped,
        [&](Context& c, Expr const* node) -> Expr const*
        {
            // First try the resolution-of-identity-with-contraction fold
            // Σ_i (X·e_i) e_i → X (vibe 000068 P2): reassemble should finish on
            // its own, so it runs the completeness fold (which self-distributes
            // over sums/signs) before the coordinate-carrier folds below.  The
            // focused `reassemble_completeness` remains for callers who want
            // only this.
            // Σ_i (X·e_i) e_i → X folds basis structure, not a named object,
            // so a targeted call skips it.
            if (!target)
                if (auto* fc = fold_completeness(c, node, basis))
                    return fc;

            // Peel nested ExplicitSums (collecting the summed indices) and
            // signs, interleaved.  A subtracted term carries its sign as a
            // Negate, which canonicalize may leave *between* two binders, e.g.
            // Σ_j −(Σ_i …).  Peeling all sums first then one sign would trap
            // the inner i-binder behind the Negate, so walk the chain in order
            // and track the running sign (an even number of Negates cancels).
            std::vector<int> summed;
            bool negated = false;
            Expr const* body = node;
            for (;;)
            {
                if (auto const* es = std::get_if<ExplicitSum>(&body->node))
                {
                    if (es->bound)
                        return node; // symbolic bound: not a basis expansion
                    summed.push_back(es->index.id);
                    body = es->body;
                    continue;
                }
                if (auto const* n = std::get_if<Negate>(&body->node))
                {
                    negated = !negated;
                    body = n->operand;
                    continue;
                }
                break;
            }
            if (summed.empty())
                return node;
            auto signed_ = [&](Expr const* r)
            { return negated ? make_negate(c, r) : r; };

            std::vector<Expr const*> factors;
            flatten_product(body, factors);

            // First fold each coordinate vector / dot / identity group on its
            // own (handles a term with several coordinate factors).  Falls
            // through when nothing matches, leaving the single higher-rank
            // coordinate to the whole-term path below.
            if (auto* g =
                    fold_reassembly_groups(c, summed, factors, basis, target))
                return signed_(g);

            // The body is one coordinate tensor times a polyad of basis
            // vectors.  Partition the flattened factors accordingly.
            std::vector<int> vec_ids;
            TensorObject const* coord = nullptr;
            for (auto const* f: factors)
            {
                if (auto bv = as_basis_vector(f, basis))
                {
                    vec_ids.push_back(bv->first.id);
                    continue;
                }
                auto const* t = std::get_if<TensorObject>(&f->node);
                if (!t || coord)
                    return node; // a non-coordinate factor, or a second one
                if (target && t->name != *target)
                    return node; // not the invariant asked for
                // The coordinate must belong to this basis (vibe 000067): a
                // foreign or two-point coordinate (slots straddling two bases)
                // is not a clean reassembly target, so leave the term unfolded.
                for (auto const& sb: t->slots)
                    if (sb.slot.basis_id != basis.basis_id())
                        return node;
                coord = t;
            }
            if (!coord)
            {
                // No coordinate factor: the resolution of identity
                // Σ_i e_i ⊗ e^i (two basis vectors sharing the one summed
                // index, nothing else) folds back to the identity tensor.
                auto const s = sorted(summed);
                if (vec_ids.size() == 2 && s.size() == 1 && vec_ids[0] == s[0]
                    && vec_ids[1] == s[0])
                    return signed_(make_identity(c, basis.space())); // vibe 82
                return node;
            }
            auto const coord_ids = countable_slot_ids(*coord);
            if (!coord_ids)
                return node;

            // Each summed index must appear exactly once as a coordinate slot
            // and once as a basis vector, with nothing left over.
            auto const s = sorted(summed);
            if (std::adjacent_find(s.begin(), s.end()) != s.end())
                return node;
            if (sorted(vec_ids) != s || sorted(*coord_ids) != s)
                return node;

            return signed_(make_tensor_object(
                c, coord->name, {}, static_cast<int>(s.size()), coord->dim));
        });
    // Surface the prepared result, not the raw input (vibe 000064 #3/#4/#6).
    // The self-prep canonicalize may simplify the input on its own — cancelling
    // equal-and-opposite terms, collapsing a sum — even when no fold then fires
    // (#6); and a partial fold leaves the unfolded terms carrying the Σ binders
    // canonicalize materialized, which must be stripped back to implicit (#3).
    // So implicitize the output, and report a no-op (returning the original
    // `e`) only when the fold did not fire (`out == prepped`, rewrite_tree
    // reuses the pointer) *and* the prep left the expression structurally
    // unchanged.
    if (out == prepped && structural_eq(prepped, e))
        return e;
    return steps::implicitize(ctx, out);
}

auto reassemble_completeness(Context& ctx, Expr const* e, Basis const& basis)
    -> Expr const*
{
    Expr const* prepped = e;
    try
    {
        prepped = steps::canonicalize(ctx, e);
    }
    catch (std::invalid_argument const&)
    {
        prepped = e;
    }
    auto const* out = rewrite_tree(
        ctx,
        prepped,
        [&](Context& c, Expr const* node) -> Expr const*
        {
            auto* f = fold_completeness(c, node, basis);
            return f ? f : node;
        });
    // Same as reassemble: surface prep + strip materialized Σ (vibe 000064
    // #3/#4/#6); a genuine no-op still returns the original `e`.
    if (out == prepped && structural_eq(prepped, e))
        return e;
    return steps::implicitize(ctx, out);
}

auto fold_resolution_of_identity(
    Context& ctx, Expr const* e, Basis const& basis) -> Expr const*
{
    if (!basis.is_orthonormal())
        return e;
    Expr const* prepped = e;
    try
    {
        prepped = steps::canonicalize(ctx, e);
    }
    catch (std::invalid_argument const&)
    {
        prepped = e;
    }
    auto const* out = rewrite_tree(
        ctx,
        prepped,
        [&](Context& c, Expr const* node) -> Expr const*
        {
            if (!std::holds_alternative<Sum>(node->node)
                && !std::holds_alternative<Difference>(node->node))
                return node;
            std::vector<std::pair<Expr const*, bool>> addends;
            collect_addends(node, false, addends);
            auto folded = fold_identity_dyads(c, addends, basis);
            return folded ? rebuild_sum(c, *folded) : node;
        });
    if (out == prepped && structural_eq(prepped, e))
        return e;
    return steps::implicitize(ctx, out);
}

auto reassemble(
    Context& ctx,
    Expr const* e,
    Basis const& basis,
    std::optional<TensorName> target) -> Expr const*
{
    // One entry point for reassembly (vibe 000106).  There were three —
    // `reassemble`, `reassemble_completeness` and `fold_resolution_of_identity`
    // — and a caller had to know which shape they were holding to pick one.
    // They are all "put this back into direct notation", so they run together,
    // to a fixed point: one fold can expose another's pattern.
    //
    // With a `target` the completeness folds are skipped: they rebuild *basis*
    // structure (Σ e_i⊗e_i → I), which is not the named object the caller asked
    // for, and folding it anyway would do work that was not requested.
    // "Same up to normalization": each pass self-prepares by canonicalizing,
    // and canon reorders a symmetric contraction (`y·a` → `a·y`).  So a pass
    // that folded nothing still returns a *structurally* different expression,
    // and a naive fixed point would loop once more and hand back the reordered
    // form — changing what every existing caller sees, for no gain.  Progress
    // therefore means "changed beyond canon", and the last productive result is
    // returned.
    auto same_modulo_canon = [](Context& c, Expr const* x, Expr const* y)
    {
        if (structural_eq(x, y))
            return true;
        try
        {
            return structural_eq(
                steps::canonicalize(c, x), steps::canonicalize(c, y));
        }
        catch (std::invalid_argument const&)
        {
            return false;
        }
    };

    // Each of the three folds self-prepares, so each *also* returns a
    // canonicalized expression when it folded nothing.  Chaining them naively
    // would let a fold that did no work reorder the previous fold's answer.  So
    // a fold's result is kept only if it changed something beyond canon.
    auto keep_if_productive =
        [&](Context& c, Expr const* x, Expr const* folded) -> Expr const*
    { return same_modulo_canon(c, folded, x) ? x : folded; };

    auto once = [&](Context& c, Expr const* x) -> Expr const*
    {
        x = reassemble_pass(c, x, basis, target);
        if (!target)
        {
            x = keep_if_productive(c, x, reassemble_completeness(c, x, basis));
            x = keep_if_productive(
                c, x, fold_resolution_of_identity(c, x, basis));
        }
        return x;
    };

    // The first pass runs unconditionally and its result stands: a pass
    // surfaces its own preparation even when no fold fired (vibe 000064 #6 —
    // the prep can cancel equal-and-opposite terms on its own), and that
    // contract predates this unification.  Only *further* passes need to earn
    // their place.
    Expr const* cur = once(ctx, e);
    for (int pass = 1; pass < 16; ++pass)
    {
        Expr const* const next = once(ctx, cur);
        if (same_modulo_canon(ctx, next, cur))
            break;
        cur = next;
    }
    return cur;
}

auto expand_identity(Context& ctx, Expr const* e, Basis const& basis)
    -> Expr const*
{
    if (!basis.is_orthonormal())
        throw std::invalid_argument(
            "expand_identity: the resolution of identity Σ_k e_k⊗e_k = I holds "
            "only for an orthonormal basis");
    return rewrite_tree(
        ctx,
        e,
        [&](Context& c, Expr const* node) -> Expr const*
        {
            if (!is_identity_tensor(node))
                return node;
            Expr const* sum = nullptr;
            for (int k = 0; k < basis.dim(); ++k)
            {
                Expr const* dyad =
                    make_tensor_product(c, basis.basis(k), basis.basis(k));
                sum = sum ? make_sum(c, sum, dyad) : dyad;
            }
            return sum ? sum : node;
        });
}

} // namespace tender
