#include <tender/basis.hpp>

#include <tender/derivation.hpp>
#include <tender/rewrite.hpp>

#include <algorithm>
#include <optional>
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
    std::vector<Expr const*> covectors) :
  realm_(realm),
  space_(space),
  symbol_(symbol),
  vectors_(std::move(vectors)),
  covectors_(std::move(covectors))
{
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
            IndexSlot{Level::Lower, realm_, space_}, IndexAssoc{index}}},
        1);
}

auto Basis::contravariant_vector(Context& ctx, CountableIndex index) const
    -> Expr const*
{
    auto const level = is_orthonormal() ? Level::Lower : Level::Upper;
    return make_tensor_object(
        ctx,
        symbol_,
        {SlotBinding{IndexSlot{level, realm_, space_}, IndexAssoc{index}}},
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

auto make_orthonormal_basis(
    IndexSpace const* space,
    std::vector<Expr const*> vectors,
    TensorName vector_symbol) -> Basis
{
    validate_basis_vectors("make_orthonormal_basis", space, vectors);

    // Orthonormal: the cobasis coincides with the basis.
    auto covectors = vectors;
    return Basis{
        Realm::Orthonormal,
        space,
        vector_symbol,
        std::move(vectors),
        std::move(covectors)};
}

auto make_oblique_basis(
    Context& ctx,
    IndexSpace const* space,
    std::vector<Expr const*> vectors,
    TensorName vector_symbol) -> Basis
{
    validate_basis_vectors("make_oblique_basis", space, vectors);
    if (vectors.size() != 3)
        throw std::invalid_argument(
            "make_oblique_basis: only 3D oblique bases are supported (cobasis "
            "derived via the cross-product formula)");

    // Reciprocal basis: e^0 = (e_1×e_2)/V, e^1 = (e_2×e_0)/V, e^2 = (e_0×e_1)/V
    // with the cell volume V = e_0·(e_1×e_2).
    Expr const* const vol =
        make_dot(ctx, vectors[0], make_cross(ctx, vectors[1], vectors[2]));
    auto cob = [&](Expr const* a, Expr const* b) -> Expr const*
    { return make_scalar_div(ctx, make_cross(ctx, a, b), vol); };
    std::vector<Expr const*> covectors{
        cob(vectors[1], vectors[2]),
        cob(vectors[2], vectors[0]),
        cob(vectors[0], vectors[1])};

    return Basis{
        Realm::Oblique,
        space,
        vector_symbol,
        std::move(vectors),
        std::move(covectors)};
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
                CountableIndex const idx{c.alloc_index_id()};
                return make_tensor_product(
                    c,
                    basis.covariant_vector(c, idx),
                    basis.contravariant_vector(c, idx));
            }

            if (!is_expandable_invariant(*t))
                return node;

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
                        coord_level_for(v, ortho), basis.realm(), basis.space()},
                    IndexAssoc{idx}});
                Expr const* const vec = v == Variance::Covariant ?
                                            basis.covariant_vector(c, idx) :
                                            basis.contravariant_vector(c, idx);
                polyad = polyad ? make_tensor_product(c, polyad, vec) : vec;
            }
            Expr const* const coord =
                make_tensor_object(c, t->name, std::move(coord_slots), 0);
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
    CountableIndex index;
    Level level;
};

// Is e a symbolic basis vector of b — its vector symbol, rank 1, one slot
// carrying a CountableIndex?  Returns the index and its level.
auto as_basis_vector(Expr const* e, Basis const& b)
    -> std::optional<std::pair<CountableIndex, Level>>
{
    auto const* t = std::get_if<TensorObject>(&e->node);
    if (!t || t->name.v.view() != b.vector_symbol().v.view()
        || t->slots.size() != 1 || !t->slots[0].index)
        return std::nullopt;
    auto const* ci = std::get_if<CountableIndex>(&*t->slots[0].index);
    if (!ci)
        return std::nullopt;
    return std::pair{*ci, t->slots[0].slot.level};
}

// Split one dot operand into an optional component factor and a basis vector.
// Accepts a bare basis vector or a product of a component value with one.
auto as_vec_side(Expr const* e, Basis const& b) -> std::optional<VecSide>
{
    if (auto bv = as_basis_vector(e, b))
        return VecSide{nullptr, bv->first, bv->second};
    auto const* tp = std::get_if<TensorProduct>(&e->node);
    if (!tp)
        return std::nullopt;
    if (auto bv = as_basis_vector(tp->right, b);
        bv && is_component_valued(tp->left))
        return VecSide{tp->left, bv->first, bv->second};
    if (auto bv = as_basis_vector(tp->left, b);
        bv && is_component_valued(tp->right))
        return VecSide{tp->right, bv->first, bv->second};
    return std::nullopt;
}

} // namespace

auto simplify_basis_dot(Context& ctx, Expr const* e, Basis const& basis)
    -> Expr const*
{
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
                IndexAssoc{l->index},
                IndexAssoc{r->index});
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

} // namespace

auto reassemble(Context& ctx, Expr const* e, Basis const& basis) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [&](Context& c, Expr const* node) -> Expr const*
        {
            // Peel nested ExplicitSums, collecting the summed indices.
            std::vector<int> summed;
            Expr const* body = node;
            while (auto const* es = std::get_if<ExplicitSum>(&body->node))
            {
                if (es->bound)
                    return node; // symbolic bound: not a basis expansion
                summed.push_back(es->index.id);
                body = es->body;
            }
            if (summed.empty())
                return node;

            // The body is one coordinate tensor times a polyad of basis
            // vectors.  Partition the flattened factors accordingly.
            std::vector<Expr const*> factors;
            flatten_product(body, factors);
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
                    return make_identity(c);
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

            return make_tensor_object(
                c, coord->name, {}, static_cast<int>(s.size()));
        });
}

} // namespace tender
