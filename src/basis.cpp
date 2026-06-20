#include <tender/basis.hpp>

#include <tender/derivation.hpp>
#include <tender/rewrite.hpp>

#include <optional>
#include <stdexcept>
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

} // namespace

auto make_orthonormal_basis(
    IndexSpace const* space,
    std::vector<Expr const*> vectors,
    std::string_view vector_symbol) -> Basis
{
    if (!space)
        throw std::invalid_argument("make_orthonormal_basis: null space");
    if (vectors.empty())
        throw std::invalid_argument(
            "make_orthonormal_basis: at least one vector is required");
    if (vectors.size() != space->values().size())
        throw std::invalid_argument(
            "make_orthonormal_basis: number of vectors must equal the index "
            "space cardinality");
    for (auto const* v: vectors)
    {
        if (!v)
            throw std::invalid_argument(
                "make_orthonormal_basis: null basis vector");
        if (!rank_ok(v))
            throw std::invalid_argument(
                "make_orthonormal_basis: basis vectors must be rank 1");
    }

    // Orthonormal: the cobasis coincides with the basis.
    auto covectors = vectors;
    return Basis{
        Realm::Orthonormal,
        space,
        make_tensor_name(vector_symbol),
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

} // namespace

auto expand_in_basis(
    Context& ctx,
    Expr const* e,
    Basis const& basis,
    Variance variance) -> Expr const*
{
    bool const ortho = basis.is_orthonormal();
    // Coordinate index level, chosen so the shared index Einstein-contracts
    // against the basis vector: orthonormal pairs two lower indices; oblique
    // pairs one upper with one lower.
    Level const coord_level = variance == Variance::Covariant ?
                                  (ortho ? Level::Lower : Level::Upper) :
                                  Level::Lower;

    return rewrite_tree(
        ctx,
        e,
        [&](Context& c, Expr const* node) -> Expr const*
        {
            auto const* t = std::get_if<TensorObject>(&node->node);
            if (!t || !is_expandable_invariant(*t))
                return node;

            int const r = *t->rank;
            std::vector<SlotBinding> coord_slots;
            coord_slots.reserve(static_cast<std::size_t>(r));
            Expr const* polyad = nullptr;
            for (int k = 0; k < r; ++k)
            {
                CountableIndex const idx{c.alloc_index_id()};
                coord_slots.push_back(SlotBinding{
                    IndexSlot{coord_level, basis.realm(), basis.space()},
                    IndexAssoc{idx}});
                Expr const* const vec = variance == Variance::Covariant ?
                                            basis.covariant_vector(c, idx) :
                                            basis.contravariant_vector(c, idx);
                polyad = polyad ? make_tensor_product(c, polyad, vec) : vec;
            }
            Expr const* const coord =
                make_tensor_object(c, t->name, std::move(coord_slots), 0);
            return make_tensor_product(c, coord, polyad);
        });
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

            Expr const* result = make_delta(
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

} // namespace tender
