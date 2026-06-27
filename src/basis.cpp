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
    Expr const* volume) :
  realm_(realm),
  space_(space),
  symbol_(symbol),
  vectors_(std::move(vectors)),
  covectors_(std::move(covectors)),
  volume_(volume)
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
    Context& ctx,
    IndexSpace const* space,
    std::vector<Expr const*> vectors,
    TensorName vector_symbol,
    Handedness handedness) -> Basis
{
    validate_basis_vectors("make_orthonormal_basis", space, vectors);

    // Orthonormal: the cobasis coincides with the basis, and √g = ±1 by
    // handedness (+1 right-handed, -1 left-handed).
    auto covectors = vectors;
    Expr const* const vol =
        make_scalar(ctx, Rational{handedness == Handedness::Right ? 1 : -1});
    return Basis{
        Realm::Orthonormal,
        space,
        vector_symbol,
        std::move(vectors),
        std::move(covectors),
        vol};
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
    // with the cell volume V = √g = e_0·(e_1×e_2).
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
        std::move(covectors),
        vol};
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
    if (auto bv = as_basis_vector(e, b))
        return VecSide{nullptr, bv->first, bv->second};
    auto const* tp = std::get_if<TensorProduct>(&e->node);
    if (!tp)
        return std::nullopt;
    if (auto bv = as_basis_vector(tp->right, b);
        bv && is_scalar_coefficient(tp->left))
        return VecSide{tp->left, bv->first, bv->second};
    if (auto bv = as_basis_vector(tp->left, b);
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
                IndexAssoc{l->index},
                IndexAssoc{r->index});
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
                {IndexAssoc{l->index}, IndexAssoc{r->index}, IndexAssoc{k}});
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
            out.push_back(make_identity(ctx));
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
        auto const* ci = std::get_if<CountableIndex>(&*sb.index);
        if (!ci)
            return std::nullopt;
        ids.push_back(ci->id);
    }
    return std::pair{t, std::move(ids)};
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
    Basis const& basis) -> Expr const*
{
    auto coord_invariant = [&](TensorObject const* c, int rank)
    { return make_tensor_object(ctx, c->name, {}, rank); };

    // Classify the factors: basis vectors (by summed id), coordinate carriers,
    // and the summed ids blocked by some other (foreign) factor.
    std::vector<Carrier> carriers;
    std::map<int, std::vector<std::pair<int, int>>> in_carrier; // id→[(car,slot)]
    std::map<int, std::vector<std::size_t>> in_basis; // id→[positions]
    std::set<int> const summed_set(summed.begin(), summed.end());
    std::set<int> blocked;
    for (std::size_t p = 0; p < factors.size(); ++p)
    {
        if (auto bv = as_basis_vector(factors[p], basis);
            bv && summed_set.count(bv->first.id))
        {
            in_basis[bv->first.id].push_back(p);
            continue;
        }
        if (auto cc = as_coord_component(factors[p], basis))
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
        for (int id: summed)
            if (mentions_index(ctx, factors[p], id))
                blocked.insert(id);
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

    std::set<std::size_t> drop;                 // factor positions removed
    std::map<std::size_t, Expr const*> replace; // basis position → invariant
    std::vector<Expr const*> scalars;           // scalar folds, emitted first
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
                auto m = contract_carriers(ctx, active[a], active[b], id);
                if (!m)
                {
                    ok = false;
                    break;
                }
                int const hi = std::max(a, b);
                int const lo = std::min(a, b);
                active.erase(active.begin() + hi);
                active.erase(active.begin() + lo);
                active.push_back(std::move(*m));
            }
        }

        // Realize each surviving carrier; accumulate into blob-local changes so
        // a failure leaves the whole blob untouched.
        std::set<std::size_t> ldrop;
        std::map<std::size_t, Expr const*> lreplace;
        std::vector<Expr const*> lscalars;
        std::set<int> lfolded;
        for (Carrier& c: active)
        {
            if (!ok)
                break;
            if (c.legs.empty()) // scalar invariant (dot / trace / bilinear)
            {
                lscalars.push_back(c.value);
                for (int o: c.origins)
                    ldrop.insert(static_cast<std::size_t>(o));
            }
            else if (c.legs.size() == 1) // vector leg → place at its basis vec
            {
                auto const& bp = in_basis[c.legs[0]];
                if (bp.size() != 1)
                {
                    ok = false;
                    break;
                }
                lreplace[bp[0]] = c.value;
                lfolded.insert(c.legs[0]);
                for (int o: c.origins)
                    ldrop.insert(static_cast<std::size_t>(o));
            }
            else if (c.legs.size() == 2) // tensor → place at the leftmost basis
            {
                auto const& b0 = in_basis[c.legs[0]];
                auto const& b1 = in_basis[c.legs[1]];
                if (b0.size() != 1 || b1.size() != 1)
                {
                    ok = false;
                    break;
                }
                std::size_t const p0 = b0[0];
                std::size_t const p1 = b1[0];
                // The leftmost basis vector fixes the first tensor slot: in
                // slot order → value; reversed → its transpose.
                Expr const* tens =
                    (p0 < p1) ? c.value : make_transpose(ctx, c.value);
                lreplace[std::min(p0, p1)] = tens;
                ldrop.insert(std::max(p0, p1));
                lfolded.insert(c.legs[0]);
                lfolded.insert(c.legs[1]);
                for (int o: c.origins)
                    ldrop.insert(static_cast<std::size_t>(o));
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
            replace[bp[0]] = make_identity(ctx);
            drop.insert(bp[1]);
            folded.insert(id);
        }

    if (folded.empty())
        return nullptr;

    std::vector<Expr const*> out = scalars;
    for (std::size_t p = 0; p < factors.size(); ++p)
    {
        if (drop.count(p))
            continue;
        auto it = replace.find(p);
        out.push_back(it != replace.end() ? it->second : factors[p]);
    }
    std::vector<int> rest;
    for (int id: summed)
        if (!folded.count(id))
            rest.push_back(id);
    return wrap_sums(ctx, rest, product_of(ctx, out));
}

} // namespace

auto reassemble(Context& ctx, Expr const* e, Basis const& basis) -> Expr const*
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
            if (auto* g = fold_reassembly_groups(c, summed, factors, basis))
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
                    return signed_(make_identity(c));
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
                c, coord->name, {}, static_cast<int>(s.size())));
        });
    return out == prepped ? e : out;
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
    return out == prepped ? e : out;
}

} // namespace tender
