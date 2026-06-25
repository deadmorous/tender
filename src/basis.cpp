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

// A bare coordinate component: a non-basis, non-well-known tensor with exactly
// one CountableIndex slot.  Its name + the index id are returned; the
// whole-term reassembly (below) handles higher-rank single coordinates, while
// this drives the per-vector folding that recognizes *each* coordinate vector
// individually (so a_i b_j e_j e_i reassembles to b ⊗ a without ever assembling
// a dyad as a special case — triads and beyond fall out the same way).
auto as_coord_vector(Expr const* e, Basis const& basis)
    -> std::optional<std::pair<TensorObject const*, int>>
{
    if (as_basis_vector(e, basis))
        return std::nullopt;
    auto const* t = std::get_if<TensorObject>(&e->node);
    // A coordinate component is rank 0 (a scalar, the way expand_in_basis emits
    // it); requiring that excludes a rank-1 basis vector — including a
    // *foreign* basis's vector, which this basis would not otherwise recognize.
    if (!t || (t->traits && t->traits->well_known) || t->rank != 0
        || t->slots.size() != 1 || !t->slots[0].index)
        return std::nullopt;
    auto const* ci = std::get_if<CountableIndex>(&*t->slots[0].index);
    if (!ci)
        return std::nullopt;
    return std::pair{t, ci->id};
}

// Fold the recognizable single-index groups of one basis-expanded term, each
// summed index independently:
//   vector  c_i e_i           → c           (replace the basis vector in place)
//   dot     c_i d_i           → c · d        (a scalar invariant)
//   ident.  e_i e_i           → I            (resolution of identity)
// Indices whose two occurrences do not form one of these clean shapes are left
// bound (e.g. a slot of a higher-rank coordinate, handled by the whole-term
// path).  Returns nullptr when nothing folds, so the caller can fall through.
auto fold_reassembly_groups(
    Context& ctx,
    std::vector<int> const& summed,
    std::vector<Expr const*> const& factors,
    Basis const& basis) -> Expr const*
{
    auto invariant = [&](TensorObject const* c)
    { return make_tensor_object(ctx, c->name, {}, 1); };

    std::set<std::size_t> drop; // factor positions removed outright
    std::map<std::size_t, Expr const*> replace; // basis-vec position →
                                                // invariant/I
    std::vector<Expr const*> scalars; // folded dot products, emitted first
    std::set<int> folded;

    for (int id: summed)
    {
        std::vector<std::size_t> bpos, cpos;
        bool foreign = false;
        for (std::size_t p = 0; p < factors.size(); ++p)
        {
            if (auto bv = as_basis_vector(factors[p], basis);
                bv && bv->first.id == id)
                bpos.push_back(p);
            else if (auto cv = as_coord_vector(factors[p], basis);
                     cv && cv->second == id)
                cpos.push_back(p);
            else if (mentions_index(ctx, factors[p], id))
                foreign = true; // a higher-rank coord or other carrier of id
        }
        if (foreign)
            continue;
        if (bpos.size() == 1 && cpos.size() == 1) // vector
        {
            auto const* c = std::get_if<TensorObject>(&factors[cpos[0]]->node);
            replace[bpos[0]] = invariant(c);
            drop.insert(cpos[0]);
            folded.insert(id);
        }
        else if (bpos.empty() && cpos.size() == 2) // dot
        {
            auto const* c0 = std::get_if<TensorObject>(&factors[cpos[0]]->node);
            auto const* c1 = std::get_if<TensorObject>(&factors[cpos[1]]->node);
            scalars.push_back(make_dot(ctx, invariant(c0), invariant(c1)));
            drop.insert(cpos[0]);
            drop.insert(cpos[1]);
            folded.insert(id);
        }
        else if (bpos.size() == 2 && cpos.empty()) // resolution of identity
        {
            replace[bpos[0]] = make_identity(ctx);
            drop.insert(bpos[1]);
            folded.insert(id);
        }
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

            // Peel one leading sign (a subtracted term carries it as a Negate);
            // re-apply it to whatever the body reassembles to.
            bool negated = false;
            if (auto const* n = std::get_if<Negate>(&body->node))
            {
                negated = true;
                body = n->operand;
            }
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
}

auto reassemble_completeness(Context& ctx, Expr const* e, Basis const& basis)
    -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [&](Context& c, Expr const* node) -> Expr const*
        {
            auto* f = fold_completeness(c, node, basis);
            return f ? f : node;
        });
}

} // namespace tender
