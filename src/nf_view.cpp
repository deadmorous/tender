#include <tender/nf_view.hpp>

#include <tender/summation.hpp>

#include <stdexcept>
#include <utility>

namespace tender::view
{

namespace
{

// Flatten a ⊗-chain (order preserved) — local: the scaled-additive peel needs
// to see a product's factor list.
void flatten_product(Expr const* e, std::vector<Expr const*>& out)
{
    if (auto const* p = std::get_if<TensorProduct>(&e->node))
    {
        flatten_product(p->left, out);
        flatten_product(p->right, out);
    }
    else
        out.push_back(e);
}

// Left-fold factors back into a ⊗-chain (precondition: non-empty).
auto refold_product(Context& ctx, std::vector<Expr const*> const& factors)
    -> Expr const*
{
    Expr const* p = nullptr;
    for (auto const* f: factors)
        p = p ? make_tensor_product(ctx, p, f) : f;
    return p;
}

// A scalar-weighted additive operand `s · (A ± B)`: its scalar factors and
// the single additive core, or nullopt when the shape does not match.
auto peel_scaled_additive(Expr const* e)
    -> std::optional<std::pair<std::vector<Expr const*>, Expr const*>>
{
    std::vector<Expr const*> flat;
    flatten_product(e, flat);
    std::vector<Expr const*> scalars, rest;
    for (auto const* f: flat)
        (infer_rank(f) == std::optional<int>{0} ? scalars : rest).push_back(f);
    if (scalars.empty() || rest.size() != 1)
        return std::nullopt;
    auto const& core = rest[0]->node;
    if (!std::holds_alternative<Sum>(core)
        && !std::holds_alternative<Difference>(core)
        && !std::holds_alternative<Negate>(core))
        return std::nullopt;
    return std::pair{std::move(scalars), rest[0]};
}

} // namespace

auto signed_addends(Expr const* e) -> std::vector<nf::SignedExpr>
{
    return nf::additive_flatten(e);
}

auto sum_of(Context& ctx, std::vector<nf::SignedExpr> const& addends)
    -> Expr const*
{
    if (addends.empty())
        return make_scalar(ctx, Rational{0});
    Expr const* acc = nullptr;
    for (auto const& [sign, body]: addends)
    {
        if (!acc)
            acc = sign < 0 ? make_negate(ctx, body) : body;
        else
            acc = sign < 0 ? make_difference(ctx, acc, body) :
                             make_sum(ctx, acc, body);
    }
    return acc;
}

auto map_additive_leaves(
    Context& ctx,
    Expr const* e,
    std::function<Expr const*(Expr const*)> const& leaf,
    AdditiveOptions opts) -> Expr const*
{
    std::function<Expr const*(Expr const*)> go =
        [&](Expr const* x) -> Expr const*
    {
        if (auto const* s = std::get_if<Sum>(&x->node))
        {
            auto const* l = go(s->left);
            auto const* r = go(s->right);
            return l == s->left && r == s->right ? x : make_sum(ctx, l, r);
        }
        if (auto const* d = std::get_if<Difference>(&x->node))
        {
            auto const* l = go(d->left);
            auto const* r = go(d->right);
            return l == d->left && r == d->right ? x :
                                                   make_difference(ctx, l, r);
        }
        if (auto const* n = std::get_if<Negate>(&x->node))
        {
            auto const* o = go(n->operand);
            return o == n->operand ? x : make_negate(ctx, o);
        }
        if (opts.descend_scalar_div)
        {
            if (auto const* d = std::get_if<ScalarDiv>(&x->node))
            {
                auto const* l = go(d->left);
                return l == d->left ? x : make_scalar_div(ctx, l, d->right);
            }
        }
        return leaf(x);
    };
    return go(e);
}

auto distribute_bilinear(
    Context& ctx,
    Expr const* l,
    Expr const* r,
    std::function<Expr const*(Expr const*, Expr const*)> const& core,
    BilinearOptions opts) -> Expr const*
{
    auto recur = [&](Expr const* nl, Expr const* nr)
    { return distribute_bilinear(ctx, nl, nr, core, opts); };

    // One side's peels; called for the left side first (the normative order).
    auto peel_side = [&](Expr const* self,
                         Expr const* other,
                         bool self_is_left) -> Expr const*
    {
        auto pair = [&](Expr const* a)
        { return self_is_left ? recur(a, other) : recur(other, a); };
        if (auto const* s = std::get_if<Sum>(&self->node))
            return make_sum(ctx, pair(s->left), pair(s->right));
        if (auto const* d = std::get_if<Difference>(&self->node))
            return make_difference(ctx, pair(d->left), pair(d->right));
        if (opts.negate)
            if (auto const* n = std::get_if<Negate>(&self->node))
                return make_negate(ctx, pair(n->operand));
        if (opts.binders)
            if (auto const* es = std::get_if<ExplicitSum>(&self->node);
                es && !es->bound)
            {
                // Pull the binder out, α-renaming to a fresh id so nothing in
                // the other operand is captured.
                CountableIndex const fresh{ctx.alloc_index_id()};
                auto const* body =
                    substitute_index_id(ctx, es->body, es->index.id, fresh.id);
                return make_explicit_sum(ctx, fresh, pair(body));
            }
        if (opts.scalar_div)
            if (auto const* d = std::get_if<ScalarDiv>(&self->node))
                return make_scalar_div(ctx, pair(d->left), d->right);
        if (opts.scaled_additive)
            if (auto p = peel_scaled_additive(self))
            {
                auto factors = p->first;
                factors.push_back(pair(p->second));
                return refold_product(ctx, factors);
            }
        return nullptr; // no peel applies on this side
    };

    if (auto const* out = peel_side(l, r, /*self_is_left=*/true))
        return out;
    if (auto const* out = peel_side(r, l, /*self_is_left=*/false))
        return out;
    return core(l, r);
}

auto map_nf_terms(
    Context& ctx,
    Expr const* e,
    std::function<void(std::vector<nf::Term>&)> const& transform) -> Expr const*
{
    nf::Nf const* const original = nf::canonicalize_nf(ctx, e);
    std::vector<nf::Term> terms = original->terms;
    transform(terms);
    nf::Nf const* const edited = nf::make_nf(ctx, std::move(terms));
    if (nf::equal(edited, original))
        return e; // nothing changed — keep the input (no-op pointer contract)
    // The transform may have broken sorting / like-term collection; a
    // re-canonicalization restores the invariants before raising.
    return nf::raise(ctx, *nf::canonicalize_nf(ctx, nf::raise(ctx, *edited)));
}

auto fixpoint(
    Context& ctx,
    Expr const* e,
    std::function<Expr const*(Context&, Expr const*)> const& step,
    int max_iterations) -> Expr const*
{
    Expr const* cur = e;
    for (int i = 0; i < max_iterations; ++i)
    {
        Expr const* const next = step(ctx, cur);
        if (next == cur)
            return cur;
        cur = next;
    }
    throw std::runtime_error(
        "view::fixpoint: step did not converge (does it obey the no-op "
        "pointer contract?)");
}

} // namespace tender::view
