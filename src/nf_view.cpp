#include <tender/nf_view.hpp>

#include <stdexcept>

namespace tender::view
{

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
