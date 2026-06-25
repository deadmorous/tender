#include <tender/identity.hpp>

#include <tender/context.hpp>
#include <tender/derivation.hpp> // steps::canonicalize, steps::implicitize
#include <tender/nf_lower.hpp>   // canonicalize_nf, raise
#include <tender/nf_match.hpp>   // fire_identity_on_term

#include <utility>
#include <vector>

namespace tender
{

auto apply_identity(Context& ctx, Expr const* e, Identity const& id)
    -> Expr const*
{
    // Match on the flat normal form (vibe 000058 / C14): the canonical `Nf` of
    // an expression is `canonicalize_nf(canonicalize(·))` (the C12 round-trip).
    // The identity's LHS becomes a single pattern term whose factors are
    // matched as a *sub-product* / *sub-chain* of a target term — so an
    // identity fires even when its product sits among extra factors of a larger
    // term, the gap the old binary-tree matcher could not reach.
    auto const* target = nf::canonicalize_nf(ctx, steps::canonicalize(ctx, e));

    // Both the fired and the no-match results leave the function in one uniform
    // canonical, implicit shape (so a derivation chain stays consistent): raise
    // the resulting `Nf` and re-canonicalize (re-α-renaming any freshened RHS
    // dummies, merging like terms), then implicitize so the explicit binders
    // the normal form carries do not leak into the user's implicit notation.
    auto finish = [&](nf::Nf const* nf) -> Expr const*
    {
        return steps::implicitize(
            ctx, steps::canonicalize(ctx, nf::raise(ctx, *nf)));
    };

    auto const* lhs =
        nf::canonicalize_nf(ctx, steps::canonicalize(ctx, id.lhs));

    // Only a single-term LHS is matched as a sub-product / sub-chain; a
    // multi-term LHS (a sub-sum pattern) has no Nf matcher yet, so it never
    // fires — the target comes back unchanged (canonical).
    if (lhs->terms.size() != 1)
        return finish(target);

    auto const* rhs =
        nf::canonicalize_nf(ctx, steps::canonicalize(ctx, id.rhs));
    nf::Term const& lhs_term = lhs->terms.front();

    // Fire at the first target term where the LHS matches; the matched part is
    // replaced (the RHS spliced back where the matched run sat — ⊗ is
    // non-commutative), the rest of the term and the other terms carried
    // through.  When nothing fires, `out` reproduces `target` and `finish`
    // returns it unchanged.
    bool fired = false;
    std::vector<nf::Term> out;
    for (auto const& tterm: target->terms)
    {
        if (!fired)
            if (auto rep = nf::fire_identity_on_term(ctx, lhs_term, rhs, tterm))
            {
                fired = true;
                for (auto& t: *rep)
                    out.push_back(std::move(t));
                continue;
            }
        out.push_back(tterm);
    }

    return finish(fired ? nf::make_nf(ctx, std::move(out)) : target);
}

namespace steps
{

auto apply_identity(Identity id)
    -> std::function<Expr const*(Context&, Expr const*)>
{
    return [id = std::move(id)](Context& ctx, Expr const* e) -> Expr const*
    { return tender::apply_identity(ctx, e, id); };
}

} // namespace steps

} // namespace tender
