#include <tender/engine.hpp>

#include <tender/derivation.hpp> // steps::canonicalize, steps::implicitize
#include <tender/nf_lower.hpp>   // raise

namespace tender::engine
{

auto prove_equal(
    Context& ctx,
    Expr const* lhs,
    Expr const* rhs,
    std::vector<Identity> const& rules,
    SaturateBudget budget) -> ProofResult
{
    nf::NfEGraph g{ctx};
    auto const l = g.add(lhs);
    auto const r = g.add(rhs);

    // Both sides live in one graph, so a rule rewriting either side toward the
    // other closes the gap — the sides meet in the middle rather than one
    // being driven all the way into the other.
    auto joined = [&] { return g.find(l) == g.find(r); };

    ProofResult out;
    if (joined())
    {
        // Already equal under canonicalization alone (theory T0) — no rule
        // firing needed.  Reported as a zero-pass proof.
        out.status = ProofStatus::Proved;
        out.report.fired.assign(rules.size(), 0);
        out.report.nodes = g.node_count();
        return out;
    }

    out.report = g.saturate(rules, budget, joined);
    if (joined())
        out.status = ProofStatus::Proved;
    else if (out.report.outcome == SaturateOutcome::Saturated)
        out.status = ProofStatus::Exhausted;
    else
        out.status = ProofStatus::Budget;
    return out;
}

auto simplify(
    Context& ctx,
    Expr const* e,
    std::vector<Identity> const& rules,
    SaturateBudget budget) -> SimplifyResult
{
    nf::NfEGraph g{ctx};
    auto const root = g.add(e);

    SimplifyResult out;
    out.report = g.saturate(rules, budget);

    // Extract the cheapest member of the root's class and raise it back to the
    // user-facing implicit form — the same final shape `apply_identity`
    // returns, so an engine result drops into a derivation chain unchanged.
    auto const* best = g.extract(g.find(root));
    out.expr =
        steps::implicitize(ctx, steps::canonicalize(ctx, nf::raise(ctx, *best)));
    return out;
}

} // namespace tender::engine
