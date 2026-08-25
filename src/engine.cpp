#include <tender/engine.hpp>

#include <tender/basis.hpp>        // expand_in_basis, simplify_basis_*
#include <tender/coord_system.hpp> // wcs
#include <tender/derivation.hpp>   // steps::canonicalize, steps::implicitize
#include <tender/nf_lower.hpp>     // raise
#include <tender/rewrite.hpp>      // children (residue walk)

#include <stdexcept>
#include <string>

namespace tender::engine
{

using nf::EClassId;

namespace
{

// Reduce an invariant expression to concrete World-Cartesian components:
// expand every abstract tensor on the frame, turn frame crosses into ε and
// frame dots into δ, then evaluate both symbols at concrete index values and
// fold the arithmetic.  What survives is a polynomial in the component
// symbols, which `canonicalize` decides.
auto to_components(Context& ctx, Basis const& frame, Expr const* e)
    -> Expr const*
{
    e = expand_in_basis(ctx, e, frame, Variance::Covariant);
    // A nested cross (`a × (b × c)`) needs one pass per level; the pass is a
    // no-op once none are left, so iterating to a fixed point is safe.
    for (int i = 0; i < 4; ++i)
    {
        auto const* next = simplify_basis_cross(ctx, e, frame);
        if (next == e)
            break;
        e = next;
    }
    e = simplify_basis_dot(ctx, e, frame);
    e = steps::canonicalize(ctx, e);
    e = steps::unroll_sums(ctx, e);
    e = steps::eval_eps_concrete(ctx, e);
    e = steps::eval_delta_concrete(ctx, e);
    e = steps::fold_arithmetic(ctx, e);
    return steps::canonicalize(ctx, e);
}

// Does `e` still hold anything the component reduction could not evaluate —
// a leftover ε / δ symbol, an unexpanded summation binder, or a differential
// operator?  Such a residue means the reduction did not decide, so a
// difference between two sides is not evidence of inequality.
auto has_residue(Expr const* e) -> bool
{
    if (auto const* t = std::get_if<TensorObject>(&e->node);
        t && t->traits && t->traits->well_known)
        switch (*t->traits->well_known)
        {
            case WellKnownKind::LeviCivita:
            case WellKnownKind::Delta:
            case WellKnownKind::Metric: return true;
            case WellKnownKind::Identity: break; // I is fully expandable
        }
    if (std::holds_alternative<ExplicitSum>(e->node)
        || std::holds_alternative<NoSum>(e->node)
        || std::holds_alternative<Nabla>(e->node)
        || std::holds_alternative<Deriv>(e->node))
        return true;
    for (auto const* c: children(e))
        if (has_residue(c))
            return true;
    return false;
}

} // namespace

auto decide_by_components(Context& ctx, Expr const* lhs, Expr const* rhs)
    -> ComponentVerdict
{
    try
    {
        auto const frame = wcs(ctx);
        auto const* l = to_components(ctx, frame, lhs);
        auto const* r = to_components(ctx, frame, rhs);
        if (algebraic_eq(ctx, l, r))
            return ComponentVerdict::Equal;
        // Only call it a refutation when both sides actually reduced: a
        // leftover binder or symbol means "not decided", not "different".
        if (!has_residue(l) && !has_residue(r))
            return ComponentVerdict::Different;
        return ComponentVerdict::Undecided;
    }
    catch (std::exception const&)
    {
        // The reduction is a best-effort decision procedure; anything it
        // cannot process (a differential operator, a chart-bound quantity, an
        // unsupported shape) is simply undecided.
        return ComponentVerdict::Undecided;
    }
}

auto prove_equal(
    Context& ctx,
    Expr const* lhs,
    Expr const* rhs,
    std::vector<Identity> const& rules,
    SaturateBudget budget) -> ProofResult
{
    // Canonicalization can reject a shape it does not yet handle.  That is a
    // fact about tender, not about the claim, so report it as such instead of
    // letting an internal message escape as an exception (vibe 000098).
    nf::NfEGraph g{ctx};
    EClassId l = 0;
    EClassId r = 0;
    try
    {
        l = g.add(lhs);
        r = g.add(rhs);
    }
    catch (std::exception const& err)
    {
        ProofResult bad;
        bad.status = ProofStatus::Unsupported;
        bad.detail =
            std::string{
                "tender cannot yet put this expression in "
                "canonical form: "}
            + err.what();
        return bad;
    }

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
    {
        out.status = ProofStatus::Proved;
        return out;
    }
    if (nf::is_budget_stop(out.report.outcome))
    {
        // A budget trip concludes nothing, so do not spend the component pass
        // on it: the rules might still have proved it given room.  Which cap
        // fired is in `report.outcome` — a caller that hit the wall clock may
        // simply retry with more room, while a node-count trip usually means
        // the rule set is explosive.
        out.status = ProofStatus::Budget;
        return out;
    }

    // The rules are exhausted.  Ask the independent decision procedure
    // whether the claim is actually false, so the caller gets a verdict
    // instead of "I could not do it" (vibe 000097).
    switch (decide_by_components(ctx, lhs, rhs))
    {
        case ComponentVerdict::Different:
            out.status = ProofStatus::Refuted;
            break;
        case ComponentVerdict::Equal:
            // True, but unreachable with these rules — the rule set is what
            // is incomplete, not the mathematics.
            out.status = ProofStatus::Exhausted;
            out.components_agree = true;
            break;
        case ComponentVerdict::Undecided:
            out.status = ProofStatus::Exhausted;
            break;
    }
    return out;
}

auto simplify(
    Context& ctx,
    Expr const* e,
    std::vector<Identity> const& rules,
    SaturateBudget budget,
    CostModel const& cost) -> SimplifyResult
{
    nf::NfEGraph g{ctx};
    SimplifyResult out;
    EClassId root = 0;
    try
    {
        root = g.add(e);
    }
    catch (std::exception const& err)
    {
        // Nothing can be simplified, but the caller still gets their
        // expression back rather than an exception from canon's internals.
        out.expr = e;
        out.unsupported =
            std::string{
                "tender cannot yet put this expression "
                "in canonical form: "}
            + err.what();
        return out;
    }

    out.report = g.saturate(rules, budget);

    // Extract the cheapest member of the root's class and raise it back to the
    // user-facing implicit form — the same final shape `apply_identity`
    // returns, so an engine result drops into a derivation chain unchanged.
    auto const* best = g.extract(g.find(root), cost);
    out.expr =
        steps::implicitize(ctx, steps::canonicalize(ctx, nf::raise(ctx, *best)));
    return out;
}

} // namespace tender::engine
