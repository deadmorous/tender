#include <tender/match.hpp>

#include <stdexcept>
#include <vector>

#include <tender/expr.hpp>
#include <tender/identity.hpp>
#include <tender/integral.hpp>

namespace tender
{

// ===========================================================================
// Helpers
// ===========================================================================

// Merge two binding sets: for each binding in `a`, extend it with each
// binding in `b`, keeping only consistent combinations (no conflicting
// assignments for the same PatternVar).
static auto merge_bindings(
    std::vector<PatternBinding> const& a,
    std::vector<PatternBinding> const& b) -> std::vector<PatternBinding>
{
    std::vector<PatternBinding> result;
    for (auto const& ba: a)
    {
        for (auto const& bb: b)
        {
            PatternBinding merged = ba;
            bool ok = true;
            for (auto const& [pv, ex]: bb)
            {
                auto it = merged.find(pv);
                if (it != merged.end() && it->second != ex)
                {
                    ok = false;
                    break;
                }
                merged[pv] = ex;
            }
            if (ok)
                result.push_back(std::move(merged));
        }
    }
    return result;
}

// Check that `expr` satisfies the constraints of `pv`.
static auto satisfies_constraints(PatternVar const* pv, Expr* expr) -> bool
{
    auto const& c = pv->constraints();
    if (c.required_rank >= 0 && expr->rank() != c.required_rank)
        return false;
    // Symmetry constraints: only NamedTensor supports these flags for now.
    if (c.symmetric)
    {
        auto* nt = dynamic_cast<NamedTensor*>(expr);
        if (!nt || !nt->is_symmetric())
            return false;
    }
    if (c.skew_symmetric)
    {
        auto* nt = dynamic_cast<NamedTensor*>(expr);
        if (!nt || !nt->is_skew_symmetric())
            return false;
    }
    return true;
}

// ===========================================================================
// match_pattern — recursive pattern matching
// ===========================================================================

// Forward declaration for mutual recursion.
static auto match_impl(Expr* pattern, Expr* expr, PatternBinding const& bindings)
    -> std::vector<PatternBinding>;

// Attempt to match two expressions that have the same structural type.
// Returns merged binding lists.
template <typename BinaryNode>
static auto match_binary(
    BinaryNode* p,
    Expr* expr_raw,
    PatternBinding const& bindings) -> std::vector<PatternBinding>
{
    auto* e = dynamic_cast<BinaryNode*>(expr_raw);
    if (!e)
        return {};
    auto lhs_bindings = match_impl(p->lhs(), e->lhs(), bindings);
    if (lhs_bindings.empty())
        return {};
    std::vector<PatternBinding> result;
    for (auto const& lb: lhs_bindings)
    {
        auto rhs_bindings = match_impl(p->rhs(), e->rhs(), lb);
        for (auto& rb: rhs_bindings)
            result.push_back(std::move(rb));
    }
    return result;
}

template <typename UnaryNode>
static auto match_unary(
    UnaryNode* p,
    Expr* expr_raw,
    PatternBinding const& bindings) -> std::vector<PatternBinding>
{
    auto* e = dynamic_cast<UnaryNode*>(expr_raw);
    if (!e)
        return {};
    return match_impl(p->arg(), e->arg(), bindings);
}

static auto match_impl(Expr* pattern, Expr* expr, PatternBinding const& bindings)
    -> std::vector<PatternBinding>
{
    // --- PatternVar ---
    if (auto* pv = dynamic_cast<PatternVar*>(pattern))
    {
        auto it = bindings.find(pv);
        if (it != bindings.end())
            return (it->second == expr) ?
                       std::vector<PatternBinding>{bindings} :
                       std::vector<PatternBinding>{};
        if (!satisfies_constraints(pv, expr))
            return {};
        PatternBinding extended = bindings;
        extended[pv] = expr;
        return {std::move(extended)};
    }

    // --- Leaf identity checks (pointer or value equality) ---
    if (dynamic_cast<IdentityTensor*>(pattern))
        return dynamic_cast<IdentityTensor*>(expr) ?
                   std::vector<PatternBinding>{bindings} :
                   std::vector<PatternBinding>{};
    if (dynamic_cast<LeviCivitaTensor*>(pattern))
        return dynamic_cast<LeviCivitaTensor*>(expr) ?
                   std::vector<PatternBinding>{bindings} :
                   std::vector<PatternBinding>{};

    if (auto* rc = dynamic_cast<RationalConst*>(pattern))
    {
        auto* re = dynamic_cast<RationalConst*>(expr);
        return (re && re->value() == rc->value()) ?
                   std::vector<PatternBinding>{bindings} :
                   std::vector<PatternBinding>{};
    }

    if (auto* nc = dynamic_cast<NamedConst*>(pattern))
    {
        auto* ne = dynamic_cast<NamedConst*>(expr);
        return (ne && ne->symbol() == nc->symbol()) ?
                   std::vector<PatternBinding>{bindings} :
                   std::vector<PatternBinding>{};
    }

    if (auto* pv2 = dynamic_cast<SymbolicVar*>(pattern))
    {
        // Concrete SymbolicVar / Parameter: must be the same object.
        return (expr == pattern) ? std::vector<PatternBinding>{bindings} :
                                   std::vector<PatternBinding>{};
    }

    if (auto* nt = dynamic_cast<NamedTensor*>(pattern))
        return (expr == pattern) ? std::vector<PatternBinding>{bindings} :
                                   std::vector<PatternBinding>{};

    // --- Sum ---
    if (auto* ps = dynamic_cast<Sum*>(pattern))
    {
        auto* es = dynamic_cast<Sum*>(expr);
        if (!es || es->terms().size() != ps->terms().size())
            return {};
        std::vector<PatternBinding> current{bindings};
        for (std::size_t i = 0; i < ps->terms().size(); ++i)
        {
            std::vector<PatternBinding> next;
            for (auto const& b: current)
            {
                auto sub = match_impl(ps->terms()[i], es->terms()[i], b);
                for (auto& sb: sub)
                    next.push_back(std::move(sb));
            }
            current = std::move(next);
            if (current.empty())
                return {};
        }
        return current;
    }

    // --- Scale ---
    if (auto* psc = dynamic_cast<Scale*>(pattern))
    {
        auto* esc = dynamic_cast<Scale*>(expr);
        if (!esc || esc->coeff() != psc->coeff())
            return {};
        return match_impl(psc->expr(), esc->expr(), bindings);
    }

    // --- TensorProduct ---
    if (auto* p = dynamic_cast<TensorProduct*>(pattern))
        return match_binary(p, expr, bindings);

    // --- Contract / DoubleContract / DoubleContractReversed / CrossProduct ---
    if (auto* p = dynamic_cast<Contract*>(pattern))
        return match_binary(p, expr, bindings);
    if (auto* p = dynamic_cast<DoubleContract*>(pattern))
        return match_binary(p, expr, bindings);
    if (auto* p = dynamic_cast<DoubleContractReversed*>(pattern))
        return match_binary(p, expr, bindings);
    if (auto* p = dynamic_cast<CrossProduct*>(pattern))
        return match_binary(p, expr, bindings);

    // --- Trace / Gradient / Divergence / Rotor ---
    if (auto* p = dynamic_cast<Trace*>(pattern))
        return match_unary(p, expr, bindings);
    if (auto* p = dynamic_cast<Gradient*>(pattern))
        return match_unary(p, expr, bindings);
    if (auto* p = dynamic_cast<Divergence*>(pattern))
        return match_unary(p, expr, bindings);
    if (auto* p = dynamic_cast<Rotor*>(pattern))
        return match_unary(p, expr, bindings);

    // --- FunctionApply ---
    if (auto* pf = dynamic_cast<FunctionApply*>(pattern))
    {
        auto* ef = dynamic_cast<FunctionApply*>(expr);
        if (!ef || ef->kind() != pf->kind())
            return {};
        return match_impl(pf->arg(), ef->arg(), bindings);
    }

    // --- Pow (exponent is Rational, not Expr*) ---
    if (auto* pp = dynamic_cast<Pow*>(pattern))
    {
        auto* ep = dynamic_cast<Pow*>(expr);
        if (!ep || ep->exponent() != pp->exponent())
            return {};
        return match_impl(pp->base(), ep->base(), bindings);
    }

    // --- Integral ---
    if (auto* pi = dynamic_cast<Integral*>(pattern))
    {
        auto* ei = dynamic_cast<Integral*>(expr);
        if (!ei || ei->domain() != pi->domain())
            return {};
        return match_impl(pi->integrand(), ei->integrand(), bindings);
    }

    return {}; // no match for unhandled types
}

auto match_pattern(Expr* pattern, Expr* expr, PatternBinding const& bindings)
    -> std::vector<PatternBinding>
{
    return match_impl(pattern, expr, bindings);
}

// ===========================================================================
// find_matches — tree search
// ===========================================================================

static auto find_matches_impl(
    Expr* pattern,
    Expr* root,
    int& budget,
    std::vector<PatternBinding>& results) -> void
{
    if (--budget < 0)
        throw std::runtime_error("find_matches: search budget exceeded");

    // Try to match at this node.
    auto bindings = match_impl(pattern, root, {});
    for (auto& b: bindings)
        results.push_back(std::move(b));

    // Recurse into children.
    if (auto* s = dynamic_cast<Sum*>(root))
        for (auto* t: s->terms())
            find_matches_impl(pattern, t, budget, results);
    else if (auto* sc = dynamic_cast<Scale*>(root))
        find_matches_impl(pattern, sc->expr(), budget, results);
    else if (auto* tp = dynamic_cast<TensorProduct*>(root))
    {
        find_matches_impl(pattern, tp->lhs(), budget, results);
        find_matches_impl(pattern, tp->rhs(), budget, results);
    }
    else if (auto* c = dynamic_cast<Contract*>(root))
    {
        find_matches_impl(pattern, c->lhs(), budget, results);
        find_matches_impl(pattern, c->rhs(), budget, results);
    }
    else if (auto* dc = dynamic_cast<DoubleContract*>(root))
    {
        find_matches_impl(pattern, dc->lhs(), budget, results);
        find_matches_impl(pattern, dc->rhs(), budget, results);
    }
    else if (auto* dcr = dynamic_cast<DoubleContractReversed*>(root))
    {
        find_matches_impl(pattern, dcr->lhs(), budget, results);
        find_matches_impl(pattern, dcr->rhs(), budget, results);
    }
    else if (auto* cp = dynamic_cast<CrossProduct*>(root))
    {
        find_matches_impl(pattern, cp->lhs(), budget, results);
        find_matches_impl(pattern, cp->rhs(), budget, results);
    }
    else if (auto* tr = dynamic_cast<Trace*>(root))
        find_matches_impl(pattern, tr->arg(), budget, results);
    else if (auto* g = dynamic_cast<Gradient*>(root))
        find_matches_impl(pattern, g->arg(), budget, results);
    else if (auto* d = dynamic_cast<Divergence*>(root))
        find_matches_impl(pattern, d->arg(), budget, results);
    else if (auto* r = dynamic_cast<Rotor*>(root))
        find_matches_impl(pattern, r->arg(), budget, results);
    else if (auto* fa = dynamic_cast<FunctionApply*>(root))
        find_matches_impl(pattern, fa->arg(), budget, results);
    else if (auto* pw = dynamic_cast<Pow*>(root))
        find_matches_impl(pattern, pw->base(), budget, results);
    else if (auto* integ = dynamic_cast<Integral*>(root))
        find_matches_impl(pattern, integ->integrand(), budget, results);
}

auto find_matches(Identity const& id, Expr* expr, int max_nodes)
    -> std::vector<PatternBinding>
{
    std::vector<PatternBinding> results;
    int budget = max_nodes;
    find_matches_impl(id.lhs(), expr, budget, results);
    return results;
}

// ===========================================================================
// apply_identity_auto
// ===========================================================================

auto apply_identity_auto(Identity const& id, Expr* expr, int max_nodes)
    -> DerivationStep
{
    auto matches = find_matches(id, expr, max_nodes);
    if (matches.empty())
        throw std::invalid_argument(
            "apply_identity_auto: no match for identity '" + id.name()
            + "' in expression");
    return apply_identity(
        id, PatternMapping{matches[0].begin(), matches[0].end()});
}

} // namespace tender
