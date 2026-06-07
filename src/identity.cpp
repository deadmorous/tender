#include <tender/identity.hpp>

#include <stdexcept>
#include <string>

#include <tender/match.hpp>

namespace tender
{

// ===========================================================================
// PatternVar
// ===========================================================================

PatternVar::PatternVar(std::string symbol) : symbol_(std::move(symbol))
{
}

auto PatternVar::latex() const -> std::string
{
    return symbol_;
}

auto PatternVar::python() const -> std::string
{
    return "pattern_var('" + symbol_ + "')";
}

auto PatternVar::constrain_rank(int r) -> PatternVar*
{
    constraints_.required_rank = r;
    return this;
}

auto PatternVar::constrain_symmetric() -> PatternVar*
{
    constraints_.symmetric = true;
    return this;
}

auto PatternVar::constrain_skew_symmetric() -> PatternVar*
{
    constraints_.skew_symmetric = true;
    return this;
}

// ===========================================================================
// Identity
// ===========================================================================

Identity::Identity(std::string name, Expr* lhs, Expr* rhs) :
  name_(std::move(name)), lhs_(lhs), rhs_(rhs)
{
}

auto Identity::from_derivation(
    std::string name, std::vector<State> const& history) -> Identity
{
    if (history.size() < 2)
        throw std::invalid_argument(
            "Identity::from_derivation: history must have at least 2 states");
    return Identity{
        std::move(name), history.front().expr(), history.back().expr()};
}

// ===========================================================================
// Pattern substitution
// ===========================================================================

static auto substitute_pattern(
    ResourceList& rl, Expr* e, PatternMapping const& mapping) -> Expr*
{
    if (auto* pv = dynamic_cast<PatternVar*>(e))
    {
        auto it = mapping.find(pv);
        return it != mapping.end() ? it->second : e;
    }

    if (auto* sc = dynamic_cast<Scale*>(e))
        return make_scale(
            rl, sc->coeff(), substitute_pattern(rl, sc->expr(), mapping));

    if (auto* s = dynamic_cast<Sum*>(e))
    {
        std::vector<Expr*> terms;
        terms.reserve(s->terms().size());
        for (auto* t: s->terms())
            terms.push_back(substitute_pattern(rl, t, mapping));
        return make_sum(rl, std::move(terms));
    }

    if (auto* tp = dynamic_cast<TensorProduct*>(e))
        return make_tensor_product(
            rl,
            substitute_pattern(rl, tp->lhs(), mapping),
            substitute_pattern(rl, tp->rhs(), mapping));

    if (auto* co = dynamic_cast<Contract*>(e))
        return make_contract(
            rl,
            substitute_pattern(rl, co->lhs(), mapping),
            substitute_pattern(rl, co->rhs(), mapping));

    if (auto* dc = dynamic_cast<DoubleContract*>(e))
        return make_double_contract(
            rl,
            substitute_pattern(rl, dc->lhs(), mapping),
            substitute_pattern(rl, dc->rhs(), mapping));

    if (auto* dcr = dynamic_cast<DoubleContractReversed*>(e))
        return make_double_contract_reversed(
            rl,
            substitute_pattern(rl, dcr->lhs(), mapping),
            substitute_pattern(rl, dcr->rhs(), mapping));

    if (auto* cp = dynamic_cast<CrossProduct*>(e))
        // Bypass make_cross_product guard: pattern substitution may produce
        // nested cross products intentionally (e.g. in the LHS of BAC-CAB).
        return rl.make<CrossProduct>(
            substitute_pattern(rl, cp->lhs(), mapping),
            substitute_pattern(rl, cp->rhs(), mapping));

    if (auto* fa = dynamic_cast<FunctionApply*>(e))
        return make_function(
            rl, fa->kind(), substitute_pattern(rl, fa->arg(), mapping));

    if (auto* pw = dynamic_cast<Pow*>(e))
        return make_pow(
            rl, substitute_pattern(rl, pw->base(), mapping), pw->exponent());

    if (auto* pr = dynamic_cast<Product*>(e))
        return make_product(
            rl,
            substitute_pattern(rl, pr->lhs(), mapping),
            substitute_pattern(rl, pr->rhs(), mapping));

    // Leaf nodes (RationalConst, NamedConst, SymbolicVar, Parameter,
    // NamedTensor, IdentityTensor, LeviCivitaTensor, Trace, MaterialDeriv,
    // PolynomialExpr, unbound PatternVar) — return unchanged.
    return e;
}

// ===========================================================================
// apply_identity
// ===========================================================================

auto apply_identity(Identity const& id, PatternMapping const& mapping)
    -> DerivationStep
{
    return DerivationStep{
        "apply(" + id.name() + ")",
        [id, mapping](ResourceList& rl, Expr* e) -> Expr*
        {
            for (auto const& [pvar, actual]: mapping)
            {
                int const req = pvar->constraints().required_rank;
                if (req >= 0 && actual->rank() != req)
                    throw std::invalid_argument(
                        "apply_identity: expression bound to '" + pvar->symbol()
                        + "' has wrong rank (expected " + std::to_string(req)
                        + ", got " + std::to_string(actual->rank()) + ")");
            }
            PatternBinding initial{mapping.begin(), mapping.end()};
            if (match_pattern(id.lhs(), e, initial).empty())
                throw std::invalid_argument(
                    "apply_identity: the provided binding does not match the LHS "
                    "of '"
                    + id.name() + "' against the current expression");
            return substitute_pattern(rl, id.rhs(), mapping);
        }};
}

auto apply_rhs(ResourceList& rl, Identity const& id, PatternMapping const& mapping)
    -> Expr*
{
    return substitute_pattern(rl, id.rhs(), mapping);
}

// ===========================================================================
// Factory
// ===========================================================================

auto make_pattern_var(ResourceList& rl, std::string symbol) -> PatternVar*
{
    return rl.make<PatternVar>(std::move(symbol));
}

} // namespace tender
