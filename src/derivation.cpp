#include <tender/derivation.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace tender
{

// ===========================================================================
// State
// ===========================================================================

State::State(Expr* expr, std::string label) :
  expr_(expr), label_(std::move(label))
{
}

auto State::latex() const -> std::string
{
    return expr_->latex();
}

// ===========================================================================
// DerivationStep
// ===========================================================================

DerivationStep::DerivationStep(std::string name, Fn fn) :
  name_(std::move(name)), fn_(std::move(fn))
{
}

auto DerivationStep::apply(ResourceList& rl, State const& s) const -> State
{
    return State{fn_(rl, s.expr()), name_};
}

// ===========================================================================
// Derivation
// ===========================================================================

Derivation::Derivation(std::vector<DerivationStep> steps) :
  steps_(std::move(steps))
{
}

auto Derivation::apply(ResourceList& rl, State const& initial) const
    -> std::vector<State>
{
    std::vector<State> history;
    history.reserve(steps_.size() + 1);
    history.push_back(initial);
    for (auto const& step: steps_)
        history.push_back(step.apply(rl, history.back()));
    return history;
}

auto Derivation::operator+(Derivation const& rhs) const -> Derivation
{
    auto combined = steps_;
    combined.insert(combined.end(), rhs.steps_.begin(), rhs.steps_.end());
    return Derivation{std::move(combined)};
}

// ===========================================================================
// show
// ===========================================================================

auto show(std::vector<State> const& history) -> std::string
{
    std::string result;
    for (auto const& s: history)
    {
        if (!result.empty())
            result += '\n';
        std::string const tag = s.label().empty() ? "initial" : s.label();
        result += "[" + tag + "]  " + s.latex();
    }
    return result;
}

auto show_final(std::vector<State> const& history) -> std::string
{
    if (history.empty())
        return "";
    return history.back().latex();
}

// ===========================================================================
// expand_poly_step helpers
// ===========================================================================

static auto expand_poly_impl(ResourceList& rl, Expr* e) -> Expr*
{
    if (auto* pe = dynamic_cast<PolynomialExpr*>(e))
        return pe->expand(rl);

    if (auto* sc = dynamic_cast<Scale*>(e))
        return make_scale(rl, sc->coeff(), expand_poly_impl(rl, sc->expr()));

    if (auto* s = dynamic_cast<Sum*>(e))
    {
        std::vector<Expr*> terms;
        terms.reserve(s->terms().size());
        for (auto* t: s->terms())
            terms.push_back(expand_poly_impl(rl, t));
        return make_sum(rl, std::move(terms));
    }

    if (auto* tp = dynamic_cast<TensorProduct*>(e))
        return make_tensor_product(
            rl,
            expand_poly_impl(rl, tp->lhs()),
            expand_poly_impl(rl, tp->rhs()));

    if (auto* co = dynamic_cast<Contract*>(e))
        return make_contract(
            rl,
            expand_poly_impl(rl, co->lhs()),
            expand_poly_impl(rl, co->rhs()));

    if (auto* pr = dynamic_cast<Product*>(e))
        return make_product(
            rl,
            expand_poly_impl(rl, pr->lhs()),
            expand_poly_impl(rl, pr->rhs()));

    return e;
}

// ===========================================================================
// Built-in step factories
// ===========================================================================

auto diff_step(Parameter const* param) -> DerivationStep
{
    return DerivationStep{
        "diff(" + param->symbol() + ")",
        [param](ResourceList& rl, Expr* e) -> Expr*
        { return deriv(rl, param, e); }};
}

auto expand_poly_step() -> DerivationStep
{
    return DerivationStep{"expand_poly", [](ResourceList& rl, Expr* e) -> Expr* {
                              return expand_poly_impl(rl, e);
                          }};
}

auto named_step(std::string name, DerivationStep::Fn fn) -> DerivationStep
{
    return DerivationStep{std::move(name), std::move(fn)};
}

} // namespace tender
