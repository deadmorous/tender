#include <tender/integral.hpp>

#include <map>
#include <stdexcept>
#include <vector>

namespace tender
{

// ===========================================================================
// Domain types
// ===========================================================================

SurfaceDomain::SurfaceDomain(std::string name, Expr* normal) :
  name_(std::move(name)), normal_(normal)
{
}

VolumeDomain::VolumeDomain(
    std::string name, Expr* outward_normal, SurfaceDomain* bdy) :
  name_(std::move(name)), normal_(outward_normal), boundary_(bdy)
{
}

auto make_surface_domain(ResourceList& rl, std::string name, Expr* normal)
    -> SurfaceDomain*
{
    return rl.make<SurfaceDomain>(std::move(name), normal);
}

auto make_volume_domain(
    ResourceList& rl, std::string name, Expr* outward_normal) -> VolumeDomain*
{
    auto* bdy = rl.make<SurfaceDomain>("\\partial " + name, outward_normal);
    return rl.make<VolumeDomain>(std::move(name), outward_normal, bdy);
}

// ===========================================================================
// Gradient
// ===========================================================================

Gradient::Gradient(Expr* arg) : arg_(arg)
{
}

auto Gradient::latex(IndexNameMap const& map) const -> std::string
{
    return "\\nabla " + arg_->latex(map);
}

auto Gradient::python() const -> std::string
{
    return "gradient(" + arg_->python() + ")";
}

auto make_gradient(ResourceList& rl, Expr* arg) -> Expr*
{
    return rl.make<Gradient>(arg);
}

// ===========================================================================
// Divergence
// ===========================================================================

Divergence::Divergence(Expr* arg) : arg_(arg)
{
    if (arg->rank() < 1)
        throw std::invalid_argument(
            "Divergence: argument must have rank >= 1, got rank 0");
}

auto Divergence::latex(IndexNameMap const& map) const -> std::string
{
    return "\\nabla \\cdot " + arg_->latex(map);
}

auto Divergence::python() const -> std::string
{
    return "divergence(" + arg_->python() + ")";
}

auto make_divergence(ResourceList& rl, Expr* arg) -> Expr*
{
    return rl.make<Divergence>(arg);
}

// ===========================================================================
// Rotor
// ===========================================================================

Rotor::Rotor(Expr* arg) : arg_(arg)
{
    if (arg->rank() < 1)
        throw std::invalid_argument(
            "Rotor: argument must have rank >= 1, got rank 0");
}

auto Rotor::latex(IndexNameMap const& map) const -> std::string
{
    return "\\nabla \\times " + arg_->latex(map);
}

auto Rotor::python() const -> std::string
{
    return "rotor(" + arg_->python() + ")";
}

auto make_rotor(ResourceList& rl, Expr* arg) -> Expr*
{
    return rl.make<Rotor>(arg);
}

// ===========================================================================
// Integral
// ===========================================================================

Integral::Integral(Domain* domain, Expr* integrand) :
  domain_(domain), integrand_(integrand)
{
}

auto Integral::latex(IndexNameMap const& map) const -> std::string
{
    return "\\int_{" + domain_->name() + "} " + integrand_->latex(map) + " \\, "
           + domain_->measure_latex();
}

auto Integral::python() const -> std::string
{
    return "integral('" + domain_->name() + "', " + integrand_->python() + ")";
}

auto make_integral(ResourceList& rl, Domain* domain, Expr* integrand) -> Expr*
{
    return rl.make<Integral>(domain, integrand);
}

// ===========================================================================
// apply_integration_by_parts_step
//
// Matches Integral(V, DoubleContract(A, Gradient(B))) and rewrites to:
//   Sum(Integral(∂V, Contract(Contract(A, n), B)),
//       Scale(-1, Integral(V, Contract(Divergence(A), B))))
// ===========================================================================

static auto apply_ibp_impl(ResourceList& rl, Expr* e, VolumeDomain* domain)
    -> Expr*;

static auto apply_ibp_to_terms(
    ResourceList& rl,
    std::vector<Expr*> const& terms,
    VolumeDomain* domain) -> std::vector<Expr*>
{
    std::vector<Expr*> result;
    result.reserve(terms.size());
    for (auto* t: terms)
        result.push_back(apply_ibp_impl(rl, t, domain));
    return result;
}

static auto apply_ibp_impl(ResourceList& rl, Expr* e, VolumeDomain* domain)
    -> Expr*
{
    if (auto* s = dynamic_cast<Sum*>(e))
    {
        auto terms = apply_ibp_to_terms(rl, s->terms(), domain);
        return make_sum(rl, std::move(terms));
    }

    if (auto* sc = dynamic_cast<Scale*>(e))
    {
        auto* inner = apply_ibp_impl(rl, sc->expr(), domain);
        return make_scale(rl, sc->coeff(), inner);
    }

    if (auto* integ = dynamic_cast<Integral*>(e))
    {
        if (integ->domain() != domain)
            return e;
        auto* dc = dynamic_cast<DoubleContract*>(integ->integrand());
        if (!dc)
            return e;
        auto* grad = dynamic_cast<Gradient*>(dc->rhs());
        if (!grad)
            return e;

        // ∫_V A:(∇B) dV = ∫_∂V (A·n)·B dS − ∫_V (∇·A)·B dV
        Expr* A = dc->lhs();
        Expr* B = grad->arg();
        Expr* n = domain->outward_normal();
        SurfaceDomain* bdy = domain->surface_boundary();

        auto* An = make_contract(rl, A, n);
        auto* An_dot_B = make_contract(rl, An, B);
        auto* divA = make_divergence(rl, A);
        auto* divA_dot_B = make_contract(rl, divA, B);

        return make_sum(
            rl,
            {make_integral(rl, bdy, An_dot_B),
             make_scale(
                 rl, Rational{-1}, make_integral(rl, domain, divA_dot_B))});
    }

    return e;
}

auto apply_integration_by_parts_step(VolumeDomain* domain) -> DerivationStep
{
    return DerivationStep{
        "ibp(" + domain->name() + ")",
        [domain](ResourceList& rl, Expr* e) -> Expr*
        { return apply_ibp_impl(rl, e, domain); }};
}

// ===========================================================================
// apply_divergence_theorem_step
//
// Matches Integral(V, Divergence(A)) and rewrites to:
//   Integral(∂V, Contract(A, n))
// ===========================================================================

static auto apply_div_thm_impl(ResourceList& rl, Expr* e, VolumeDomain* domain)
    -> Expr*
{
    if (auto* s = dynamic_cast<Sum*>(e))
    {
        std::vector<Expr*> terms;
        terms.reserve(s->terms().size());
        for (auto* t: s->terms())
            terms.push_back(apply_div_thm_impl(rl, t, domain));
        return make_sum(rl, std::move(terms));
    }

    if (auto* sc = dynamic_cast<Scale*>(e))
    {
        auto* inner = apply_div_thm_impl(rl, sc->expr(), domain);
        return make_scale(rl, sc->coeff(), inner);
    }

    if (auto* integ = dynamic_cast<Integral*>(e))
    {
        if (integ->domain() != domain)
            return e;
        auto* dv = dynamic_cast<Divergence*>(integ->integrand());
        if (!dv)
            return e;

        // ∫_V ∇·A dV = ∫_∂V A·n dS
        Expr* n = domain->outward_normal();
        auto* A_dot_n = make_contract(rl, dv->arg(), n);
        return make_integral(rl, domain->surface_boundary(), A_dot_n);
    }

    return e;
}

auto apply_divergence_theorem_step(VolumeDomain* domain) -> DerivationStep
{
    return DerivationStep{
        "div_thm(" + domain->name() + ")",
        [domain](ResourceList& rl, Expr* e) -> Expr*
        { return apply_div_thm_impl(rl, e, domain); }};
}

// ===========================================================================
// localize_step
// ===========================================================================

// Try to extract the integrand from a single (non-Sum) term if it is an
// Integral over `domain` (possibly wrapped in Scale).
// Returns the extracted integrand (with scale folded in) or nullptr.
static auto try_extract_integrand(ResourceList& rl, Expr* t, Domain* domain)
    -> Expr*
{
    Rational sc{1};
    if (auto* s = dynamic_cast<Scale*>(t))
    {
        sc = s->coeff();
        t = s->expr();
    }
    if (auto* integ = dynamic_cast<Integral*>(t))
        if (integ->domain() == domain)
        {
            Expr* inner = integ->integrand();
            return (sc == Rational{1}) ? inner : make_scale(rl, sc, inner);
        }
    return nullptr;
}

static auto localize_impl(ResourceList& rl, Expr* e, Domain* domain) -> Expr*
{
    // For a Sum: keep only the terms that are Integrals over `domain`,
    // extracting their integrands.  Terms from other domains are discarded —
    // by the support argument of the fundamental lemma, choosing test functions
    // with support strictly inside `domain` makes all other domain integrals
    // vanish, so they do not appear in the localized equation.
    if (auto* s = dynamic_cast<Sum*>(e))
    {
        std::vector<Expr*> terms;
        for (auto* t: s->terms())
            if (auto* extracted = try_extract_integrand(rl, t, domain))
                terms.push_back(extracted);
        if (terms.empty())
            return e; // nothing matched — leave unchanged
        return make_sum(rl, std::move(terms));
    }

    // Single term: extract if it matches, leave unchanged otherwise.
    if (auto* extracted = try_extract_integrand(rl, e, domain))
        return extracted;
    return e;
}

auto localize_step(Domain* domain) -> DerivationStep
{
    return DerivationStep{
        "localize(" + domain->name() + ")",
        [domain](ResourceList& rl, Expr* e) -> Expr*
        { return localize_impl(rl, e, domain); }};
}

// ===========================================================================
// collect_step
// ===========================================================================

// Try to extract the coefficient of v from a single (non-Sum) term.
// Returns nullptr if v is not an immediate contraction factor of term.
// The domain pointer is set to the Domain* if term is wrapped in an Integral,
// or left as nullptr for pointwise terms.
// Folds any Scale coefficient into the returned coefficient expression.
static auto extract_coeff(
    ResourceList& rl, Expr* term, Expr* v, Domain*& domain_out) -> Expr*
{
    Rational scale{1};
    Expr* inner = term;

    // Peel off a Scale wrapper (from the outermost level).
    if (auto* sc = dynamic_cast<Scale*>(inner))
    {
        scale = sc->coeff();
        inner = sc->expr();
    }

    // Peel off an Integral wrapper.
    Domain* dom = nullptr;
    if (auto* integ = dynamic_cast<Integral*>(inner))
    {
        dom = integ->domain();
        inner = integ->integrand();
        // Integrand may itself be scaled.
        if (auto* sc = dynamic_cast<Scale*>(inner))
        {
            scale = scale * sc->coeff();
            inner = sc->expr();
        }
    }

    // inner must now be Contract(A, v) or Contract(v, A).
    Expr* coeff = nullptr;
    if (auto* c = dynamic_cast<Contract*>(inner))
    {
        if (c->rhs() == v)
            coeff = c->lhs();
        else if (c->lhs() == v)
            coeff = c->rhs();
    }

    if (!coeff)
        return nullptr;

    domain_out = dom;
    if (scale != Rational{1})
        coeff = make_scale(rl, scale, coeff);
    return coeff;
}

static auto collect_impl(ResourceList& rl, Expr* e, Expr* v) -> Expr*
{
    // Gather top-level terms.
    std::vector<Expr*> all_terms;
    if (auto* s = dynamic_cast<Sum*>(e))
        all_terms = s->terms();
    else
        all_terms = {e};

    // Map each domain pointer to its list of collected coefficients.
    // nullptr key = pointwise (no integral wrapper).
    std::vector<Domain*> domain_order; // preserves first-seen order
    std::map<Domain*, std::vector<Expr*>> groups;
    std::vector<Expr*> residuals;

    for (Expr* t: all_terms)
    {
        Domain* dom = nullptr;
        Expr* coeff = extract_coeff(rl, t, v, dom);
        if (!coeff)
        {
            residuals.push_back(t);
            continue;
        }
        if (groups.find(dom) == groups.end())
            domain_order.push_back(dom);
        groups[dom].push_back(coeff);
    }

    // Reassemble grouped terms, keeping domain order stable.
    std::vector<Expr*> result_terms;
    result_terms.reserve(domain_order.size() + residuals.size());

    for (Domain* dom: domain_order)
    {
        auto& coeffs = groups[dom];
        Expr* coeff_expr =
            (coeffs.size() == 1) ? coeffs[0] : make_sum(rl, coeffs);
        Expr* contracted = make_contract(rl, coeff_expr, v);
        if (dom)
            result_terms.push_back(make_integral(rl, dom, contracted));
        else
            result_terms.push_back(contracted);
    }

    for (Expr* r: residuals)
        result_terms.push_back(r);

    return make_sum(rl, std::move(result_terms));
}

auto collect_step(Expr* v) -> DerivationStep
{
    if (v->rank() == 0)
        throw std::invalid_argument(
            "collect_step: variation must have rank >= 1");
    return DerivationStep{
        "collect(" + v->latex() + ")", [v](ResourceList& rl, Expr* e) -> Expr* {
            return collect_impl(rl, e, v);
        }};
}

} // namespace tender
