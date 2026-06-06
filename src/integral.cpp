#include <tender/integral.hpp>

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

auto Gradient::latex() const -> std::string
{
    return "\\nabla " + arg_->latex();
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

auto Divergence::latex() const -> std::string
{
    return "\\nabla \\cdot " + arg_->latex();
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

auto Rotor::latex() const -> std::string
{
    return "\\nabla \\times " + arg_->latex();
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

auto Integral::latex() const -> std::string
{
    return "\\int_{" + domain_->name() + "} " + integrand_->latex() + " \\, "
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

static auto localize_impl(ResourceList& rl, Expr* e, Domain* domain) -> Expr*
{
    if (auto* integ = dynamic_cast<Integral*>(e))
    {
        if (integ->domain() == domain)
            return integ->integrand();
        return e;
    }

    if (auto* s = dynamic_cast<Sum*>(e))
    {
        std::vector<Expr*> terms;
        terms.reserve(s->terms().size());
        for (auto* t: s->terms())
            terms.push_back(localize_impl(rl, t, domain));
        return make_sum(rl, std::move(terms));
    }

    if (auto* sc = dynamic_cast<Scale*>(e))
    {
        auto* inner = localize_impl(rl, sc->expr(), domain);
        return make_scale(rl, sc->coeff(), inner);
    }

    return e;
}

auto localize_step(Domain* domain) -> DerivationStep
{
    return DerivationStep{
        "localize(" + domain->name() + ")",
        [domain](ResourceList& rl, Expr* e) -> Expr*
        { return localize_impl(rl, e, domain); }};
}

} // namespace tender
