#include <tender/basis.hpp>

#include <tender/expr.hpp>
#include <tender/integral.hpp>

#include <vector>

namespace tender
{

// ===========================================================================
// simplify_basis_dot_step
// ===========================================================================

static auto simplify_basis_dot_impl(
    ResourceList& rl, Expr* e, CoordSystem const& cs) -> Expr*;

// Try to recognise e as cs.basis(i) or cs.cobasis(i); return i or -1.
static auto find_basis_index(Expr* e, CoordSystem const& cs) -> int
{
    for (int i = 0; i < cs.dim(); ++i)
        if (e == cs.basis(i) || e == cs.cobasis(i))
            return i;
    return -1;
}

// Evaluate Contract(l, r) assuming both are already simplified.
// Handles:
//   Contract(TensorProduct(s, v), r)  →  make_product(s, Contract(v, r))
//   Contract(l, TensorProduct(s, w))  →  make_product(s, Contract(l, w))
//   Contract(e_i, e^j)                →  RationalConst(i==j ? 1 : 0)
static auto simplify_contract(
    ResourceList& rl, Expr* l, Expr* r, CoordSystem const& cs) -> Expr*
{
    // Pull rank-0 scalar out of lhs TensorProduct.
    if (auto* tpl = dynamic_cast<TensorProduct*>(l))
    {
        if (tpl->lhs()->rank() == 0)
        {
            auto* inner = simplify_basis_dot_impl(
                rl, make_contract(rl, tpl->rhs(), r), cs);
            return make_product(rl, tpl->lhs(), inner);
        }
    }

    // Pull rank-0 scalar out of rhs TensorProduct.
    if (auto* tpr = dynamic_cast<TensorProduct*>(r))
    {
        if (tpr->lhs()->rank() == 0)
        {
            auto* inner = simplify_basis_dot_impl(
                rl, make_contract(rl, l, tpr->rhs()), cs);
            return make_product(rl, tpr->lhs(), inner);
        }
    }

    // Basis dot: e_i · e^j  →  δ_i^j  (1 if i == j, else 0).
    // For orthonormal systems cobasis(i) == basis(i), so both directions match.
    int li = find_basis_index(l, cs);
    int ri = find_basis_index(r, cs);
    if (li >= 0 && ri >= 0)
        return make_rational(rl, Rational{li == ri ? 1 : 0});

    return make_contract(rl, l, r);
}

static auto simplify_basis_dot_impl(
    ResourceList& rl, Expr* e, CoordSystem const& cs) -> Expr*
{
    if (auto* co = dynamic_cast<Contract*>(e))
    {
        auto* l = simplify_basis_dot_impl(rl, co->lhs(), cs);
        auto* r = simplify_basis_dot_impl(rl, co->rhs(), cs);
        return simplify_contract(rl, l, r, cs);
    }

    if (auto* sc = dynamic_cast<Scale*>(e))
    {
        auto* inner = simplify_basis_dot_impl(rl, sc->expr(), cs);
        return inner == sc->expr() ? e : make_scale(rl, sc->coeff(), inner);
    }

    if (auto* s = dynamic_cast<Sum*>(e))
    {
        std::vector<Expr*> terms;
        terms.reserve(s->terms().size());
        bool changed = false;
        for (auto* t: s->terms())
        {
            auto* sub = simplify_basis_dot_impl(rl, t, cs);
            if (sub != t)
                changed = true;
            terms.push_back(sub);
        }
        // make_sum's Flattener removes zero terms automatically.
        return changed ? make_sum(rl, std::move(terms)) : e;
    }

    if (auto* tp = dynamic_cast<TensorProduct*>(e))
    {
        auto* l = simplify_basis_dot_impl(rl, tp->lhs(), cs);
        auto* r = simplify_basis_dot_impl(rl, tp->rhs(), cs);
        return (l == tp->lhs() && r == tp->rhs()) ?
                   e :
                   make_tensor_product(rl, l, r);
    }

    if (auto* dc = dynamic_cast<DoubleContract*>(e))
    {
        auto* l = simplify_basis_dot_impl(rl, dc->lhs(), cs);
        auto* r = simplify_basis_dot_impl(rl, dc->rhs(), cs);
        return (l == dc->lhs() && r == dc->rhs()) ?
                   e :
                   make_double_contract(rl, l, r);
    }

    if (auto* dcr = dynamic_cast<DoubleContractReversed*>(e))
    {
        auto* l = simplify_basis_dot_impl(rl, dcr->lhs(), cs);
        auto* r = simplify_basis_dot_impl(rl, dcr->rhs(), cs);
        return (l == dcr->lhs() && r == dcr->rhs()) ?
                   e :
                   make_double_contract_reversed(rl, l, r);
    }

    if (auto* cp = dynamic_cast<CrossProduct*>(e))
    {
        auto* l = simplify_basis_dot_impl(rl, cp->lhs(), cs);
        auto* r = simplify_basis_dot_impl(rl, cp->rhs(), cs);
        return (l == cp->lhs() && r == cp->rhs()) ? e :
                                                    rl.make<CrossProduct>(l, r);
    }

    if (auto* pr = dynamic_cast<Product*>(e))
    {
        auto* l = simplify_basis_dot_impl(rl, pr->lhs(), cs);
        auto* r = simplify_basis_dot_impl(rl, pr->rhs(), cs);
        return (l == pr->lhs() && r == pr->rhs()) ? e : make_product(rl, l, r);
    }

    if (auto* tr = dynamic_cast<Trace*>(e))
    {
        auto* inner = simplify_basis_dot_impl(rl, tr->arg(), cs);
        return inner == tr->arg() ? e : make_trace(rl, inner);
    }

    if (auto* pw = dynamic_cast<Pow*>(e))
    {
        auto* base = simplify_basis_dot_impl(rl, pw->base(), cs);
        return base == pw->base() ? e : make_pow(rl, base, pw->exponent());
    }

    if (auto* fa = dynamic_cast<FunctionApply*>(e))
    {
        auto* arg = simplify_basis_dot_impl(rl, fa->arg(), cs);
        return arg == fa->arg() ? e : make_function(rl, fa->kind(), arg);
    }

    if (auto* integ = dynamic_cast<Integral*>(e))
    {
        auto* body = simplify_basis_dot_impl(rl, integ->integrand(), cs);
        return body == integ->integrand() ?
                   e :
                   make_integral(rl, integ->domain(), body);
    }

    return e;
}

auto simplify_basis_dot_step(CoordSystem const& cs) -> DerivationStep
{
    return DerivationStep{
        "simplify_basis_dot", [&cs](ResourceList& rl, Expr* e) -> Expr* {
            return simplify_basis_dot_impl(rl, e, cs);
        }};
}

// ===========================================================================
// collect_zero_terms_step
// ===========================================================================

static auto collect_zeros_impl(ResourceList& rl, Expr* e) -> Expr*
{
    if (auto* s = dynamic_cast<Sum*>(e))
    {
        std::vector<Expr*> terms;
        terms.reserve(s->terms().size());
        for (auto* t: s->terms())
            terms.push_back(collect_zeros_impl(rl, t));
        // make_sum's Flattener removes RationalConst(0) terms and unwraps
        // single-element sums.
        return make_sum(rl, std::move(terms));
    }

    if (auto* sc = dynamic_cast<Scale*>(e))
    {
        auto* inner = collect_zeros_impl(rl, sc->expr());
        return inner == sc->expr() ? e : make_scale(rl, sc->coeff(), inner);
    }

    if (auto* tp = dynamic_cast<TensorProduct*>(e))
    {
        auto* l = collect_zeros_impl(rl, tp->lhs());
        auto* r = collect_zeros_impl(rl, tp->rhs());
        return (l == tp->lhs() && r == tp->rhs()) ?
                   e :
                   make_tensor_product(rl, l, r);
    }

    if (auto* co = dynamic_cast<Contract*>(e))
    {
        auto* l = collect_zeros_impl(rl, co->lhs());
        auto* r = collect_zeros_impl(rl, co->rhs());
        return (l == co->lhs() && r == co->rhs()) ? e : make_contract(rl, l, r);
    }

    if (auto* dc = dynamic_cast<DoubleContract*>(e))
    {
        auto* l = collect_zeros_impl(rl, dc->lhs());
        auto* r = collect_zeros_impl(rl, dc->rhs());
        return (l == dc->lhs() && r == dc->rhs()) ?
                   e :
                   make_double_contract(rl, l, r);
    }

    if (auto* pr = dynamic_cast<Product*>(e))
    {
        auto* l = collect_zeros_impl(rl, pr->lhs());
        auto* r = collect_zeros_impl(rl, pr->rhs());
        return (l == pr->lhs() && r == pr->rhs()) ? e : make_product(rl, l, r);
    }

    if (auto* tr = dynamic_cast<Trace*>(e))
    {
        auto* inner = collect_zeros_impl(rl, tr->arg());
        return inner == tr->arg() ? e : make_trace(rl, inner);
    }

    if (auto* integ = dynamic_cast<Integral*>(e))
    {
        auto* body = collect_zeros_impl(rl, integ->integrand());
        return body == integ->integrand() ?
                   e :
                   make_integral(rl, integ->domain(), body);
    }

    return e;
}

auto collect_zero_terms_step() -> DerivationStep
{
    return DerivationStep{
        "collect_zero_terms", [](ResourceList& rl, Expr* e) -> Expr* {
            return collect_zeros_impl(rl, e);
        }};
}

// ===========================================================================
// reassemble_from_components_step
// ===========================================================================

// Check if s is exactly Sum_i TensorProduct(cs.basis(i), cs.cobasis(i))
// in any order.  Returns true and sets each slot in `found`.
static auto is_identity_sum(Sum* s, CoordSystem const& cs) -> bool
{
    int const d = cs.dim();
    if (static_cast<int>(s->terms().size()) != d)
        return false;

    std::vector<bool> seen(d, false);
    for (auto* t: s->terms())
    {
        auto* tp = dynamic_cast<TensorProduct*>(t);
        if (!tp)
            return false;

        bool matched = false;
        for (int i = 0; i < d; ++i)
        {
            if (tp->lhs() == cs.basis(i) && tp->rhs() == cs.cobasis(i))
            {
                if (seen[i])
                    return false; // duplicate direction
                seen[i] = true;
                matched = true;
                break;
            }
        }
        if (!matched)
            return false;
    }
    return true;
}

static auto reassemble_impl(ResourceList& rl, Expr* e, CoordSystem const& cs)
    -> Expr*
{
    if (auto* s = dynamic_cast<Sum*>(e))
    {
        // Recurse into terms first.
        std::vector<Expr*> terms;
        terms.reserve(s->terms().size());
        bool changed = false;
        for (auto* t: s->terms())
        {
            auto* r = reassemble_impl(rl, t, cs);
            if (r != t)
                changed = true;
            terms.push_back(r);
        }
        Sum* candidate =
            changed ? dynamic_cast<Sum*>(make_sum(rl, std::move(terms))) : s;

        if (candidate && is_identity_sum(candidate, cs))
            return make_identity(rl);

        return candidate ? static_cast<Expr*>(candidate) : e;
    }

    if (auto* tp = dynamic_cast<TensorProduct*>(e))
    {
        auto* l = reassemble_impl(rl, tp->lhs(), cs);
        auto* r = reassemble_impl(rl, tp->rhs(), cs);
        return (l == tp->lhs() && r == tp->rhs()) ?
                   e :
                   make_tensor_product(rl, l, r);
    }

    if (auto* sc = dynamic_cast<Scale*>(e))
    {
        auto* inner = reassemble_impl(rl, sc->expr(), cs);
        return inner == sc->expr() ? e : make_scale(rl, sc->coeff(), inner);
    }

    if (auto* co = dynamic_cast<Contract*>(e))
    {
        auto* l = reassemble_impl(rl, co->lhs(), cs);
        auto* r = reassemble_impl(rl, co->rhs(), cs);
        return (l == co->lhs() && r == co->rhs()) ? e : make_contract(rl, l, r);
    }

    if (auto* dc = dynamic_cast<DoubleContract*>(e))
    {
        auto* l = reassemble_impl(rl, dc->lhs(), cs);
        auto* r = reassemble_impl(rl, dc->rhs(), cs);
        return (l == dc->lhs() && r == dc->rhs()) ?
                   e :
                   make_double_contract(rl, l, r);
    }

    if (auto* tr = dynamic_cast<Trace*>(e))
    {
        auto* inner = reassemble_impl(rl, tr->arg(), cs);
        return inner == tr->arg() ? e : make_trace(rl, inner);
    }

    if (auto* integ = dynamic_cast<Integral*>(e))
    {
        auto* body = reassemble_impl(rl, integ->integrand(), cs);
        return body == integ->integrand() ?
                   e :
                   make_integral(rl, integ->domain(), body);
    }

    return e;
}

auto reassemble_from_components_step(CoordSystem const& cs) -> DerivationStep
{
    return DerivationStep{
        "reassemble_from_components",
        [&cs](ResourceList& rl, Expr* e) -> Expr*
        { return reassemble_impl(rl, e, cs); }};
}

} // namespace tender
