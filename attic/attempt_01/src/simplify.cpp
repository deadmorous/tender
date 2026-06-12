#include <tender/derivation.hpp>
#include <tender/integral.hpp>

#include <vector>

namespace tender
{

// ===========================================================================
// simplify_identity
// ===========================================================================

static auto simplify_id_impl(ResourceList& rl, Expr* e) -> Expr*
{
    if (auto* co = dynamic_cast<Contract*>(e))
    {
        if (dynamic_cast<IdentityTensor*>(co->lhs()))
            return simplify_id_impl(rl, co->rhs()); // GCOV_EXCL_LINE
        if (dynamic_cast<IdentityTensor*>(co->rhs()))
            return simplify_id_impl(rl, co->lhs()); // GCOV_EXCL_LINE
        return make_contract(
            rl,
            simplify_id_impl(rl, co->lhs()),
            simplify_id_impl(rl, co->rhs()));
    }

    if (auto* dc = dynamic_cast<DoubleContract*>(e))
    {
        if (dynamic_cast<IdentityTensor*>(dc->lhs()))
            return make_trace(
                rl, simplify_id_impl(rl, dc->rhs())); // GCOV_EXCL_LINE
        if (dynamic_cast<IdentityTensor*>(dc->rhs()))
            return make_trace(rl, simplify_id_impl(rl, dc->lhs()));
        return make_double_contract(
            rl,
            simplify_id_impl(rl, dc->lhs()),
            simplify_id_impl(rl, dc->rhs()));
    }

    if (auto* dcr = dynamic_cast<DoubleContractReversed*>(e))
    {
        if (dynamic_cast<IdentityTensor*>(dcr->lhs()))
            return make_trace(rl, simplify_id_impl(rl, dcr->rhs()));
        if (dynamic_cast<IdentityTensor*>(dcr->rhs()))
            return make_trace(rl, simplify_id_impl(rl, dcr->lhs()));
        return make_double_contract_reversed(
            rl,
            simplify_id_impl(rl, dcr->lhs()),
            simplify_id_impl(rl, dcr->rhs()));
    }

    if (auto* sc = dynamic_cast<Scale*>(e))
        return make_scale(rl, sc->coeff(), simplify_id_impl(rl, sc->expr()));

    if (auto* s = dynamic_cast<Sum*>(e))
    {
        std::vector<Expr*> terms;
        terms.reserve(s->terms().size());
        for (auto* t: s->terms())
            terms.push_back(simplify_id_impl(rl, t));
        return make_sum(rl, std::move(terms));
    }

    if (auto* tp = dynamic_cast<TensorProduct*>(e))
        return make_tensor_product(
            rl,
            simplify_id_impl(rl, tp->lhs()),
            simplify_id_impl(rl, tp->rhs()));

    if (auto* cp = dynamic_cast<CrossProduct*>(e))
        return rl.make<CrossProduct>(
            simplify_id_impl(rl, cp->lhs()), simplify_id_impl(rl, cp->rhs()));

    if (auto* pr = dynamic_cast<Product*>(e))
        return make_product(
            rl,
            simplify_id_impl(rl, pr->lhs()),
            simplify_id_impl(rl, pr->rhs()));

    if (auto* tr = dynamic_cast<Trace*>(e))
        return make_trace(rl, simplify_id_impl(rl, tr->arg()));

    if (auto* pw = dynamic_cast<Pow*>(e))
        return make_pow(rl, simplify_id_impl(rl, pw->base()), pw->exponent());

    if (auto* fa = dynamic_cast<FunctionApply*>(e))
        return make_function(rl, fa->kind(), simplify_id_impl(rl, fa->arg()));

    return e;
}

auto simplify_identity_step() -> DerivationStep
{
    return DerivationStep{
        "simplify_identity", [](ResourceList& rl, Expr* e) -> Expr* {
            return simplify_id_impl(rl, e);
        }};
}

// ===========================================================================
// substitute
// ===========================================================================

static auto substitute_impl(
    ResourceList& rl, Expr* e, Expr* what, Expr* with_what) -> Expr*
{
    if (e == what)
        return with_what;

    if (auto* sc = dynamic_cast<Scale*>(e))
    {
        auto* inner = substitute_impl(rl, sc->expr(), what, with_what);
        return inner == sc->expr() ? e : make_scale(rl, sc->coeff(), inner);
    }

    if (auto* s = dynamic_cast<Sum*>(e))
    {
        std::vector<Expr*> terms;
        bool changed = false;
        for (auto* t: s->terms())
        {
            auto* sub = substitute_impl(rl, t, what, with_what);
            if (sub != t)
                changed = true;
            terms.push_back(sub);
        }
        return changed ? make_sum(rl, std::move(terms)) : e;
    }

    if (auto* tp = dynamic_cast<TensorProduct*>(e))
    {
        auto* l = substitute_impl(rl, tp->lhs(), what, with_what);
        auto* r = substitute_impl(rl, tp->rhs(), what, with_what);
        return (l == tp->lhs() && r == tp->rhs()) ?
                   e :
                   make_tensor_product(rl, l, r);
    }

    if (auto* co = dynamic_cast<Contract*>(e))
    {
        auto* l = substitute_impl(rl, co->lhs(), what, with_what);
        auto* r = substitute_impl(rl, co->rhs(), what, with_what);
        return (l == co->lhs() && r == co->rhs()) ? e : make_contract(rl, l, r);
    }

    if (auto* dc = dynamic_cast<DoubleContract*>(e))
    {
        auto* l = substitute_impl(rl, dc->lhs(), what, with_what);
        auto* r = substitute_impl(rl, dc->rhs(), what, with_what);
        return (l == dc->lhs() && r == dc->rhs()) ?
                   e :
                   make_double_contract(rl, l, r);
    }

    if (auto* dcr = dynamic_cast<DoubleContractReversed*>(e))
    {
        auto* l = substitute_impl(rl, dcr->lhs(), what, with_what);
        auto* r = substitute_impl(rl, dcr->rhs(), what, with_what);
        return (l == dcr->lhs() && r == dcr->rhs()) ?
                   e :
                   make_double_contract_reversed(rl, l, r);
    }

    if (auto* cp = dynamic_cast<CrossProduct*>(e))
    {
        auto* l = substitute_impl(rl, cp->lhs(), what, with_what);
        auto* r = substitute_impl(rl, cp->rhs(), what, with_what);
        return (l == cp->lhs() && r == cp->rhs()) ? e :
                                                    rl.make<CrossProduct>(l, r);
    }

    if (auto* pr = dynamic_cast<Product*>(e))
    {
        auto* l = substitute_impl(rl, pr->lhs(), what, with_what);
        auto* r = substitute_impl(rl, pr->rhs(), what, with_what);
        return (l == pr->lhs() && r == pr->rhs()) ? e : make_product(rl, l, r);
    }

    if (auto* tr = dynamic_cast<Trace*>(e))
    {
        auto* inner = substitute_impl(rl, tr->arg(), what, with_what);
        return inner == tr->arg() ? e : make_trace(rl, inner);
    }

    if (auto* pw = dynamic_cast<Pow*>(e))
    {
        auto* base = substitute_impl(rl, pw->base(), what, with_what);
        return base == pw->base() ? e : make_pow(rl, base, pw->exponent());
    }

    if (auto* fa = dynamic_cast<FunctionApply*>(e))
    {
        auto* arg = substitute_impl(rl, fa->arg(), what, with_what);
        return arg == fa->arg() ? e : make_function(rl, fa->kind(), arg);
    }

    if (auto* integ = dynamic_cast<Integral*>(e))
    {
        auto* body = substitute_impl(rl, integ->integrand(), what, with_what);
        return body == integ->integrand() ?
                   e :
                   make_integral(rl, integ->domain(), body);
    }

    return e;
}

auto substitute_step(Expr* what, Expr* with_what) -> DerivationStep
{
    return DerivationStep{
        "substitute", [what, with_what](ResourceList& rl, Expr* e) -> Expr* {
            return substitute_impl(rl, e, what, with_what);
        }};
}

auto replace_in_tree(ResourceList& rl, Expr* root, Expr* what, Expr* with_what)
    -> Expr*
{
    return substitute_impl(rl, root, what, with_what);
}

auto capture_step(std::string name, Expr* result) -> DerivationStep
{
    return DerivationStep{
        std::move(name),
        [result](ResourceList&, Expr*) -> Expr* { return result; }};
}

// ===========================================================================
// expand — distribute binary operations over Sum (bottom-up)
// ===========================================================================

static auto expand_impl(ResourceList& rl, Expr* e) -> Expr*;

// Distribute `op(l, r)` when l or r is a Sum.
// `rebuild` re-creates the node when neither is a Sum.
static auto distribute_binary(
    ResourceList& rl,
    Expr* l,
    Expr* r,
    auto rebuild,
    auto make_left,  // make_left(rl, term, r)  — term from lhs Sum
    auto make_right) // make_right(rl, l, term)  — term from rhs Sum
    -> Expr*
{
    if (auto* sl = dynamic_cast<Sum*>(l))
    {
        std::vector<Expr*> terms;
        terms.reserve(sl->terms().size());
        for (auto* t: sl->terms())
            terms.push_back(expand_impl(rl, make_left(rl, t, r)));
        return make_sum(rl, std::move(terms));
    }
    if (auto* sr = dynamic_cast<Sum*>(r))
    {
        std::vector<Expr*> terms;
        terms.reserve(sr->terms().size());
        for (auto* t: sr->terms())
            terms.push_back(expand_impl(rl, make_right(rl, l, t)));
        return make_sum(rl, std::move(terms));
    }
    return rebuild(rl, l, r);
}

static auto expand_impl(ResourceList& rl, Expr* e) -> Expr*
{
    // Scale: recurse first, then distribute over sum
    if (auto* sc = dynamic_cast<Scale*>(e))
    {
        auto* inner = expand_impl(rl, sc->expr());
        if (auto* s = dynamic_cast<Sum*>(inner))
        {
            std::vector<Expr*> terms;
            terms.reserve(s->terms().size());
            for (auto* t: s->terms())
                terms.push_back(make_scale(rl, sc->coeff(), t));
            return make_sum(rl, std::move(terms));
        }
        return make_scale(rl, sc->coeff(), inner);
    }

    if (auto* s = dynamic_cast<Sum*>(e))
    {
        std::vector<Expr*> terms;
        terms.reserve(s->terms().size());
        for (auto* t: s->terms())
            terms.push_back(expand_impl(rl, t));
        return make_sum(rl, std::move(terms));
    }

    if (auto* tp = dynamic_cast<TensorProduct*>(e))
    {
        auto* l = expand_impl(rl, tp->lhs());
        auto* r = expand_impl(rl, tp->rhs());
        return distribute_binary(
            rl,
            l,
            r,
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return make_tensor_product(rl2, a, b); },
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return make_tensor_product(rl2, a, b); },
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return make_tensor_product(rl2, a, b); });
    }

    if (auto* co = dynamic_cast<Contract*>(e))
    {
        auto* l = expand_impl(rl, co->lhs());
        auto* r = expand_impl(rl, co->rhs());
        return distribute_binary(
            rl,
            l,
            r,
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return make_contract(rl2, a, b); },
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return make_contract(rl2, a, b); },
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return make_contract(rl2, a, b); });
    }

    if (auto* dc = dynamic_cast<DoubleContract*>(e))
    {
        auto* l = expand_impl(rl, dc->lhs());
        auto* r = expand_impl(rl, dc->rhs());
        return distribute_binary(
            rl,
            l,
            r,
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return make_double_contract(rl2, a, b); },
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return make_double_contract(rl2, a, b); },
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return make_double_contract(rl2, a, b); });
    }

    if (auto* dcr = dynamic_cast<DoubleContractReversed*>(e))
    {
        auto* l = expand_impl(rl, dcr->lhs());
        auto* r = expand_impl(rl, dcr->rhs());
        return distribute_binary(
            rl,
            l,
            r,
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return make_double_contract_reversed(rl2, a, b); },
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return make_double_contract_reversed(rl2, a, b); },
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return make_double_contract_reversed(rl2, a, b); });
    }

    if (auto* cp = dynamic_cast<CrossProduct*>(e))
    {
        auto* l = expand_impl(rl, cp->lhs());
        auto* r = expand_impl(rl, cp->rhs());
        return distribute_binary(
            rl,
            l,
            r,
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return rl2.make<CrossProduct>(a, b); },
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return rl2.make<CrossProduct>(a, b); },
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return rl2.make<CrossProduct>(a, b); });
    }

    if (auto* pr = dynamic_cast<Product*>(e))
    {
        auto* l = expand_impl(rl, pr->lhs());
        auto* r = expand_impl(rl, pr->rhs());
        return distribute_binary(
            rl,
            l,
            r,
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return make_product(rl2, a, b); },
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return make_product(rl2, a, b); },
            [](ResourceList& rl2, Expr* a, Expr* b)
            { return make_product(rl2, a, b); });
    }

    if (auto* tr = dynamic_cast<Trace*>(e))
        return make_trace(rl, expand_impl(rl, tr->arg()));

    if (auto* pw = dynamic_cast<Pow*>(e))
        return make_pow(rl, expand_impl(rl, pw->base()), pw->exponent());

    if (auto* fa = dynamic_cast<FunctionApply*>(e))
        return make_function(rl, fa->kind(), expand_impl(rl, fa->arg()));

    return e;
}

auto expand_step() -> DerivationStep
{
    return DerivationStep{"expand", [](ResourceList& rl, Expr* e) -> Expr* {
                              return expand_impl(rl, e);
                          }};
}

} // namespace tender
