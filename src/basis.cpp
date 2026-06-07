#include <tender/basis.hpp>

#include <tender/expr.hpp>
#include <tender/integral.hpp>

#include <algorithm>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
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

// ===========================================================================
// collect_repeated_sum_step
// ===========================================================================

struct ComponentInfo
{
    std::string base;
    std::string sep; // "^" or "_"
    int idx;         // 1-based integer suffix
};

static auto parse_component(std::string const& sym)
    -> std::optional<ComponentInfo>
{
    auto try_split = [&](char delim) -> std::optional<ComponentInfo>
    {
        auto pos = sym.find(delim);
        if (pos == std::string::npos)
            return std::nullopt;
        std::string base = sym.substr(0, pos);
        std::string suffix = sym.substr(pos + 1);
        if (base.empty() || suffix.empty())
            return std::nullopt;
        if (!std::all_of(suffix.begin(), suffix.end(), ::isdigit))
            return std::nullopt;
        try
        {
            int idx = std::stoi(suffix);
            return ComponentInfo{base, std::string(1, delim), idx};
        }
        catch (...)
        {
            return std::nullopt;
        }
    };

    if (auto r = try_split('^'))
        return r;
    return try_split('_');
}

static auto collect_used_index_letters(Expr* e) -> std::set<std::string>
{
    std::set<std::string> result;
    if (!e)
        return result;

    if (auto* es = dynamic_cast<ExplicitSum*>(e))
    {
        result.insert(es->index()->letter());
        auto sub = collect_used_index_letters(es->body());
        result.insert(sub.begin(), sub.end());
    }
    else if (auto* is = dynamic_cast<IndexedSum*>(e))
    {
        result.insert(is->index_letter());
    }
    else if (auto* s = dynamic_cast<Sum*>(e))
    {
        for (auto* t: s->terms())
        {
            auto sub = collect_used_index_letters(t);
            result.insert(sub.begin(), sub.end());
        }
    }
    else if (auto* tp = dynamic_cast<TensorProduct*>(e))
    {
        auto l = collect_used_index_letters(tp->lhs());
        auto r = collect_used_index_letters(tp->rhs());
        result.insert(l.begin(), l.end());
        result.insert(r.begin(), r.end());
    }
    else if (auto* co = dynamic_cast<Contract*>(e))
    {
        auto l = collect_used_index_letters(co->lhs());
        auto r = collect_used_index_letters(co->rhs());
        result.insert(l.begin(), l.end());
        result.insert(r.begin(), r.end());
    }
    else if (auto* pr = dynamic_cast<Product*>(e))
    {
        auto l = collect_used_index_letters(pr->lhs());
        auto r = collect_used_index_letters(pr->rhs());
        result.insert(l.begin(), l.end());
        result.insert(r.begin(), r.end());
    }
    else if (auto* sc = dynamic_cast<Scale*>(e))
    {
        auto sub = collect_used_index_letters(sc->expr());
        result.insert(sub.begin(), sub.end());
    }
    return result;
}

static auto pick_index_letter(Expr* e, std::string const& requested)
    -> std::string
{
    static char const candidates[] = "ijklmnpqrs";
    auto used = collect_used_index_letters(e);

    if (!requested.empty())
    {
        if (used.count(requested))
            throw std::invalid_argument(
                "collect_repeated_sum_step: index letter '" + requested
                + "' is already in use");
        return requested;
    }

    for (char c: candidates)
    {
        std::string letter(1, c);
        if (!used.count(letter))
            return letter;
    }
    throw std::runtime_error(
        "collect_repeated_sum_step: all candidate index letters exhausted");
}

static auto collect_repeated_sum_impl(
    ResourceList& rl,
    Expr* e,
    CoordSystem const& cs,
    std::string const& letter) -> Expr*
{
    auto* s = dynamic_cast<Sum*>(e);
    if (!s || static_cast<int>(s->terms().size()) != cs.dim())
        return e;

    // Parse each term as Product(NamedTensor(rank=0), NamedTensor(rank=0))
    struct TermData
    {
        ComponentInfo lhs;
        ComponentInfo rhs;
    };
    std::vector<TermData> data;
    data.reserve(cs.dim());

    for (auto* term: s->terms())
    {
        auto* pr = dynamic_cast<Product*>(term);
        if (!pr)
            return e;
        auto* lhs_nt = dynamic_cast<NamedTensor*>(pr->lhs());
        auto* rhs_nt = dynamic_cast<NamedTensor*>(pr->rhs());
        if (!lhs_nt || lhs_nt->rank() != 0 || !rhs_nt || rhs_nt->rank() != 0)
            return e;
        auto lc = parse_component(lhs_nt->symbol());
        auto rc = parse_component(rhs_nt->symbol());
        if (!lc || !rc)
            return e;
        data.push_back({*lc, *rc});
    }

    // Consistency: same base/sep across all terms; paired indices must match
    std::string const& lb = data[0].lhs.base;
    std::string const& ls = data[0].lhs.sep;
    std::string const& rb = data[0].rhs.base;
    std::string const& rs = data[0].rhs.sep;

    for (auto const& d: data)
    {
        if (d.lhs.base != lb || d.lhs.sep != ls)
            return e;
        if (d.rhs.base != rb || d.rhs.sep != rs)
            return e;
        if (d.lhs.idx != d.rhs.idx)
            return e;
    }

    // Indices must cover {1..dim} exactly (any order)
    std::vector<bool> seen(cs.dim() + 1, false);
    for (auto const& d: data)
    {
        if (d.lhs.idx < 1 || d.lhs.idx > cs.dim())
            return e;
        if (seen[d.lhs.idx])
            return e;
        seen[d.lhs.idx] = true;
    }

    return make_indexed_sum(rl, lb, ls, rb, rs, letter, 0);
}

auto collect_repeated_sum_step(CoordSystem const& cs, std::string index_letter)
    -> DerivationStep
{
    return DerivationStep{
        "collect_repeated_sum",
        [&cs, req = std::move(index_letter)](ResourceList& rl, Expr* e) -> Expr*
        {
            std::string letter = pick_index_letter(e, req);
            return collect_repeated_sum_impl(rl, e, cs, letter);
        }};
}

// ===========================================================================
// reassemble_vector_step
// ===========================================================================

// Try to match e as Sum of dim TensorProduct(component_scalar, basis_vector)
// terms.  Returns a fresh NamedTensor(base, rank=1) on match, nullptr on miss.
// The separator ("^" or "_") in the component symbol determines covariance:
//   "^" (upper) → matched against cs.basis(k)
//   "_" (lower) → matched against cs.cobasis(k)
static auto try_match_vector_sum(
    ResourceList& rl, Expr* e, CoordSystem const& cs) -> Expr*
{
    auto* s = dynamic_cast<Sum*>(e);
    if (!s || static_cast<int>(s->terms().size()) != cs.dim())
        return nullptr;

    struct TermData
    {
        ComponentInfo comp;
        int basis_idx; // 0-based
    };
    std::vector<TermData> data;
    data.reserve(cs.dim());

    for (auto* term: s->terms())
    {
        auto* tp = dynamic_cast<TensorProduct*>(term);
        if (!tp)
            return nullptr;
        auto* comp_nt = dynamic_cast<NamedTensor*>(tp->lhs());
        if (!comp_nt || comp_nt->rank() != 0)
            return nullptr;
        auto comp = parse_component(comp_nt->symbol());
        if (!comp)
            return nullptr;

        // Use the separator to choose basis vs cobasis
        int bidx = -1;
        for (int i = 0; i < cs.dim(); ++i)
        {
            Expr* expected = (comp->sep == "^") ? cs.basis(i) : cs.cobasis(i);
            if (tp->rhs() == expected)
            {
                bidx = i;
                break;
            }
        }
        if (bidx < 0)
            return nullptr;

        data.push_back({*comp, bidx});
    }

    // Consistency: same base and sep across all terms
    std::string const& base = data[0].comp.base;
    std::string const& sep = data[0].comp.sep;

    for (auto const& d: data)
    {
        if (d.comp.base != base || d.comp.sep != sep)
            return nullptr;
        // Component index (1-based) must equal basis index (0-based) + 1
        if (d.comp.idx != d.basis_idx + 1)
            return nullptr;
    }

    // Indices must cover {0..dim-1} exactly
    std::vector<bool> seen(cs.dim(), false);
    for (auto const& d: data)
    {
        if (seen[d.basis_idx])
            return nullptr;
        seen[d.basis_idx] = true;
    }

    return make_named_tensor(rl, base, 1, {});
}

static auto reassemble_vector_impl(
    ResourceList& rl, Expr* e, CoordSystem const& cs) -> Expr*
{
    // Try to match the current node first.
    if (auto* v = try_match_vector_sum(rl, e, cs))
        return v;

    // Otherwise recurse into known compound nodes.
    if (auto* s = dynamic_cast<Sum*>(e))
    {
        std::vector<Expr*> terms;
        terms.reserve(s->terms().size());
        bool changed = false;
        for (auto* t: s->terms())
        {
            auto* r = reassemble_vector_impl(rl, t, cs);
            if (r != t)
                changed = true;
            terms.push_back(r);
        }
        return changed ? make_sum(rl, std::move(terms)) : e;
    }
    if (auto* tp = dynamic_cast<TensorProduct*>(e))
    {
        auto* l = reassemble_vector_impl(rl, tp->lhs(), cs);
        auto* r = reassemble_vector_impl(rl, tp->rhs(), cs);
        return (l == tp->lhs() && r == tp->rhs()) ?
                   e :
                   make_tensor_product(rl, l, r);
    }
    if (auto* co = dynamic_cast<Contract*>(e))
    {
        auto* l = reassemble_vector_impl(rl, co->lhs(), cs);
        auto* r = reassemble_vector_impl(rl, co->rhs(), cs);
        return (l == co->lhs() && r == co->rhs()) ? e : make_contract(rl, l, r);
    }
    if (auto* sc = dynamic_cast<Scale*>(e))
    {
        auto* inner = reassemble_vector_impl(rl, sc->expr(), cs);
        return inner == sc->expr() ? e : make_scale(rl, sc->coeff(), inner);
    }
    return e;
}

auto reassemble_vector_step(CoordSystem const& cs) -> DerivationStep
{
    return DerivationStep{
        "reassemble_vector", [&cs](ResourceList& rl, Expr* e) -> Expr* {
            return reassemble_vector_impl(rl, e, cs);
        }};
}

// ===========================================================================
// reassemble_dot_step
// ===========================================================================

static auto reassemble_dot_impl(
    ResourceList& rl, Expr* e, CoordSystem const& cs) -> Expr*
{
    if (auto* co = dynamic_cast<Contract*>(e))
    {
        auto* l = try_match_vector_sum(rl, co->lhs(), cs);
        auto* r = try_match_vector_sum(rl, co->rhs(), cs);
        if (l || r)
        {
            auto* new_lhs = l ? l : reassemble_dot_impl(rl, co->lhs(), cs);
            auto* new_rhs = r ? r : reassemble_dot_impl(rl, co->rhs(), cs);
            return make_contract(rl, new_lhs, new_rhs);
        }
        auto* rl2 = reassemble_dot_impl(rl, co->lhs(), cs);
        auto* rr2 = reassemble_dot_impl(rl, co->rhs(), cs);
        return (rl2 == co->lhs() && rr2 == co->rhs()) ?
                   e :
                   make_contract(rl, rl2, rr2);
    }
    if (auto* s = dynamic_cast<Sum*>(e))
    {
        // Try vector match first, then recurse.
        if (auto* v = try_match_vector_sum(rl, e, cs))
            return v;
        std::vector<Expr*> terms;
        terms.reserve(s->terms().size());
        bool changed = false;
        for (auto* t: s->terms())
        {
            auto* r = reassemble_dot_impl(rl, t, cs);
            if (r != t)
                changed = true;
            terms.push_back(r);
        }
        return changed ? make_sum(rl, std::move(terms)) : e;
    }
    if (auto* tp = dynamic_cast<TensorProduct*>(e))
    {
        auto* l = reassemble_dot_impl(rl, tp->lhs(), cs);
        auto* r = reassemble_dot_impl(rl, tp->rhs(), cs);
        return (l == tp->lhs() && r == tp->rhs()) ?
                   e :
                   make_tensor_product(rl, l, r);
    }
    if (auto* sc = dynamic_cast<Scale*>(e))
    {
        auto* inner = reassemble_dot_impl(rl, sc->expr(), cs);
        return inner == sc->expr() ? e : make_scale(rl, sc->coeff(), inner);
    }
    return e;
}

auto reassemble_dot_step(CoordSystem const& cs) -> DerivationStep
{
    return DerivationStep{
        "reassemble_dot", [&cs](ResourceList& rl, Expr* e) -> Expr* {
            return reassemble_dot_impl(rl, e, cs);
        }};
}

} // namespace tender
