#include <tender/basis.hpp>

#include <tender/coord_system.hpp>
#include <tender/expr.hpp>
#include <tender/integral.hpp>

#include <algorithm>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
//   Contract(TP(AbstractComp, SBV_basis), TP(AbstractComp, SBV_cobasis))
//     →  AbstractIndexedSum
//   Contract(TensorProduct(s, v), r)        →  make_product(s, Contract(v, r))
//   Contract(l, TensorProduct(s, w))        →  make_product(s, Contract(l, w))
//   Contract(e_i, e^j)                      →  RationalConst(i==j ? 1 : 0)
static auto simplify_contract(
    ResourceList& rl, Expr* l, Expr* r, CoordSystem const& cs) -> Expr*
{
    // Abstract-basis dot: Contract(TP(AbstractComp, SBV_basis),
    //                              TP(AbstractComp, SBV_cobasis))
    // → AbstractIndexedSum using the shared index_id from both SBVs.
    if (auto* tpl = dynamic_cast<TensorProduct*>(l))
    {
        if (auto* sbvl = dynamic_cast<SymBasisVec*>(tpl->rhs()))
        {
            if (auto* tpr = dynamic_cast<TensorProduct*>(r))
            {
                if (auto* sbvr = dynamic_cast<SymBasisVec*>(tpr->rhs()))
                {
                    if (&sbvl->cs() == &sbvr->cs()
                        && sbvl->is_cobasis() != sbvr->is_cobasis())
                    {
                        auto* ac_a = dynamic_cast<AbstractComp*>(tpl->lhs());
                        auto* ac_b = dynamic_cast<AbstractComp*>(tpr->lhs());
                        if (ac_a && ac_b)
                        {
                            int const lhs_id = sbvl->index_id();
                            int const rhs_id = sbvr->index_id();
                            if (lhs_id == rhs_id)
                            {
                                // Same ID: direct shortcut — skip
                                // KroneckerDelta
                                return make_abstract_indexed_sum(
                                    rl, ac_a, ac_b, lhs_id, 0);
                            }
                            // Different IDs: general path via KroneckerDelta.
                            // cobasis SBV is the lower index (covariant),
                            // basis SBV is the upper index (contravariant).
                            int lo_id = sbvl->is_cobasis() ? rhs_id : lhs_id;
                            int hi_id = sbvl->is_cobasis() ? lhs_id : rhs_id;
                            auto* kd = make_kronecker_delta(rl, lo_id, hi_id);
                            Expr* prod = make_product(rl, ac_a, ac_b);
                            return make_product(rl, prod, kd);
                        }
                    }
                }
            }
        }
    }

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

    // Pull rank-0 scalar out of rhs TensorProduct (scalar on lhs of rhs TP).
    if (auto* tpr = dynamic_cast<TensorProduct*>(r))
    {
        if (tpr->lhs()->rank() == 0)
        {
            auto* inner = simplify_basis_dot_impl(
                rl, make_contract(rl, l, tpr->rhs()), cs);
            if (inner->rank() == 0)
                return make_product(rl, tpr->lhs(), inner);
        }
    }

    // Pull rank-0 scalar out of rhs TensorProduct (scalar on rhs of rhs TP).
    if (auto* tpr = dynamic_cast<TensorProduct*>(r))
    {
        if (tpr->rhs()->rank() == 0)
        {
            auto* inner = simplify_basis_dot_impl(
                rl, make_contract(rl, l, tpr->lhs()), cs);
            if (inner->rank() == 0)
                return make_product(rl, tpr->rhs(), inner);
        }
    }

    // Basis dot: e_i · e^j  →  δ_i^j  (1 if i == j, else 0).
    // For orthonormal systems cobasis(i) == basis(i), so both directions match.
    int li = find_basis_index(l, cs);
    int ri = find_basis_index(r, cs);
    if (li >= 0 && ri >= 0)
        return make_rational(rl, Rational{li == ri ? 1 : 0});

    // Abstract SBV dot: Contract(SBV(id_l, cob_l), SBV(id_r, cob_r)).
    // Opposite cobasis → KroneckerDelta (works in any CS).
    // Same cobasis in orthonormal CS → KroneckerDelta (g^{ij}=δ^{ij}).
    if (auto* sbvl = dynamic_cast<SymBasisVec*>(l))
    {
        if (auto* sbvr = dynamic_cast<SymBasisVec*>(r))
        {
            if (&sbvl->cs() == &cs && &sbvr->cs() == &cs)
            {
                bool opp = (sbvl->is_cobasis() != sbvr->is_cobasis());
                bool same_ortho =
                    (sbvl->is_cobasis() == sbvr->is_cobasis()
                     && cs.is_orthonormal());
                if (opp || same_ortho)
                {
                    // Lower id = the basis (non-cobasis) one; upper = cobasis.
                    // For same-cobasis orthonormal, pick lo < hi by convention.
                    int lo_id, hi_id;
                    if (opp)
                    {
                        lo_id = sbvl->is_cobasis() ? sbvr->index_id() :
                                                     sbvl->index_id();
                        hi_id = sbvl->is_cobasis() ? sbvl->index_id() :
                                                     sbvr->index_id();
                    }
                    else
                    {
                        lo_id = sbvl->index_id();
                        hi_id = sbvr->index_id();
                    }
                    return make_kronecker_delta(rl, lo_id, hi_id);
                }
            }
        }
    }

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
        // Fold TP(1, x) → x after basis-dot simplification.
        if (auto* lrc = dynamic_cast<RationalConst*>(l))
            if (lrc->value() == Rational{1})
                return r;
        return (l == tp->lhs() && r == tp->rhs()) ?
                   e :
                   make_tensor_product(rl, l, r);
    }

    if (auto* dc = dynamic_cast<DoubleContract*>(e))
    {
        auto* l = simplify_basis_dot_impl(rl, dc->lhs(), cs);
        auto* r = simplify_basis_dot_impl(rl, dc->rhs(), cs);
        // Try to expand ddot(TP(TP(A_prefix, A_mid), A_last), TP(B_first,
        // B_second)) → TP(A_prefix, Product(Contract(A_mid, B_first),
        // Contract(A_last, B_second)))
        auto* tp_l_outer = dynamic_cast<TensorProduct*>(l);
        auto* tp_r = dynamic_cast<TensorProduct*>(r);
        if (tp_l_outer && tp_r)
        {
            auto* a_last = tp_l_outer->rhs();
            auto* tp_l_inner = dynamic_cast<TensorProduct*>(tp_l_outer->lhs());
            if (tp_l_inner && a_last->rank() == 1)
            {
                auto* a_mid = tp_l_inner->rhs();
                auto* a_prefix = tp_l_inner->lhs();
                auto* b_first = tp_r->lhs();
                auto* b_second = tp_r->rhs();
                if (a_mid->rank() == 1 && b_first->rank() == 1
                    && b_second->rank() == 1)
                {
                    auto* dot1 = simplify_basis_dot_impl(
                        rl, make_contract(rl, a_mid, b_first), cs);
                    auto* dot2 = simplify_basis_dot_impl(
                        rl, make_contract(rl, a_last, b_second), cs);
                    if (dot1->rank() == 0 && dot2->rank() == 0)
                        return make_tensor_product(
                            rl, a_prefix, make_product(rl, dot1, dot2));
                }
            }
        }
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

// ===========================================================================
// contract_kronecker_step
// ===========================================================================

// Count every abstract index ID occurrence in the tree.
static auto count_all_index_ids(
    Expr const* e, std::unordered_map<int, int>& counts) -> void
{
    if (!e)
        return;
    if (auto const* ac = dynamic_cast<AbstractComp const*>(e))
    {
        for (auto const& [id, up]: ac->indices())
            counts[id]++;
        return;
    }
    if (auto const* sbv = dynamic_cast<SymBasisVec const*>(e))
    {
        counts[sbv->index_id()]++;
        return;
    }
    if (auto const* kd = dynamic_cast<KroneckerDelta const*>(e))
    {
        counts[kd->lower_id()]++;
        counts[kd->upper_id()]++;
        return;
    }
    if (auto const* lcs = dynamic_cast<LeviCivitaSymbol const*>(e))
    {
        for (int id: lcs->ids())
            counts[id]++;
        return;
    }
    if (auto const* ais = dynamic_cast<AbstractIndexedSum const*>(e))
    {
        count_all_index_ids(ais->lhs(), counts);
        count_all_index_ids(ais->rhs(), counts);
        return;
    }
    if (auto const* tp = dynamic_cast<TensorProduct const*>(e))
    {
        count_all_index_ids(tp->lhs(), counts);
        count_all_index_ids(tp->rhs(), counts);
        return;
    }
    if (auto const* pr = dynamic_cast<Product const*>(e))
    {
        count_all_index_ids(pr->lhs(), counts);
        count_all_index_ids(pr->rhs(), counts);
        return;
    }
    if (auto const* s = dynamic_cast<Sum const*>(e))
    {
        for (auto const* t: s->terms())
            count_all_index_ids(t, counts);
        return;
    }
    if (auto const* sc = dynamic_cast<Scale const*>(e))
    {
        count_all_index_ids(sc->expr(), counts);
        return;
    }
    if (auto const* co = dynamic_cast<Contract const*>(e))
    {
        count_all_index_ids(co->lhs(), counts);
        count_all_index_ids(co->rhs(), counts);
        return;
    }
    if (auto const* dc = dynamic_cast<DoubleContract const*>(e))
    {
        count_all_index_ids(dc->lhs(), counts);
        count_all_index_ids(dc->rhs(), counts);
        return;
    }
    if (auto const* dr = dynamic_cast<DoubleContractReversed const*>(e))
    {
        count_all_index_ids(dr->lhs(), counts);
        count_all_index_ids(dr->rhs(), counts);
        return;
    }
    if (auto const* tr = dynamic_cast<Trace const*>(e))
    {
        count_all_index_ids(tr->arg(), counts);
        return;
    }
}

// Find the first KroneckerDelta in the tree (DFS order).
static auto find_first_kronecker(Expr* e) -> KroneckerDelta*
{
    if (!e)
        return nullptr;
    if (auto* kd = dynamic_cast<KroneckerDelta*>(e))
        return kd;
    if (auto* tp = dynamic_cast<TensorProduct*>(e))
    {
        if (auto* kd = find_first_kronecker(tp->lhs()))
            return kd;
        return find_first_kronecker(tp->rhs());
    }
    if (auto* pr = dynamic_cast<Product*>(e))
    {
        if (auto* kd = find_first_kronecker(pr->lhs()))
            return kd;
        return find_first_kronecker(pr->rhs());
    }
    if (auto* s = dynamic_cast<Sum*>(e))
    {
        for (auto* t: s->terms())
            if (auto* kd = find_first_kronecker(t))
                return kd;
        return nullptr;
    }
    if (auto* sc = dynamic_cast<Scale*>(e))
        return find_first_kronecker(sc->expr());
    if (auto* co = dynamic_cast<Contract*>(e))
    {
        if (auto* kd = find_first_kronecker(co->lhs()))
            return kd;
        return find_first_kronecker(co->rhs());
    }
    if (auto* dc = dynamic_cast<DoubleContract*>(e))
    {
        if (auto* kd = find_first_kronecker(dc->lhs()))
            return kd;
        return find_first_kronecker(dc->rhs());
    }
    if (auto* dr = dynamic_cast<DoubleContractReversed*>(e))
    {
        if (auto* kd = find_first_kronecker(dr->lhs()))
            return kd;
        return find_first_kronecker(dr->rhs());
    }
    if (auto* tr = dynamic_cast<Trace*>(e))
        return find_first_kronecker(tr->arg());
    return nullptr;
}

// Substitute old_id → new_id in the tree AND replace target_kd with 1.
static auto substitute_and_remove(
    ResourceList& rl,
    Expr* e,
    int old_id,
    int new_id,
    KroneckerDelta const* target_kd) -> Expr*
{
    if (e == target_kd)
        return make_rational(rl, Rational{1});
    return substitute_index(rl, e, old_id, new_id);
}

// Full tree walk for substitute_and_remove: recurses through compound nodes.
static auto substitute_remove_impl(
    ResourceList& rl,
    Expr* e,
    int old_id,
    int new_id,
    KroneckerDelta const* target_kd) -> Expr*
{
    if (e == target_kd)
        return make_rational(rl, Rational{1});

    // For leaf-level index nodes, substitute without further recursion.
    if (dynamic_cast<AbstractComp*>(e) || dynamic_cast<SymBasisVec*>(e)
        || dynamic_cast<KroneckerDelta*>(e)
        || dynamic_cast<LeviCivitaSymbol*>(e)
        || dynamic_cast<AbstractIndexedSum*>(e))
    {
        return substitute_index(rl, e, old_id, new_id);
    }
    if (auto* tp = dynamic_cast<TensorProduct*>(e))
    {
        auto* l =
            substitute_remove_impl(rl, tp->lhs(), old_id, new_id, target_kd);
        auto* r =
            substitute_remove_impl(rl, tp->rhs(), old_id, new_id, target_kd);
        return (l == tp->lhs() && r == tp->rhs()) ?
                   e :
                   make_tensor_product(rl, l, r);
    }
    if (auto* pr = dynamic_cast<Product*>(e))
    {
        auto* l =
            substitute_remove_impl(rl, pr->lhs(), old_id, new_id, target_kd);
        auto* r =
            substitute_remove_impl(rl, pr->rhs(), old_id, new_id, target_kd);
        return (l == pr->lhs() && r == pr->rhs()) ? e : make_product(rl, l, r);
    }
    if (auto* s = dynamic_cast<Sum*>(e))
    {
        std::vector<Expr*> terms;
        terms.reserve(s->terms().size());
        bool changed = false;
        for (auto* t: s->terms())
        {
            auto* sub =
                substitute_remove_impl(rl, t, old_id, new_id, target_kd);
            if (sub != t)
                changed = true;
            terms.push_back(sub);
        }
        return changed ? make_sum(rl, std::move(terms)) : e;
    }
    if (auto* sc = dynamic_cast<Scale*>(e))
    {
        auto* inner =
            substitute_remove_impl(rl, sc->expr(), old_id, new_id, target_kd);
        return inner == sc->expr() ? e : make_scale(rl, sc->coeff(), inner);
    }
    if (auto* co = dynamic_cast<Contract*>(e))
    {
        auto* l =
            substitute_remove_impl(rl, co->lhs(), old_id, new_id, target_kd);
        auto* r =
            substitute_remove_impl(rl, co->rhs(), old_id, new_id, target_kd);
        return (l == co->lhs() && r == co->rhs()) ? e : make_contract(rl, l, r);
    }
    if (auto* dc = dynamic_cast<DoubleContract*>(e))
    {
        auto* l =
            substitute_remove_impl(rl, dc->lhs(), old_id, new_id, target_kd);
        auto* r =
            substitute_remove_impl(rl, dc->rhs(), old_id, new_id, target_kd);
        return (l == dc->lhs() && r == dc->rhs()) ?
                   e :
                   make_double_contract(rl, l, r);
    }
    if (auto* dr = dynamic_cast<DoubleContractReversed*>(e))
    {
        auto* l =
            substitute_remove_impl(rl, dr->lhs(), old_id, new_id, target_kd);
        auto* r =
            substitute_remove_impl(rl, dr->rhs(), old_id, new_id, target_kd);
        return (l == dr->lhs() && r == dr->rhs()) ?
                   e :
                   make_double_contract_reversed(rl, l, r);
    }
    if (auto* tr = dynamic_cast<Trace*>(e))
    {
        auto* inner =
            substitute_remove_impl(rl, tr->arg(), old_id, new_id, target_kd);
        return inner == tr->arg() ? e : make_trace(rl, inner);
    }
    return e;
}

auto contract_kronecker_step() -> DerivationStep
{
    return DerivationStep{
        "contract_kronecker",
        [](ResourceList& rl, Expr* e) -> Expr*
        {
            KroneckerDelta* kd = find_first_kronecker(e);
            if (!kd)
                return e;

            // Count all index-ID occurrences in the full tree.
            std::unordered_map<int, int> counts;
            count_all_index_ids(e, counts);

            int lo = kd->lower_id();
            int hi = kd->upper_id();
            // Subtract the delta's own contributions to get external counts.
            int lo_ext = counts[lo] - 1;
            int hi_ext = counts[hi] - 1;

            int dummy_id, free_id;
            if (lo_ext == 1 && hi_ext != 1)
                dummy_id = lo, free_id = hi;
            else if (hi_ext == 1 && lo_ext != 1)
                dummy_id = hi, free_id = lo;
            else if (lo_ext == 1 && hi_ext == 1)
                // Both are dummy; prefer to eliminate hi → lo.
                dummy_id = hi, free_id = lo;
            else
                return e; // no dummy index; leave as-is

            return substitute_remove_impl(rl, e, dummy_id, free_id, kd);
        }};
}

// ===========================================================================
// replace_first_lct_step
// ===========================================================================

static auto replace_first_lct_impl(
    ResourceList& rl, Expr* e, Expr* expansion, bool& replaced) -> Expr*
{
    if (replaced)
        return e;
    if (dynamic_cast<LeviCivitaTensor*>(e))
    {
        replaced = true;
        return expansion;
    }
    if (auto* tp = dynamic_cast<TensorProduct*>(e))
    {
        auto* l = replace_first_lct_impl(rl, tp->lhs(), expansion, replaced);
        auto* r = replace_first_lct_impl(rl, tp->rhs(), expansion, replaced);
        return (l == tp->lhs() && r == tp->rhs()) ?
                   e :
                   make_tensor_product(rl, l, r);
    }
    if (auto* sc = dynamic_cast<Scale*>(e))
    {
        auto* inner =
            replace_first_lct_impl(rl, sc->expr(), expansion, replaced);
        return inner == sc->expr() ? e : make_scale(rl, sc->coeff(), inner);
    }
    if (auto* s = dynamic_cast<Sum*>(e))
    {
        std::vector<Expr*> terms;
        terms.reserve(s->terms().size());
        bool changed = false;
        for (auto* t: s->terms())
        {
            auto* sub = replace_first_lct_impl(rl, t, expansion, replaced);
            if (sub != t)
                changed = true;
            terms.push_back(sub);
        }
        return changed ? make_sum(rl, std::move(terms)) : e;
    }
    if (auto* co = dynamic_cast<Contract*>(e))
    {
        auto* l = replace_first_lct_impl(rl, co->lhs(), expansion, replaced);
        auto* r = replace_first_lct_impl(rl, co->rhs(), expansion, replaced);
        return (l == co->lhs() && r == co->rhs()) ? e : make_contract(rl, l, r);
    }
    if (auto* dc = dynamic_cast<DoubleContract*>(e))
    {
        auto* l = replace_first_lct_impl(rl, dc->lhs(), expansion, replaced);
        auto* r = replace_first_lct_impl(rl, dc->rhs(), expansion, replaced);
        return (l == dc->lhs() && r == dc->rhs()) ?
                   e :
                   make_double_contract(rl, l, r);
    }
    if (auto* pr = dynamic_cast<Product*>(e))
    {
        auto* l = replace_first_lct_impl(rl, pr->lhs(), expansion, replaced);
        auto* r = replace_first_lct_impl(rl, pr->rhs(), expansion, replaced);
        return (l == pr->lhs() && r == pr->rhs()) ? e : make_product(rl, l, r);
    }
    return e;
}

auto replace_first_lct_step(Expr* expansion) -> DerivationStep
{
    return DerivationStep{
        "expand_levi_civita_first",
        [expansion](ResourceList& rl, Expr* e) -> Expr*
        {
            bool replaced = false;
            return replace_first_lct_impl(rl, e, expansion, replaced);
        }};
}

// ===========================================================================
// contract_eps_pair_step
// ===========================================================================

static void collect_lcs_nodes(Expr* e, std::vector<LeviCivitaSymbol*>& out)
{
    if (auto* lcs = dynamic_cast<LeviCivitaSymbol*>(e))
    {
        out.push_back(lcs);
        return;
    }
    if (auto* tp = dynamic_cast<TensorProduct*>(e))
    {
        collect_lcs_nodes(tp->lhs(), out);
        collect_lcs_nodes(tp->rhs(), out);
    }
    else if (auto* sc = dynamic_cast<Scale*>(e))
    {
        collect_lcs_nodes(sc->expr(), out);
    }
    else if (auto* s = dynamic_cast<Sum*>(e))
    {
        for (auto* t: s->terms())
            collect_lcs_nodes(t, out);
    }
    else if (auto* co = dynamic_cast<Contract*>(e))
    {
        collect_lcs_nodes(co->lhs(), out);
        collect_lcs_nodes(co->rhs(), out);
    }
    else if (auto* dc = dynamic_cast<DoubleContract*>(e))
    {
        collect_lcs_nodes(dc->lhs(), out);
        collect_lcs_nodes(dc->rhs(), out);
    }
    else if (auto* pr = dynamic_cast<Product*>(e))
    {
        collect_lcs_nodes(pr->lhs(), out);
        collect_lcs_nodes(pr->rhs(), out);
    }
}

auto contract_eps_pair_step() -> DerivationStep
{
    return DerivationStep{
        "contract_eps_pair",
        [](ResourceList& rl, Expr* e) -> Expr*
        {
            std::vector<LeviCivitaSymbol*> lcs_nodes;
            collect_lcs_nodes(e, lcs_nodes);
            if (lcs_nodes.size() < 2)
                return e;

            auto* lcs1 = lcs_nodes[0];
            auto* lcs2 = lcs_nodes[1];

            // Find shared dummy index
            int dummy = -1;
            int p1 = -1, p2 = -1;
            for (int i = 0; i < 3 && dummy < 0; ++i)
                for (int j = 0; j < 3 && dummy < 0; ++j)
                    if (lcs1->ids()[i] == lcs2->ids()[j])
                    {
                        dummy = lcs1->ids()[i];
                        p1 = i;
                        p2 = j;
                    }

            if (dummy < 0)
                return e;

            // Free indices from each ε (cyclic positions after removing dummy)
            int a1 = lcs1->ids()[(p1 + 1) % 3];
            int a2 = lcs1->ids()[(p1 + 2) % 3];
            int b1 = lcs2->ids()[(p2 + 1) % 3];
            int b2 = lcs2->ids()[(p2 + 2) % 3];

            // Σ_s ε_{..s..} ε_{..s..} = δ_{a1,b1}δ_{a2,b2} - δ_{a1,b2}δ_{a2,b1}
            auto* kd_a1b1 = make_kronecker_delta(rl, a1, b1);
            auto* kd_a2b2 = make_kronecker_delta(rl, a2, b2);
            auto* kd_a1b2 = make_kronecker_delta(rl, a1, b2);
            auto* kd_a2b1 = make_kronecker_delta(rl, a2, b1);

            auto* term1_kd = make_product(rl, kd_a1b1, kd_a2b2);
            auto* term2_kd = make_product(rl, kd_a1b2, kd_a2b1);
            auto* one = make_rational(rl, Rational{1});

            // vA: replace lcs1 → term1_kd, lcs2 → 1
            Expr* vA = replace_in_tree(rl, e, lcs1, term1_kd);
            vA = replace_in_tree(rl, vA, lcs2, one);

            // vB: replace lcs1 → term2_kd, lcs2 → 1
            Expr* vB = replace_in_tree(rl, e, lcs1, term2_kd);
            vB = replace_in_tree(rl, vB, lcs2, one);

            return make_sum(rl, {vA, make_scale(rl, Rational{-1}, vB)});
        }};
}

} // namespace tender
