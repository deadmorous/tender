#include <tender/nf_match.hpp>

#include <mpk/mix/util/overloads.hpp>
#include <tender/expr.hpp> // TensorObject, SlotBinding

#include <functional>

using namespace mpk::mix;

namespace tender::nf
{

auto NfBinding::find(int id) const -> std::optional<IndexAssoc>
{
    for (auto const& [pid, target]: indices)
        if (pid == id)
            return target;
    return std::nullopt;
}

auto NfBinding::find_subtree(std::string_view name) const -> Factor const*
{
    for (auto const& [n, target]: subtrees)
        if (n == name)
            return target;
    return nullptr;
}

namespace
{

// ---- index-level matching (mirrors identity.cpp, over NfBinding) -----------

auto assoc_eq(IndexAssoc const& a, IndexAssoc const& b) -> bool
{
    return visit(
        Overloads{
            [&](CountableIndex const& ca)
            {
                auto const* cb = std::get_if<CountableIndex>(&b);
                return cb && ca.id == cb->id;
            },
            [&](ConcreteIndex const& ca)
            {
                auto const* cb = std::get_if<ConcreteIndex>(&b);
                return cb && ca.value == cb->value;
            },
            [&](LabelIndex const& la)
            {
                auto const* lb = std::get_if<LabelIndex>(&b);
                return lb && la.name == lb->name;
            },
        },
        a);
}

auto try_bind(NfBinding& bnd, int id, IndexAssoc const& tgt) -> bool
{
    if (auto cur = bnd.find(id))
        return assoc_eq(*cur, tgt);
    bnd.indices.emplace_back(id, tgt);
    return true;
}

// A pattern CountableIndex is a variable (binds to anything); a Concrete/Label
// pattern index must equal the target exactly.
auto match_index(IndexAssoc const& pat, IndexAssoc const& tgt, NfBinding& bnd)
    -> bool
{
    if (auto const* ci = std::get_if<CountableIndex>(&pat))
        return try_bind(bnd, ci->id, tgt);
    return assoc_eq(pat, tgt);
}

auto match_slot(SlotBinding const& pat, SlotBinding const& tgt, NfBinding& bnd)
    -> bool
{
    if (pat.slot.level != tgt.slot.level)
        return false;
    if (pat.slot.realm != tgt.slot.realm)
        return false;
    if (pat.slot.space != tgt.slot.space)
        return false;
    if (pat.index.has_value() != tgt.index.has_value())
        return false;
    if (!pat.index)
        return true;
    return match_index(*pat.index, *tgt.index, bnd);
}

// ---- subtree variables ------------------------------------------------------

// A slot-less, non-well-known named tensor in the LHS is a subtree variable: it
// matches any whole target factor.  Well-known tensors (I, δ, ε) and slotted
// tensors stay literal.
auto is_subtree_var(TensorObject const& t) -> bool
{
    return t.slots.empty() && !(t.traits && t.traits->well_known);
}

auto try_bind_subtree(NfBinding& bnd, std::string_view name, Factor const* tgt)
    -> bool
{
    if (auto const* cur = bnd.find_subtree(name))
        return equal(cur, tgt);
    bnd.subtrees.emplace_back(std::string{name}, tgt);
    return true;
}

// ---- structural factor matching --------------------------------------------

auto match_nf(Nf const* pat, Nf const* tgt, NfBinding& bnd) -> bool;

// Match pattern factors against target factors modulo order (bounded AC
// backtracking): used for the commutative `scalars` region.
auto match_factors_ac(
    std::vector<Factor const*> const& pats,
    std::vector<Factor const*> const& tgts,
    NfBinding& bnd) -> bool
{
    if (pats.size() != tgts.size())
        return false;
    std::vector<bool> used(tgts.size(), false);
    auto rec = [&](auto&& self, std::size_t i, NfBinding acc) -> bool
    {
        if (i == pats.size())
        {
            bnd = std::move(acc);
            return true;
        }
        for (std::size_t j = 0; j < tgts.size(); ++j)
        {
            if (used[j])
                continue;
            NfBinding trial = acc;
            if (match_factor(pats[i], tgts[j], trial))
            {
                used[j] = true;
                if (self(self, i + 1, std::move(trial)))
                    return true;
                used[j] = false;
            }
        }
        return false;
    };
    return rec(rec, 0, bnd);
}

} // namespace

auto match_factor(Factor const* pat, Factor const* tgt, NfBinding& bnd) -> bool
{
    return visit(
        Overloads{
            [&](Atom const& p) -> bool
            {
                // Subtree variable: bind the whole target factor, consistently.
                if (is_subtree_var(p.obj))
                    return try_bind_subtree(bnd, p.obj.name.v.view(), tgt);
                auto const* t = std::get_if<Atom>(&tgt->node);
                if (!t)
                    return false;
                auto const& po = p.obj;
                auto const& to = t->obj;
                if (po.name != to.name || po.rank != to.rank
                    || po.slots.size() != to.slots.size())
                    return false;
                for (std::size_t i = 0; i < po.slots.size(); ++i)
                    if (!match_slot(po.slots[i], to.slots[i], bnd))
                        return false;
                return true;
            },
            [&](Contraction const& p) -> bool
            {
                auto const* t = std::get_if<Contraction>(&tgt->node);
                if (!t || p.factors.size() != t->factors.size()
                    || p.ops != t->ops)
                    return false;
                for (std::size_t i = 0; i < p.factors.size(); ++i)
                    if (!match_factor(p.factors[i], t->factors[i], bnd))
                        return false;
                return true;
            },
            [&](Cross const& p) -> bool
            {
                auto const* t = std::get_if<Cross>(&tgt->node);
                if (!t || p.factors.size() != t->factors.size())
                    return false;
                for (std::size_t i = 0; i < p.factors.size(); ++i)
                    if (!match_factor(p.factors[i], t->factors[i], bnd))
                        return false;
                return true;
            },
            [&](Paren const& p) -> bool
            {
                auto const* t = std::get_if<Paren>(&tgt->node);
                return t && match_nf(p.body, t->body, bnd);
            },
            [&](Unary const& p) -> bool
            {
                auto const* t = std::get_if<Unary>(&tgt->node);
                return t && p.op == t->op
                       && match_factor(p.operand, t->operand, bnd);
            },
            [&](Div const& p) -> bool
            {
                auto const* t = std::get_if<Div>(&tgt->node);
                return t && match_nf(p.num, t->num, bnd)
                       && match_nf(p.den, t->den, bnd);
            },
        },
        *pat);
}

auto match_term(Term const& pat, Term const& tgt, NfBinding& bnd) -> bool
{
    if (!(pat.coeff == tgt.coeff))
        return false;
    if (pat.bound.size() != tgt.bound.size())
        return false;
    for (std::size_t i = 0; i < pat.bound.size(); ++i)
    {
        if (pat.bound[i].mode != tgt.bound[i].mode)
            return false;
        if (!match_index(
                IndexAssoc{pat.bound[i].index},
                IndexAssoc{tgt.bound[i].index},
                bnd))
            return false;
        if ((pat.bound[i].range == nullptr) != (tgt.bound[i].range == nullptr))
            return false;
        if (pat.bound[i].range
            && !match_nf(pat.bound[i].range, tgt.bound[i].range, bnd))
            return false;
    }
    if (pat.tensors.size() != tgt.tensors.size())
        return false;
    for (std::size_t i = 0; i < pat.tensors.size(); ++i)
        if (!match_factor(pat.tensors[i], tgt.tensors[i], bnd))
            return false;
    return match_factors_ac(pat.scalars, tgt.scalars, bnd);
}

namespace
{

auto match_nf(Nf const* pat, Nf const* tgt, NfBinding& bnd) -> bool
{
    if (!pat || !tgt)
        return pat == tgt;
    if (pat->terms.size() != tgt->terms.size())
        return false;
    for (std::size_t i = 0; i < pat->terms.size(); ++i)
        if (!match_term(pat->terms[i], tgt->terms[i], bnd))
            return false;
    return true;
}

} // namespace

} // namespace tender::nf
