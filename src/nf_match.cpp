#include <tender/nf_match.hpp>

#include <mpk/mix/util/overloads.hpp>
#include <tender/expr.hpp>         // TensorObject, SlotBinding
#include <tender/tensor_order.hpp> // tensor_object_cmp

#include <algorithm>
#include <functional>
#include <map>
#include <set>

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
    // basis_id is a don't-care when the pattern leaves it unset (0): a
    // basis-generic identity matches a basis-tagged term (vibe 000067).  A
    // pattern that pins a basis must match it exactly.
    if (pat.slot.basis_id != 0 && pat.slot.basis_id != tgt.slot.basis_id)
        return false;
    if (pat.index.has_value() != tgt.index.has_value())
        return false;
    if (!pat.index)
        return true;
    return match_index(*pat.index, *tgt.index, bnd);
}

// ---- structural rank ----------------------------------------------------

// The invariant rank of a factor, inferred structurally (no Context, so the
// matcher need not thread one): mirrors the Expr-side `infer_rank` arithmetic —
// Dot subtracts 2, a double-dot subtracts 4, an n-way Cross subtracts n-1, a
// `⊗` (term juxtaposition) adds ranks.  Returns nullopt when a leaf rank is
// unknown, so callers can stay permissive on undeclared tensors.
auto nf_rank(Nf const* nf) -> std::optional<int>;

auto factor_rank(Factor const* f) -> std::optional<int>
{
    return visit(
        Overloads{
            [](Atom const& a) -> std::optional<int> { return a.obj.rank; },
            [](Contraction const& c) -> std::optional<int>
            {
                int r = 0;
                for (auto const* g: c.factors)
                {
                    auto gr = factor_rank(g);
                    if (!gr)
                        return std::nullopt;
                    r += *gr;
                }
                for (COp op: c.ops)
                    r -= (op == COp::Dot) ? 2 : 4;
                return r;
            },
            [](Cross const& c) -> std::optional<int>
            {
                int r = 0;
                for (auto const* g: c.factors)
                {
                    auto gr = factor_rank(g);
                    if (!gr)
                        return std::nullopt;
                    r += *gr;
                }
                return r - (static_cast<int>(c.factors.size()) - 1);
            },
            [](Paren const& p) -> std::optional<int>
            { return nf_rank(p.body); },
            [](Unary const& u) -> std::optional<int>
            {
                switch (u.op)
                {
                    case UnaryOp::Trace: return 0;
                    case UnaryOp::VectorInvariant: return 1;
                    case UnaryOp::Transpose: return factor_rank(u.operand);
                }
                return std::nullopt;
            },
            [](Div const& d) -> std::optional<int> { return nf_rank(d.num); },
            // Scalar fields are rank 0.
            [](ScalarFn const&) -> std::optional<int> { return 0; },
            [](Pow const&) -> std::optional<int> { return 0; },
            // A ∂ operator's rank is that of its wrt-object (vibe 000077).
            [](Deriv const& d) -> std::optional<int> { return d.wrt.rank; },
            // ∇ is a rank-1 vector operator (vibe 000078).
            [](Nabla const&) -> std::optional<int> { return 1; }},
        *f);
}

auto nf_rank(Nf const* nf) -> std::optional<int>
{
    // Every term shares the same rank; read it off the first.
    if (nf->terms.empty())
        return std::nullopt;
    auto const& t = nf->terms.front();
    int r = 0;
    for (auto const* g: t.scalars)
    {
        auto gr = factor_rank(g);
        if (!gr)
            return std::nullopt;
        r += *gr;
    }
    for (auto const* g: t.tensors)
    {
        auto gr = factor_rank(g);
        if (!gr)
            return std::nullopt;
        r += *gr;
    }
    return r;
}

// ---- subtree variables ------------------------------------------------------

// A slot-less, non-well-known named tensor in the LHS is a subtree variable: it
// matches any whole target factor.  Well-known tensors (I, δ, ε), slotted
// tensors, and *constrained symbols* stay literal.
//
// The last of those is what lets a declared constraint be stated as a rule at
// all (vibe 000110 I4).  `P·Pᵀ → I` written with an ordinary abstract P would
// be read as "for any X, X·Xᵀ = I" — measured, and it proved the orthogonality
// of every tensor in sight.  A declared symbol is a specific object, so the
// pattern must mean *that* symbol.
auto is_subtree_var(TensorObject const& t) -> bool
{
    return t.slots.empty()
           && !(t.traits && (t.traits->well_known || t.traits->constraint));
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

// Whether a *binary* contraction of these two factors is commutative, so canon
// may have put its operands in name order.  Mirrors `nf_lower`'s
// `contraction_commutes` exactly, at the `Factor` level: `·` commutes only
// between two rank-1 vectors, `:` / `··` only between two rank-2 tensors (a
// directional `C:ε` keeps its order).  Kept in lockstep with that function —
// if one gains a case, so must the other.
auto contraction_commutes_factors(COp op, Factor const* l, Factor const* r)
    -> bool
{
    auto const lr = factor_rank(l);
    auto const rr = factor_rank(r);
    if (!lr || !rr)
        return false;
    switch (op)
    {
        case COp::Dot: return *lr == 1 && *rr == 1;
        case COp::DDot:
        case COp::DDotAlt: return *lr == 2 && *rr == 2;
    }
    return false;
}

auto match_factor(Factor const* pat, Factor const* tgt, NfBinding& bnd) -> bool
{
    return visit(
        Overloads{
            [&](Atom const& p) -> bool
            {
                // Subtree variable: bind the whole target factor, consistently.
                if (is_subtree_var(p.obj))
                {
                    // Rank-gate the bind: a variable with a declared rank must
                    // not capture a target factor of a different (known) rank.
                    // This stops the rank-1 vars of bac-cab (a×(b×c)) from
                    // binding the rank-2 identity in the fenced chain a×(I×b),
                    // where the triple-product expansion is invalid (000059).
                    // Unknown ranks stay permissive.
                    if (p.obj.rank)
                        if (auto tr = factor_rank(tgt);
                            tr && *tr != *p.obj.rank)
                            return false;
                    return try_bind_subtree(bnd, p.obj.name.v.view(), tgt);
                }
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
                auto positional = [&](NfBinding& b)
                {
                    for (std::size_t i = 0; i < p.factors.size(); ++i)
                        if (!match_factor(p.factors[i], t->factors[i], b))
                            return false;
                    return true;
                };
                {
                    NfBinding trial = bnd;
                    if (positional(trial))
                    {
                        bnd = std::move(trial);
                        return true;
                    }
                }
                // Symmetric chains are AC-matched (vibe 000096 increment 3).
                // Canon puts a *commutative* binary contraction's operands in
                // name order, and a pattern variable's name sorts arbitrarily
                // among them — so a positional-only match makes the rule fire
                // on some targets and silently miss others purely by naming
                // (measured: `X··I = tr X` matched targets named A–H and not
                // J–Z).  Trying the swapped order removes that dependence.
                // Restricted to exactly the shape canon may reorder, so a
                // directional contraction (`A·b`, `C:ε`) is never mismatched.
                if (!contraction_commutes_factors(
                        t->ops.front(), t->factors[0], t->factors[1]))
                    return false;
                NfBinding swapped = bnd;
                if (!match_factor(p.factors[0], t->factors[1], swapped))
                    return false;
                if (!match_factor(p.factors[1], t->factors[0], swapped))
                    return false;
                bnd = std::move(swapped);
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
                return t && p.kind == t->kind && match_nf(p.body, t->body, bnd);
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
            [&](ScalarFn const& p) -> bool
            {
                auto const* t = std::get_if<ScalarFn>(&tgt->node);
                return t && p.kind == t->kind
                       && match_nf(p.operand, t->operand, bnd);
            },
            [&](Pow const& p) -> bool
            {
                auto const* t = std::get_if<Pow>(&tgt->node);
                return t && match_nf(p.base, t->base, bnd)
                       && match_nf(p.exponent, t->exponent, bnd);
            },
            [&](Deriv const& p) -> bool
            {
                auto const* t = std::get_if<Deriv>(&tgt->node);
                return t && tensor_object_cmp(p.wrt, t->wrt) == 0;
            },
            [&](Nabla const&) -> bool
            { return std::holds_alternative<Nabla>(tgt->node); },
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

// ---- index occurrence (for partial-match soundness) ------------------------

void collect_nf_ids(Nf const* nf, std::set<int>& out);

void collect_factor_ids(Factor const* f, std::set<int>& out)
{
    visit(
        Overloads{
            [&](Atom const& a)
            {
                for (auto const& s: a.obj.slots)
                    if (s.index)
                        if (auto const* ci =
                                std::get_if<CountableIndex>(&*s.index))
                            out.insert(ci->id);
            },
            [&](Contraction const& c)
            {
                for (auto const* x: c.factors)
                    collect_factor_ids(x, out);
            },
            [&](Cross const& c)
            {
                for (auto const* x: c.factors)
                    collect_factor_ids(x, out);
            },
            [&](Paren const& p) { collect_nf_ids(p.body, out); },
            [&](Unary const& u) { collect_factor_ids(u.operand, out); },
            [&](Div const& d)
            {
                collect_nf_ids(d.num, out);
                collect_nf_ids(d.den, out);
            },
            [&](ScalarFn const& s) { collect_nf_ids(s.operand, out); },
            [&](Pow const& p)
            {
                collect_nf_ids(p.base, out);
                collect_nf_ids(p.exponent, out);
            },
            [&](Deriv const& d)
            {
                for (auto const& s: d.wrt.slots)
                    if (s.index)
                        if (auto const* ci =
                                std::get_if<CountableIndex>(&*s.index))
                            out.insert(ci->id);
            },
            [&](Nabla const&) {},
        },
        *f);
}

void collect_nf_ids(Nf const* nf, std::set<int>& out)
{
    if (!nf)
        return;
    for (auto const& t: nf->terms)
    {
        for (auto const& b: t.bound)
            out.insert(b.index.id);
        for (auto const* f: t.scalars)
            collect_factor_ids(f, out);
        for (auto const* f: t.tensors)
            collect_factor_ids(f, out);
    }
}

auto factor_ids(std::vector<Factor const*> const& fs) -> std::set<int>
{
    std::set<int> out;
    for (auto const* f: fs)
        collect_factor_ids(f, out);
    return out;
}

} // namespace

auto match_term_partial(Term const& pat, Term const& tgt)
    -> std::optional<PartialMatch>
{
    if (pat.coeff == Rational{0})
        return std::nullopt;
    if (pat.tensors.size() > tgt.tensors.size()
        || pat.scalars.size() > tgt.scalars.size()
        || pat.bound.size() > tgt.bound.size())
        return std::nullopt;

    std::set<int> pat_bound_ids;
    for (auto const& b: pat.bound)
        pat_bound_ids.insert(b.index.id);

    std::size_t const ptn = pat.tensors.size();
    std::size_t const ttn = tgt.tensors.size();

    // Reconcile the bound indices and assemble the leftover, given a completed
    // factor binding (`bnd`), the chosen tensor offset `k`, and which target
    // scalars the pattern consumed (`used`).  Returns the `PartialMatch` if the
    // assignment is sound, else nullopt (caller backtracks).
    auto finalize = [&](std::size_t k,
                        std::vector<bool> const& used,
                        NfBinding const& bnd) -> std::optional<PartialMatch>
    {
        std::vector<bool> consumed(tgt.bound.size(), false);
        std::set<int> consumed_ids;
        for (auto const& pb: pat.bound)
        {
            auto a = bnd.find(pb.index.id);
            if (!a)
                return std::nullopt;
            auto const* ci = std::get_if<CountableIndex>(&*a);
            if (!ci)
                return std::nullopt;
            bool found = false;
            for (std::size_t j = 0; j < tgt.bound.size(); ++j)
            {
                if (tgt.bound[j].index.id != ci->id)
                    continue;
                if (consumed[j] || tgt.bound[j].mode != pb.mode)
                    return std::nullopt;
                consumed[j] = true;
                consumed_ids.insert(ci->id);
                found = true;
                break;
            }
            if (!found)
                return std::nullopt;
        }

        // A pattern *free* index must not capture a consumed target dummy — the
        // RHS reintroduces it, so it must survive.
        for (auto const& [pid, target]: bnd.indices)
        {
            if (pat_bound_ids.count(pid))
                continue;
            if (auto const* ci = std::get_if<CountableIndex>(&target))
                if (consumed_ids.count(ci->id))
                    return std::nullopt;
        }

        PartialMatch pm;
        pm.binding = bnd;
        pm.tensor_at = k;
        pm.leftover.coeff = tgt.coeff / pat.coeff;
        for (std::size_t j = 0; j < tgt.scalars.size(); ++j)
            if (!used[j])
                pm.leftover.scalars.push_back(tgt.scalars[j]);
        for (std::size_t j = 0; j < ttn; ++j)
            if (j < k || j >= k + ptn)
                pm.leftover.tensors.push_back(tgt.tensors[j]);
        for (std::size_t j = 0; j < tgt.bound.size(); ++j)
            if (!consumed[j])
                pm.leftover.bound.push_back(tgt.bound[j]);

        // A consumed dummy that still occurs in a leftover factor would mean we
        // tore apart a live contraction — reject.
        std::set<int> left_ids = factor_ids(pm.leftover.scalars);
        for (int id: factor_ids(pm.leftover.tensors))
            left_ids.insert(id);
        for (int id: consumed_ids)
            if (left_ids.count(id))
                return std::nullopt;

        return pm;
    };

    for (std::size_t k = 0; k + ptn <= ttn; ++k)
    {
        NfBinding b0;
        bool ok = true;
        for (std::size_t i = 0; i < ptn && ok; ++i)
            if (!match_factor(pat.tensors[i], tgt.tensors[k + i], b0))
                ok = false;
        if (!ok)
            continue;

        std::vector<bool> used(tgt.scalars.size(), false);
        std::optional<PartialMatch> result;
        std::function<bool(std::size_t, NfBinding)> rec =
            [&](std::size_t i, NfBinding acc) -> bool
        {
            if (i == pat.scalars.size())
            {
                result = finalize(k, used, acc);
                return result.has_value();
            }
            for (std::size_t j = 0; j < tgt.scalars.size(); ++j)
            {
                if (used[j])
                    continue;
                NfBinding trial = acc;
                if (match_factor(pat.scalars[i], tgt.scalars[j], trial))
                {
                    used[j] = true;
                    if (rec(i + 1, std::move(trial)))
                        return true;
                    used[j] = false;
                }
            }
            return false;
        };
        if (rec(0, b0))
            return result;
    }
    return std::nullopt;
}

// ---- instantiation ---------------------------------------------------------

namespace
{

using FreshMap = std::map<int, CountableIndex>;

// Map one index assoc under the binding (pattern free index → target) and the
// per-term freshening of RHS bound dummies.
auto remap_assoc(IndexAssoc a, NfBinding const& bnd, FreshMap const& fresh)
    -> IndexAssoc
{
    if (auto const* ci = std::get_if<CountableIndex>(&a))
    {
        if (auto it = fresh.find(ci->id); it != fresh.end())
            return IndexAssoc{it->second};
        if (auto t = bnd.find(ci->id))
            return *t;
    }
    return a;
}

auto inst_nf(Context& ctx, Nf const* nf, NfBinding const& bnd) -> Nf const*;

auto inst_factor(
    Context& ctx,
    Factor const* f,
    NfBinding const& bnd,
    FreshMap const& fresh) -> Factor const*
{
    return visit(
        Overloads{
            [&](Atom const& a) -> Factor const*
            {
                if (is_subtree_var(a.obj))
                    if (auto const* sub = bnd.find_subtree(a.obj.name.v.view()))
                        return sub;
                auto slots = a.obj.slots;
                for (auto& s: slots)
                    if (s.index)
                        s.index = remap_assoc(*s.index, bnd, fresh);
                // Copy the whole object and only replace the slots, so its
                // dimension-awareness (`dim`, vibe 000082) and applied ∂-marks
                // survive instantiation — a positional rebuild dropped them.
                TensorObject obj = a.obj;
                obj.slots = std::move(slots);
                return make_atom(ctx, std::move(obj));
            },
            [&](Contraction const& c) -> Factor const*
            {
                std::vector<Factor const*> fs;
                fs.reserve(c.factors.size());
                for (auto const* x: c.factors)
                    fs.push_back(inst_factor(ctx, x, bnd, fresh));
                return make_contraction(ctx, std::move(fs), c.ops);
            },
            [&](Cross const& c) -> Factor const*
            {
                std::vector<Factor const*> fs;
                fs.reserve(c.factors.size());
                for (auto const* x: c.factors)
                    fs.push_back(inst_factor(ctx, x, bnd, fresh));
                return make_cross(ctx, std::move(fs));
            },
            [&](Paren const& p) -> Factor const*
            { return make_paren(ctx, inst_nf(ctx, p.body, bnd), p.kind); },
            [&](Unary const& u) -> Factor const* {
                return make_unary(
                    ctx, u.op, inst_factor(ctx, u.operand, bnd, fresh));
            },
            [&](Div const& d) -> Factor const* {
                return make_div(
                    ctx, inst_nf(ctx, d.num, bnd), inst_nf(ctx, d.den, bnd));
            },
            [&](ScalarFn const& s) -> Factor const* {
                return make_scalar_fn(ctx, s.kind, inst_nf(ctx, s.operand, bnd));
            },
            [&](Pow const& p) -> Factor const*
            {
                return make_pow(
                    ctx,
                    inst_nf(ctx, p.base, bnd),
                    inst_nf(ctx, p.exponent, bnd));
            },
            [&](Deriv const& d) -> Factor const*
            {
                auto slots = d.wrt.slots;
                for (auto& s: slots)
                    if (s.index)
                        s.index = remap_assoc(*s.index, bnd, fresh);
                return make_deriv(
                    ctx,
                    TensorObject{
                        d.wrt.name, d.wrt.rank, d.wrt.traits, std::move(slots)});
            },
            [&](Nabla const&) -> Factor const* { return nf::make_nabla(ctx); },
        },
        *f);
}

auto inst_nf(Context& ctx, Nf const* nf, NfBinding const& bnd) -> Nf const*
{
    if (!nf)
        return nf;
    std::vector<Term> out;
    out.reserve(nf->terms.size());
    for (auto const& t: nf->terms)
    {
        // Freshen this term's bound dummies so the spliced result cannot
        // collide with a leftover term's surviving dummies.
        FreshMap fresh;
        Term nt;
        nt.coeff = t.coeff;
        for (auto const& b: t.bound)
        {
            CountableIndex fid{ctx.alloc_index_id()};
            fresh.emplace(b.index.id, fid);
            nt.bound.push_back({fid, b.mode, b.range});
        }
        for (auto const* f: t.scalars)
            nt.scalars.push_back(inst_factor(ctx, f, bnd, fresh));
        for (auto const* f: t.tensors)
            nt.tensors.push_back(inst_factor(ctx, f, bnd, fresh));
        out.push_back(std::move(nt));
    }
    return make_nf(ctx, std::move(out));
}

} // namespace

auto instantiate_nf(Context& ctx, Nf const* rhs, NfBinding const& bnd)
    -> Nf const*
{
    return inst_nf(ctx, rhs, bnd);
}

// ---- sub-chain rewrite -----------------------------------------------------

namespace
{

// A view of a `Contraction` / `Cross` factor as a chain: its factor sequence
// and (for a contraction) its join ops.  `ops` is null for a cross (all `×`).
struct ChainView final
{
    bool is_cross;
    std::vector<Factor const*> const* factors;
    std::vector<COp> const* ops;
};

auto as_chain(Factor const* f) -> std::optional<ChainView>
{
    if (auto const* c = std::get_if<Cross>(&f->node))
        return ChainView{true, &c->factors, nullptr};
    if (auto const* c = std::get_if<Contraction>(&f->node))
        return ChainView{false, &c->factors, &c->ops};
    return std::nullopt;
}

// A rule side qualifies as a chain rule iff it is a single term with no
// scalars, no bound indices, and exactly one tensor that is a `Contraction` /
// `Cross`.
auto chain_rule_side(Term const& t) -> std::optional<ChainView>
{
    if (!t.scalars.empty() || !t.bound.empty() || t.tensors.size() != 1)
        return std::nullopt;
    return as_chain(t.tensors.front());
}

// The single tensor factor of a rule side, when it is not a chain: a term with
// no scalars, no bound indices, and one tensor that is an atom, a unary, a
// paren…  Such a pattern still has to reach *inside* a chain — `(A·B)ᵀ → Bᵀ·Aᵀ`
// must fire on the second factor of `(P·Q)·(P·Q)ᵀ` — which is the mirror of the
// replacement case above and was missing for the same reason (vibe 000110 I4b).
//
// A bare subtree variable is excluded: `U → …` as a pattern matches every
// factor everywhere, which is not a rule anyone means to write.
auto lone_factor_rule_side(Term const& t) -> Factor const*
{
    if (!t.scalars.empty() || !t.bound.empty() || t.tensors.size() != 1)
        return nullptr;
    auto const* f = t.tensors.front();
    if (as_chain(f))
        return nullptr;
    if (auto const* a = std::get_if<Atom>(&f->node);
        a && is_subtree_var(a->obj))
        return nullptr;
    return f;
}

// A resolved chain rule: the pattern and replacement factor/op sequences (the
// op vectors are null for a cross).
struct ChainRule final
{
    bool is_cross;
    std::vector<Factor const*> const* pat;
    std::vector<COp> const* pat_ops;
    std::vector<Factor const*> const* rep;
    std::vector<COp> const* rep_ops;
};

// Splice the instantiated replacement chain into `C` at the run `[k, k+pn)`,
// preserving the boundary ops that joined the run to the rest of the chain
// (`CO` is the original ops, null for a cross).
auto splice_chain(
    Context& ctx,
    ChainRule const& cr,
    std::vector<Factor const*> const& C,
    std::vector<COp> const* CO,
    std::size_t k,
    NfBinding const& b) -> Factor const*
{
    std::size_t const pn = cr.pat->size();
    std::vector<Factor const*> fs;
    fs.insert(fs.end(), C.begin(), C.begin() + static_cast<std::ptrdiff_t>(k));
    for (auto const* rf: *cr.rep)
        fs.push_back(inst_factor(ctx, rf, b, {}));
    fs.insert(
        fs.end(), C.begin() + static_cast<std::ptrdiff_t>(k + pn), C.end());
    if (cr.is_cross)
        return make_cross(ctx, std::move(fs));

    std::vector<COp> ops;
    ops.insert(
        ops.end(), CO->begin(), CO->begin() + static_cast<std::ptrdiff_t>(k));
    if (cr.rep_ops)
        ops.insert(ops.end(), cr.rep_ops->begin(), cr.rep_ops->end());
    ops.insert(
        ops.end(),
        CO->begin() + static_cast<std::ptrdiff_t>(k + pn - 1),
        CO->end());
    return make_contraction(ctx, std::move(fs), std::move(ops));
}

// Rewrite the first sub-chain match within factor `f`, recursing into nested
// chain factors: the encapsulation keeps a `Cross`/`Contraction` chain as
// nested binary factors, so the `I × b` run hides one level down inside `a × (I
// × b)`. Returns the rewritten factor, or nullptr if nothing matched.
auto rewrite_in_factor(Context& ctx, ChainRule const& cr, Factor const* f)
    -> Factor const*
{
    std::size_t const pn = cr.pat->size();

    // Match the pattern as a contiguous run at this chain level.
    if (auto cv = as_chain(f); cv && cv->is_cross == cr.is_cross)
    {
        auto const& C = *cv->factors;
        for (std::size_t k = 0; pn <= C.size() && k + pn <= C.size(); ++k)
        {
            NfBinding b;
            bool ok = true;
            for (std::size_t i = 0; i < pn && ok; ++i)
                if (!match_factor((*cr.pat)[i], C[k + i], b))
                    ok = false;
            if (ok && !cr.is_cross)
                for (std::size_t i = 0; i + 1 < pn && ok; ++i)
                    if ((*cr.pat_ops)[i] != (*cv->ops)[k + i])
                        ok = false;
            if (ok)
                return splice_chain(ctx, cr, C, cv->ops, k, b);
        }
    }

    // Otherwise recurse into the factor's children, rewriting the first hit.
    return visit(
        Overloads{
            [&](Atom const&) -> Factor const* { return nullptr; },
            [&](Contraction const& c) -> Factor const*
            {
                for (std::size_t i = 0; i < c.factors.size(); ++i)
                    if (auto const* nw =
                            rewrite_in_factor(ctx, cr, c.factors[i]))
                    {
                        auto fs = c.factors;
                        fs[i] = nw;
                        return make_contraction(ctx, std::move(fs), c.ops);
                    }
                return nullptr;
            },
            [&](Cross const& c) -> Factor const*
            {
                for (std::size_t i = 0; i < c.factors.size(); ++i)
                    if (auto const* nw =
                            rewrite_in_factor(ctx, cr, c.factors[i]))
                    {
                        auto fs = c.factors;
                        fs[i] = nw;
                        return make_cross(ctx, std::move(fs));
                    }
                return nullptr;
            },
            [&](Unary const& u) -> Factor const*
            {
                if (auto const* nw = rewrite_in_factor(ctx, cr, u.operand))
                    return make_unary(ctx, u.op, nw);
                return nullptr;
            },
            [&](Paren const&) -> Factor const* { return nullptr; },
            [&](Div const&) -> Factor const* { return nullptr; },
            [&](ScalarFn const&) -> Factor const* { return nullptr; },
            [&](Pow const&) -> Factor const* { return nullptr; },
            [&](Deriv const&) -> Factor const* { return nullptr; },
            [&](Nabla const&) -> Factor const* { return nullptr; },
        },
        *f);
}

} // namespace

auto rewrite_subchain(
    Context& ctx,
    Term const& lhs_term,
    Nf const* rhs,
    Term const& tgt) -> std::optional<Term>
{
    std::vector<Factor const*> pat_storage;
    std::vector<COp> const* pat_ops = nullptr;
    bool pat_is_cross = false;
    if (auto pat = chain_rule_side(lhs_term))
    {
        pat_storage = *pat->factors;
        pat_ops = pat->ops;
        pat_is_cross = pat->is_cross;
    }
    else if (auto const* lone = lone_factor_rule_side(lhs_term))
        pat_storage = {lone};
    else
        return std::nullopt;
    if (!rhs || rhs->terms.size() != 1)
        return std::nullopt;
    Term const& rhs_term = rhs->terms.front();

    // The replacement, in one of two shapes (vibe 000110 I4b).
    //
    // A chain of the *same* kind splices inline, its factors and internal ops
    // joining the surrounding chain — the original case.  Anything else that is
    // a single tensor factor splices as one opaque factor: `A·B → I` replaces
    // the run by the atom I, and a cross-valued replacement drops whole into a
    // contraction chain.
    //
    // That second shape is what was missing, and it is the common one: a rule
    // whose right-hand side is a single symbol could not fire inside a longer
    // chain at all, so `A·B → I` did nothing to `a·A·B·b` — vibe 000100's
    // context-blocking, measured again in vibe 000110 I4 and blocking the
    // rotation arc.
    //
    // Still not handled: a replacement with *no* tensor factor (`n·n → 1`),
    // which would shorten the chain rather than substitute within it and needs
    // a boundary op dropped — and, when the run is the whole chain, the factor
    // to disappear into the term's coefficient.  No consumer yet.
    std::vector<Factor const*> rep_storage;
    std::vector<COp> const* rep_ops = nullptr;
    // (The chain-inlining branch is for a chain *pattern* only: a lone-factor
    // pattern replaces one element, and dropping a chain in as a single factor
    // there is both correct and simpler than splicing its ops into the host —
    // canon flattens the nesting on the next pass anyway.)
    if (auto rep = chain_rule_side(rhs_term);
        rep && rep->is_cross == pat_is_cross && pat_storage.size() > 1)
    {
        rep_storage = *rep->factors;
        rep_ops = rep->ops;
    }
    else if (
        rhs_term.scalars.empty() && rhs_term.bound.empty()
        && rhs_term.tensors.size() == 1)
    {
        rep_storage = {rhs_term.tensors.front()};
    }
    else
        return std::nullopt;

    ChainRule const cr{
        pat_is_cross, &pat_storage, pat_ops, &rep_storage, rep_ops};

    auto try_region = [&](std::vector<Factor const*> const& region,
                          std::size_t& hit) -> Factor const*
    {
        for (std::size_t i = 0; i < region.size(); ++i)
            if (auto const* nw = rewrite_in_factor(ctx, cr, region[i]))
            {
                hit = i;
                return nw;
            }
        return nullptr;
    };

    std::size_t hit = 0;
    if (auto const* nw = try_region(tgt.tensors, hit))
    {
        Term out = tgt;
        out.coeff = out.coeff * rhs_term.coeff;
        out.tensors[hit] = nw;
        return out;
    }
    if (auto const* nw = try_region(tgt.scalars, hit))
    {
        Term out = tgt;
        out.coeff = out.coeff * rhs_term.coeff;
        out.scalars[hit] = nw;
        return out;
    }
    return std::nullopt;
}

auto fire_identity_on_term(
    Context& ctx,
    Term const& lhs_term,
    Nf const* rhs,
    Term const& tgt) -> std::optional<std::vector<Term>>
{
    // Sub-product path: the LHS matches a sub-multiset of the target's factors;
    // splice the instantiated RHS into the leftover where the matched run sat
    // (⊗ is non-commutative), one result term per RHS term.
    if (auto pm = match_term_partial(lhs_term, tgt))
    {
        auto const* rhs_inst = instantiate_nf(ctx, rhs, pm->binding);
        Term const& L = pm->leftover;
        auto const pos = static_cast<std::ptrdiff_t>(
            std::min(pm->tensor_at, L.tensors.size()));
        std::vector<Term> out;
        out.reserve(rhs_inst->terms.size());
        for (auto const& r: rhs_inst->terms)
        {
            Term m;
            m.coeff = r.coeff * L.coeff;
            m.bound = L.bound;
            m.bound.insert(m.bound.end(), r.bound.begin(), r.bound.end());
            m.scalars = L.scalars;
            m.scalars.insert(
                m.scalars.end(), r.scalars.begin(), r.scalars.end());
            m.tensors.insert(
                m.tensors.end(), L.tensors.begin(), L.tensors.begin() + pos);
            m.tensors.insert(
                m.tensors.end(), r.tensors.begin(), r.tensors.end());
            m.tensors.insert(
                m.tensors.end(), L.tensors.begin() + pos, L.tensors.end());
            out.push_back(std::move(m));
        }
        return out;
    }

    // Chain path: a single Contraction/Cross-factor LHS rewrites a contiguous
    // sub-run inside one of this term's chain factors.
    if (auto rt = rewrite_subchain(ctx, lhs_term, rhs, tgt))
        return std::vector<Term>{std::move(*rt)};

    return std::nullopt;
}

} // namespace tender::nf
