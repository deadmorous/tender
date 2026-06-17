#include <tender/identity.hpp>

#include <mpk/mix/util/overloads.hpp>
#include <tender/context.hpp>
#include <tender/derivation.hpp> // steps::canonicalize, is_component_valued
#include <tender/rewrite.hpp>

#include <vector>

using namespace mpk::mix;

namespace tender
{

auto MatchBinding::find(int id) const -> std::optional<IndexAssoc>
{
    for (auto const& [pid, target]: indices)
        if (pid == id)
            return target;
    return std::nullopt;
}

namespace
{

// ---- index-level matching ----------------------------------------------

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

// Bind pattern index `id` to target index `tgt`, or require consistency with an
// existing binding.
auto try_bind(MatchBinding& bnd, int id, IndexAssoc const& tgt) -> bool
{
    if (auto cur = bnd.find(id))
        return assoc_eq(*cur, tgt);
    bnd.indices.emplace_back(id, tgt);
    return true;
}

// A pattern CountableIndex is a variable (binds to anything); a Concrete/Label
// pattern index must equal the target exactly.
auto match_index(
    IndexAssoc const& pat, IndexAssoc const& tgt, MatchBinding& bnd) -> bool
{
    if (auto const* ci = std::get_if<CountableIndex>(&pat))
        return try_bind(bnd, ci->id, tgt);
    return assoc_eq(pat, tgt);
}

auto match_slot(
    SlotBinding const& pat, SlotBinding const& tgt, MatchBinding& bnd) -> bool
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

// ---- structural matching with bounded AC backtracking ------------------

template <typename NodeT>
void flatten(Expr const* e, std::vector<Expr const*>& out)
{
    if (auto const* n = std::get_if<NodeT>(&e->node))
    {
        flatten<NodeT>(n->left, out);
        flatten<NodeT>(n->right, out);
    }
    else
        out.push_back(e);
}

auto match_node(Expr const* pat, Expr const* tgt, MatchBinding& bnd) -> bool;

// Match a multiset of pattern children against a multiset of target children,
// modulo order.  Bounded backtracking — the child lists are tiny after
// canonicalization, so the worst-case factorial cost is negligible.
auto match_commutative(
    std::vector<Expr const*> const& pats,
    std::vector<Expr const*> const& tgts,
    MatchBinding& bnd) -> bool
{
    if (pats.size() != tgts.size())
        return false;

    std::vector<bool> used(tgts.size(), false);
    auto rec = [&](auto&& self, std::size_t i, MatchBinding acc) -> bool
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
            MatchBinding trial = acc;
            if (match_node(pats[i], tgts[j], trial))
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

auto match_bin(
    Expr const* pl,
    Expr const* pr,
    Expr const* tl,
    Expr const* tr,
    MatchBinding& bnd) -> bool
{
    return match_node(pl, tl, bnd) && match_node(pr, tr, bnd);
}

auto match_node(Expr const* pat, Expr const* tgt, MatchBinding& bnd) -> bool
{
    return visit(
        Overloads{
            [&](TensorObject const& p) -> bool
            {
                auto const* t = std::get_if<TensorObject>(&tgt->node);
                if (!t)
                    return false;
                if (p.name != t->name || p.rank != t->rank
                    || p.slots.size() != t->slots.size())
                    return false;
                for (std::size_t i = 0; i < p.slots.size(); ++i)
                    if (!match_slot(p.slots[i], t->slots[i], bnd))
                        return false;
                return true;
            },
            [&](ScalarLiteral const& p) -> bool
            {
                auto const* t = std::get_if<ScalarLiteral>(&tgt->node);
                return t && p.value == t->value;
            },
            [&](Negate const& p) -> bool
            {
                auto const* t = std::get_if<Negate>(&tgt->node);
                return t && match_node(p.operand, t->operand, bnd);
            },
            [&](Sum const&) -> bool
            {
                if (!std::get_if<Sum>(&tgt->node))
                    return false;
                std::vector<Expr const*> ps, ts;
                flatten<Sum>(pat, ps);
                flatten<Sum>(tgt, ts);
                return match_commutative(ps, ts, bnd);
            },
            [&](TensorProduct const& p) -> bool
            {
                auto const* t = std::get_if<TensorProduct>(&tgt->node);
                if (!t)
                    return false;
                // Component products commute (match modulo factor order);
                // products with invariant factors are positional.
                if (is_component_valued(pat) && is_component_valued(tgt))
                {
                    std::vector<Expr const*> ps, ts;
                    flatten<TensorProduct>(pat, ps);
                    flatten<TensorProduct>(tgt, ts);
                    return match_commutative(ps, ts, bnd);
                }
                return match_bin(p.left, p.right, t->left, t->right, bnd);
            },
            [&](Difference const& p) -> bool
            {
                auto const* t = std::get_if<Difference>(&tgt->node);
                return t && match_bin(p.left, p.right, t->left, t->right, bnd);
            },
            [&](ScalarDiv const& p) -> bool
            {
                auto const* t = std::get_if<ScalarDiv>(&tgt->node);
                return t && match_bin(p.left, p.right, t->left, t->right, bnd);
            },
            [&](Dot const& p) -> bool
            {
                auto const* t = std::get_if<Dot>(&tgt->node);
                return t && match_bin(p.left, p.right, t->left, t->right, bnd);
            },
            [&](DDot const& p) -> bool
            {
                auto const* t = std::get_if<DDot>(&tgt->node);
                return t && match_bin(p.left, p.right, t->left, t->right, bnd);
            },
            [&](DDotAlt const& p) -> bool
            {
                auto const* t = std::get_if<DDotAlt>(&tgt->node);
                return t && match_bin(p.left, p.right, t->left, t->right, bnd);
            },
            [&](Cross const& p) -> bool
            {
                auto const* t = std::get_if<Cross>(&tgt->node);
                return t && match_bin(p.left, p.right, t->left, t->right, bnd);
            },
            [&](ExplicitSum const& p) -> bool
            {
                auto const* t = std::get_if<ExplicitSum>(&tgt->node);
                if (!t)
                    return false;
                if (!try_bind(bnd, p.index.id, CountableIndex{t->index.id}))
                    return false;
                if ((p.bound == nullptr) != (t->bound == nullptr))
                    return false;
                if (p.bound && !match_node(p.bound, t->bound, bnd))
                    return false;
                return match_node(p.body, t->body, bnd);
            },
            [&](NoSum const& p) -> bool
            {
                auto const* t = std::get_if<NoSum>(&tgt->node);
                if (!t)
                    return false;
                if (!try_bind(bnd, p.index.id, CountableIndex{t->index.id}))
                    return false;
                return match_node(p.body, t->body, bnd);
            },
        },
        *pat);
}

} // namespace

auto match(Expr const* pattern, Expr const* target)
    -> std::optional<MatchBinding>
{
    MatchBinding bnd;
    if (match_node(pattern, target, bnd))
        return bnd;
    return std::nullopt;
}

auto instantiate(Context& ctx, Expr const* rhs, MatchBinding const& bnd)
    -> Expr const*
{
    return rewrite_tree(
        ctx,
        rhs,
        [&bnd](Context& ctx, Expr const* e) -> Expr const*
        {
            return visit(
                Overloads{
                    [&](TensorObject const& t) -> Expr const*
                    {
                        auto slots = t.slots;
                        bool changed = false;
                        for (auto& sb: slots)
                        {
                            if (!sb.index)
                                continue;
                            auto const* ci =
                                std::get_if<CountableIndex>(&*sb.index);
                            if (!ci)
                                continue;
                            if (auto target = bnd.find(ci->id))
                            {
                                sb.index = *target;
                                changed = true;
                            }
                        }
                        if (!changed)
                            return e;
                        return ctx.make<Expr>(TensorObject{
                            t.name, t.rank, t.traits, std::move(slots)});
                    },
                    [&](ExplicitSum const& s) -> Expr const*
                    {
                        auto target = bnd.find(s.index.id);
                        if (!target)
                            return e;
                        auto const* ci = std::get_if<CountableIndex>(&*target);
                        if (!ci || ci->id == s.index.id)
                            return e;
                        return make_explicit_sum(ctx, *ci, s.body, s.bound);
                    },
                    [&](NoSum const& s) -> Expr const*
                    {
                        auto target = bnd.find(s.index.id);
                        if (!target)
                            return e;
                        auto const* ci = std::get_if<CountableIndex>(&*target);
                        if (!ci || ci->id == s.index.id)
                            return e;
                        return make_no_sum(ctx, *ci, s.body);
                    },
                    [&](auto const&) -> Expr const* { return e; },
                },
                *e);
        });
}

auto apply_identity(Context& ctx, Expr const* e, Identity const& id)
    -> Expr const*
{
    auto const* target = steps::canonicalize(ctx, e);
    auto const* lhs = steps::canonicalize(ctx, id.lhs);

    bool done = false;
    auto const* rewritten = rewrite_tree(
        ctx,
        target,
        [&](Context& ctx, Expr const* node) -> Expr const*
        {
            if (done)
                return node;
            if (auto bnd = match(lhs, node))
            {
                done = true;
                return instantiate(ctx, id.rhs, *bnd);
            }
            return node;
        });

    return done ? steps::canonicalize(ctx, rewritten) : target;
}

namespace steps
{

auto apply_identity(Identity id)
    -> std::function<Expr const*(Context&, Expr const*)>
{
    return [id = std::move(id)](Context& ctx, Expr const* e) -> Expr const*
    { return tender::apply_identity(ctx, e, id); };
}

} // namespace steps

} // namespace tender
