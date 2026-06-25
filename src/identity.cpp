#include <tender/identity.hpp>

#include <mpk/mix/util/overloads.hpp>
#include <tender/context.hpp>
#include <tender/derivation.hpp> // steps::canonicalize, is_component_valued, infer_rank, structural_eq
#include <tender/nf_lower.hpp> // canonicalize_nf, raise
#include <tender/nf_match.hpp> // match_term_partial, instantiate_nf
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

auto MatchBinding::find_subtree(std::string_view name) const -> Expr const*
{
    for (auto const& [n, target]: subtrees)
        if (n == name)
            return target;
    return nullptr;
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

// A slot-less, non-well-known named tensor in the LHS is a *subtree variable*:
// it matches any whole subtree.  Well-known tensors (I, δ, ε) and slotted
// tensors stay literal.
auto is_subtree_var(TensorObject const& t) -> bool
{
    return t.slots.empty() && !(t.traits && t.traits->well_known);
}

// Bind subtree variable `name` to target `tgt`, or require consistency
// (structural equality) with an existing binding.
auto try_bind_subtree(MatchBinding& bnd, std::string_view name, Expr const* tgt)
    -> bool
{
    if (auto const* cur = bnd.find_subtree(name))
        return structural_eq(cur, tgt);
    bnd.subtrees.emplace_back(std::string{name}, tgt);
    return true;
}

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
                // Subtree variable: bind the whole target subtree (rank-checked
                // when both ranks are known), consistently across the match.
                if (is_subtree_var(p))
                {
                    if (p.rank)
                        if (auto tr = infer_rank(tgt); tr && *tr != *p.rank)
                            return false;
                    return try_bind_subtree(bnd, p.name.v.view(), tgt);
                }
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
            [&](Trace const& p) -> bool
            {
                auto const* t = std::get_if<Trace>(&tgt->node);
                return t && match_node(p.operand, t->operand, bnd);
            },
            [&](VectorInvariant const& p) -> bool
            {
                auto const* t = std::get_if<VectorInvariant>(&tgt->node);
                return t && match_node(p.operand, t->operand, bnd);
            },
            [&](Transpose const& p) -> bool
            {
                auto const* t = std::get_if<Transpose>(&tgt->node);
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

auto match_into(Expr const* pattern, Expr const* target, MatchBinding& bnd)
    -> bool
{
    return match_node(pattern, target, bnd);
}

auto bind_pattern_index(
    MatchBinding& bnd, int pattern_id, IndexAssoc const& target) -> bool
{
    return try_bind(bnd, pattern_id, target);
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
                        // A slot-less subtree variable expands to its bound
                        // target subtree.
                        if (t.slots.empty())
                            if (auto const* sub =
                                    bnd.find_subtree(t.name.v.view()))
                                return sub;
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

// The original binary-tree matcher: canonicalize, walk the `Expr` bottom-up,
// and at the first subtree where the LHS matches, splice in the instantiated
// RHS. Still the path for sub-*chain* rewrites — an identity matching a
// contiguous run *inside* a flat `Contraction` / `Cross` factor (e.g. `I × x =
// x × I` on the `I × b` of `a × I × b`), which the flat `Nf` partial matcher
// does not yet reach.  apply_identity falls back here when the `Nf` path does
// not fire.
auto apply_identity_expr(Context& ctx, Expr const* e, Identity const& id)
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

    return done ? steps::implicitize(ctx, steps::canonicalize(ctx, rewritten)) :
                  e;
}

auto apply_identity(Context& ctx, Expr const* e, Identity const& id)
    -> Expr const*
{
    // Match on the flat normal form (vibe 000058 / C14): the canonical `Nf` of
    // an expression is `canonicalize_nf(canonicalize(·))` (the C12 round-trip).
    // The identity's LHS becomes a single pattern term whose factors are
    // matched as a *sub-product* of a target term — so an identity fires even
    // when its product sits among extra factors of a larger term, the gap the
    // binary-tree matcher could not reach.
    auto const* target = nf::canonicalize_nf(ctx, steps::canonicalize(ctx, e));
    auto const* lhs =
        nf::canonicalize_nf(ctx, steps::canonicalize(ctx, id.lhs));
    auto const* rhs =
        nf::canonicalize_nf(ctx, steps::canonicalize(ctx, id.rhs));

    // Only single-term LHS rules are matched as sub-products; a multi-term LHS
    // (matching a sub-sum) is handled by the binary-tree fallback.
    if (lhs->terms.size() != 1)
        return apply_identity_expr(ctx, e, id);
    nf::Term const& lhs_term = lhs->terms.front();

    // Fire at the first target term where the LHS partially matches, splice the
    // instantiated RHS into the leftover (the RHS tensor block goes back where
    // the matched run sat — ⊗ is non-commutative), and carry the rest through.
    bool fired = false;
    std::vector<nf::Term> out;
    for (auto const& tterm: target->terms)
    {
        if (!fired)
            if (auto rep = nf::fire_identity_on_term(ctx, lhs_term, rhs, tterm))
            {
                fired = true;
                for (auto& t: *rep)
                    out.push_back(std::move(t));
                continue;
            }
        out.push_back(tterm);
    }

    // With no flat-form match, fall back to the binary-tree matcher (a
    // multi-term LHS, or a match nested deeper than a term / chain factor).  On
    // a match the spliced `Nf` is raised, re-canonicalized (re-α-renaming the
    // freshened RHS dummies, merging like terms), and implicitized so the
    // explicit binders the normal form carries do not leak into the user's
    // implicit notation.
    if (!fired)
        return apply_identity_expr(ctx, e, id);
    auto const* result = nf::make_nf(ctx, std::move(out));
    return steps::implicitize(
        ctx, steps::canonicalize(ctx, nf::raise(ctx, *result)));
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
