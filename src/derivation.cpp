#include <tender/derivation.hpp>

#include <mpk/mix/util/overloads.hpp>
#include <tender/rewrite.hpp>

#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

using namespace mpk::mix;

namespace tender
{
namespace
{

// Walk e looking for the IndexSpace of the CountableIndex with the given id.
// Returns nullptr if not found (e.g. the index appears in no concrete slot).
auto find_index_space(Expr const* e, int id) -> IndexSpace const*
{
    IndexSpace const* found = nullptr;
    std::function<void(Expr const*)> go = [&](Expr const* node)
    {
        if (found)
            return;
        visit(
            Overloads{
                [&](TensorObject const& t)
                {
                    for (auto const& sb: t.slots)
                    {
                        if (!sb.index || !sb.slot.space)
                            continue;
                        if (auto const* ci =
                                std::get_if<CountableIndex>(&*sb.index))
                            if (ci->id == id)
                            {
                                found = sb.slot.space;
                                return;
                            }
                    }
                },
                [&](ScalarLiteral const&) {},
                [&](Negate const& n) { go(n.operand); },
                [&](Trace const& u) { go(u.operand); },
                [&](VectorInvariant const& u) { go(u.operand); },
                [&](Transpose const& u) { go(u.operand); },
                [&](Sum const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](Difference const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](TensorProduct const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](ScalarDiv const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](Dot const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](DDot const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](DDotAlt const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](Cross const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](ExplicitSum const& s)
                {
                    go(s.body);
                    if (s.bound)
                        go(s.bound);
                },
                [&](NoSum const& s) { go(s.body); },
            },
            *node);
    };
    go(e);
    return found;
}

// Replace all CountableIndex references with id == idx_id with concrete val.
auto substitute(Context& ctx, Expr const* e, int idx_id, ConcreteIndex val)
    -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [idx_id, val](Context& ctx, Expr const* e) -> Expr const*
        {
            auto const* t = std::get_if<TensorObject>(&e->node);
            if (!t)
                return e;

            auto slots = t->slots;
            bool changed = false;
            for (auto& sb: slots)
            {
                if (!sb.index)
                    continue;
                if (auto const* ci = std::get_if<CountableIndex>(&*sb.index))
                    if (ci->id == idx_id)
                    {
                        sb.index = val;
                        changed = true;
                    }
            }
            if (!changed)
                return e;
            return ctx.make<Expr>(
                TensorObject{t->name, t->rank, t->traits, std::move(slots)});
        });
}

// Collect all addends of a left/right Sum tree into a flat vector.
void collect_addends(Expr const* e, std::vector<Expr const*>& out)
{
    if (auto const* s = std::get_if<Sum>(&e->node))
    {
        collect_addends(s->left, out);
        collect_addends(s->right, out);
    }
    else
    {
        out.push_back(e);
    }
}

// Flatten an additive expression into signed leaf addends: Sum keeps the sign,
// Difference flips it on the right operand, Negate flips it.  Non-additive
// nodes are leaves.  This lets like-term collection see through subtraction and
// negation uniformly (X - X, 2X - X, X + Y - X), independently of whether a
// sign is encoded as Sum+Negate or as Difference.
void collect_signed_addends(
    Expr const* e, int sign, std::vector<std::pair<int, Expr const*>>& out)
{
    if (auto const* s = std::get_if<Sum>(&e->node))
    {
        collect_signed_addends(s->left, sign, out);
        collect_signed_addends(s->right, sign, out);
    }
    else if (auto const* d = std::get_if<Difference>(&e->node))
    {
        collect_signed_addends(d->left, sign, out);
        collect_signed_addends(d->right, -sign, out);
    }
    else if (auto const* n = std::get_if<Negate>(&e->node))
    {
        collect_signed_addends(n->operand, -sign, out);
    }
    else
    {
        out.push_back({sign, e});
    }
}

// Find an IndexSpace from any ConcreteIndex-bearing TensorObject slot.
auto find_space_from_concrete(Expr const* e) -> IndexSpace const*
{
    IndexSpace const* found = nullptr;
    std::function<void(Expr const*)> go = [&](Expr const* node)
    {
        if (found)
            return;
        visit(
            Overloads{
                [&](TensorObject const& t)
                {
                    for (auto const& sb: t.slots)
                    {
                        if (!sb.slot.space || !sb.index)
                            continue;
                        if (std::get_if<ConcreteIndex>(&*sb.index))
                        {
                            found = sb.slot.space;
                            return;
                        }
                    }
                },
                [&](ScalarLiteral const&) {},
                [&](Negate const& n) { go(n.operand); },
                [&](Trace const& u) { go(u.operand); },
                [&](VectorInvariant const& u) { go(u.operand); },
                [&](Transpose const& u) { go(u.operand); },
                [&](Sum const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](Difference const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](TensorProduct const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](ScalarDiv const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](Dot const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](DDot const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](DDotAlt const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](Cross const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](ExplicitSum const& s)
                {
                    go(s.body);
                    if (s.bound)
                        go(s.bound);
                },
                [&](NoSum const& s) { go(s.body); },
            },
            *node);
    };
    go(e);
    return found;
}

// Collect unique ConcreteIndex values appearing in TensorObject slots of e.
void collect_concrete_values(Expr const* e, std::vector<int>& out)
{
    std::function<void(Expr const*)> go = [&](Expr const* node)
    {
        visit(
            Overloads{
                [&](TensorObject const& t)
                {
                    for (auto const& sb: t.slots)
                    {
                        if (!sb.index)
                            continue;
                        if (auto const* ci =
                                std::get_if<ConcreteIndex>(&*sb.index))
                        {
                            if (std::find(out.begin(), out.end(), ci->value)
                                == out.end())
                                out.push_back(ci->value);
                        }
                    }
                },
                [&](ScalarLiteral const&) {},
                [&](Negate const& n) { go(n.operand); },
                [&](Trace const& u) { go(u.operand); },
                [&](VectorInvariant const& u) { go(u.operand); },
                [&](Transpose const& u) { go(u.operand); },
                [&](Sum const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](Difference const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](TensorProduct const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](ScalarDiv const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](Dot const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](DDot const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](DDotAlt const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](Cross const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](ExplicitSum const& s)
                {
                    go(s.body);
                    if (s.bound)
                        go(s.bound);
                },
                [&](NoSum const& s) { go(s.body); },
            },
            *node);
    };
    go(e);
}

} // namespace

// structural_eq and its index helper live at namespace scope (not in the
// anonymous namespace above) so the public API and the identity matcher can
// reuse them.  structural_eq is declared in derivation.hpp; index_assoc_eq is
// kept file-local via `static`.
//
// Deep structural equality of two expression trees.  CountableIndex ids must
// match exactly (free variables are not alpha-renamed; ExplicitSum binders are
// compared by id).
static auto index_assoc_eq(
    std::optional<IndexAssoc> const& a,
    std::optional<IndexAssoc> const& b) -> bool
{
    if (!a && !b)
        return true;
    if (!a || !b)
        return false;
    return visit(
        Overloads{
            [&](CountableIndex const& ca) -> bool
            {
                auto const* cb = std::get_if<CountableIndex>(&*b);
                return cb && ca.id == cb->id;
            },
            [&](ConcreteIndex const& ca) -> bool
            {
                auto const* cb = std::get_if<ConcreteIndex>(&*b);
                return cb && ca.value == cb->value;
            },
            [&](LabelIndex const& la) -> bool
            {
                auto const* lb = std::get_if<LabelIndex>(&*b);
                return lb && la.name == lb->name;
            },
        },
        *a);
}

auto structural_eq(Expr const* a, Expr const* b) -> bool
{
    if (a == b)
        return true;
    return visit(
        Overloads{
            [&](TensorObject const& ta) -> bool
            {
                auto const* tb = std::get_if<TensorObject>(&b->node);
                if (!tb)
                    return false;
                if (ta.name != tb->name)
                    return false;
                if (ta.rank != tb->rank)
                    return false;
                if (ta.slots.size() != tb->slots.size())
                    return false;
                for (std::size_t i = 0; i < ta.slots.size(); ++i)
                {
                    auto const& sa = ta.slots[i];
                    auto const& sb = tb->slots[i];
                    if (sa.slot.level != sb.slot.level)
                        return false;
                    if (sa.slot.realm != sb.slot.realm)
                        return false;
                    if (sa.slot.space != sb.slot.space)
                        return false;
                    if (!index_assoc_eq(sa.index, sb.index))
                        return false;
                }
                return true;
            },
            [&](ScalarLiteral const& la) -> bool
            {
                auto const* lb = std::get_if<ScalarLiteral>(&b->node);
                return lb && la.value == lb->value;
            },
            [&](Negate const& na) -> bool
            {
                auto const* nb = std::get_if<Negate>(&b->node);
                return nb && structural_eq(na.operand, nb->operand);
            },
            [&](Trace const& na) -> bool
            {
                auto const* nb = std::get_if<Trace>(&b->node);
                return nb && structural_eq(na.operand, nb->operand);
            },
            [&](VectorInvariant const& na) -> bool
            {
                auto const* nb = std::get_if<VectorInvariant>(&b->node);
                return nb && structural_eq(na.operand, nb->operand);
            },
            [&](Transpose const& na) -> bool
            {
                auto const* nb = std::get_if<Transpose>(&b->node);
                return nb && structural_eq(na.operand, nb->operand);
            },
            [&](Sum const& sa) -> bool
            {
                auto const* sb = std::get_if<Sum>(&b->node);
                return sb && structural_eq(sa.left, sb->left)
                       && structural_eq(sa.right, sb->right);
            },
            [&](Difference const& da) -> bool
            {
                auto const* db = std::get_if<Difference>(&b->node);
                return db && structural_eq(da.left, db->left)
                       && structural_eq(da.right, db->right);
            },
            [&](TensorProduct const& pa) -> bool
            {
                auto const* pb = std::get_if<TensorProduct>(&b->node);
                return pb && structural_eq(pa.left, pb->left)
                       && structural_eq(pa.right, pb->right);
            },
            [&](ScalarDiv const& da) -> bool
            {
                auto const* db = std::get_if<ScalarDiv>(&b->node);
                return db && structural_eq(da.left, db->left)
                       && structural_eq(da.right, db->right);
            },
            [&](Dot const& da) -> bool
            {
                auto const* db = std::get_if<Dot>(&b->node);
                return db && structural_eq(da.left, db->left)
                       && structural_eq(da.right, db->right);
            },
            [&](DDot const& da) -> bool
            {
                auto const* db = std::get_if<DDot>(&b->node);
                return db && structural_eq(da.left, db->left)
                       && structural_eq(da.right, db->right);
            },
            [&](DDotAlt const& da) -> bool
            {
                auto const* db = std::get_if<DDotAlt>(&b->node);
                return db && structural_eq(da.left, db->left)
                       && structural_eq(da.right, db->right);
            },
            [&](Cross const& da) -> bool
            {
                auto const* db = std::get_if<Cross>(&b->node);
                return db && structural_eq(da.left, db->left)
                       && structural_eq(da.right, db->right);
            },
            [&](ExplicitSum const& sa) -> bool
            {
                auto const* sb = std::get_if<ExplicitSum>(&b->node);
                if (!sb)
                    return false;
                if (sa.index.id != sb->index.id)
                    return false;
                if (!structural_eq(sa.body, sb->body))
                    return false;
                if ((sa.bound == nullptr) != (sb->bound == nullptr))
                    return false;
                return !sa.bound || structural_eq(sa.bound, sb->bound);
            },
            [&](NoSum const& sa) -> bool
            {
                auto const* sb = std::get_if<NoSum>(&b->node);
                return sb && sa.index.id == sb->index.id
                       && structural_eq(sa.body, sb->body);
            },
        },
        a->node);
}

// Algebraic equality in theory T0: structural_eq of the two canonical forms.
auto algebraic_eq(Context& ctx, Expr const* a, Expr const* b) -> bool
{
    return structural_eq(
        steps::canonicalize(ctx, a), steps::canonicalize(ctx, b));
}

namespace
{

// Replace all ConcreteIndex{old_val} occurrences in TensorObject slots with
// CountableIndex{new_idx}.
auto substitute_concrete(
    Context& ctx,
    Expr const* e,
    int old_val,
    CountableIndex new_idx) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [old_val, new_idx](Context& ctx, Expr const* e) -> Expr const*
        {
            auto const* t = std::get_if<TensorObject>(&e->node);
            if (!t)
                return e;

            auto slots = t->slots;
            bool changed = false;
            for (auto& sb: slots)
            {
                if (!sb.index)
                    continue;
                if (auto const* ci = std::get_if<ConcreteIndex>(&*sb.index))
                    if (ci->value == old_val)
                    {
                        sb.index = new_idx;
                        changed = true;
                    }
            }
            if (!changed)
                return e;
            return ctx.make<Expr>(
                TensorObject{t->name, t->rank, t->traits, std::move(slots)});
        });
}

// Generic product distribution: distribute any binary product over
// Sum/Difference. make_prod(l, r) creates the leaf product node.
template <typename F>
auto distribute_any(
    Context& ctx, Expr const* l, Expr const* r, F const& make_prod)
    -> Expr const*
{
    if (auto const* s = std::get_if<Sum>(&l->node))
        return make_sum(
            ctx,
            distribute_any(ctx, s->left, r, make_prod),
            distribute_any(ctx, s->right, r, make_prod));
    if (auto const* d = std::get_if<Difference>(&l->node))
        return make_difference(
            ctx,
            distribute_any(ctx, d->left, r, make_prod),
            distribute_any(ctx, d->right, r, make_prod));
    if (auto const* s = std::get_if<Sum>(&r->node))
        return make_sum(
            ctx,
            distribute_any(ctx, l, s->left, make_prod),
            distribute_any(ctx, l, s->right, make_prod));
    if (auto const* d = std::get_if<Difference>(&r->node))
        return make_difference(
            ctx,
            distribute_any(ctx, l, d->left, make_prod),
            distribute_any(ctx, l, d->right, make_prod));
    return make_prod(l, r);
}

// Extract (rational coefficient, core expression) from an addend.
// Returns {1, e}       for a plain expression.
// Returns {c, core}    for TensorProduct(ScalarLiteral(c), core) — scalar on
//                      either side (so X·2 reads the same as 2·X).
// Returns {-1, core}   for Negate(core).
// The coefficient is an exact Rational, so e.g. ½X + ½X collects to X.
auto extract_coeff(Expr const* e) -> std::pair<Rational, Expr const*>
{
    if (auto const* tp = std::get_if<TensorProduct>(&e->node))
    {
        if (auto const* sl = std::get_if<ScalarLiteral>(&tp->left->node))
            return {sl->value, tp->right};
        if (auto const* sr = std::get_if<ScalarLiteral>(&tp->right->node))
            return {sr->value, tp->left};
    }
    if (auto const* neg = std::get_if<Negate>(&e->node))
        return {Rational{-1}, neg->operand};
    return {Rational{1}, e};
}

// Return true if any of `indices` appears free in `e` — i.e., in a tensor
// slot that is not bound by an enclosing ExplicitSum or suppressed by NoSum.
auto has_free_index_for(
    Expr const* e,
    std::vector<CountableIndex> const& indices,
    std::set<int> const& bound) -> bool
{
    return visit(
        Overloads{
            [&](TensorObject const& t) -> bool
            {
                for (auto const& sb: t.slots)
                {
                    if (!sb.index)
                        continue;
                    if (auto const* ci =
                            std::get_if<CountableIndex>(&*sb.index))
                    {
                        if (bound.count(ci->id))
                            continue;
                        for (auto const& idx: indices)
                            if (idx.id == ci->id)
                                return true;
                    }
                }
                return false;
            },
            [&](ScalarLiteral const&) -> bool { return false; },
            [&](Negate const& n) -> bool
            { return has_free_index_for(n.operand, indices, bound); },
            [&](Trace const& u) -> bool
            { return has_free_index_for(u.operand, indices, bound); },
            [&](VectorInvariant const& u) -> bool
            { return has_free_index_for(u.operand, indices, bound); },
            [&](Transpose const& u) -> bool
            { return has_free_index_for(u.operand, indices, bound); },
            [&](Sum const& s) -> bool
            {
                return has_free_index_for(s.left, indices, bound)
                       || has_free_index_for(s.right, indices, bound);
            },
            [&](Difference const& s) -> bool
            {
                return has_free_index_for(s.left, indices, bound)
                       || has_free_index_for(s.right, indices, bound);
            },
            [&](TensorProduct const& s) -> bool
            {
                return has_free_index_for(s.left, indices, bound)
                       || has_free_index_for(s.right, indices, bound);
            },
            [&](ScalarDiv const& s) -> bool
            {
                return has_free_index_for(s.left, indices, bound)
                       || has_free_index_for(s.right, indices, bound);
            },
            [&](Dot const& s) -> bool
            {
                return has_free_index_for(s.left, indices, bound)
                       || has_free_index_for(s.right, indices, bound);
            },
            [&](DDot const& s) -> bool
            {
                return has_free_index_for(s.left, indices, bound)
                       || has_free_index_for(s.right, indices, bound);
            },
            [&](DDotAlt const& s) -> bool
            {
                return has_free_index_for(s.left, indices, bound)
                       || has_free_index_for(s.right, indices, bound);
            },
            [&](Cross const& s) -> bool
            {
                return has_free_index_for(s.left, indices, bound)
                       || has_free_index_for(s.right, indices, bound);
            },
            [&](ExplicitSum const& s) -> bool
            {
                auto b2 = bound;
                b2.insert(s.index.id);
                // The bound expression lives in the outer scope.
                if (s.bound && has_free_index_for(s.bound, indices, bound))
                    return true;
                return has_free_index_for(s.body, indices, b2);
            },
            [&](NoSum const& s) -> bool
            {
                // NoSum suppresses summation — treat as bound for our purposes.
                auto b2 = bound;
                b2.insert(s.index.id);
                return has_free_index_for(s.body, indices, b2);
            },
        },
        *e);
}

// ===== Algebraic normal form (vibe 000037) ===============================

// ---- total order over expressions (for sorting commutative operands) ----

auto name_view_cmp(std::string_view a, std::string_view b) -> int
{
    return a.compare(b);
}

auto space_cmp(IndexSpace const* a, IndexSpace const* b) -> int
{
    if (a == b)
        return 0;
    auto va = a->values(), vb = b->values();
    if (va.size() != vb.size())
        return va.size() < vb.size() ? -1 : 1;
    for (std::size_t i = 0; i < va.size(); ++i)
        if (va[i] != vb[i])
            return va[i] < vb[i] ? -1 : 1;
    // Same value set but distinct instances: fall back to pointer for a stable
    // within-run order.
    return a < b ? -1 : 1;
}

auto index_assoc_cmp(
    std::optional<IndexAssoc> const& a,
    std::optional<IndexAssoc> const& b) -> int
{
    if (!a && !b)
        return 0;
    if (!a)
        return -1;
    if (!b)
        return 1;
    if (a->index() != b->index())
        return a->index() < b->index() ? -1 : 1;
    return std::visit(
        Overloads{
            [&](CountableIndex const& ca) -> int
            {
                auto id = std::get<CountableIndex>(*b).id;
                return ca.id == id ? 0 : (ca.id < id ? -1 : 1);
            },
            [&](ConcreteIndex const& ca) -> int
            {
                auto v = std::get<ConcreteIndex>(*b).value;
                return ca.value == v ? 0 : (ca.value < v ? -1 : 1);
            },
            [&](LabelIndex const& la) -> int
            {
                return name_view_cmp(
                    la.name.v.view(), std::get<LabelIndex>(*b).name.v.view());
            }},
        *a);
}

auto expr_cmp(Expr const* a, Expr const* b) -> int;

auto binop_cmp(Expr const* al, Expr const* ar, Expr const* bl, Expr const* br)
    -> int
{
    if (int c = expr_cmp(al, bl))
        return c;
    return expr_cmp(ar, br);
}

// Total order: first by node kind (variant index), then structurally.  Children
// are assumed already canonical, so this is a well-defined order on canonical
// forms.
auto expr_cmp(Expr const* a, Expr const* b) -> int
{
    if (a == b)
        return 0;
    auto ka = a->node.index(), kb = b->node.index();
    if (ka != kb)
        return ka < kb ? -1 : 1;
    return visit(
        Overloads{
            [&](TensorObject const& ta) -> int
            {
                auto const& tb = std::get<TensorObject>(b->node);
                if (int c = name_view_cmp(ta.name.v.view(), tb.name.v.view()))
                    return c;
                if (ta.rank != tb.rank)
                    return ta.rank < tb.rank ? -1 : 1;
                if (ta.slots.size() != tb.slots.size())
                    return ta.slots.size() < tb.slots.size() ? -1 : 1;
                for (std::size_t i = 0; i < ta.slots.size(); ++i)
                {
                    auto const& sa = ta.slots[i];
                    auto const& sb = tb.slots[i];
                    if (sa.slot.level != sb.slot.level)
                        return sa.slot.level < sb.slot.level ? -1 : 1;
                    if (sa.slot.realm != sb.slot.realm)
                        return sa.slot.realm < sb.slot.realm ? -1 : 1;
                    if (int c = space_cmp(sa.slot.space, sb.slot.space))
                        return c;
                    if (int c = index_assoc_cmp(sa.index, sb.index))
                        return c;
                }
                return 0;
            },
            [&](ScalarLiteral const& la) -> int
            {
                auto const& lb = std::get<ScalarLiteral>(b->node);
                auto o = la.value <=> lb.value;
                return o < 0 ? -1 : (o > 0 ? 1 : 0);
            },
            [&](Negate const& na) -> int
            { return expr_cmp(na.operand, std::get<Negate>(b->node).operand); },
            [&](Trace const& na) -> int
            { return expr_cmp(na.operand, std::get<Trace>(b->node).operand); },
            [&](VectorInvariant const& na) -> int {
                return expr_cmp(
                    na.operand, std::get<VectorInvariant>(b->node).operand);
            },
            [&](Transpose const& na) -> int {
                return expr_cmp(
                    na.operand, std::get<Transpose>(b->node).operand);
            },
            [&](Sum const& sa) -> int
            {
                auto const& sb = std::get<Sum>(b->node);
                return binop_cmp(sa.left, sa.right, sb.left, sb.right);
            },
            [&](Difference const& da) -> int
            {
                auto const& db = std::get<Difference>(b->node);
                return binop_cmp(da.left, da.right, db.left, db.right);
            },
            [&](TensorProduct const& pa) -> int
            {
                auto const& pb = std::get<TensorProduct>(b->node);
                return binop_cmp(pa.left, pa.right, pb.left, pb.right);
            },
            [&](ScalarDiv const& da) -> int
            {
                auto const& db = std::get<ScalarDiv>(b->node);
                return binop_cmp(da.left, da.right, db.left, db.right);
            },
            [&](Dot const& da) -> int
            {
                auto const& db = std::get<Dot>(b->node);
                return binop_cmp(da.left, da.right, db.left, db.right);
            },
            [&](DDot const& da) -> int
            {
                auto const& db = std::get<DDot>(b->node);
                return binop_cmp(da.left, da.right, db.left, db.right);
            },
            [&](DDotAlt const& da) -> int
            {
                auto const& db = std::get<DDotAlt>(b->node);
                return binop_cmp(da.left, da.right, db.left, db.right);
            },
            [&](Cross const& da) -> int
            {
                auto const& db = std::get<Cross>(b->node);
                return binop_cmp(da.left, da.right, db.left, db.right);
            },
            [&](ExplicitSum const& sa) -> int
            {
                auto const& sb = std::get<ExplicitSum>(b->node);
                if (sa.index.id != sb.index.id)
                    return sa.index.id < sb.index.id ? -1 : 1;
                if (int c = expr_cmp(sa.body, sb.body))
                    return c;
                if ((sa.bound == nullptr) != (sb.bound == nullptr))
                    return sa.bound == nullptr ? -1 : 1;
                return sa.bound ? expr_cmp(sa.bound, sb.bound) : 0;
            },
            [&](NoSum const& sa) -> int
            {
                auto const& sb = std::get<NoSum>(b->node);
                if (sa.index.id != sb.index.id)
                    return sa.index.id < sb.index.id ? -1 : 1;
                return expr_cmp(sa.body, sb.body);
            },
        },
        *a);
}

} // namespace

// ---- component-valued predicate (vibe 000036 coordinate/invariant line) --

// True iff e denotes a scalar/coordinate value: a scalar, a fully-indexed
// coordinate tensor, or a combination thereof.  A slot-less rank >= 1 object is
// an invariant (false).  Component-valued factors commute with everything;
// invariant factors do not commute among themselves.
//
// Declared in derivation.hpp; lives at namespace scope (not anonymous) so the
// identity matcher can reuse the coordinate/invariant distinction when deciding
// which products match modulo factor order.
auto is_component_valued(Expr const* e) -> bool
{
    return visit(
        Overloads{
            [](ScalarLiteral const&) { return true; },
            [](TensorObject const& t)
            {
                if (t.rank && *t.rank == 0)
                    return true; // rank 0: a scalar coordinate
                if (t.rank && *t.rank >= 1)
                    return false; // rank >= 1: a basis vector / indexed tensor,
                                  // not a commuting scalar (even fully indexed)
                if (t.slots.empty())
                    return false; // slot-less, rank unknown: invariant
                for (auto const& sb: t.slots)
                    if (!sb.index)
                        return false; // partially indexed: be conservative
                return true;          // fully indexed, rank unknown: coordinate
            },
            [](Negate const& n) { return is_component_valued(n.operand); },
            // tr/vec/transpose are invariant tensor operations, not
            // coordinates.
            [](Trace const&) { return false; },
            [](VectorInvariant const&) { return false; },
            [](Transpose const&) { return false; },
            [](Sum const& s) {
                return is_component_valued(s.left)
                       && is_component_valued(s.right);
            },
            [](Difference const& d) {
                return is_component_valued(d.left)
                       && is_component_valued(d.right);
            },
            [](TensorProduct const& p) {
                return is_component_valued(p.left)
                       && is_component_valued(p.right);
            },
            [](ScalarDiv const& d) {
                return is_component_valued(d.left)
                       && is_component_valued(d.right);
            },
            [](ExplicitSum const& s) { return is_component_valued(s.body); },
            [](NoSum const& s) { return is_component_valued(s.body); },
            // Invariant operations: conservatively non-component.
            [](Dot const&) { return false; },
            [](DDot const&) { return false; },
            [](DDotAlt const&) { return false; },
            [](Cross const&) { return false; },
        },
        *e);
}

auto infer_rank(Expr const* e) -> std::optional<int>
{
    // Rank arithmetic for a contraction that removes `removed` indices from the
    // outer-product rank of its two operands; nullopt if either is unknown or
    // the result is negative (ill-formed).
    auto contracted = [](std::optional<int> l,
                         std::optional<int> r,
                         int removed) -> std::optional<int>
    {
        if (!l || !r || *l + *r - removed < 0)
            return std::nullopt;
        return *l + *r - removed;
    };
    return visit(
        Overloads{
            [](TensorObject const& t) -> std::optional<int> { return t.rank; },
            [](ScalarLiteral const&) -> std::optional<int> { return 0; },
            [](Negate const& n) -> std::optional<int>
            { return infer_rank(n.operand); },
            // tr(A) is a scalar; vec(A) is a vector; transpose keeps the rank.
            [](Trace const&) -> std::optional<int> { return 0; },
            [](VectorInvariant const&) -> std::optional<int> { return 1; },
            [](Transpose const& u) -> std::optional<int>
            { return infer_rank(u.operand); },
            // A sum keeps the shared rank of its operands; trust the known
            // side.
            [](Sum const& s) -> std::optional<int>
            {
                auto const l = infer_rank(s.left);
                return l ? l : infer_rank(s.right);
            },
            [](Difference const& s) -> std::optional<int>
            {
                auto const l = infer_rank(s.left);
                return l ? l : infer_rank(s.right);
            },
            // Outer product adds ranks; scalar division keeps the left rank.
            [&](TensorProduct const& s) -> std::optional<int>
            { return contracted(infer_rank(s.left), infer_rank(s.right), 0); },
            [](ScalarDiv const& s) -> std::optional<int>
            { return infer_rank(s.left); },
            [&](Dot const& s) -> std::optional<int>
            { return contracted(infer_rank(s.left), infer_rank(s.right), 2); },
            [&](DDot const& s) -> std::optional<int>
            { return contracted(infer_rank(s.left), infer_rank(s.right), 4); },
            [&](DDotAlt const& s) -> std::optional<int>
            { return contracted(infer_rank(s.left), infer_rank(s.right), 4); },
            [&](Cross const& s) -> std::optional<int>
            { return contracted(infer_rank(s.left), infer_rank(s.right), 1); },
            [](ExplicitSum const& s) -> std::optional<int>
            { return infer_rank(s.body); },
            [](NoSum const& s) -> std::optional<int>
            { return infer_rank(s.body); },
        },
        *e);
}

namespace
{

// ---- the canonicalizer --------------------------------------------------

// Replace every CountableIndex{old_id} in TensorObject slots with
// CountableIndex{new_id} — used to α-normalize a binder's dummy index.
auto substitute_index_id(Context& ctx, Expr const* e, int old_id, int new_id)
    -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [old_id, new_id](Context& ctx, Expr const* e) -> Expr const*
        {
            auto const* t = std::get_if<TensorObject>(&e->node);
            if (!t)
                return e;
            auto slots = t->slots;
            bool changed = false;
            for (auto& sb: slots)
            {
                if (!sb.index)
                    continue;
                if (auto const* ci = std::get_if<CountableIndex>(&*sb.index))
                    if (ci->id == old_id)
                    {
                        sb.index = CountableIndex{new_id};
                        changed = true;
                    }
            }
            if (!changed)
                return e;
            return ctx.make<Expr>(
                TensorObject{t->name, t->rank, t->traits, std::move(slots)});
        });
}

// Like substitute_index_id but applies a whole id→id map atomically (each slot
// id looked up once), so a remap that permutes ids (e.g. -1↔-2) is correct.
auto substitute_index_ids(
    Context& ctx, Expr const* e, std::map<int, int> const& remap) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [&remap](Context& ctx, Expr const* e) -> Expr const*
        {
            auto const* t = std::get_if<TensorObject>(&e->node);
            if (!t)
                return e;
            auto slots = t->slots;
            bool changed = false;
            for (auto& sb: slots)
            {
                if (!sb.index)
                    continue;
                if (auto const* ci = std::get_if<CountableIndex>(&*sb.index))
                    if (auto it = remap.find(ci->id); it != remap.end())
                    {
                        sb.index = CountableIndex{it->second};
                        changed = true;
                    }
            }
            if (!changed)
                return e;
            return ctx.make<Expr>(
                TensorObject{t->name, t->rank, t->traits, std::move(slots)});
        });
}

// Canonical (reserved, negative) id for a bound index at binder nesting depth
// d.  Free ids are non-negative, so this namespace never collides with them,
// and distinct depths get distinct ids (so Σ_i Σ_j keeps i and j apart).
auto bound_canon_id(int depth) -> int
{
    return -(depth + 1);
}

auto canon(Context& ctx, Expr const* e, int depth) -> Expr const*;

// Canonicalize a stack of consecutive null-bound ExplicitSums (Σ_a Σ_b … body).
// The binders are interchangeable (Fubini), so this fixes a canonical order:
// it tries every permutation of the binders onto the depth-numbered canonical
// ids and keeps the one whose canonicalized body is smallest under expr_cmp.
// Two Fubini-equivalent stacks therefore produce the identical canonical form.
// The permutation search is bounded; very tall stacks keep their given order.
auto canon_sum_stack(Context& ctx, Expr const* e, int depth) -> Expr const*
{
    std::vector<int> ids;
    Expr const* body = e;
    while (auto const* s = std::get_if<ExplicitSum>(&body->node))
    {
        if (s->bound)
            break; // a symbolic bound is not part of the free stack
        ids.push_back(s->index.id);
        body = s->body;
    }
    int const n = static_cast<int>(ids.size());

    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    bool const search = n <= 6; // n! permutations; tall stacks fall back to
                                // order
    Expr const* best = nullptr;
    do
    {
        std::map<int, int> remap;
        for (int pos = 0; pos < n; ++pos)
            remap[ids[static_cast<std::size_t>(
                order[static_cast<std::size_t>(pos)])]] =
                bound_canon_id(depth + pos);
        auto const* cbody =
            canon(ctx, substitute_index_ids(ctx, body, remap), depth + n);
        if (!best || expr_cmp(cbody, best) < 0)
            best = cbody;
    } while (search && std::next_permutation(order.begin(), order.end()));

    for (int pos = n - 1; pos >= 0; --pos)
        best = make_explicit_sum(
            ctx, CountableIndex{bound_canon_id(depth + pos)}, best, nullptr);
    return best;
}

// ---- symmetry orbit canonicalization (vibe 000047) ----------------------

// Total order on slot sequences (same key as expr_cmp's TensorObject arm).
auto slot_seq_cmp(
    std::vector<SlotBinding> const& a, std::vector<SlotBinding> const& b) -> int
{
    // a and b are arrangements of one tensor's slots, so they share the same
    // multiset of (realm, space) — only level and index ever differ between two
    // orbit elements.  The realm/space arms are kept for generality but cannot
    // fire for any symmetric tensor we build.
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (a[i].slot.level != b[i].slot.level)
            return a[i].slot.level < b[i].slot.level ? -1 : 1;
        // GCOV_EXCL_START  (realm/space uniform across one tensor's slots)
        if (a[i].slot.realm != b[i].slot.realm)
            return a[i].slot.realm < b[i].slot.realm ? -1 : 1;
        if (int c = space_cmp(a[i].slot.space, b[i].slot.space))
            return c;
        // GCOV_EXCL_STOP
        if (int c = index_assoc_cmp(a[i].index, b[i].index))
            return c;
    }
    return 0;
}

// Apply a permutation (image[i] = destination of slot i) to a slot sequence.
auto permute_slots(
    std::vector<SlotBinding> const& slots,
    PermutationView const& p) -> std::vector<SlotBinding>
{
    std::vector<SlotBinding> out(slots.size());
    for (std::size_t i = 0; i < slots.size(); ++i)
        out[p[i]] = slots[i];
    return out;
}

// Canonicalize a TensorObject under its declared (anti)symmetry: pick the
// orbit-minimal slot arrangement and fold the sign of an antisymmetric
// reordering into a leading −1.  The symmetry/antisymmetry generators induce a
// sign homomorphism (+1 / −1) on the generated permutation group; the orbit is
// the group orbit of the original slot sequence.  Returns:
//   - e unchanged if there are no generators or e is already orbit-minimal,
//   - a reordered TensorObject (sign +1),
//   - Negate(reordered) (sign −1), or
//   - scalar 0 when an arrangement is reachable with both signs — the object is
//     identically zero (e.g. ε with a repeated index).
auto canon_symmetry(Context& ctx, Expr const* e) -> Expr const*
{
    auto const* t = std::get_if<TensorObject>(&e->node);
    if (!t || !t->traits)
        return e;
    auto const& sym = t->traits->symmetry.generators;
    auto const& asym = t->traits->antisymmetry.generators;
    if (sym.empty() && asym.empty())
        return e;

    struct Reached final
    {
        std::vector<SlotBinding> slots;
        int sign;
    };
    std::vector<Reached> seen;
    seen.push_back({t->slots, +1});

    auto find_seen = [&](std::vector<SlotBinding> const& s) -> Reached*
    {
        for (auto& r: seen)
            if (slot_seq_cmp(r.slots, s) == 0)
                return &r;
        return nullptr;
    };

    // Breadth-first closure of the slot sequence under the generators.  The
    // group is finite, so the forward closure (no explicit inverses) reaches
    // every element.  Orbits are tiny (≤ |group|), so linear search suffices.
    for (std::size_t head = 0; head < seen.size(); ++head)
    {
        Reached const cur = seen[head]; // copy: seen may reallocate
        auto step = [&](PermutationView const& p, int gsign) -> bool
        {
            // Defensive: every generator we install matches the slot count.
            if (p.size() != cur.slots.size()) // GCOV_EXCL_LINE
                return false;                 // GCOV_EXCL_LINE
            auto next = permute_slots(cur.slots, p);
            int const nsign = cur.sign * gsign;
            if (auto* r = find_seen(next))
                return r->sign != nsign; // conflict ⇒ identically zero
            seen.push_back({std::move(next), nsign});
            return false;
        };
        // A symmetry step alone never sign-conflicts (all +1); a conflict
        // through the symmetry loop needs a tensor with *both* symmetry and
        // antisymmetry generators, which none we build has yet.
        for (auto const& p: sym)
            if (step(p, +1))                          // GCOV_EXCL_LINE
                return make_scalar(ctx, Rational{0}); // GCOV_EXCL_LINE
        for (auto const& p: asym)
            if (step(p, -1))
                return make_scalar(ctx, Rational{0});
    }

    Reached const* best = &seen.front();
    for (auto const& r: seen)
        if (slot_seq_cmp(r.slots, best->slots) < 0)
            best = &r;

    Expr const* canon_t =
        slot_seq_cmp(best->slots, t->slots) == 0 ?
            e :
            ctx.make<Expr>(
                TensorObject{t->name, t->rank, t->traits, best->slots});
    return best->sign < 0 ? make_negate(ctx, canon_t) : canon_t;
}

// Flatten a left/right TensorProduct tree into its factor leaves.
void flatten_factors(Expr const* e, std::vector<Expr const*>& out)
{
    if (auto const* p = std::get_if<TensorProduct>(&e->node))
    {
        flatten_factors(p->left, out);
        flatten_factors(p->right, out);
    }
    else
        out.push_back(e);
}

// Build coeff * (factors folded left-associatively), with identity/zero rules.
auto build_term(
    Context& ctx,
    Rational coeff,
    std::vector<Expr const*> const& factors) -> Expr const*
{
    if (coeff == 0)
        return make_scalar(ctx, Rational{0});
    Expr const* prod = nullptr;
    for (auto const* f: factors)
        prod = prod ? make_tensor_product(ctx, prod, f) : f;
    if (!prod)
        return make_scalar(ctx, coeff);
    if (coeff == 1)
        return prod;
    if (coeff == -1)
        return make_negate(ctx, prod);
    return make_tensor_product(ctx, make_scalar(ctx, coeff), prod);
}

auto canon_product(Context& ctx, Expr const* e, int depth) -> Expr const*
{
    std::vector<Expr const*> flat;
    flatten_factors(e, flat);

    Rational coeff{1};
    std::vector<Expr const*> comp; // component-valued factors (sortable)
    std::vector<Expr const*> inv;  // invariant factors (order preserved)
    for (auto const* f: flat)
    {
        auto const* cf = canon(ctx, f, depth);
        // Pull out a leading numeric coefficient / sign, then re-flatten the
        // core (a collapsed sub-sum may have become a scaled monomial).
        auto [c, core] = extract_coeff(cf);
        coeff *= c;
        std::vector<Expr const*> sub;
        flatten_factors(core, sub);
        for (auto const* g: sub)
        {
            if (auto const* sl = std::get_if<ScalarLiteral>(&g->node))
            {
                coeff *= sl->value;
                continue;
            }
            if (is_component_valued(g))
                comp.push_back(g);
            else
                inv.push_back(g);
        }
    }
    std::sort(
        comp.begin(),
        comp.end(),
        [](Expr const* x, Expr const* y) { return expr_cmp(x, y) < 0; });

    std::vector<Expr const*> ordered;
    ordered.reserve(comp.size() + inv.size());
    ordered.insert(ordered.end(), comp.begin(), comp.end());
    ordered.insert(ordered.end(), inv.begin(), inv.end());
    return build_term(ctx, coeff, ordered);
}

auto canon_additive(Context& ctx, Expr const* e, int depth) -> Expr const*
{
    std::vector<std::pair<int, Expr const*>> addends;
    collect_signed_addends(e, +1, addends);

    Rational constant{0};                                // numeric terms folded
    std::vector<std::pair<Rational, Expr const*>> terms; // (coeff, core)
    for (auto const& [sign, sub]: addends)
    {
        auto const* cs = canon(ctx, sub, depth);
        auto [c, core] = extract_coeff(cs);
        Rational coeff = c * Rational{sign};
        if (auto const* sl = std::get_if<ScalarLiteral>(&core->node))
        {
            constant += coeff * sl->value;
            continue;
        }
        bool merged = false;
        for (auto& [tc, tcore]: terms)
            if (structural_eq(core, tcore))
            {
                tc += coeff;
                merged = true;
                break;
            }
        if (!merged)
            terms.emplace_back(coeff, core);
    }

    // Drop zero terms.
    std::vector<std::pair<Rational, Expr const*>> kept;
    for (auto const& t: terms)
        if (!(t.first == 0))
            kept.push_back(t);
    if (kept.empty())
        return make_scalar(ctx, constant);

    std::sort(
        kept.begin(),
        kept.end(),
        [](auto const& x, auto const& y)
        { return expr_cmp(x.second, y.second) < 0; });

    Expr const* result = nullptr;
    for (auto const& [coeff, core]: kept)
    {
        Expr const* term = build_term(ctx, coeff, {core});
        result = result ? make_sum(ctx, result, term) : term;
    }
    // A non-zero numeric constant is appended last (deterministic position).
    if (!(constant == 0))
        result = make_sum(ctx, result, make_scalar(ctx, constant));
    return result;
}

// Canonicalize a rank-aware invariant binary op (Dot commutes, Cross
// anticommutes) when both operands are rank-1 invariant vectors.
auto is_rank1_vector(Expr const* e) -> bool
{
    auto const* t = std::get_if<TensorObject>(&e->node);
    return t && t->rank && *t->rank == 1 && t->slots.empty();
}

// Re-associate a cross chain around a rank-≥2 fence: (x × M) × z → x × (M × z).
// When the middle operand M is rank ≥ 2, the ⊗ inside it fences the two crosses
// onto M's disjoint outer legs (x onto the first, z onto the last), so the
// bracketing is immaterial — (x×M)×z and x×(M×z) are equal.  We normalize to
// the right-associated form, which exposes the `M × z` subterm (e.g. `I × b`)
// for the matcher.  The rank-1 middle case (x×y)×z is the vector triple
// product, genuinely non-associative (bac-cab), and is deliberately left
// untouched. Returns the re-associated expr, or nullptr when the pattern does
// not apply. Operands are assumed already canonicalized.
auto reassociate_cross_fence(Context& ctx, Expr const* l, Expr const* r)
    -> Expr const*
{
    auto const* inner = std::get_if<Cross>(&l->node);
    if (!inner)
        return nullptr;
    auto const rx = infer_rank(inner->left);
    auto const rm = infer_rank(inner->right);
    auto const rz = infer_rank(r);
    if (rx == std::optional<int>{1} && rm && *rm >= 2
        && rz == std::optional<int>{1})
        return make_cross(ctx, inner->left, make_cross(ctx, inner->right, r));
    return nullptr;
}

auto canon(Context& ctx, Expr const* e, int depth) -> Expr const*
{
    return visit(
        Overloads{
            [&](ScalarLiteral const&) -> Expr const* { return e; },
            [&](TensorObject const&) -> Expr const*
            { return canon_symmetry(ctx, e); },
            [&](Negate const&) -> Expr const*
            { return canon_additive(ctx, e, depth); },
            [&](Trace const& u) -> Expr const*
            { return make_trace(ctx, canon(ctx, u.operand, depth)); },
            [&](VectorInvariant const& u) -> Expr const* {
                return make_vector_invariant(ctx, canon(ctx, u.operand, depth));
            },
            [&](Transpose const& u) -> Expr const*
            { return make_transpose(ctx, canon(ctx, u.operand, depth)); },
            [&](Sum const&) -> Expr const*
            { return canon_additive(ctx, e, depth); },
            [&](Difference const&) -> Expr const*
            { return canon_additive(ctx, e, depth); },
            [&](TensorProduct const&) -> Expr const*
            { return canon_product(ctx, e, depth); },
            [&](ScalarDiv const& d) -> Expr const*
            {
                auto const* l = canon(ctx, d.left, depth);
                auto const* r = canon(ctx, d.right, depth);
                auto const* sl = std::get_if<ScalarLiteral>(&l->node);
                auto const* sr = std::get_if<ScalarLiteral>(&r->node);
                if (sl && sr)
                    return make_scalar(ctx, sl->value / sr->value);
                return make_scalar_div(ctx, l, r);
            },
            [&](Dot const& d) -> Expr const*
            {
                auto const* l = canon(ctx, d.left, depth);
                auto const* r = canon(ctx, d.right, depth);
                if (is_rank1_vector(l) && is_rank1_vector(r)
                    && expr_cmp(l, r) > 0)
                    return make_dot(ctx, r, l); // a·b = b·a
                return make_dot(ctx, l, r);
            },
            [&](Cross const& c) -> Expr const*
            {
                auto const* l = canon(ctx, c.left, depth);
                auto const* r = canon(ctx, c.right, depth);
                if (is_rank1_vector(l) && is_rank1_vector(r)
                    && expr_cmp(l, r) > 0)
                    return make_negate(ctx, make_cross(ctx, r, l)); // a×b =
                                                                    // -(b×a)
                if (auto const* ra = reassociate_cross_fence(ctx, l, r))
                    return ra; // (x×M)×z → x×(M×z) around a rank-≥2 fence
                return make_cross(ctx, l, r);
            },
            [&](DDot const& d) -> Expr const*
            {
                return make_ddot(
                    ctx, canon(ctx, d.left, depth), canon(ctx, d.right, depth));
            },
            [&](DDotAlt const& d) -> Expr const*
            {
                return make_ddot_alt(
                    ctx, canon(ctx, d.left, depth), canon(ctx, d.right, depth));
            },
            [&](ExplicitSum const& s) -> Expr const*
            {
                // α-normalize: relabel the dummy to a canonical id *before*
                // canonicalizing the body, so the body's sort order does not
                // depend on the original dummy id.  A null-bound stack of
                // nested sums is order-normalized too (Fubini), via
                // canon_sum_stack.
                if (!s.bound)
                    return canon_sum_stack(ctx, e, depth);
                int cid = bound_canon_id(depth);
                auto const* relabeled =
                    substitute_index_id(ctx, s.body, s.index.id, cid);
                auto const* body = canon(ctx, relabeled, depth + 1);
                return make_explicit_sum(
                    ctx, CountableIndex{cid}, body, canon(ctx, s.bound, depth));
            },
            // NoSum suppresses summation, so its index stays a free reference
            // (not α-renamed); only its body is canonicalized.
            [&](NoSum const& s) -> Expr const*
            { return make_no_sum(ctx, s.index, canon(ctx, s.body, depth)); },
        },
        *e);
}

// ---- implicit Einstein summation (vibe 000028) --------------------------
//
// An index is either contracted or free — never unspecified.  The realm rule
// decides; ExplicitSum/NoSum are overrides.  This pass materializes every
// implicit contraction as an ExplicitSum, so the canonical form is uniform and
// the matcher need only understand explicit binders.  It runs before `canon`.
//
// Scope: a *term* is a pure multilinear tree of tensors — products and
// contractions (⊗ · : ·· ×) and the linear unary ops — with no Sum/Difference/
// ScalarDiv/binder.  Contraction is detected across the whole term (so the
// indices of e_i·e_j and another e_i·e_j sum together).  Indices free only
// inside an un-distributed Sum (a scope boundary) are still deferred.

// One free occurrence of a CountableIndex within a term.
struct IndexUse final
{
    Level level;
    Realm realm;
};

// A *term*: a pure multilinear expression (no Sum/Difference/ScalarDiv/binder),
// so the whole thing is one Einstein-summation scope.
auto is_term(Expr const* e) -> bool
{
    return visit(
        Overloads{
            [](TensorObject const&) { return true; },
            [](ScalarLiteral const&) { return true; },
            [](Negate const& n) { return is_term(n.operand); },
            [](Trace const& u) { return is_term(u.operand); },
            [](VectorInvariant const& u) { return is_term(u.operand); },
            [](Transpose const& u) { return is_term(u.operand); },
            [](TensorProduct const& p)
            { return is_term(p.left) && is_term(p.right); },
            [](Dot const& d) { return is_term(d.left) && is_term(d.right); },
            [](DDot const& d) { return is_term(d.left) && is_term(d.right); },
            [](DDotAlt const& d)
            { return is_term(d.left) && is_term(d.right); },
            [](Cross const& c) { return is_term(c.left) && is_term(c.right); },
            [](auto const&) { return false; }, // Sum/Diff/ScalarDiv/binders
        },
        *e);
}

// Collect every free CountableIndex occurrence in a term, descending through
// its multilinear structure.  Stops at scope boundaries (a non-term node is
// opaque).
void collect_term_uses(
    Expr const* e,
    std::set<int> const& bound,
    std::map<int, std::vector<IndexUse>>& uses)
{
    auto bin = [&](Expr const* l, Expr const* r)
    {
        collect_term_uses(l, bound, uses);
        collect_term_uses(r, bound, uses);
    };
    visit(
        Overloads{
            [&](TensorObject const& t)
            {
                for (auto const& sb: t.slots)
                {
                    if (!sb.index)
                        continue;
                    auto const* ci = std::get_if<CountableIndex>(&*sb.index);
                    if (!ci || bound.count(ci->id))
                        continue;
                    uses[ci->id].push_back({sb.slot.level, sb.slot.realm});
                }
            },
            [&](Negate const& n) { collect_term_uses(n.operand, bound, uses); },
            [&](Trace const& u) { collect_term_uses(u.operand, bound, uses); },
            [&](VectorInvariant const& u)
            { collect_term_uses(u.operand, bound, uses); },
            [&](Transpose const& u)
            { collect_term_uses(u.operand, bound, uses); },
            [&](TensorProduct const& p) { bin(p.left, p.right); },
            [&](Dot const& d) { bin(d.left, d.right); },
            [&](DDot const& d) { bin(d.left, d.right); },
            [&](DDotAlt const& d) { bin(d.left, d.right); },
            [&](Cross const& c) { bin(c.left, c.right); },
            [&](auto const&) {}, // ScalarLiteral / scope boundaries: opaque
        },
        *e);
}

// Decide which free ids among `factors` are implicitly contracted, per each
// id's realm rule.  Ids already bound by an enclosing ExplicitSum/NoSum are in
// `bound` and excluded — so an explicit override both suppresses contraction
// and silences the error checks.  Throws for ill-formed terms with no such
// override.  Returns the contracted ids ascending (std::map key order).
auto contracted_ids(Expr const* term, std::set<int> const& bound)
    -> std::vector<int>
{
    std::map<int, std::vector<IndexUse>> uses;
    collect_term_uses(term, bound, uses);

    std::vector<int> result;
    for (auto const& [id, us]: uses)
    {
        std::size_t const n = us.size();
        switch (us.front().realm)
        {
            case Realm::Oblique:
                if (n == 1)
                    break;
                if (n == 2 && us[0].level != us[1].level)
                {
                    result.push_back(id);
                    break;
                }
                throw std::invalid_argument(
                    "implicit summation: an Oblique index must contract exactly "
                    "one upper with one lower slot; a same-level pair or three or "
                    "more occurrences requires an explicit ExplicitSum/NoSum");
            case Realm::Orthonormal:
                if (n == 1)
                    break;
                if (n == 2)
                {
                    result.push_back(id);
                    break;
                }
                throw std::invalid_argument(
                    "implicit summation: an Orthonormal index occurs three or more "
                    "times; this requires an explicit ExplicitSum/NoSum");
            case Realm::Collection:
            case Realm::Label: break; // never auto-contract
        }
    }
    return result;
}

auto materialize(Context& ctx, Expr const* e, std::set<int> bound)
    -> Expr const*;

// Wrap `body` in an ExplicitSum per id (ids already ascending).
auto wrap_sums(Context& ctx, Expr const* body, std::vector<int> const& ids)
    -> Expr const*
{
    for (int id: ids)
        body = make_explicit_sum(ctx, CountableIndex{id}, body, nullptr);
    return body;
}

auto materialize(Context& ctx, Expr const* e, std::set<int> bound) -> Expr const*
{
    // A pure multilinear term is one Einstein-summation scope: contract its
    // repeated free indices (counted across the whole term, through the
    // contractions) and wrap once.
    if (is_term(e))
        return wrap_sums(ctx, e, contracted_ids(e, bound));

    // Otherwise descend.  A multilinear node with a scope-boundary child scopes
    // each operand independently (the un-distributed-sum deferral); Sum /
    // Difference / ScalarDiv parts are separate scopes; a binder adds its
    // index.
    auto rec = [&](Expr const* x) { return materialize(ctx, x, bound); };
    return visit(
        Overloads{
            [&](Negate const& n) -> Expr const*
            { return make_negate(ctx, rec(n.operand)); },
            [&](Trace const& u) -> Expr const*
            { return make_trace(ctx, rec(u.operand)); },
            [&](VectorInvariant const& u) -> Expr const*
            { return make_vector_invariant(ctx, rec(u.operand)); },
            [&](Transpose const& u) -> Expr const*
            { return make_transpose(ctx, rec(u.operand)); },
            [&](Sum const& s) -> Expr const*
            { return make_sum(ctx, rec(s.left), rec(s.right)); },
            [&](Difference const& d) -> Expr const*
            { return make_difference(ctx, rec(d.left), rec(d.right)); },
            [&](TensorProduct const& p) -> Expr const*
            { return make_tensor_product(ctx, rec(p.left), rec(p.right)); },
            [&](ScalarDiv const& d) -> Expr const*
            { return make_scalar_div(ctx, rec(d.left), rec(d.right)); },
            [&](Dot const& d) -> Expr const*
            { return make_dot(ctx, rec(d.left), rec(d.right)); },
            [&](DDot const& d) -> Expr const*
            { return make_ddot(ctx, rec(d.left), rec(d.right)); },
            [&](DDotAlt const& d) -> Expr const*
            { return make_ddot_alt(ctx, rec(d.left), rec(d.right)); },
            [&](Cross const& c) -> Expr const*
            { return make_cross(ctx, rec(c.left), rec(c.right)); },
            [&](ExplicitSum const& s) -> Expr const*
            {
                bound.insert(s.index.id);
                return make_explicit_sum(
                    ctx, s.index, materialize(ctx, s.body, bound), s.bound);
            },
            [&](NoSum const& s) -> Expr const*
            {
                bound.insert(s.index.id);
                return make_no_sum(
                    ctx, s.index, materialize(ctx, s.body, bound));
            },
            // ScalarLiteral / TensorObject are always terms (handled above).
            [&](auto const&) -> Expr const* { return e; },
        },
        *e);
}

// If `operand` is a null-bound ExplicitSum, return a fresh binder and the body
// renamed to it (so floating the binder up cannot capture a free index in a
// sibling operand).  nullopt otherwise.
auto as_floatable_sum(Context& ctx, Expr const* operand)
    -> std::optional<std::pair<CountableIndex, Expr const*>>
{
    auto const* s = std::get_if<ExplicitSum>(&operand->node);
    if (!s || s->bound)
        return std::nullopt;
    CountableIndex const fresh{ctx.alloc_index_id()};
    return std::pair{
        fresh, substitute_index_id(ctx, s->body, s->index.id, fresh.id)};
}

// Pull a null-bound ExplicitSum out of a *multilinear* operand position of
// `node` to the head: op(Σ_i A, B) → Σ_i op(A, B), recursively, so a term ends
// up as a stack of binders over a sum-free body (Fubini-ordered later by
// canon).  The floatable positions are exactly those linear in the summed
// operand — both legs of TensorProduct / Dot / DDot / DDotAlt / Cross, the
// numerator of ScalarDiv, and the operand of Negate / Trace / VectorInvariant /
// Transpose.  Sum/Difference operands and a ScalarDiv denominator are NOT
// floatable (Σ would change meaning).
auto pull_sums_up(Context& ctx, Expr const* node) -> Expr const*
{
    using Mk2 = Expr const* (*)(Context&, Expr const*, Expr const*);
    using Mk1 = Expr const* (*)(Context&, Expr const*);

    // Float from a binary op's left or right leg (both multilinear).
    auto bin = [&](Expr const* l, Expr const* r, Mk2 mk) -> Expr const*
    {
        if (auto f = as_floatable_sum(ctx, l))
            return make_explicit_sum(
                ctx, f->first, pull_sums_up(ctx, mk(ctx, f->second, r)));
        if (auto f = as_floatable_sum(ctx, r))
            return make_explicit_sum(
                ctx, f->first, pull_sums_up(ctx, mk(ctx, l, f->second)));
        return node;
    };
    // Float from a unary op's operand.
    auto un = [&](Expr const* x, Mk1 mk) -> Expr const*
    {
        if (auto f = as_floatable_sum(ctx, x))
            return make_explicit_sum(
                ctx, f->first, pull_sums_up(ctx, mk(ctx, f->second)));
        return node;
    };

    return visit(
        Overloads{
            [&](TensorProduct const& p) -> Expr const*
            { return bin(p.left, p.right, &make_tensor_product); },
            [&](Dot const& p) -> Expr const*
            { return bin(p.left, p.right, &make_dot); },
            [&](DDot const& p) -> Expr const*
            { return bin(p.left, p.right, &make_ddot); },
            [&](DDotAlt const& p) -> Expr const*
            { return bin(p.left, p.right, &make_ddot_alt); },
            [&](Cross const& p) -> Expr const*
            { return bin(p.left, p.right, &make_cross); },
            // ScalarDiv: numerator only (the denominator is not linear in Σ).
            [&](ScalarDiv const& d) -> Expr const*
            {
                if (auto f = as_floatable_sum(ctx, d.left))
                    return make_explicit_sum(
                        ctx,
                        f->first,
                        pull_sums_up(
                            ctx, make_scalar_div(ctx, f->second, d.right)));
                return node;
            },
            [&](Negate const& n) -> Expr const*
            { return un(n.operand, &make_negate); },
            [&](Trace const& u) -> Expr const*
            { return un(u.operand, &make_trace); },
            [&](VectorInvariant const& u) -> Expr const*
            { return un(u.operand, &make_vector_invariant); },
            [&](Transpose const& u) -> Expr const*
            { return un(u.operand, &make_transpose); },
            [&](auto const&) -> Expr const* { return node; },
        },
        *node);
}

// Float every null-bound ExplicitSum out of multilinear positions to the head
// of its term (bottom-up, so an already-floated operand is itself a binder by
// the time its parent is visited).
auto float_sums(Context& ctx, Expr const* e) -> Expr const*
{
    return rewrite_tree(
        ctx, e, [](Context& c, Expr const* n) { return pull_sums_up(c, n); });
}

} // namespace

namespace steps
{

auto canonicalize(Context& ctx, Expr const* e) -> Expr const*
{
    return canon(ctx, float_sums(ctx, materialize(ctx, e, {})), 0);
}

auto implicitize(Context& ctx, Expr const* e) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [](Context&, Expr const* node) -> Expr const*
        {
            auto const* s = std::get_if<ExplicitSum>(&node->node);
            if (!s || s->bound)
                return node;
            std::vector<int> ids;
            try
            {
                ids = contracted_ids(s->body, {});
            }
            catch (std::invalid_argument const&)
            {
                return node;
            }
            if (std::find(ids.begin(), ids.end(), s->index.id) != ids.end())
                return s->body; // redundant explicit sum → back to implicit
            return node;
        });
}

auto contract_identity(Context& ctx, Expr const* e) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [](Context&, Expr const* node) -> Expr const*
        {
            auto const* d = std::get_if<Dot>(&node->node);
            if (!d)
                return node;
            auto is_identity = [](Expr const* x)
            {
                auto const* t = std::get_if<TensorObject>(&x->node);
                return t && t->traits
                       && t->traits->well_known == WellKnownKind::Identity;
            };
            // I · x → x  and  x · I → x  (the identity acts as identity).
            if (is_identity(d->left))
                return d->right;
            if (is_identity(d->right))
                return d->left;
            return node;
        });
}

auto distribute_contraction(Context& ctx, Expr const* e) -> Expr const*
{
    using Mk = Expr const* (*)(Context&, Expr const*, Expr const*);
    auto is_scalar = [](Expr const* x)
    { return infer_rank(x) == std::optional<int>{0}; };
    auto distribute = [&](Context& c,
                          Expr const* node,
                          Expr const* l,
                          Expr const* r,
                          Mk mk) -> Expr const*
    {
        if (auto const* rp = std::get_if<TensorProduct>(&r->node))
        {
            // op(L, A⊗B): float a scalar near leg out, else contract with the
            // near vector A.
            if (is_scalar(rp->left))
                return make_tensor_product(c, rp->left, mk(c, l, rp->right));
            return make_tensor_product(c, mk(c, l, rp->left), rp->right);
        }
        if (auto const* lp = std::get_if<TensorProduct>(&l->node))
        {
            // op(A⊗B, R): float a scalar near leg (B) out, else contract with
            // the near vector B.
            if (is_scalar(lp->right))
                return make_tensor_product(c, mk(c, lp->left, r), lp->right);
            return make_tensor_product(c, lp->left, mk(c, lp->right, r));
        }
        return node;
    };
    auto one_pass = [&](Expr const* x)
    {
        return rewrite_tree(
            ctx,
            x,
            [&](Context& c, Expr const* node) -> Expr const*
            {
                if (auto const* d = std::get_if<Dot>(&node->node))
                    return distribute(c, node, d->left, d->right, &make_dot);
                if (auto const* cr = std::get_if<Cross>(&node->node))
                    return distribute(c, node, cr->left, cr->right, &make_cross);
                return node;
            });
    };
    // Distribute one level per pass; iterate to a fixpoint (rewrite_tree reuses
    // the pointer when nothing changes).
    for (Expr const* cur = e;;)
    {
        Expr const* next = one_pass(cur);
        if (next == cur)
            return cur;
        cur = next;
    }
}

namespace
{

// Split a (possibly scalar-weighted) dyad into its scalar factors and exactly
// two non-scalar legs, or nullopt if it is not a 2-leg product.  E.g.
// s ⊗ (a ⊗ b) → ({s}, a, b).
struct DyadSplit final
{
    std::vector<Expr const*> scalars;
    Expr const* leg0;
    Expr const* leg1;
};
auto split_dyad(Expr const* e) -> std::optional<DyadSplit>
{
    std::vector<Expr const*> flat;
    flatten_factors(e, flat);
    std::vector<Expr const*> scalars;
    std::vector<Expr const*> legs;
    for (auto const* f: flat)
    {
        if (infer_rank(f) == std::optional<int>{0})
            scalars.push_back(f);
        else
            legs.push_back(f);
    }
    if (legs.size() != 2)
        return std::nullopt;
    return DyadSplit{std::move(scalars), legs[0], legs[1]};
}

// Left-fold factors into a TensorProduct (one factor returns itself).
auto product_of(Context& ctx, std::vector<Expr const*> const& factors)
    -> Expr const*
{
    Expr const* p = nullptr;
    for (auto const* f: factors)
        p = p ? make_tensor_product(ctx, p, f) : f;
    return p;
}

// Expand one double-dot of two dyads by definition, distributing through sums
// and summation binders.  `alt` selects the pairing:
//   :  (a⊗b):(c⊗d)  = (a·c)(b·d)
//   ·· (a⊗b)··(c⊗d) = (a·d)(b·c)
auto dd_expand(Context& ctx, Expr const* l, Expr const* r, bool alt)
    -> Expr const*
{
    auto rebuild = [&](Expr const* a, Expr const* b)
    { return alt ? make_ddot_alt(ctx, a, b) : make_ddot(ctx, a, b); };

    // Distribute over sums on either side.
    if (auto const* s = std::get_if<Sum>(&l->node))
        return make_sum(
            ctx,
            dd_expand(ctx, s->left, r, alt),
            dd_expand(ctx, s->right, r, alt));
    if (auto const* s = std::get_if<Sum>(&r->node))
        return make_sum(
            ctx,
            dd_expand(ctx, l, s->left, alt),
            dd_expand(ctx, l, s->right, alt));

    // Pull a summation binder out (renaming to a fresh id to avoid capture).
    if (auto const* es = std::get_if<ExplicitSum>(&l->node); es && !es->bound)
    {
        CountableIndex fresh{ctx.alloc_index_id()};
        auto const* body =
            substitute_index_id(ctx, es->body, es->index.id, fresh.id);
        return make_explicit_sum(ctx, fresh, dd_expand(ctx, body, r, alt));
    }
    if (auto const* es = std::get_if<ExplicitSum>(&r->node); es && !es->bound)
    {
        CountableIndex fresh{ctx.alloc_index_id()};
        auto const* body =
            substitute_index_id(ctx, es->body, es->index.id, fresh.id);
        return make_explicit_sum(ctx, fresh, dd_expand(ctx, l, body, alt));
    }

    // Core dyad rule.
    auto const ls = split_dyad(l);
    auto const rs = split_dyad(r);
    if (!ls || !rs)
        return rebuild(l, r); // not a dyad pair — leave it

    std::vector<Expr const*> factors = ls->scalars;
    factors.insert(factors.end(), rs->scalars.begin(), rs->scalars.end());
    factors.push_back(make_dot(ctx, ls->leg0, alt ? rs->leg1 : rs->leg0));
    factors.push_back(make_dot(ctx, ls->leg1, alt ? rs->leg0 : rs->leg1));
    return product_of(ctx, factors);
}

} // namespace

auto expand_double_dot(Context& ctx, Expr const* e) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [](Context& c, Expr const* node) -> Expr const*
        {
            Expr const* out = node;
            if (auto const* d = std::get_if<DDot>(&node->node))
                out = dd_expand(c, d->left, d->right, /*alt=*/false);
            else if (auto const* d = std::get_if<DDotAlt>(&node->node))
                out = dd_expand(c, d->left, d->right, /*alt=*/true);
            // Preserve the no-op pointer contract when nothing actually
            // changed.
            return structural_eq(out, node) ? node : out;
        });
}

namespace
{

// Is e a well-known symmetric tensor (I, δ, g), so transpose(e) = e?
auto is_symmetric_well_known(Expr const* e) -> bool
{
    auto const* t = std::get_if<TensorObject>(&e->node);
    if (!t || !t->traits || !t->traits->well_known)
        return false;
    switch (*t->traits->well_known)
    {
        case WellKnownKind::Identity:
        case WellKnownKind::Delta:
        case WellKnownKind::Metric: return true;
        case WellKnownKind::LeviCivita: return false;
    }
    return false; // GCOV_EXCL_LINE
}

// Apply a linear rank-2 operation (tr/vec/transpose) by its definition,
// distributing over sums/negation and acting on each dyad.  `dyad_rule` builds
// the result from a dyad's scalar factors and its two legs; `make_node`
// rebuilds the original unary node when the operand is not reducible.
template <typename DyadRule, typename MakeNode>
auto expand_unary(
    Context& ctx, Expr const* operand, DyadRule dyad_rule, MakeNode make_node)
    -> Expr const*
{
    auto recur = [&](Expr const* x)
    { return expand_unary(ctx, x, dyad_rule, make_node); };

    if (auto const* s = std::get_if<Sum>(&operand->node))
        return make_sum(ctx, recur(s->left), recur(s->right));
    if (auto const* d = std::get_if<Difference>(&operand->node))
        return make_difference(ctx, recur(d->left), recur(d->right));
    if (auto const* n = std::get_if<Negate>(&operand->node))
        return make_negate(ctx, recur(n->operand));
    if (auto const sp = split_dyad(operand))
        return dyad_rule(*sp);
    return make_node(operand); // not reducible — leave the op in place
}

} // namespace

auto expand_dyad_ops(Context& ctx, Expr const* e) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [](Context& c, Expr const* node) -> Expr const*
        {
            Expr const* out = node;
            if (auto const* u = std::get_if<Trace>(&node->node))
            {
                // tr(s a⊗b) = s (a·b)
                out = expand_unary(
                    c,
                    u->operand,
                    [&](DyadSplit const& sp)
                    {
                        auto fs = sp.scalars;
                        fs.push_back(make_dot(c, sp.leg0, sp.leg1));
                        return product_of(c, fs);
                    },
                    [&](Expr const* op) { return make_trace(c, op); });
            }
            else if (auto const* u = std::get_if<VectorInvariant>(&node->node))
            {
                // vec(s a⊗b) = s (a×b)
                out = expand_unary(
                    c,
                    u->operand,
                    [&](DyadSplit const& sp)
                    {
                        auto fs = sp.scalars;
                        fs.push_back(make_cross(c, sp.leg0, sp.leg1));
                        return product_of(c, fs);
                    },
                    [&](Expr const* op)
                    { return make_vector_invariant(c, op); });
            }
            else if (auto const* u = std::get_if<Transpose>(&node->node))
            {
                // (s a⊗b)^T = s b⊗a; a symmetric well-known transposes to
                // itself.
                out = expand_unary(
                    c,
                    u->operand,
                    [&](DyadSplit const& sp)
                    {
                        auto fs = sp.scalars;
                        fs.push_back(sp.leg1);
                        fs.push_back(sp.leg0);
                        return product_of(c, fs);
                    },
                    [&](Expr const* op) -> Expr const* {
                        return is_symmetric_well_known(op) ?
                                   op :
                                   make_transpose(c, op);
                    });
            }
            return structural_eq(out, node) ? node : out;
        });
}

auto unroll_sums(Context& ctx, Expr const* e) -> Expr const*
{
    // Expose implicit Einstein sums first, so an implicit δ_ii (a trace)
    // unrolls the same as an explicit Σ_i δ_ii.  Ill-formed implicit sums
    // (which materialize rejects) just have nothing to unroll, so fall back.
    Expr const* m = e;
    try
    {
        m = materialize(ctx, e, {});
    }
    catch (std::invalid_argument const&)
    {
        m = e;
    }
    auto const* out = rewrite_tree(
        ctx,
        m,
        [](Context& ctx, Expr const* e) -> Expr const*
        {
            auto const* s = std::get_if<ExplicitSum>(&e->node);
            if (!s || s->bound)
                return e; // symbolic bound — cannot unroll

            IndexSpace const* space = find_index_space(e, s->index.id);
            if (!space)
                return e; // index space not found in body — leave unchanged

            auto const vals = space->values();
            Expr const* result = nullptr;
            for (int v: vals)
            {
                Expr const* term =
                    substitute(ctx, s->body, s->index.id, ConcreteIndex{v});
                result = result ? make_sum(ctx, result, term) : term;
            }
            return result ? result : make_scalar(ctx, Rational{0});
        });
    // Preserve the no-op: if materialization + unrolling changed nothing
    // (no concrete sum, implicit or explicit), return the original input.
    return structural_eq(out, e) ? e : out;
}

auto eval_delta_concrete(Context& ctx, Expr const* e) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [](Context& ctx, Expr const* e) -> Expr const*
        {
            auto const* t = std::get_if<TensorObject>(&e->node);
            if (!t || !t->traits)
                return e;
            if (t->traits->well_known != WellKnownKind::Delta)
                return e;
            if (t->slots.size() != 2)
                return e;

            auto const* c0 = std::get_if<ConcreteIndex>(&*t->slots[0].index);
            auto const* c1 = std::get_if<ConcreteIndex>(&*t->slots[1].index);
            if (!c0 || !c1)
                return e;

            return make_scalar(ctx, Rational{c0->value == c1->value ? 1 : 0});
        });
}

auto eval_eps_concrete(Context& ctx, Expr const* e) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [](Context& ctx, Expr const* e) -> Expr const*
        {
            auto const* t = std::get_if<TensorObject>(&e->node);
            if (!t || !t->traits
                || t->traits->well_known != WellKnownKind::LeviCivita)
                return e;

            // Every slot must carry a concrete value to evaluate the symbol.
            std::vector<int> vals;
            vals.reserve(t->slots.size());
            for (auto const& sb: t->slots)
            {
                if (!sb.index)
                    return e;
                auto const* c = std::get_if<ConcreteIndex>(&*sb.index);
                if (!c)
                    return e; // a symbolic index remains — cannot evaluate
                vals.push_back(c->value);
            }

            // The permutation symbol: 0 on any repeated value, else the sign of
            // the permutation (parity of the inversion count).
            int sign = 1;
            for (std::size_t i = 0; i < vals.size(); ++i)
                for (std::size_t j = i + 1; j < vals.size(); ++j)
                {
                    if (vals[i] == vals[j])
                        return make_scalar(ctx, Rational{0});
                    if (vals[i] > vals[j])
                        sign = -sign;
                }
            return make_scalar(ctx, Rational{sign});
        });
}

auto fold_arithmetic(Context& ctx, Expr const* e) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [](Context& ctx, Expr const* e) -> Expr const*
        {
            return visit(
                Overloads{
                    [&](Sum const& s) -> Expr const*
                    {
                        auto const* l =
                            std::get_if<ScalarLiteral>(&s.left->node);
                        auto const* r =
                            std::get_if<ScalarLiteral>(&s.right->node);
                        if (l && r)
                            return make_scalar(ctx, l->value + r->value);
                        if (l && l->value == Rational{0})
                            return s.right;
                        if (r && r->value == Rational{0})
                            return s.left;
                        // X + (-Y) → X - Y
                        if (std::holds_alternative<Negate>(s.right->node))
                        {
                            auto const& neg = std::get<Negate>(s.right->node);
                            return make_difference(ctx, s.left, neg.operand);
                        }
                        // (-X) + Y → Y - X  (Negate on left: move to right as
                        // Difference)
                        if (std::holds_alternative<Negate>(s.left->node))
                        {
                            auto const& neg = std::get<Negate>(s.left->node);
                            return make_difference(ctx, s.right, neg.operand);
                        }
                        return e;
                    },
                    [&](Difference const& s) -> Expr const*
                    {
                        auto const* l =
                            std::get_if<ScalarLiteral>(&s.left->node);
                        auto const* r =
                            std::get_if<ScalarLiteral>(&s.right->node);
                        if (l && r)
                            return make_scalar(ctx, l->value - r->value);
                        if (r && r->value == Rational{0})
                            return s.left;
                        // X - (-Y) → X + Y
                        if (std::holds_alternative<Negate>(s.right->node))
                        {
                            auto const& neg = std::get<Negate>(s.right->node);
                            return make_sum(ctx, s.left, neg.operand);
                        }
                        return e;
                    },
                    [&](TensorProduct const& s) -> Expr const*
                    {
                        auto const* l =
                            std::get_if<ScalarLiteral>(&s.left->node);
                        auto const* r =
                            std::get_if<ScalarLiteral>(&s.right->node);
                        if (l && r)
                            return make_scalar(ctx, l->value * r->value);
                        if (l && l->value == Rational{0})
                            return make_scalar(ctx, Rational{0});
                        if (r && r->value == Rational{0})
                            return make_scalar(ctx, Rational{0});
                        if (l && l->value == Rational{1})
                            return s.right;
                        if (r && r->value == Rational{1})
                            return s.left;
                        // (-A)(-B) = A*B
                        auto const* nl = std::get_if<Negate>(&s.left->node);
                        auto const* nr = std::get_if<Negate>(&s.right->node);
                        if (nl && nr)
                            return make_tensor_product(
                                ctx, nl->operand, nr->operand);
                        // (-A) * B → -(A * B)  and  A * (-B) → -(A * B)
                        // Pull the sign out so it can be absorbed by Sum/Diff.
                        if (nl)
                            return make_negate(
                                ctx,
                                make_tensor_product(ctx, nl->operand, s.right));
                        if (nr)
                            return make_negate(
                                ctx,
                                make_tensor_product(ctx, s.left, nr->operand));
                        return e;
                    },
                    [&](ScalarDiv const& s) -> Expr const*
                    {
                        auto const* l =
                            std::get_if<ScalarLiteral>(&s.left->node);
                        auto const* r =
                            std::get_if<ScalarLiteral>(&s.right->node);
                        if (!l || !r)
                            return e;
                        return make_scalar(ctx, l->value / r->value);
                    },
                    [&](Negate const& n) -> Expr const*
                    {
                        auto const* v =
                            std::get_if<ScalarLiteral>(&n.operand->node);
                        if (v)
                            return make_scalar(ctx, -v->value);
                        // -(-A) = A
                        if (auto const* nn =
                                std::get_if<Negate>(&n.operand->node))
                            return nn->operand;
                        return e;
                    },
                    [&](auto const&) -> Expr const* { return e; },
                },
                *e);
        });
}

auto expand_products(Context& ctx, Expr const* e) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [](Context& ctx, Expr const* e) -> Expr const*
        {
            return visit(
                Overloads{
                    [&](TensorProduct const& p) -> Expr const*
                    {
                        return distribute_any(
                            ctx,
                            p.left,
                            p.right,
                            [&](Expr const* a, Expr const* b)
                            { return make_tensor_product(ctx, a, b); });
                    },
                    [&](Dot const& p) -> Expr const*
                    {
                        return distribute_any(
                            ctx,
                            p.left,
                            p.right,
                            [&](Expr const* a, Expr const* b)
                            { return make_dot(ctx, a, b); });
                    },
                    [&](DDot const& p) -> Expr const*
                    {
                        return distribute_any(
                            ctx,
                            p.left,
                            p.right,
                            [&](Expr const* a, Expr const* b)
                            { return make_ddot(ctx, a, b); });
                    },
                    [&](DDotAlt const& p) -> Expr const*
                    {
                        return distribute_any(
                            ctx,
                            p.left,
                            p.right,
                            [&](Expr const* a, Expr const* b)
                            { return make_ddot_alt(ctx, a, b); });
                    },
                    [&](Cross const& p) -> Expr const*
                    {
                        return distribute_any(
                            ctx,
                            p.left,
                            p.right,
                            [&](Expr const* a, Expr const* b)
                            { return make_cross(ctx, a, b); });
                    },
                    [&](auto const&) -> Expr const* { return e; },
                },
                *e);
        });
}

auto expand_eps(Context& ctx, Expr const* e) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [](Context& ctx, Expr const* e) -> Expr const*
        {
            auto const* t = std::get_if<TensorObject>(&e->node);
            if (!t || !t->traits)
                return e;
            if (t->traits->well_known != WellKnownKind::LeviCivita)
                return e;
            if (t->slots.size() != 3)
                return e; // only rank-3 supported

            Realm realm = t->slots[0].slot.realm;
            IndexSpace const* space = t->slots[0].slot.space;
            if (!space)
                return e;

            auto const vals = space->values();
            if (vals.size() != 3)
                return e; // only 3-value spaces supported

            // Verify all slots are bound.
            for (auto const& sb: t->slots)
                if (!sb.index)
                    return e;

            // All 3! permutations of {0,1,2} mapped to val indices, with sign.
            struct Perm
            {
                std::array<int, 3> p;
                int sign;
            };
            static constexpr std::array<Perm, 6> perms = {{
                {{0, 1, 2}, +1},
                {{0, 2, 1}, -1},
                {{1, 0, 2}, -1},
                {{1, 2, 0}, +1},
                {{2, 0, 1}, +1},
                {{2, 1, 0}, -1},
            }};

            Expr const* result = nullptr;
            for (auto const& perm: perms)
            {
                // δ^{vals[perm[k]]}_{slot_k} for k = 0,1,2
                Expr const* term = nullptr;
                for (int k = 0; k < 3; ++k)
                {
                    auto const& sb = t->slots[k];
                    Expr const* d = make_delta(
                        ctx,
                        realm,
                        space,
                        Level::Upper,
                        sb.slot.level,
                        IndexAssoc{ConcreteIndex{vals[perm.p[k]]}},
                        *sb.index);
                    term = term ? make_tensor_product(ctx, term, d) : d;
                }
                if (perm.sign < 0)
                    term = make_negate(ctx, term);
                result = result ? make_sum(ctx, result, term) : term;
            }
            return result ? result : make_scalar(ctx, Rational{0});
        });
}

auto fold_sums(Context& ctx, Expr const* e) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [](Context& ctx, Expr const* e) -> Expr const*
        {
            // Collect flat list of addends from this Sum tree.
            std::vector<Expr const*> addends;
            collect_addends(e, addends);
            if (addends.size() < 2)
                return e;

            CountableIndex fresh{ctx.alloc_index_id()};

            // Try each addend as the anchor for a cycle search.
            for (std::size_t anchor = 0; anchor < addends.size(); ++anchor)
            {
                IndexSpace const* space =
                    find_space_from_concrete(addends[anchor]);
                if (!space)
                    continue;

                auto const vals = space->values();
                if (vals.size() < 2 || vals.size() > addends.size())
                    continue;

                std::vector<int> cvals;
                collect_concrete_values(addends[anchor], cvals);

                for (int try_val: cvals)
                {
                    if (std::find(vals.begin(), vals.end(), try_val)
                        == vals.end())
                        continue;

                    auto const* templ = substitute_concrete(
                        ctx, addends[anchor], try_val, fresh);
                    if (templ == addends[anchor])
                        continue;

                    // Find one addend per remaining value, without reuse.
                    std::set<int> used = {try_val};
                    std::vector<std::size_t> matched = {anchor};

                    for (int v: vals)
                    {
                        if (used.count(v))
                            continue;
                        for (std::size_t k = 0; k < addends.size(); ++k)
                        {
                            if (std::find(matched.begin(), matched.end(), k)
                                != matched.end())
                                continue;
                            auto const* cand = substitute(
                                ctx, templ, fresh.id, ConcreteIndex{v});
                            if (structural_eq(cand, addends[k]))
                            {
                                used.insert(v);
                                matched.push_back(k);
                                break;
                            }
                        }
                    }

                    if (used.size() != vals.size())
                        continue;

                    // Cycle found: fold matched addends into ExplicitSum.
                    auto const* folded = make_explicit_sum(ctx, fresh, templ);
                    if (matched.size() == addends.size())
                        return folded;

                    // Rebuild sum with unmatched addends + folded.
                    Expr const* result = folded;
                    for (std::size_t k = 0; k < addends.size(); ++k)
                    {
                        if (std::find(matched.begin(), matched.end(), k)
                            != matched.end())
                            continue;
                        result = make_sum(ctx, result, addends[k]);
                    }
                    return result;
                }
            }

            return e; // no fold pattern matched
        });
}

auto contract_delta(Context& ctx, Expr const* e) -> Expr const*
{
    // Expose implicit Einstein sums first, so the step fires on a pre-canonical
    // δ_ij δ_ij (sums still implicit) the same as on an explicit Σ form.  Track
    // whether any contraction actually fired, so a genuine no-op returns the
    // original input untouched (no materialized sums leak out).
    //
    // materialize throws on ill-formed implicit sums (e.g. an Oblique
    // same-level pair); for this step that just means "nothing to contract", so
    // we fall back to the original expression rather than propagating the
    // error.
    bool fired = false;
    Expr const* m = e;
    try
    {
        m = materialize(ctx, e, {});
    }
    catch (std::invalid_argument const&)
    {
        return e;
    }
    auto const* out = rewrite_tree(
        ctx,
        m,
        [&fired](Context& ctx, Expr const* e) -> Expr const*
        {
            auto const* s = std::get_if<ExplicitSum>(&e->node);
            if (!s || s->bound)
                return e;

            auto const* prod = std::get_if<TensorProduct>(&s->body->node);
            if (!prod)
                return e;

            auto const* d1 = std::get_if<TensorObject>(&prod->left->node);
            auto const* d2 = std::get_if<TensorObject>(&prod->right->node);
            if (!d1 || !d2)
                return e;
            if (!d1->traits || d1->traits->well_known != WellKnownKind::Delta)
                return e;
            if (!d2->traits || d2->traits->well_known != WellKnownKind::Delta)
                return e;
            if (d1->slots.size() != 2 || d2->slots.size() != 2)
                return e;

            int sum_id = s->index.id;

            // Find which slot carries the summation index (returns slot index,
            // or -1 when not found).
            auto find_sum_slot = [sum_id](TensorObject const& d) -> int
            {
                for (int i = 0; i < 2; ++i)
                {
                    if (!d.slots[i].index)
                        continue;
                    if (auto const* ci =
                            std::get_if<CountableIndex>(&*d.slots[i].index))
                        if (ci->id == sum_id)
                            return i;
                }
                return -1;
            };

            int s1 = find_sum_slot(*d1);
            int s2 = find_sum_slot(*d2);
            if (s1 < 0 || s2 < 0)
                return e;

            // The contracted slots must share level, realm, and space.
            if (d1->slots[s1].slot.level != d2->slots[s2].slot.level)
                return e;
            if (d1->slots[s1].slot.realm != d2->slots[s2].slot.realm)
                return e;
            if (d1->slots[s1].slot.space != d2->slots[s2].slot.space)
                return e;

            auto const& sur1 = d1->slots[1 - s1]; // surviving slot in d1
            auto const& sur2 = d2->slots[1 - s2]; // surviving slot in d2
            if (!sur1.index || !sur2.index)
                return e;

            fired = true;
            return make_delta(
                ctx,
                sur1.slot.realm,
                sur1.slot.space,
                sur1.slot.level,
                sur2.slot.level,
                *sur1.index,
                *sur2.index);
        });
    // No contraction fired — return the original input, untouched.  When it
    // did, strip any explicit sums materialization added but the contraction
    // left behind, so the result stays in implicit form (e.g. δ_ij δ_ij → δ_ii,
    // not Σ_i δ_ii).
    return fired ? implicitize(ctx, out) : e;
}

namespace
{

// A non-contracted slot of a Levi-Civita symbol: its level and index, used to
// build the surviving Kronecker deltas of the generalized-Kronecker result.
struct FreeSlot
{
    Level level;
    IndexAssoc index;
};

// Sign of the permutation that re-orders `orig` (a list of index ids) into
// `target` (the same ids, permuted): +1 if even, -1 if odd.  Computed as the
// parity of the inversion count of the rank sequence.
auto reorder_sign(std::vector<int> const& orig, std::vector<int> const& target)
    -> int
{
    std::vector<int> seq;
    seq.reserve(orig.size());
    for (int id: orig)
    {
        auto it = std::find(target.begin(), target.end(), id);
        seq.push_back(static_cast<int>(it - target.begin()));
    }
    int inversions = 0;
    for (std::size_t a = 0; a < seq.size(); ++a)
        for (std::size_t b = a + 1; b < seq.size(); ++b)
            if (seq[a] > seq[b])
                ++inversions;
    return (inversions % 2 == 0) ? +1 : -1;
}

// Expand the q×q Kronecker determinant det(D), where
// D[r][c] = δ connecting free_a[r] (upper-side) and free_b[c] (lower-side).
// Returns a Sum/Negate tree of signed δ-products (scalar 1 when q == 0),
// mirroring the construction style of expand_eps.
auto build_kronecker_det(
    Context& ctx,
    Realm realm,
    IndexSpace const* space,
    std::vector<FreeSlot> const& free_a,
    std::vector<FreeSlot> const& free_b) -> Expr const*
{
    int const q = static_cast<int>(free_a.size());
    if (q == 0)
        return make_scalar(ctx, Rational{1});

    std::vector<int> perm(q);
    for (int i = 0; i < q; ++i)
        perm[i] = i;

    Expr const* result = nullptr;
    do
    {
        int inversions = 0;
        for (int a = 0; a < q; ++a)
            for (int b = a + 1; b < q; ++b)
                if (perm[a] > perm[b])
                    ++inversions;

        Expr const* term = nullptr;
        for (int r = 0; r < q; ++r)
        {
            auto const& fa = free_a[r];
            auto const& fb = free_b[perm[r]];
            Expr const* d = make_delta(
                ctx, realm, space, fa.level, fb.level, fa.index, fb.index);
            term = term ? make_tensor_product(ctx, term, d) : d;
        }
        if (inversions % 2 != 0)
            term = make_negate(ctx, term);
        result = result ? make_sum(ctx, result, term) : term;
    } while (std::next_permutation(perm.begin(), perm.end()));

    return result;
}

// Rebuild a node by recursing into its children with `rec` (an
// Expr const*(Expr const*) callable).  Pointer-preserving: returns e unchanged
// when no child changed.  Used to drive a top-down rewrite.
template <typename Rec>
auto map_children(Context& ctx, Expr const* e, Rec const& rec) -> Expr const*
{
    return visit(
        Overloads{
            [&](TensorObject const&) -> Expr const* { return e; },
            [&](ScalarLiteral const&) -> Expr const* { return e; },
            [&](Negate const& n) -> Expr const*
            {
                auto* o = rec(n.operand);
                return o == n.operand ? e : make_negate(ctx, o);
            },
            [&](Trace const& u) -> Expr const*
            {
                auto* o = rec(u.operand);
                return o == u.operand ? e : make_trace(ctx, o);
            },
            [&](VectorInvariant const& u) -> Expr const*
            {
                auto* o = rec(u.operand);
                return o == u.operand ? e : make_vector_invariant(ctx, o);
            },
            [&](Transpose const& u) -> Expr const*
            {
                auto* o = rec(u.operand);
                return o == u.operand ? e : make_transpose(ctx, o);
            },
            [&](Sum const& s) -> Expr const*
            {
                auto* l = rec(s.left);
                auto* r = rec(s.right);
                return (l == s.left && r == s.right) ? e : make_sum(ctx, l, r);
            },
            [&](Difference const& s) -> Expr const*
            {
                auto* l = rec(s.left);
                auto* r = rec(s.right);
                return (l == s.left && r == s.right) ?
                           e :
                           make_difference(ctx, l, r);
            },
            [&](TensorProduct const& s) -> Expr const*
            {
                auto* l = rec(s.left);
                auto* r = rec(s.right);
                return (l == s.left && r == s.right) ?
                           e :
                           make_tensor_product(ctx, l, r);
            },
            [&](ScalarDiv const& s) -> Expr const*
            {
                auto* l = rec(s.left);
                auto* r = rec(s.right);
                return (l == s.left && r == s.right) ?
                           e :
                           make_scalar_div(ctx, l, r);
            },
            [&](Dot const& s) -> Expr const*
            {
                auto* l = rec(s.left);
                auto* r = rec(s.right);
                return (l == s.left && r == s.right) ? e : make_dot(ctx, l, r);
            },
            [&](DDot const& s) -> Expr const*
            {
                auto* l = rec(s.left);
                auto* r = rec(s.right);
                return (l == s.left && r == s.right) ? e : make_ddot(ctx, l, r);
            },
            [&](DDotAlt const& s) -> Expr const*
            {
                auto* l = rec(s.left);
                auto* r = rec(s.right);
                return (l == s.left && r == s.right) ? e :
                                                       make_ddot_alt(ctx, l, r);
            },
            [&](Cross const& s) -> Expr const*
            {
                auto* l = rec(s.left);
                auto* r = rec(s.right);
                return (l == s.left && r == s.right) ? e :
                                                       make_cross(ctx, l, r);
            },
            [&](ExplicitSum const& s) -> Expr const*
            {
                auto* body = rec(s.body);
                auto* bound = s.bound ? rec(s.bound) : nullptr;
                return (body == s.body && bound == s.bound) ?
                           e :
                           make_explicit_sum(ctx, s.index, body, bound);
            },
            [&](NoSum const& s) -> Expr const*
            {
                auto* body = rec(s.body);
                return body == s.body ? e : make_no_sum(ctx, s.index, body);
            },
        },
        *e);
}

// Attempt the ε-pair contraction anchored at node `e`: peel every nested
// null-bound ExplicitSum (and a leading −1 sign), then look inside the product
// body for exactly two rank-3 Levi-Civita symbols.  Contract them over the
// indices they *share* (summed in both), via the generalized-Kronecker formula,
// keeping any other factors and re-wrapping the non-contracted sums.  So it
// fires on a bare Σ ε ε *and* on Σ ε ε buried in a coordinate product (e.g.
// bac-cab's Σ −ε ε a b c e).  Returns nullptr on no match.
auto try_contract_eps_pair(Context& ctx, Expr const* e) -> Expr const*
{
    // Peel nested null-bound ExplicitSum nodes; collect dummy ids.
    std::vector<int> summed;
    Expr const* body = e;
    while (auto const* s = std::get_if<ExplicitSum>(&body->node))
    {
        if (s->bound)
            return nullptr; // symbolic bound — not handled
        summed.push_back(s->index.id);
        body = s->body;
    }
    if (summed.empty())
        return nullptr;

    // Peel one leading sign (canonical forms carry it as a Negate wrapper).
    bool negated = false;
    if (auto const* n = std::get_if<Negate>(&body->node))
    {
        negated = true;
        body = n->operand;
    }

    // Flatten the product; find exactly two ε's, collect the other factors.
    std::vector<Expr const*> factors;
    flatten_factors(body, factors);
    auto as_eps = [](Expr const* f) -> TensorObject const*
    {
        auto const* t = std::get_if<TensorObject>(&f->node);
        return (t && t->traits
                && t->traits->well_known == WellKnownKind::LeviCivita
                && t->slots.size() == 3) ?
                   t :
                   nullptr;
    };
    TensorObject const* ea = nullptr;
    TensorObject const* eb = nullptr;
    std::vector<Expr const*> others;
    for (auto const* f: factors)
    {
        if (auto const* t = as_eps(f))
        {
            if (!ea)
                ea = t;
            else if (!eb)
                eb = t;
            else
                return nullptr; // more than two ε's — not handled
        }
        else
            others.push_back(f);
    }
    if (!ea || !eb)
        return nullptr;

    // All ε slots must be CountableIndex; gather each ε's id set.
    auto eps_ids = [](TensorObject const& t, std::set<int>& s) -> bool
    {
        for (auto const& sb: t.slots)
        {
            if (!sb.index)
                return false;
            auto const* ci = std::get_if<CountableIndex>(&*sb.index);
            if (!ci)
                return false;
            s.insert(ci->id);
        }
        return true;
    };
    std::set<int> ida, idb;
    if (!eps_ids(*ea, ida) || !eps_ids(*eb, idb))
        return nullptr;

    // The ε-pair contracts over the indices shared by both ε's that are summed.
    std::set<int> const summed_set(summed.begin(), summed.end());
    std::set<int> shared;
    for (int id: ida)
        if (idb.count(id) && summed_set.count(id))
            shared.insert(id);
    if (shared.empty())
        return nullptr;

    // Split each ε into contracted (shared) ids and free slots, in slot order.
    auto split = [&](TensorObject const& eps,
                     std::vector<int>& cids,
                     std::vector<FreeSlot>& free)
    {
        for (auto const& sb: eps.slots)
        {
            int id = std::get<CountableIndex>(*sb.index).id;
            if (shared.count(id))
                cids.push_back(id);
            else
                free.push_back(FreeSlot{sb.slot.level, *sb.index});
        }
    };
    std::vector<int> ca, cb;
    std::vector<FreeSlot> fa, fb;
    split(*ea, ca, fa);
    split(*eb, cb, fb);

    // Each shared index must appear exactly once in each ε.
    int const p = static_cast<int>(shared.size());
    if (static_cast<int>(ca.size()) != p || static_cast<int>(cb.size()) != p)
        return nullptr;
    if (fa.size() != fb.size())
        return nullptr;

    // Sign of re-ordering each ε to [shared-in-ca-order, free].
    std::vector<int> ids_a, ids_b, target_a, target_b;
    for (auto const& sb: ea->slots)
        ids_a.push_back(std::get<CountableIndex>(*sb.index).id);
    for (auto const& sb: eb->slots)
        ids_b.push_back(std::get<CountableIndex>(*sb.index).id);
    target_a = ca;
    for (auto const& f: fa)
        target_a.push_back(std::get<CountableIndex>(f.index).id);
    target_b = ca; // same shared order for the second ε
    for (auto const& f: fb)
        target_b.push_back(std::get<CountableIndex>(f.index).id);
    int sign = reorder_sign(ids_a, target_a) * reorder_sign(ids_b, target_b);

    Realm realm = ea->slots[0].slot.realm;
    IndexSpace const* space = ea->slots[0].slot.space;

    Expr const* det = build_kronecker_det(ctx, realm, space, fa, fb);

    // Scalar weight: p!  (factorial small: p ∈ {1,2,3}).
    int fact = 1;
    for (int k = 2; k <= p; ++k)
        fact *= k;
    Expr const* core =
        (fact == 1) ?
            det :
            make_tensor_product(ctx, make_scalar(ctx, Rational{fact}), det);

    // Re-attach the other (non-ε) factors of the product, preserving their
    // original left-to-right order.  This matters for the non-commuting basis
    // vectors of a dyad: prepending in forward order would reverse them and
    // transpose the result (invisible at rank 1, wrong at rank ≥ 2).
    Expr const* result = core;
    for (auto it = others.rbegin(); it != others.rend(); ++it)
        result = make_tensor_product(ctx, *it, result);
    if (sign < 0)
        result = make_negate(ctx, result);
    if (negated)
        result = make_negate(ctx, result);

    // Re-wrap the summed indices that were not contracted by the ε-pair.
    for (int id: summed)
        if (!shared.count(id))
            result =
                make_explicit_sum(ctx, CountableIndex{id}, result, nullptr);
    return result;
}

// Top-down walk: try the contraction at each node; on a match the whole
// nested-sum subtree is consumed (no recursion into the replacement), so a
// multi-index contraction is performed in one shot at the outermost sum.
auto contract_eps_pair_walk(Context& ctx, Expr const* e) -> Expr const*
{
    if (auto const* r = try_contract_eps_pair(ctx, e))
        return r;
    return map_children(
        ctx,
        e,
        [&ctx](Expr const* x) { return contract_eps_pair_walk(ctx, x); });
}

} // namespace

auto contract_eps_pair(Context& ctx, Expr const* e) -> Expr const*
{
    return contract_eps_pair_walk(ctx, e);
}

auto unroll_sums_for(
    Context& ctx,
    Expr const* e,
    std::vector<CountableIndex> const& indices) -> Expr const*
{
    // Pass 1: unroll ExplicitSum nodes whose index is in `indices`.
    Expr const* result = rewrite_tree(
        ctx,
        e,
        [&indices](Context& ctx, Expr const* e) -> Expr const*
        {
            auto const* s = std::get_if<ExplicitSum>(&e->node);
            if (!s || s->bound)
                return e;

            bool requested = false;
            for (auto const& idx: indices)
                if (idx.id == s->index.id)
                {
                    requested = true;
                    break;
                }
            if (!requested)
                return e;

            IndexSpace const* space = find_index_space(e, s->index.id);
            if (!space)
                return e;

            auto const vals = space->values();
            Expr const* r = nullptr;
            for (int v: vals)
            {
                Expr const* term =
                    substitute(ctx, s->body, s->index.id, ConcreteIndex{v});
                r = r ? make_sum(ctx, r, term) : term;
            }
            return r ? r : make_scalar(ctx, Rational{0});
        });

    // Pass 2: for any requested index that appears free (implicit Einstein
    // sum) in the result, substitute concrete values and form a sum.
    for (auto const& idx: indices)
    {
        if (!has_free_index_for(result, {idx}, {}))
            continue;
        IndexSpace const* space = find_index_space(result, idx.id);
        if (!space)
            continue;
        Expr const* summed = nullptr;
        for (int v: space->values())
        {
            Expr const* term =
                substitute(ctx, result, idx.id, ConcreteIndex{v});
            summed = summed ? make_sum(ctx, summed, term) : term;
        }
        if (summed)
            result = summed;
    }
    return result;
}

auto has_explicit_sum_for(
    Expr const* e, std::vector<CountableIndex> const& indices) -> bool
{
    // Also accept implicit (Einstein-convention) free indices.
    if (has_free_index_for(e, indices, {}))
        return true;

    bool found = false;
    std::function<void(Expr const*)> go = [&](Expr const* node)
    {
        if (found)
            return;
        visit(
            Overloads{
                [&](ExplicitSum const& s)
                {
                    for (auto const& idx: indices)
                        if (idx.id == s.index.id)
                        {
                            found = true;
                            return;
                        }
                    if (!found)
                        go(s.body);
                    if (!found && s.bound)
                        go(s.bound);
                },
                [&](TensorObject const&) {},
                [&](ScalarLiteral const&) {},
                [&](Negate const& n) { go(n.operand); },
                [&](Trace const& u) { go(u.operand); },
                [&](VectorInvariant const& u) { go(u.operand); },
                [&](Transpose const& u) { go(u.operand); },
                [&](Sum const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](Difference const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](TensorProduct const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](ScalarDiv const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](Dot const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](DDot const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](DDotAlt const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](Cross const& s)
                {
                    go(s.left);
                    go(s.right);
                },
                [&](NoSum const& s) { go(s.body); },
            },
            *node);
    };
    go(e);
    return found;
}

auto fold_equal_addends(Context& ctx, Expr const* e) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [](Context& ctx, Expr const* e) -> Expr const*
        {
            std::vector<std::pair<int, Expr const*>> addends;
            collect_signed_addends(e, +1, addends);
            if (addends.size() < 2)
                return e;

            // Group signed addends by structural equality of core expression.
            // Each group accumulates the rational coefficient total.
            std::vector<bool> used(addends.size(), false);
            std::vector<std::pair<Rational, Expr const*>> groups;

            for (std::size_t i = 0; i < addends.size(); ++i)
            {
                if (used[i])
                    continue;
                auto [si, ei] = addends[i];
                auto [ci, corei] = extract_coeff(ei);
                Rational total = ci * Rational{si};
                used[i] = true;

                for (std::size_t j = i + 1; j < addends.size(); ++j)
                {
                    if (used[j])
                        continue;
                    auto [sj, ej] = addends[j];
                    auto [cj, corej] = extract_coeff(ej);
                    if (structural_eq(corei, corej))
                    {
                        total += cj * Rational{sj};
                        used[j] = true;
                    }
                }
                groups.emplace_back(total, corei);
            }

            if (groups.size() == addends.size())
                return e; // no merging happened

            // Rebuild sum from groups.
            Expr const* result = nullptr;
            for (auto const& [coeff, core]: groups)
            {
                if (coeff == 0)
                    continue;
                Expr const* term;
                if (coeff == 1)
                    term = core;
                else if (coeff == -1)
                    term = make_negate(ctx, core);
                else
                    term =
                        make_tensor_product(ctx, make_scalar(ctx, coeff), core);
                result = result ? make_sum(ctx, result, term) : term;
            }
            return result ? result : make_scalar(ctx, Rational{0});
        });
}

} // namespace steps
} // namespace tender
