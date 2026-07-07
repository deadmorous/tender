#include <tender/derivation.hpp>

#include <mpk/mix/util/overloads.hpp>
#include <tender/context.hpp>
#include <tender/nf_lower.hpp>
#include <tender/rewrite.hpp>
#include <tender/summation.hpp>
#include <tender/tensor_order.hpp>

#include <algorithm>
#include <functional>
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
                [&](ScalarFn const& s) { go(s.operand); },
                [&](Pow const& s)
                {
                    go(s.base);
                    go(s.exponent);
                },
                [&](Deriv const& s) { go(s.wrt); },
                [&](Nabla const&) {},
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
            // Copy the whole object so deriv marks (the applied ∂s of a
            // field, vibe 000073) survive index substitution — rebuilding with
            // only name/rank/traits/slots silently drops the derivative.
            TensorObject obj = *t;
            obj.slots = std::move(slots);
            return ctx.make<Expr>(std::move(obj));
        });
}

// Replace every CountableIndex with id `from_id` by the index assoc `to`,
// throughout the tensor-object slots of `e`.  The index→index sibling of
// `substitute` (which maps an index to a concrete value); used by
// `contract_delta` to identify a Kronecker δ's two indices.
auto substitute_index(Context& ctx, Expr const* e, int from_id, IndexAssoc to)
    -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [from_id, &to](Context& ctx, Expr const* e) -> Expr const*
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
                    if (ci->id == from_id)
                    {
                        sb.index = to;
                        changed = true;
                    }
            }
            if (!changed)
                return e;
            // Copy the whole object so deriv marks (the applied ∂s of a
            // field, vibe 000073) survive index substitution — rebuilding with
            // only name/rank/traits/slots silently drops the derivative.
            TensorObject obj = *t;
            obj.slots = std::move(slots);
            return ctx.make<Expr>(std::move(obj));
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
                [&](ScalarFn const& s) { go(s.operand); },
                [&](Pow const& s)
                {
                    go(s.base);
                    go(s.exponent);
                },
                [&](Deriv const& s) { go(s.wrt); },
                [&](Nabla const&) {},
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
                [&](ScalarFn const& s) { go(s.operand); },
                [&](Pow const& s)
                {
                    go(s.base);
                    go(s.exponent);
                },
                [&](Deriv const& s) { go(s.wrt); },
                [&](Nabla const&) {},
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
                    if (sa.slot.basis_id != sb.slot.basis_id)
                        return false;
                    if (!index_assoc_eq(sa.index, sb.index))
                        return false;
                }
                // Applied-derivative marks bear identity (vibe 000077 step D).
                if (ta.deriv_marks.size() != tb->deriv_marks.size())
                    return false;
                for (std::size_t i = 0; i < ta.deriv_marks.size(); ++i)
                {
                    auto const& da = ta.deriv_marks[i];
                    auto const& db = tb->deriv_marks[i];
                    if (da.wrt.chart_id != db.wrt.chart_id
                        || da.wrt.slot != db.wrt.slot || da.free != db.free
                        || da.link != db.link)
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
            [&](ScalarFn const& fa) -> bool
            {
                auto const* fb = std::get_if<ScalarFn>(&b->node);
                return fb && fa.kind == fb->kind
                       && structural_eq(fa.operand, fb->operand);
            },
            [&](Pow const& pa) -> bool
            {
                auto const* pb = std::get_if<Pow>(&b->node);
                return pb && structural_eq(pa.base, pb->base)
                       && structural_eq(pa.exponent, pb->exponent);
            },
            [&](Deriv const& da) -> bool
            {
                auto const* db = std::get_if<Deriv>(&b->node);
                return db && structural_eq(da.wrt, db->wrt);
            },
            // ∇ carries no data — any two ∇ operators are structurally equal.
            [&](Nabla const&) -> bool
            { return std::holds_alternative<Nabla>(b->node); },
        },
        a->node);
}

namespace
{

// The T0 decision procedure: structural_eq of the two canonical forms.  This
// is the comparator simplify_scalars' own helpers use (trig-square pairing,
// fraction-factor bagging) — those must not re-enter simplify_scalars, so
// they call this instead of the public algebraic_eq below.
auto algebraic_eq_t0(Context& ctx, Expr const* a, Expr const* b) -> bool
{
    return structural_eq(
        steps::canonicalize(ctx, a), steps::canonicalize(ctx, b));
}

} // namespace

// Algebraic equality: the T0 decision procedure, plus a scalar fallback for
// what T0 keeps apart — fraction shapes.  Canonical forms distinguish
// x/r + y/r from (x+y)/r, so when T0 says "different", check whether the
// difference simplifies to the literal 0 (vibe 000074).
auto algebraic_eq(Context& ctx, Expr const* a, Expr const* b) -> bool
{
    if (algebraic_eq_t0(ctx, a, b))
        return true;
    Expr const* d = steps::simplify_scalars(
        ctx, steps::canonicalize(ctx, make_difference(ctx, a, b)));
    auto const* s = std::get_if<ScalarLiteral>(&d->node);
    return s && s->value.is_zero();
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
            // Copy the whole object so deriv marks (the applied ∂s of a
            // field, vibe 000073) survive index substitution — rebuilding with
            // only name/rank/traits/slots silently drops the derivative.
            TensorObject obj = *t;
            obj.slots = std::move(slots);
            return ctx.make<Expr>(std::move(obj));
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
            [&](ScalarFn const& s) -> bool
            { return has_free_index_for(s.operand, indices, bound); },
            [&](Pow const& s) -> bool
            {
                return has_free_index_for(s.base, indices, bound)
                       || has_free_index_for(s.exponent, indices, bound);
            },
            [&](Deriv const& s) -> bool
            { return has_free_index_for(s.wrt, indices, bound); },
            [&](Nabla const&) -> bool { return false; },
        },
        *e);
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
            // Scalar fields are component (commuting) values.
            [](ScalarFn const& s) { return is_component_valued(s.operand); },
            [](Pow const& p) {
                return is_component_valued(p.base)
                       && is_component_valued(p.exponent);
            },
            // A differential operator is not a commuting component value
            // (vibe 000077): its position carries meaning.
            [](Deriv const&) { return false; },
            [](Nabla const&) { return false; },
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
            // Scalar fields are rank 0.
            [](ScalarFn const&) -> std::optional<int> { return 0; },
            [](Pow const&) -> std::optional<int> { return 0; },
            // A differential operator's rank is that of the object it
            // differentiates with respect to (vibe 000077): a coordinate ⇒ 0.
            [](Deriv const& d) -> std::optional<int>
            { return infer_rank(d.wrt); },
            // ∇ is a rank-1 vector operator (vibe 000078).
            [](Nabla const&) -> std::optional<int> { return 1; },
        },
        *e);
}

namespace
{

// ---- shared canonicalization helpers ------------------------------------
//
// Index α-renaming (`substitute_index_id` / `substitute_index_ids`), the
// canonical dummy id (`bound_canon_id`), and the implicit-summation detection
// (`is_term` / `collect_term_uses` / `contracted_ids`) live in
// tender/summation.hpp; the symmetry-orbit search (`canon_symmetry_slots`)
// lives in tender/tensor_symmetry.hpp.  Both are shared with the `Nf` lowering,
// which the public `canonicalize` (raise ∘ lower) now drives.  The old
// binary-tree canonicalizer — `canon` / `canon_sum_stack` / `canon_product` /
// `canon_additive`, the `canon_symmetry` Expr wrapper, and the `build_term` /
// `is_rank1_vector` / `reassociate_cross_fence` helpers — was pruned at C15
// (vibe 000058) once the flip made it unreachable.

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

// `is_term`, `collect_term_uses`, and `contracted_ids` now live in
// tender/summation.hpp (shared with the Nf lowering, vibe 000058 / C8).

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
    // The flip (vibe 000058 / C13): canonicalize is now `raise ∘ lower` over
    // the all-`*` normal form `Nf`, replacing the binary-tree `canon`.  The
    // prep `materialize` (realm-implicit → explicit `ExplicitSum`) +
    // `float_sums` (binders to term heads) puts the input in the shape
    // `canonicalize_nf` expects; lowering re-derives implicit summation and the
    // all-`*` regions, and `raise` rebuilds an `Expr` carrying the same
    // explicit binders the rest of the `Expr` pipeline reads.  (`canon` /
    // `canon_sum_stack` are now dead; pruned at C15.)
    return nf::raise(
        ctx,
        *nf::canonicalize_nf(ctx, float_sums(ctx, materialize(ctx, e, {}))));
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
        // See through a Negate or a scalar quotient wrapping either operand — a
        // connection term such as ∂_θ e_θ = −e_r yields op(L, −(A⊗B)), and a
        // symmetrized term like (A + Aᵀ)/2 yields op(L, X/2); in both cases the
        // ⊗ inside must still be fence-distributed (vibes 000073/000075).  Peel
        // the wrappers, distribute, then re-apply: op(L, −X) = −op(L, X) and
        // op(L, X/s) = op(L, X)/s.
        int sign = 1;
        std::vector<Expr const*> divisors;
        auto peel = [&](Expr const*& side)
        {
            for (;;)
            {
                if (auto const* n = std::get_if<Negate>(&side->node))
                {
                    side = n->operand;
                    sign = -sign;
                }
                else if (auto const* q = std::get_if<ScalarDiv>(&side->node))
                {
                    side = q->left;
                    divisors.push_back(q->right);
                }
                else
                    break;
            }
        };
        peel(l);
        peel(r);
        Expr const* res = nullptr;
        if (auto const* rp = std::get_if<TensorProduct>(&r->node))
        {
            // op(L, A⊗B): float a scalar near leg out, else contract with the
            // near vector A.
            res = is_scalar(rp->left) ?
                      make_tensor_product(c, rp->left, mk(c, l, rp->right)) :
                      make_tensor_product(c, mk(c, l, rp->left), rp->right);
        }
        else if (auto const* lp = std::get_if<TensorProduct>(&l->node))
        {
            // op(A⊗B, R): float a scalar near leg (B) out, else contract with
            // the near vector B.
            res = is_scalar(lp->right) ?
                      make_tensor_product(c, mk(c, lp->left, r), lp->right) :
                      make_tensor_product(c, lp->left, mk(c, lp->right, r));
        }
        // Peeling a wrapper may expose a sum that the entry expand_products
        // never saw (it was hidden under the quotient): distribute over it so
        // the next fixpoint pass reaches each addend's ⊗ fence.
        bool const peeled = sign < 0 || !divisors.empty();
        if (!res && peeled)
        {
            if (auto const* s = std::get_if<Sum>(&l->node))
                res = make_sum(c, mk(c, s->left, r), mk(c, s->right, r));
            else if (auto const* d = std::get_if<Difference>(&l->node))
                res = make_difference(c, mk(c, d->left, r), mk(c, d->right, r));
            else if (auto const* s2 = std::get_if<Sum>(&r->node))
                res = make_sum(c, mk(c, l, s2->left), mk(c, l, s2->right));
            else if (auto const* d2 = std::get_if<Difference>(&r->node))
                res =
                    make_difference(c, mk(c, l, d2->left), mk(c, l, d2->right));
            else if (!divisors.empty())
                res = mk(c, l, r); // hoist the quotient out of the contraction
        }
        if (!res)
            return node; // no ⊗ fence to distribute → unchanged
        for (auto const* dv: divisors)
            res = make_scalar_div(c, res, dv);
        return sign < 0 ? make_negate(c, res) : res;
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
    // Self-prepare: distribute products over sums first, so a contraction
    // hidden behind an un-distributed sum — op(L, A⊗B + C⊗D) — becomes
    // reachable (the caller need not run expand_products).  If no contraction
    // actually distributes, undo the prep and return the original (no-op
    // identity).
    Expr const* const prepped = expand_products(ctx, e);
    // Distribute one level per pass; iterate to a fixpoint (rewrite_tree reuses
    // the pointer when nothing changes).
    Expr const* cur = prepped;
    for (;;)
    {
        Expr const* next = one_pass(cur);
        if (next == cur)
            break;
        cur = next;
    }
    return cur == prepped ? e : cur;
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
    // Self-prepare so the caller never has to: distribute products over sums
    // (expand_products) and materialize the implicit Einstein sums with
    // per-term binders (canonicalize).  Distribution matters because the ε-pair
    // contraction leaves a δ-determinant Σ_m … (δ_ab δ_cd − δ_ad δ_cb);
    // expanding it into separate terms lets each contract (a single distributed
    // sum under a binder is otherwise left alone — see the guard below).  Track
    // whether any contraction fired, so a genuine no-op returns the original
    // input untouched (no materialized sums leak out).
    //
    // canonicalize throws on an ill-formed implicit sum (e.g. an Oblique
    // same-level pair); for this step that just means "nothing to contract", so
    // we fall back to the original expression rather than propagating the
    // error.
    bool fired = false;
    Expr const* m = e;
    try
    {
        m = canonicalize(ctx, expand_products(ctx, e));
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
            // Fire at a summation binder Σ_m whose body carries a Kronecker δ
            // with m in one slot: δ identifies m with its other index n, so the
            // sum collapses — drop δ, substitute m := n in the rest, and shed
            // the Σ_m binder.  This generalizes the old δ·δ → δ rule (which is
            // the case where the "rest" is itself a single δ) to contracting a
            // δ against *any* factor: a_i δ_ij → a_j, δ_ij e_i ⊗ e_j → I, etc.
            auto const* s = std::get_if<ExplicitSum>(&e->node);
            if (!s || s->bound)
                return e;
            int const m = s->index.id;

            // Only contract within a single multiplicative term.  If the body
            // is a distributed sum (e.g. the δ-determinant the ε-pair leaves
            // behind, Σ_m … (δ_ab δ_cd − δ_ad δ_cb)), substituting m across the
            // ± would wrongly identify one addend's indices in another.  Peel
            // the remaining binders and sign to the multiplicative core and
            // bail on a distributed sum; let expand_products split the term
            // first.
            {
                Expr const* core = s->body;
                for (bool peeled = true; peeled;)
                {
                    peeled = false;
                    if (auto const* es = std::get_if<ExplicitSum>(&core->node);
                        es && !es->bound)
                        core = es->body, peeled = true;
                    else if (auto const* ns = std::get_if<NoSum>(&core->node))
                        core = ns->body, peeled = true;
                    else if (auto const* ng = std::get_if<Negate>(&core->node))
                        core = ng->operand, peeled = true;
                }
                std::vector<Expr const*> facs;
                flatten_factors(core, facs);
                for (auto const* f: facs)
                    if (std::holds_alternative<Sum>(f->node)
                        || std::holds_alternative<Difference>(f->node))
                        return e;
            }

            // Locate the first δ in the body that carries index m, returning
            // the δ node and the partner index n in its other slot (n must
            // differ from m — a δ_mm self-trace is a dimension count, not a
            // contraction).  rewrite_tree drives the (read-only) traversal, so
            // every node kind is descended without a per-kind arm here.
            Expr const* delta = nullptr;
            IndexAssoc partner;
            IndexSlot m_slot{}; // descriptor of the δ slot carrying m
            rewrite_tree(
                ctx,
                s->body,
                [&](Context&, Expr const* node) -> Expr const*
                {
                    if (delta)
                        return node;
                    auto const* t = std::get_if<TensorObject>(&node->node);
                    if (!t || !t->traits
                        || t->traits->well_known != WellKnownKind::Delta
                        || t->slots.size() != 2)
                        return node;
                    auto const& s0 = t->slots[0];
                    auto const& s1 = t->slots[1];
                    if (!s0.index || !s1.index)
                        return node;
                    // δ joins its two slots, so they must share realm and space
                    // (always true for a make_delta δ).
                    if (s0.slot.realm != s1.slot.realm
                        || s0.slot.space != s1.slot.space)
                        return node;
                    // It must be a genuine Kronecker, not an oblique same-level
                    // "δ" (really the metric g): in an Oblique realm the two
                    // slots must straddle the upper/lower divide; Orthonormal
                    // makes them interchangeable, so any pairing is a
                    // Kronecker.
                    if (s0.slot.realm != Realm::Orthonormal
                        && s0.slot.level == s1.slot.level)
                        return node;
                    auto carries = [&](IndexAssoc const& a) -> bool
                    {
                        auto const* ci = std::get_if<CountableIndex>(&a);
                        return ci && ci->id == m;
                    };
                    if (carries(*s0.index) && !carries(*s1.index))
                        delta = node, partner = *s1.index, m_slot = s0.slot;
                    else if (carries(*s1.index) && !carries(*s0.index))
                        delta = node, partner = *s0.index, m_slot = s1.slot;
                    return node;
                });
            if (!delta)
                return e;

            // Drop the located δ from a multiplicative position (it is rank 0,
            // so removing it leaves the surrounding tensor product intact).
            std::function<Expr const*(Expr const*)> drop =
                [&](Expr const* node) -> Expr const*
            {
                if (node == delta)
                    return nullptr; // signal: this leg was the δ
                auto const* p = std::get_if<TensorProduct>(&node->node);
                if (p)
                {
                    auto const* l = drop(p->left);
                    auto const* r = drop(p->right);
                    if (l == p->left && r == p->right)
                        return node;
                    if (!l)
                        return r;
                    if (!r)
                        return l;
                    return make_tensor_product(ctx, l, r);
                }
                if (auto const* es = std::get_if<ExplicitSum>(&node->node))
                {
                    auto const* b = drop(es->body);
                    return b == es->body ?
                               node :
                               make_explicit_sum(ctx, es->index, b, es->bound);
                }
                if (auto const* ns = std::get_if<NoSum>(&node->node))
                {
                    auto const* b = drop(ns->body);
                    return b == ns->body ? node :
                                           make_no_sum(ctx, ns->index, b);
                }
                if (auto const* ng = std::get_if<Negate>(&node->node))
                {
                    auto const* b = drop(ng->operand);
                    if (b == ng->operand)
                        return node;
                    return b ? make_negate(ctx, b) : nullptr;
                }
                return node;
            };
            auto const* without = drop(s->body);
            // The δ was the sole factor (e.g. Σ_m δ_mn with n free) —
            // degenerate; leave it for another step.
            if (!without)
                return e;

            // The contraction is genuine only if m has a partner occurrence in
            // the rest, at a matching realm and space (Σ_m δ^m_k = 1 with no
            // partner, and contractions across mismatched realms, are not ours
            // to collapse).  Levels need not match — δ identifies its indices
            // regardless of which slot is up or down.
            IndexSlot partner_slot{};
            bool partner_found = false;
            rewrite_tree(
                ctx,
                without,
                [&](Context&, Expr const* node) -> Expr const*
                {
                    if (partner_found)
                        return node;
                    auto const* t = std::get_if<TensorObject>(&node->node);
                    if (!t)
                        return node;
                    for (auto const& sb: t->slots)
                    {
                        if (!sb.index)
                            continue;
                        auto const* ci =
                            std::get_if<CountableIndex>(&*sb.index);
                        if (ci && ci->id == m)
                        {
                            partner_slot = sb.slot;
                            partner_found = true;
                            return node;
                        }
                    }
                    return node;
                });
            if (!partner_found || partner_slot.realm != m_slot.realm
                || partner_slot.space != m_slot.space)
                return e;

            fired = true;
            // Identify m with n in what remains, and shed the now-spent Σ_m.
            return substitute_index(ctx, without, m, partner);
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
            [&](ScalarFn const& s) -> Expr const*
            {
                auto* o = rec(s.operand);
                return o == s.operand ? e : make_scalar_fn(ctx, s.kind, o);
            },
            [&](Pow const& s) -> Expr const*
            {
                auto* base = rec(s.base);
                auto* exp = rec(s.exponent);
                return (base == s.base && exp == s.exponent) ?
                           e :
                           make_pow(ctx, base, exp);
            },
            [&](Deriv const& s) -> Expr const*
            {
                auto* w = rec(s.wrt);
                return w == s.wrt ? e : make_deriv(ctx, w);
            },
            [&](Nabla const&) -> Expr const* { return e; },
        },
        *e);
}

// Attempt the ε-pair contraction anchored at node `e`: peel every nested
// null-bound ExplicitSum (and a leading −1 sign), then look inside the product
// body for rank-3 Levi-Civita symbols.  Pick the first pair (in factor order)
// that share a summed index and contract them over those shared indices, via
// the generalized-Kronecker formula, keeping every other factor — including any
// further ε's — and re-wrapping the non-contracted sums.  So it fires on a bare
// Σ ε ε, on Σ ε ε buried in a coordinate product (bac-cab's Σ −ε ε a b c e),
// and on products of more than two ε's (the driver iterates pass-by-pass until
// no ε-pair shares a summed index; vibe 000063).  Returns nullptr on no match.
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
    // Collect the positions of every ε factor (a product may hold more than two
    // — e.g. after the dyad-identity insertion in a × B × c; vibe 000063).
    std::vector<int> eps_pos;
    for (int i = 0; i < static_cast<int>(factors.size()); ++i)
        if (as_eps(factors[i]))
            eps_pos.push_back(i);
    if (eps_pos.size() < 2)
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

    // Pick the first pair of ε's (in factor order) that share a summed index;
    // that pair is the one we contract this pass.  Any remaining ε's are left
    // as ordinary factors and contracted on a later pass once the driver
    // iterates.
    std::set<int> const summed_set(summed.begin(), summed.end());
    int pa = -1;
    int pb = -1;
    std::set<int> shared;
    for (std::size_t i = 0; i < eps_pos.size() && pa < 0; ++i)
    {
        std::set<int> ida;
        if (!eps_ids(*as_eps(factors[eps_pos[i]]), ida))
            continue;
        for (std::size_t j = i + 1; j < eps_pos.size(); ++j)
        {
            std::set<int> idb;
            if (!eps_ids(*as_eps(factors[eps_pos[j]]), idb))
                continue;
            std::set<int> sh;
            for (int id: ida)
                if (idb.count(id) && summed_set.count(id))
                    sh.insert(id);
            if (!sh.empty())
            {
                pa = eps_pos[i];
                pb = eps_pos[j];
                shared = std::move(sh);
                break;
            }
        }
    }
    if (pa < 0)
        return nullptr;

    TensorObject const* ea = as_eps(factors[pa]);
    TensorObject const* eb = as_eps(factors[pb]);

    // Every other factor (non-ε factors *and* the un-contracted ε's) is kept in
    // its original left-to-right order — basis vectors of a dyad do not
    // commute.
    std::vector<Expr const*> others;
    for (int i = 0; i < static_cast<int>(factors.size()); ++i)
        if (i != pa && i != pb)
            others.push_back(factors[i]);

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
    // Self-prepare: the contraction reads the summation binders off explicit
    // ExplicitSum nodes, so materialize the implicit Einstein sums first (via
    // canonicalize) — the caller never has to.  canonicalize throws on an
    // ill-formed implicit sum; that just means "nothing to prepare", so fall
    // back to the original.  A genuine no-op returns the input untouched.
    Expr const* prepped = e;
    try
    {
        prepped = canonicalize(ctx, e);
    }
    catch (std::invalid_argument const&)
    {
        prepped = e;
    }
    // Iterate the walk to a fixpoint: each pass contracts one ε-pair, so a
    // product of N ε's (vibe 000063) is reduced pair-by-pair until no two ε's
    // share a summed index.
    Expr const* cur = prepped;
    for (;;)
    {
        Expr const* const next = contract_eps_pair_walk(ctx, cur);
        if (next == cur)
            break;
        cur = next;
    }
    if (cur == prepped)
        return e; // genuine no-op: no ε-pair contracted
    // The contraction emits a Kronecker determinant (δδ − δδ), a Sum factor.
    // Its summation binders straddle that Sum (a scope boundary, vibe 000052),
    // so implicitize cannot strip them and they leak as explicit Σ (vibe 000064
    // #2).  Distribute the determinant so each emitted term is a single product
    // — re-canonicalize to settle the freed binders at each term head — then
    // the implicit-summation convention applies and implicitize clears them.
    // The δ's remain for the next contract_delta.
    return implicitize(ctx, canonicalize(ctx, expand_products(ctx, cur)));
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
                [&](ScalarFn const& s) { go(s.operand); },
                [&](Pow const& s)
                {
                    go(s.base);
                    go(s.exponent);
                },
                [&](Deriv const& s) { go(s.wrt); },
                [&](Nabla const&) {},
            },
            *node);
    };
    go(e);
    return found;
}

auto fold_equal_addends_structural(Context& ctx, Expr const* e) -> Expr const*
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

auto fold_equal_addends(Context& ctx, Expr const* e) -> Expr const*
{
    // Self-prepare (vibe 000065): the structural fold only merges addends that
    // are written identically, so two terms equal merely up to dummy-index
    // renaming or factor/sign ordering (e.g. the I×a playthrough's
    // a_k e_j ε_{jki} e_i vs a_k ε_{kij} e_j e_i) escape it.  Canonicalize
    // first so equal terms collapse to one normal form, then fold, then restore
    // the implicit-sum convention.  canonicalize throws on an ill-formed
    // implicit sum; that just means "nothing to prepare", so fall back to the
    // input.
    Expr const* prepped = e;
    try
    {
        prepped = canonicalize(ctx, e);
    }
    catch (std::invalid_argument const&)
    {
        prepped = e;
    }
    return implicitize(ctx, fold_equal_addends_structural(ctx, prepped));
}

auto collect_terms(Context& ctx, Expr const* e) -> Expr const*
{
    Expr const* prepped = e;
    try
    {
        prepped = canonicalize(ctx, e);
    }
    catch (std::invalid_argument const&)
    {
        prepped = e;
    }
    std::vector<std::pair<int, Expr const*>> addends;
    collect_signed_addends(prepped, 1, addends);

    // Group by the tensor (non-scalar) part; accumulate the scalar coefficient.
    struct Group final
    {
        Expr const* tensor; // nullptr = pure scalar term
        Expr const* coeff;
    };
    std::vector<Group> groups;
    for (auto const& [sign, term]: addends)
    {
        std::vector<Expr const*> facs;
        flatten_factors(term, facs);
        std::vector<Expr const*> scalars;
        std::vector<Expr const*> tensors;
        for (auto const* f: facs)
            (infer_rank(f) == std::optional<int>{0} ? scalars : tensors)
                .push_back(f);
        Expr const* tpart =
            tensors.empty() ? nullptr : product_of(ctx, tensors);
        Expr const* coeff = scalars.empty() ? make_scalar(ctx, Rational{1}) :
                                              product_of(ctx, scalars);
        if (sign < 0)
            coeff = make_negate(ctx, coeff);

        bool merged = false;
        for (auto& g: groups)
            if ((g.tensor == nullptr && tpart == nullptr)
                || (g.tensor && tpart && structural_eq(g.tensor, tpart)))
            {
                g.coeff = make_sum(ctx, g.coeff, coeff);
                merged = true;
                break;
            }
        if (!merged)
            groups.push_back({tpart, coeff});
    }

    Expr const* out = nullptr;
    for (auto const& g: groups)
    {
        Expr const* c = simplify_scalars(ctx, canonicalize(ctx, g.coeff));
        if (auto const* s = std::get_if<ScalarLiteral>(&c->node);
            s && s->value.is_zero())
            continue;
        Expr const* term = g.tensor ? make_tensor_product(ctx, c, g.tensor) : c;
        out = out ? make_sum(ctx, out, term) : term;
    }
    return out ? out : make_scalar(ctx, Rational{0});
}

// ---- partial differentiation (vibe 000069 M2) --------------------------

namespace
{

// The coordinate we differentiate by: its display name and chart identity.
// `free_index` is set for a *free-index* ∂_i (vibe 000078) — differentiation
// along a summed frame direction rather than a fixed coordinate; `free_slot`
// then describes that index occurrence (for the mark it stamps).
struct DiffCoord final
{
    TensorName name;
    CoordinateRef ref;
    std::optional<CountableIndex> free_index = {};
    IndexSlot free_slot = {};
};

// Recognise a coordinate variable (rank-0 TensorObject with a CoordinateRef).
// A coordinate carrying a CountableIndex slot is a free-index frame direction
// (make_coordinate_direction, vibe 000078); a plain coordinate is a fixed one.
auto as_diff_coord(Expr const* e) -> std::optional<DiffCoord>
{
    auto const* t = std::get_if<TensorObject>(&e->node);
    if (!t || !t->traits || !t->traits->coordinate)
        return std::nullopt;
    for (auto const& sb: t->slots)
        if (sb.index)
            if (auto const* ci = std::get_if<CountableIndex>(&*sb.index))
                return DiffCoord{t->name, *t->traits->coordinate, *ci, sb.slot};
    return DiffCoord{t->name, *t->traits->coordinate};
}

// Does the target object `t` denote the same coordinate as `q`?  Identity is
// the chart slot (chart_id, slot) plus the display name — distinct coordinates
// of a chart differ in slot, and unbound (chart_id 0) coordinates differ in
// name.
auto is_same_coord(TensorObject const& t, DiffCoord const& q) -> bool
{
    if (!t.traits || !t.traits->coordinate)
        return false;
    auto const& c = *t.traits->coordinate;
    return t.name == q.name && c.chart_id == q.ref.chart_id
           && c.slot == q.ref.slot;
}

// Intrinsic derivative of a physical-frame basis vector (vibe 000071).  If `e`
// is a frame-vector atom eᵢ — a rank-1 TensorObject with a single basis-tagged
// slot carrying a ConcreteIndex — whose basis has a registered connection whose
// chart owns the coordinate `q`, return ∂_{q^j} eᵢ = deriv[i][j] from that
// table (the pre-expanded Σ_k γ^k_{ij} eₖ).  nullptr if `e` is not such an
// atom, or the coordinate belongs to a different chart (then eᵢ is constant in
// q here — cross-chart coupling is a later stage), so the caller falls through
// to 0.
auto diff_frame_vector(Context& ctx, Expr const* e, DiffCoord const& q)
    -> Expr const*
{
    auto const* t = std::get_if<TensorObject>(&e->node);
    if (!t || t->slots.size() != 1 || !t->slots[0].index)
        return nullptr;
    int const basis_id = t->slots[0].slot.basis_id;
    if (basis_id == 0)
        return nullptr;
    auto const* ci = std::get_if<ConcreteIndex>(&*t->slots[0].index);
    if (!ci)
        return nullptr;
    auto const* conn = ctx.connection(basis_id);
    if (!conn || conn->chart_id != q.ref.chart_id)
        return nullptr;
    int dir = -1;
    for (std::size_t k = 0; k < conn->values.size(); ++k)
        if (conn->values[k] == ci->value)
        {
            dir = static_cast<int>(k);
            break;
        }
    if (dir < 0 || q.ref.slot < 0
        || q.ref.slot >= static_cast<int>(
               conn->deriv[static_cast<std::size_t>(dir)].size()))
        return nullptr;
    return conn->deriv[static_cast<std::size_t>(dir)]
                      [static_cast<std::size_t>(q.ref.slot)];
}

// The outer derivative d/du of an elementary function f, as an Expr in `u`
// (the chain rule then multiplies by u' = ∂u).
auto scalar_fn_derivative(Context& ctx, ScalarFnKind kind, Expr const* u)
    -> Expr const*
{
    auto one = [&] { return make_scalar(ctx, Rational{1}); };
    switch (kind)
    {
        case ScalarFnKind::Sin:
            return make_scalar_fn(ctx, ScalarFnKind::Cos, u);
        case ScalarFnKind::Cos:
            return make_negate(ctx, make_scalar_fn(ctx, ScalarFnKind::Sin, u));
        case ScalarFnKind::Tan: // 1 / cos(u)^2
            return make_scalar_div(
                ctx,
                one(),
                make_pow(
                    ctx,
                    make_scalar_fn(ctx, ScalarFnKind::Cos, u),
                    make_scalar(ctx, Rational{2})));
        case ScalarFnKind::Exp:
            return make_scalar_fn(ctx, ScalarFnKind::Exp, u);
        case ScalarFnKind::Log: // 1 / u
            return make_scalar_div(ctx, one(), u);
        case ScalarFnKind::Sqrt: // 1 / (2 sqrt(u))
            return make_scalar_div(
                ctx,
                one(),
                make_tensor_product(
                    ctx,
                    make_scalar(ctx, Rational{2}),
                    make_scalar_fn(ctx, ScalarFnKind::Sqrt, u)));
    }
    return make_scalar(ctx, Rational{0}); // unreachable
}

// The raw (un-canonicalized) derivative tree.
auto diff(Context& ctx, Expr const* e, DiffCoord const& q) -> Expr const*
{
    auto zero = [&] { return make_scalar(ctx, Rational{0}); };
    // Leibniz over a binary operator `op(l, r)`: op(l', r) + op(l, r').
    auto leibniz =
        [&](auto make_op, Expr const* l, Expr const* r) -> Expr const*
    {
        return make_sum(
            ctx,
            make_op(ctx, diff(ctx, l, q), r),
            make_op(ctx, l, diff(ctx, r, q)));
    };
    return visit(
        Overloads{
            [&](TensorObject const& t) -> Expr const*
            {
                // A coordinate atom: ∂_q q = 1 for the matching coordinate, 0
                // for any other (sibling coordinate, unbound symbol).
                if (t.traits && t.traits->coordinate)
                    return is_same_coord(t, q) ? make_scalar(ctx, Rational{1}) :
                                                 zero();
                // A tensor field (vibe 000070 P7): if it depends on q, the
                // derivative is the opaque derivative field ∂_q T (a fresh
                // field of the same rank, so it can be differentiated again);
                // if it provably does not, it is constant in q → 0.  An opaque
                // field carries no e_j(q) inside it, so there are no connection
                // terms.
                if (t.traits && t.traits->field)
                {
                    auto const& fd = *t.traits->field;
                    // Free-index ∂_i (vibe 000078): stamp an abstract-direction
                    // mark.  A uniform ∂_i is only meaningful for a field that
                    // depends on every coordinate.
                    if (q.free_index)
                    {
                        if (!fd.all)
                            throw std::invalid_argument(
                                "diff: free-index ∂_i needs a field depending "
                                "on all coordinates");
                        return make_field_derivative_free(
                            ctx, e, *q.free_index, q.free_slot);
                    }
                    bool const depends =
                        fd.all
                        || std::any_of(
                            fd.only.begin(),
                            fd.only.end(),
                            [&](CoordinateRef const& d) {
                                return d.chart_id == q.ref.chart_id
                                       && d.slot == q.ref.slot;
                            });
                    return depends ?
                               make_field_derivative(ctx, e, q.name, q.ref) :
                               zero();
                }
                // A physical-frame basis vector eᵢ (vibe 000071): differentiate
                // it through the connection, ∂_{q^j} eᵢ = Σ_k γ^k_{ij} eₖ,
                // staying intrinsic to the curvilinear basis — never expanding
                // it in the reference frame.
                if (auto const* de = diff_frame_vector(ctx, e, q))
                    return de;
                // Everything else (reference vectors of an unregistered basis,
                // parameters, I/δ/ε) is constant.
                return zero();
            },
            [&](ScalarLiteral const&) -> Expr const* { return zero(); },
            [&](Negate const& n) -> Expr const*
            { return make_negate(ctx, diff(ctx, n.operand, q)); },
            [&](Trace const& u) -> Expr const*
            { return make_trace(ctx, diff(ctx, u.operand, q)); },
            [&](VectorInvariant const& u) -> Expr const*
            { return make_vector_invariant(ctx, diff(ctx, u.operand, q)); },
            [&](Transpose const& u) -> Expr const*
            { return make_transpose(ctx, diff(ctx, u.operand, q)); },
            [&](Sum const& s) -> Expr const* {
                return make_sum(
                    ctx, diff(ctx, s.left, q), diff(ctx, s.right, q));
            },
            [&](Difference const& s) -> Expr const* {
                return make_difference(
                    ctx, diff(ctx, s.left, q), diff(ctx, s.right, q));
            },
            [&](TensorProduct const& s) -> Expr const* // Leibniz; ⊗ order kept
            { return leibniz(make_tensor_product, s.left, s.right); },
            [&](ScalarDiv const& s) -> Expr const*
            {
                // (l/r)' = (l' r − l r') / r².
                auto* num = make_difference(
                    ctx,
                    make_tensor_product(ctx, diff(ctx, s.left, q), s.right),
                    make_tensor_product(ctx, s.left, diff(ctx, s.right, q)));
                return make_scalar_div(
                    ctx,
                    num,
                    make_pow(ctx, s.right, make_scalar(ctx, Rational{2})));
            },
            [&](Dot const& s) -> Expr const*
            { return leibniz(make_dot, s.left, s.right); },
            [&](DDot const& s) -> Expr const*
            { return leibniz(make_ddot, s.left, s.right); },
            [&](DDotAlt const& s) -> Expr const*
            { return leibniz(make_ddot_alt, s.left, s.right); },
            [&](Cross const& s) -> Expr const*
            { return leibniz(make_cross, s.left, s.right); },
            [&](ExplicitSum const& s) -> Expr const*
            {
                // ∂ commutes with summation; the bound (cardinality) is
                // constant.
                return make_explicit_sum(
                    ctx, s.index, diff(ctx, s.body, q), s.bound);
            },
            [&](NoSum const& s) -> Expr const*
            { return make_no_sum(ctx, s.index, diff(ctx, s.body, q)); },
            [&](ScalarFn const& f) -> Expr const*
            {
                // Chain rule: f(u)' = f'(u) · u'.
                return make_tensor_product(
                    ctx,
                    scalar_fn_derivative(ctx, f.kind, f.operand),
                    diff(ctx, f.operand, q));
            },
            [&](Pow const& p) -> Expr const*
            {
                auto* bprime = diff(ctx, p.base, q);
                if (std::holds_alternative<ScalarLiteral>(p.exponent->node))
                {
                    // Constant (literal) exponent: (b^n)' = n · b^{n−1} · b'.
                    auto* n_minus_1 = make_difference(
                        ctx, p.exponent, make_scalar(ctx, Rational{1}));
                    return make_tensor_product(
                        ctx,
                        make_tensor_product(
                            ctx, p.exponent, make_pow(ctx, p.base, n_minus_1)),
                        bprime);
                }
                // General power rule:
                //   (b^e)' = b^e · ( e'·log(b) + e·b'/b ).
                auto* eprime = diff(ctx, p.exponent, q);
                auto* inner = make_sum(
                    ctx,
                    make_tensor_product(
                        ctx,
                        eprime,
                        make_scalar_fn(ctx, ScalarFnKind::Log, p.base)),
                    make_scalar_div(
                        ctx,
                        make_tensor_product(ctx, p.exponent, bprime),
                        p.base));
                return make_tensor_product(
                    ctx, make_pow(ctx, p.base, p.exponent), inner);
            },
            [&](Deriv const&) -> Expr const*
            {
                // Differentiating an (unapplied) operator is operator
                // composition, whose Leibniz/commutation rules arrive with
                // application (vibe 000077, step B).  Until then, refuse rather
                // than guess.
                throw std::invalid_argument(
                    "diff: differentiating a ∂ operator (operator composition) "
                    "is not supported yet");
            },
            [&](Nabla const&) -> Expr const*
            {
                // Likewise for the ∇ operator: expand it in a chart first.
                throw std::invalid_argument(
                    "diff: differentiating a ∇ operator is not supported; "
                    "expand ∇ in a chart first");
            },
        },
        *e);
}

} // namespace

auto partial(Context& ctx, Expr const* e, Expr const* coord) -> Expr const*
{
    auto q = as_diff_coord(coord);
    if (!q)
        throw std::invalid_argument(
            "partial: coord must be a coordinate variable (make_coordinate)");
    auto const* raw = diff(ctx, e, *q);
    // Canonicalize to fold the 0/1 noise the rules generate.  canonicalize
    // throws on an ill-formed implicit sum; then the raw form is the best we
    // can return (it is still a correct derivative).
    try
    {
        return canonicalize(ctx, raw);
    }
    catch (std::invalid_argument const&)
    {
        return raw;
    }
}

// ---- operator application (vibe 000077 step B) -------------------------
//
// Application is Leibniz is commutation: an unapplied ∂ operator (a `Deriv`
// factor) acts on everything to its right in its product.  `apply_operators`
// carries that out by reusing the existing Leibniz engine (`partial`/`diff`):
// in each additive product term it finds the *rightmost* operator, hands the
// product of the factors to its right to `partial`, and splices the result back
// (the left factors — everything the operator does not reach — are untouched).
// Rightmost-first means the operand handed to `partial` never itself contains
// an operator, so the elementary rules (∂_x x = 1, ∂_x f formal, connection
// terms) fire cleanly; each pass removes one operator, so the recursion
// terminates.  A trailing operator with nothing to its right stays bare (an
// unapplied operator, e.g. a ∇ being passed around).

namespace
{

auto apply_operators_impl(Context& ctx, Expr const* e) -> Expr const*;

// Fold a contraction (·, :, ··, ×) with a literal-0 operand to 0 (whole tree).
// Differentiating a product/contraction of constants leaves `0 × b` / `a · 0`
// terms; folding them keeps canonicalize (which rejects a bare 0 operand) from
// tripping, exactly as reduce_dot/reduce_cross do inline.
auto fold_zero_contractions(Context& ctx, Expr const* e) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
        [](Context& c, Expr const* n) -> Expr const*
        {
            auto is_zero = [](Expr const* z)
            {
                auto const* s = std::get_if<ScalarLiteral>(&z->node);
                return s && s->value.is_zero();
            };
            auto zero_if = [&](Expr const* l, Expr const* r) -> Expr const* {
                return (is_zero(l) || is_zero(r)) ?
                           make_scalar(c, Rational{0}) :
                           n;
            };
            if (auto const* d = std::get_if<Dot>(&n->node))
                return zero_if(d->left, d->right);
            if (auto const* d = std::get_if<DDot>(&n->node))
                return zero_if(d->left, d->right);
            if (auto const* d = std::get_if<DDotAlt>(&n->node))
                return zero_if(d->left, d->right);
            if (auto const* d = std::get_if<Cross>(&n->node))
                return zero_if(d->left, d->right);
            return n;
        });
}

// canonicalize that tolerates an ill-formed intermediate (like `partial`): on a
// throw, fold zero contractions and retry, else return the input unchanged so
// the caller's own reductions can finish.
auto canon_tolerant(Context& ctx, Expr const* e) -> Expr const*
{
    try
    {
        return canonicalize(ctx, e);
    }
    catch (std::invalid_argument const&)
    {
        try
        {
            return canonicalize(ctx, fold_zero_contractions(ctx, e));
        }
        catch (std::invalid_argument const&)
        {
            return fold_zero_contractions(ctx, e);
        }
    }
}

// Apply the operators of one flat product `factors` (left-to-right order).
auto apply_product_operators(Context& ctx, std::vector<Expr const*> const& facs)
    -> Expr const*
{
    auto product = [&](std::vector<Expr const*> const& fs) -> Expr const*
    {
        Expr const* acc = nullptr;
        for (auto const* f: fs)
            acc = acc ? make_tensor_product(ctx, acc, f) : f;
        return acc;
    };

    // Rightmost operator in the product.
    int op = -1;
    for (int k = 0; k < static_cast<int>(facs.size()); ++k)
        if (std::holds_alternative<Deriv>(facs[k]->node))
            op = k;
    if (op < 0)
        return product(facs); // no operator: nothing to apply
    if (op == static_cast<int>(facs.size()) - 1)
        return product(facs); // trailing bare operator: nothing to its right

    auto const& d = std::get<Deriv>(facs[op]->node);
    std::vector<Expr const*> right(facs.begin() + op + 1, facs.end());
    // Apply this operator to its rightward operand (Leibniz + elementary
    // rules).  The operand may itself contain unapplied operators (e.g. the
    // inner ∇·ε of a grad-div ∇⊗(∇·ε)); resolve those first — rightmost-first —
    // so `partial` never meets a still-unapplied ∂ to differentiate.
    Expr const* applied =
        partial(ctx, apply_operators_impl(ctx, product(right)), d.wrt);
    // Splice: the untouched left factors, then the applied result.
    std::vector<Expr const*> left(facs.begin(), facs.begin() + op);
    Expr const* combined = left.empty() ?
                               applied :
                               make_tensor_product(ctx, product(left), applied);
    // The result may hold new sums (Leibniz) and further (left) operators;
    // re-distribute and recurse.
    return apply_operators_impl(ctx, combined);
}

// The contraction family a differential operator can commute out of: an
// operator at the end of the left operand acts on the right operand,
// `op(A ∂, B) → op(A, ∂▷B)` (∇·T, ∇×T, …).  Returns the operands and a
// rebuilder for the specific product; nullopt for a non-contraction node.
struct ContractionView final
{
    Expr const* left;
    Expr const* right;
    Expr const* (*rebuild)(Context&, Expr const*, Expr const*);
};

auto as_contraction(Expr const* e) -> std::optional<ContractionView>
{
    if (auto const* d = std::get_if<Dot>(&e->node))
        return ContractionView{d->left, d->right, &make_dot};
    if (auto const* d = std::get_if<DDot>(&e->node))
        return ContractionView{d->left, d->right, &make_ddot};
    if (auto const* d = std::get_if<DDotAlt>(&e->node))
        return ContractionView{d->left, d->right, &make_ddot_alt};
    if (auto const* d = std::get_if<Cross>(&e->node))
        return ContractionView{d->left, d->right, &make_cross};
    return std::nullopt;
}

auto apply_operators_impl(Context& ctx, Expr const* e) -> Expr const*
{
    // Distribute products over sums *first*, before any canonicalization — an
    // operand of ∇ (`∇⊗f`, `∇·v`) may be a plain scalar, and canon would
    // commute it to the *left* of the operator (which is hidden inside ∇'s
    // sum), leaving the ∂ trailing and unapplied.  Distributing first pins each
    // operand to the right of its operator, and the operator-aware canon then
    // keeps it there.
    e = expand_products(ctx, e);
    // Walk the additive structure; apply operators inside each product term.
    std::function<Expr const*(Expr const*)> go =
        [&](Expr const* n) -> Expr const*
    {
        if (auto const* s = std::get_if<Sum>(&n->node))
            return make_sum(ctx, go(s->left), go(s->right));
        if (auto const* s = std::get_if<Difference>(&n->node))
            return make_difference(ctx, go(s->left), go(s->right));
        if (auto const* s = std::get_if<Negate>(&n->node))
            return make_negate(ctx, go(s->operand));
        // Recurse into the linear unary invariants (transpose/trace/vec): an
        // operator nested inside, e.g. the ∇ of the inner (∇×ε) in ∇×(∇×ε)ᵀ,
        // must be applied before the outer operator reaches this operand.
        if (auto const* u = std::get_if<Transpose>(&n->node))
            return make_transpose(ctx, go(u->operand));
        if (auto const* u = std::get_if<Trace>(&n->node))
            return make_trace(ctx, go(u->operand));
        if (auto const* u = std::get_if<VectorInvariant>(&n->node))
            return make_vector_invariant(ctx, go(u->operand));
        // A contraction `op(L, R)`.
        if (auto cv = as_contraction(n))
        {
            // Distribute the contraction over an additive left operand (∇ is a
            // sum of terms, so `(Σ_i term_i) · T` → Σ_i term_i · T), bringing
            // each operator adjacent to the operand it acts on.
            if (auto const* s = std::get_if<Sum>(&cv->left->node))
                return go(make_sum(
                    ctx,
                    cv->rebuild(ctx, s->left, cv->right),
                    cv->rebuild(ctx, s->right, cv->right)));
            if (auto const* s = std::get_if<Difference>(&cv->left->node))
                return go(make_difference(
                    ctx,
                    cv->rebuild(ctx, s->left, cv->right),
                    cv->rebuild(ctx, s->right, cv->right)));
            if (auto const* s = std::get_if<Negate>(&cv->left->node))
                return go(
                    make_negate(ctx, cv->rebuild(ctx, s->operand, cv->right)));
            // The left operand ends in an operator: it commutes out onto the
            // (resolved) right operand — `∇·T`, `∇×T`.  Everything left of the
            // operator stays put.
            Expr const* right = go(cv->right);
            std::vector<Expr const*> lf;
            flatten_factors(cv->left, lf);
            if (!lf.empty() && std::holds_alternative<Deriv>(lf.back()->node))
            {
                auto const& d = std::get<Deriv>(lf.back()->node);
                Expr const* dR = partial(ctx, right, d.wrt);
                Expr const* lpre = nullptr;
                for (std::size_t k = 0; k + 1 < lf.size(); ++k)
                    lpre = lpre ? make_tensor_product(ctx, lpre, lf[k]) : lf[k];
                Expr const* contracted = lpre ? cv->rebuild(ctx, lpre, dR) : dR;
                // lpre may still carry operators; re-distribute and recurse.
                return apply_operators_impl(ctx, contracted);
            }
            return cv->rebuild(ctx, go(cv->left), right);
        }
        std::vector<Expr const*> facs;
        flatten_factors(n, facs);
        return apply_product_operators(ctx, facs);
    };
    return go(e);
}

} // namespace

auto apply_operators(Context& ctx, Expr const* e) -> Expr const*
{
    return canon_tolerant(ctx, apply_operators_impl(ctx, e));
}

// ---- targeted scalar simplifier (vibe 000069 M3) -----------------------

namespace
{

// Is `e` manifestly non-negative?  Used to license √(x²) → x.  Conservative:
// only the cases the geometry pipeline produces.
auto is_nonneg(Expr const* e) -> bool
{
    return visit(
        Overloads{
            [](ScalarLiteral const& s) -> bool
            { return !(s.value < Rational{0}); },
            [](TensorObject const& t) -> bool {
                return t.traits && t.traits->coordinate
                       && t.traits->coordinate->nonneg;
            },
            [&](Pow const& p) -> bool
            {
                // x^(2k) ≥ 0 for any real x; otherwise inherit the base's sign.
                if (auto const* n =
                        std::get_if<ScalarLiteral>(&p.exponent->node))
                    if (n->value.is_integer() && n->value.num() % 2 == 0)
                        return true;
                return is_nonneg(p.base);
            },
            [&](TensorProduct const& s) -> bool
            { return is_nonneg(s.left) && is_nonneg(s.right); },
            [](ScalarFn const& f) -> bool {
                return f.kind == ScalarFnKind::Sqrt
                       || f.kind == ScalarFnKind::Exp;
            },
            [](auto const&) -> bool { return false; },
        },
        *e);
}

// Local (single-node) folds, applied bottom-up by rewrite_tree:
//   x⁰ → 1, x¹ → x, and √(x^{2k}) → x^k when x is known ≥ 0.
auto fold_local_scalar(Context& ctx, Expr const* e) -> Expr const*
{
    if (auto const* p = std::get_if<Pow>(&e->node))
    {
        if (auto const* n = std::get_if<ScalarLiteral>(&p->exponent->node))
        {
            if (n->value == Rational{0})
                return make_scalar(ctx, Rational{1});
            if (n->value == Rational{1})
                return p->base;
        }
        return e;
    }
    if (auto const* f = std::get_if<ScalarFn>(&e->node);
        f && f->kind == ScalarFnKind::Sqrt)
    {
        if (auto const* p = std::get_if<Pow>(&f->operand->node))
            if (auto const* n = std::get_if<ScalarLiteral>(&p->exponent->node))
                if (n->value.is_integer() && n->value.num() > 0
                    && n->value.num() % 2 == 0 && is_nonneg(p->base))
                {
                    auto const half = n->value.num() / 2;
                    if (half == 1)
                        return p->base;
                    return make_pow(
                        ctx, p->base, make_scalar(ctx, Rational{half}));
                }
    }
    return e;
}

// A term split as C · fn²(u): the trig kind, its argument u, and the remaining
// product C (carrying coefficient and any other factors, plus the sign).
struct TrigSquare final
{
    ScalarFnKind kind;
    Expr const* arg;
    Expr const* remainder;
};

// Every way `term` reads as C · fn²(u): one TrigSquare per sin²/cos² factor.
// A term may hold several (e.g. spherical r² sin²θ sin²φ), and the Pythagorean
// pair may be on any of them, so all candidates are enumerated rather than
// peeling only the first.
auto enumerate_trig_squares(Context& ctx, Expr const* term)
    -> std::vector<TrigSquare>
{
    bool neg = false;
    Expr const* inner = term;
    if (auto const* n = std::get_if<Negate>(&term->node))
    {
        neg = true;
        inner = n->operand;
    }
    std::vector<Expr const*> factors;
    flatten_factors(inner, factors);
    std::vector<TrigSquare> out;
    for (std::size_t i = 0; i < factors.size(); ++i)
    {
        auto const* p = std::get_if<Pow>(&factors[i]->node);
        if (!p)
            continue;
        auto const* n = std::get_if<ScalarLiteral>(&p->exponent->node);
        if (!n || !(n->value == Rational{2}))
            continue;
        auto const* fn = std::get_if<ScalarFn>(&p->base->node);
        if (!fn
            || (fn->kind != ScalarFnKind::Sin && fn->kind != ScalarFnKind::Cos))
            continue;
        std::vector<Expr const*> rest;
        for (std::size_t j = 0; j < factors.size(); ++j)
            if (j != i)
                rest.push_back(factors[j]);
        Expr const* rem = rest.empty() ? make_scalar(ctx, Rational{1}) :
                                         product_of(ctx, rest);
        if (neg)
            rem = make_negate(ctx, rem);
        out.push_back(TrigSquare{fn->kind, fn->operand, rem});
    }
    return out;
}

void collect_sum(Expr const* e, std::vector<Expr const*>& out)
{
    if (auto const* s = std::get_if<Sum>(&e->node))
    {
        collect_sum(s->left, out);
        collect_sum(s->right, out);
    }
    else
        out.push_back(e);
}

// cos²(u)·C + sin²(u)·C → C for every matching pair among the addends.
auto pythagorean_fold(Context& ctx, Expr const* e) -> Expr const*
{
    std::vector<Expr const*> addends;
    collect_sum(e, addends);
    std::size_t const n = addends.size();
    if (n < 2)
        return e;
    std::vector<bool> used(n, false);
    std::vector<Expr const*> result;
    bool changed = false;
    std::vector<std::vector<TrigSquare>> cands(n);
    for (std::size_t i = 0; i < n; ++i)
        cands[i] = enumerate_trig_squares(ctx, addends[i]);
    for (std::size_t i = 0; i < n; ++i)
    {
        if (used[i])
            continue;
        bool matched = false;
        for (auto const& ci: cands[i])
        {
            for (std::size_t j = i + 1; j < n && !matched; ++j)
            {
                if (used[j])
                    continue;
                for (auto const& cj: cands[j])
                {
                    if (ci.kind == cj.kind) // need one sin and one cos
                        continue;
                    if (algebraic_eq_t0(ctx, ci.arg, cj.arg)
                        && algebraic_eq_t0(ctx, ci.remainder, cj.remainder))
                    {
                        result.push_back(ci.remainder);
                        used[i] = used[j] = true;
                        matched = changed = true;
                        break;
                    }
                }
            }
            if (matched)
                break;
        }
        if (!matched)
        {
            result.push_back(addends[i]);
            used[i] = true;
        }
    }
    if (!changed)
        return e;
    Expr const* acc = nullptr;
    for (auto const* t: result)
        acc = acc ? make_sum(ctx, acc, t) : t;
    return acc ? acc : make_scalar(ctx, Rational{0});
}

// --- Squared / higher-power Pythagorean fold (vibe 000072) ---------------
// The linear pythagorean_fold above only pairs C·sin²u + C·cos²u.  A *power*
// of the identity — e.g. (sin²θ+cos²θ)² expanded as sin⁴θ+2sin²θcos²θ+cos⁴θ —
// leaves no such matching pair, so it survives simplify_scalars.  We close this
// generally: substitute cos²u → 1−sin²u for a co-occurring argument, expand the
// resulting products into a plain sum of monomials, re-collect them into a
// pure-sin polynomial; a genuine identity power collapses to 1.  The move is
// *guarded* by a strict node-cost decrease, so it fires only when it actually
// simplifies (a lone cos²u, which merely inflates to 1−sin²u, is rejected).

auto normalize_scalar(Context& ctx, Expr const* e) -> Expr const*; // fwd

// Re-collect a scalar polynomial after substitution/expansion: merge each
// monomial's repeated factors into powers (normalize_scalar) so like terms
// (e.g. sin·sin·sin² and sin⁴) become structurally identical, then canonicalize
// to sum the coefficients.  No Pythagorean rule here, so it cannot recurse back
// into the fold.  Iterates to a structural fixpoint.
auto collect_scalar_polynomial(Context& ctx, Expr const* e) -> Expr const*
{
    Expr const* cur = canonicalize(ctx, e);
    auto cb = [](Context& c, Expr const* node) -> Expr const*
    {
        Expr const* a = fold_local_scalar(c, node);
        if (std::holds_alternative<ScalarDiv>(a->node)
            || std::holds_alternative<TensorProduct>(a->node))
            a = normalize_scalar(c, a);
        return a;
    };
    for (int iter = 0; iter < 32; ++iter)
    {
        Expr const* next = canonicalize(ctx, rewrite_tree(ctx, cur, cb));
        if (structural_eq(next, cur))
            break;
        cur = next;
    }
    return cur;
}

// Total node count of `e` (every hash-consed subnode counted once per
// position), used as the cost the Pythagorean substitution must strictly
// reduce.
auto expr_cost(Context& ctx, Expr const* e) -> std::size_t
{
    std::size_t n = 0;
    rewrite_tree(
        ctx,
        e,
        [&n](Context&, Expr const* node) -> Expr const*
        {
            ++n;
            return node;
        });
    return n;
}

// Collect the arguments u carried by even powers of sin (into `sinargs`) and of
// cos (into `cosargs`) anywhere in `e`.  Only Pow(sin/cos(u), 2k) qualifies —
// the granularity the substitution below rewrites.
void collect_even_trig_args(
    Context& ctx,
    Expr const* e,
    std::vector<Expr const*>& sinargs,
    std::vector<Expr const*>& cosargs)
{
    rewrite_tree(
        ctx,
        e,
        [&](Context&, Expr const* node) -> Expr const*
        {
            auto const* p = std::get_if<Pow>(&node->node);
            if (!p)
                return node;
            auto const* n = std::get_if<ScalarLiteral>(&p->exponent->node);
            if (!n || !n->value.is_integer() || n->value.num() < 2
                || n->value.num() % 2 != 0)
                return node;
            auto const* fn = std::get_if<ScalarFn>(&p->base->node);
            if (!fn)
                return node;
            if (fn->kind == ScalarFnKind::Sin)
                sinargs.push_back(fn->operand);
            else if (fn->kind == ScalarFnKind::Cos)
                cosargs.push_back(fn->operand);
            return node;
        });
}

// Replace every Pow(cos(arg), 2k) in `e` with (1 − sin²(arg))^k, leaving all
// other factors (including odd cos powers and cos of a *different* argument)
// untouched.
auto substitute_cos2(Context& ctx, Expr const* e, Expr const* arg) -> Expr const*
{
    Expr const* sin2 = make_pow(
        ctx,
        make_scalar_fn(ctx, ScalarFnKind::Sin, arg),
        make_scalar(ctx, Rational{2}));
    Expr const* oms = make_difference(ctx, make_scalar(ctx, Rational{1}), sin2);
    return rewrite_tree(
        ctx,
        e,
        [&](Context& c, Expr const* node) -> Expr const*
        {
            auto const* p = std::get_if<Pow>(&node->node);
            if (!p)
                return node;
            auto const* n = std::get_if<ScalarLiteral>(&p->exponent->node);
            if (!n || !n->value.is_integer() || n->value.num() < 2
                || n->value.num() % 2 != 0)
                return node;
            auto const* fn = std::get_if<ScalarFn>(&p->base->node);
            if (!fn || fn->kind != ScalarFnKind::Cos
                || !structural_eq(fn->operand, arg))
                return node;
            // Emit (1−sin²)·…·(1−sin²) as a *product* of `half` factors rather
            // than a Pow: expand_products distributes a product but leaves a
            // Pow(sum, k) intact, so the identity would not collect otherwise.
            auto const half = n->value.num() / 2;
            Expr const* acc = oms;
            for (long i = 1; i < half; ++i)
                acc = make_tensor_product(c, acc, oms);
            return acc;
        });
}

// Collapse a power of the Pythagorean identity (see block comment above).  For
// each argument appearing under both an even sin power and an even cos power,
// try the cos²→1−sin² substitution; keep the expanded+canonicalized result iff
// it is strictly cheaper than the input sum.  Returns `e` unchanged otherwise.
auto pythagorean_power_fold(Context& ctx, Expr const* e) -> Expr const*
{
    std::vector<Expr const*> sinargs;
    std::vector<Expr const*> cosargs;
    collect_even_trig_args(ctx, e, sinargs, cosargs);

    // Candidates: each distinct argument carrying both an even sin and an even
    // cos power (only such an argument can pair off under the identity).
    std::vector<Expr const*> candidates;
    for (auto const* a: sinargs)
    {
        bool in_cos = false;
        for (auto const* c: cosargs)
            if (structural_eq(a, c))
            {
                in_cos = true;
                break;
            }
        bool seen = false;
        for (auto const* t: candidates)
            if (structural_eq(a, t))
            {
                seen = true;
                break;
            }
        if (in_cos && !seen)
            candidates.push_back(a);
    }
    if (candidates.empty())
        return e;

    std::size_t const base_cost = expr_cost(ctx, e);
    for (auto const* a: candidates)
    {
        Expr const* sub = substitute_cos2(ctx, e, a);
        Expr const* cand =
            collect_scalar_polynomial(ctx, expand_products(ctx, sub));
        if (expr_cost(ctx, cand) < base_cost)
            return cand;
    }
    return e;
}

// --- Scalar fraction cancellation (vibe 000069 M4) ----------------------
// Cancel common multiplicative factors of a scalar quotient so the physical
// basis e_i = g_i / h_i comes out clean: (r cos φ)/r → cos φ, r²/r → r.

// A scalar product split into a numeric coefficient and a set of
// (base, integer exponent) factors; a denominator factor has a negative
// exponent.
struct FactorBag final
{
    Rational coeff{1};
    std::vector<std::pair<Expr const*, long>> bases;
};

void bag_add(Context& ctx, FactorBag& bag, Expr const* base, long exp)
{
    for (auto& kv: bag.bases)
        if (algebraic_eq_t0(ctx, kv.first, base))
        {
            kv.second += exp;
            return;
        }
    bag.bases.push_back({base, exp});
}

// Accumulate the factors of scalar product `e` into `bag`; `sign` is +1 when
// `e` sits in the numerator, -1 in the denominator (exponents and the numeric
// coefficient are taken to that power).
void decompose_scalar(Context& ctx, Expr const* e, long sign, FactorBag& bag)
{
    std::vector<Expr const*> facs;
    flatten_factors(e, facs);
    for (auto const* f: facs)
    {
        if (auto const* s = std::get_if<ScalarLiteral>(&f->node))
            bag.coeff = sign > 0 ? bag.coeff * s->value : bag.coeff / s->value;
        else if (auto const* neg = std::get_if<Negate>(&f->node))
        {
            bag.coeff = -bag.coeff;
            decompose_scalar(ctx, neg->operand, sign, bag);
        }
        else if (auto const* d = std::get_if<ScalarDiv>(&f->node))
        {
            decompose_scalar(ctx, d->left, sign, bag);
            decompose_scalar(ctx, d->right, -sign, bag);
        }
        else if (auto const* p = std::get_if<Pow>(&f->node))
        {
            auto const* ne = std::get_if<ScalarLiteral>(&p->exponent->node);
            if (ne && ne->value.is_integer())
                bag_add(ctx, bag, p->base, sign * ne->value.num());
            else
                bag_add(ctx, bag, f, sign);
        }
        else
            bag_add(ctx, bag, f, sign);
    }
}

auto pow_expr(Context& ctx, Expr const* base, long exp) -> Expr const*
{
    if (exp == 1)
        return base;
    return make_pow(ctx, base, make_scalar(ctx, Rational{exp}));
}

// Normalise a scalar product/quotient into a single fraction with repeated
// factors collected into powers: x·x → x², (r cos φ)/r → cos φ, r²/r → r.  A
// no-op (returns `e`) unless `e` is a rank-0 scalar product or quotient and the
// rebuilt form differs from the input.
auto normalize_scalar(Context& ctx, Expr const* e) -> Expr const*
{
    FactorBag bag;
    if (auto const* d = std::get_if<ScalarDiv>(&e->node))
    {
        if (infer_rank(d->left) != std::optional<int>{0}
            || infer_rank(d->right) != std::optional<int>{0})
            return e;
        decompose_scalar(ctx, d->left, +1, bag);
        decompose_scalar(ctx, d->right, -1, bag);
    }
    else if (std::holds_alternative<TensorProduct>(e->node))
    {
        if (infer_rank(e) != std::optional<int>{0})
            return e;
        decompose_scalar(ctx, e, +1, bag);
    }
    else
        return e;

    if (bag.coeff.is_zero())
        return make_scalar(ctx, Rational{0});

    std::vector<Expr const*> numf;
    std::vector<Expr const*> denf;
    if (bag.coeff.num() != 1)
        numf.push_back(make_scalar(ctx, Rational{bag.coeff.num()}));
    if (bag.coeff.den() != 1)
        denf.push_back(make_scalar(ctx, Rational{bag.coeff.den()}));
    for (auto const& kv: bag.bases)
    {
        if (kv.second > 0)
            numf.push_back(pow_expr(ctx, kv.first, kv.second));
        else if (kv.second < 0)
            denf.push_back(pow_expr(ctx, kv.first, -kv.second));
    }
    Expr const* num =
        numf.empty() ? make_scalar(ctx, Rational{1}) : product_of(ctx, numf);
    Expr const* result =
        denf.empty() ? num : make_scalar_div(ctx, num, product_of(ctx, denf));
    return structural_eq(result, e) ? e : result;
}

// --- Common-denominator fraction combination (vibe 000072) --------------
// normalize_scalar cancels within a single quotient, but a *sum* of quotients
// stays split (A/r² + B/r), so two algebraically-equal coefficients — e.g. the
// two transposed dyads of a symmetric ∇∇f — never become structurally equal.
// combine_fractions puts a rank-0 sum over one common denominator: A/r² + B/r →
// (A + B r)/r².  Only a scalar sum with a genuine denominator is combined;
// purely polynomial sums and every rank>0 (tensor) sum are left untouched.
auto combine_fractions(Context& ctx, Expr const* e) -> Expr const*
{
    if (infer_rank(e) != std::optional<int>{0})
        return e;
    std::vector<Expr const*> addends;
    collect_sum(e, addends); // a Sum always yields ≥ 2 addends

    // Decompose each addend; note whether any carries a denominator at all.
    std::vector<FactorBag> bags(addends.size());
    bool any_denom = false;
    for (std::size_t i = 0; i < addends.size(); ++i)
    {
        decompose_scalar(ctx, addends[i], +1, bags[i]);
        if (bags[i].coeff.den() != 1)
            any_denom = true;
        for (auto const& kv: bags[i].bases)
            if (kv.second < 0)
                any_denom = true;
    }
    if (!any_denom)
        return e;

    // The least common denominator: the numeric lcm of the coefficient
    // denominators, times each denominator base raised to the largest power any
    // addend needs.
    std::int64_t dnum = 1;
    std::vector<std::pair<Expr const*, long>> dbases;
    for (auto const& bag: bags)
    {
        dnum = std::lcm(dnum, bag.coeff.den());
        for (auto const& kv: bag.bases)
        {
            if (kv.second >= 0)
                continue;
            long const need = -kv.second;
            bool found = false;
            for (auto& db: dbases)
                if (algebraic_eq_t0(ctx, db.first, kv.first))
                {
                    db.second = std::max(db.second, need);
                    found = true;
                    break;
                }
            if (!found)
                dbases.push_back({kv.first, need});
        }
    }

    std::vector<Expr const*> dfacs;
    if (dnum != 1)
        dfacs.push_back(make_scalar(ctx, Rational{dnum}));
    for (auto const& db: dbases)
        dfacs.push_back(pow_expr(ctx, db.first, db.second));
    Expr const* denom = product_of(ctx, dfacs);

    // Each numerator term is addend·denom with its own denominator cancelled —
    // exactly what normalize_scalar does — so, denom being the LCD, no fraction
    // survives in the numerator.
    Expr const* numerator = nullptr;
    for (auto const* ad: addends)
    {
        Expr const* term =
            normalize_scalar(ctx, make_tensor_product(ctx, ad, denom));
        numerator = numerator ? make_sum(ctx, numerator, term) : term;
    }
    Expr const* result = make_scalar_div(ctx, numerator, denom);
    return structural_eq(result, e) ? e : result;
}

} // namespace

auto simplify_scalars(Context& ctx, Expr const* e) -> Expr const*
{
    Expr const* cur = e;
    try
    {
        cur = canonicalize(ctx, e);
    }
    catch (std::invalid_argument const&)
    {
        cur = e;
    }
    // One bottom-up pass applies the local folds and, at every Sum node, the
    // Pythagorean fold — so a sum buried in √(…) (a scale factor h = √(g)) is
    // folded before its enclosing root.  Re-canonicalize and repeat to a fixed
    // point; the rules strictly shrink the expression, so this terminates (the
    // iteration cap is a belt-and-braces guard).
    auto step_cb = [](Context& c, Expr const* node) -> Expr const*
    {
        Expr const* a = fold_local_scalar(c, node);
        if (std::holds_alternative<ScalarDiv>(a->node)
            || std::holds_alternative<TensorProduct>(a->node))
            a = normalize_scalar(c, a);
        if (std::holds_alternative<Sum>(a->node))
        {
            a = pythagorean_fold(c, a);
            if (std::holds_alternative<Sum>(a->node))
                a = pythagorean_power_fold(c, a);
            if (std::holds_alternative<Sum>(a->node))
                a = combine_fractions(c, a);
        }
        return a;
    };
    for (int iter = 0; iter < 64; ++iter)
    {
        auto const* stepped = rewrite_tree(ctx, cur, step_cb);
        Expr const* next = stepped;
        try
        {
            next = canonicalize(ctx, stepped);
        }
        catch (std::invalid_argument const&)
        {
            next = stepped;
        }
        if (structural_eq(next, cur))
            break;
        cur = next;
    }
    return implicitize(ctx, cur);
}

} // namespace steps
} // namespace tender
