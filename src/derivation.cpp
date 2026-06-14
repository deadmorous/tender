#include <tender/derivation.hpp>

#include <mpk/mix/util/overloads.hpp>

#include <algorithm>
#include <functional>
#include <set>

using namespace mpk::mix;

namespace tender
{
namespace
{

// Rebuild an expression bottom-up: first recurse into each child, then
// apply f(ctx, rebuilt_node).  If f returns e unchanged, the pointer is
// reused without extra allocation.
template <typename F>
auto rewrite_tree(Context& ctx, Expr const* e, F const& f) -> Expr const*
{
    Expr const* rebuilt = visit(
        Overloads{
            // Leaf nodes — no children.
            [&](TensorObject const&) -> Expr const* { return e; },
            [&](ScalarLiteral const&) -> Expr const* { return e; },

            // Unary.
            [&](Negate const& n) -> Expr const*
            {
                auto* op = rewrite_tree(ctx, n.operand, f);
                return op == n.operand ? e : make_negate(ctx, op);
            },

            // Binary.
            [&](Sum const& s) -> Expr const*
            {
                auto* l = rewrite_tree(ctx, s.left, f);
                auto* r = rewrite_tree(ctx, s.right, f);
                return (l == s.left && r == s.right) ? e : make_sum(ctx, l, r);
            },
            [&](Difference const& s) -> Expr const*
            {
                auto* l = rewrite_tree(ctx, s.left, f);
                auto* r = rewrite_tree(ctx, s.right, f);
                return (l == s.left && r == s.right) ?
                           e :
                           make_difference(ctx, l, r);
            },
            [&](TensorProduct const& s) -> Expr const*
            {
                auto* l = rewrite_tree(ctx, s.left, f);
                auto* r = rewrite_tree(ctx, s.right, f);
                return (l == s.left && r == s.right) ?
                           e :
                           make_tensor_product(ctx, l, r);
            },
            [&](ScalarDiv const& s) -> Expr const*
            {
                auto* l = rewrite_tree(ctx, s.left, f);
                auto* r = rewrite_tree(ctx, s.right, f);
                return (l == s.left && r == s.right) ?
                           e :
                           make_scalar_div(ctx, l, r);
            },
            [&](Dot const& s) -> Expr const*
            {
                auto* l = rewrite_tree(ctx, s.left, f);
                auto* r = rewrite_tree(ctx, s.right, f);
                return (l == s.left && r == s.right) ? e : make_dot(ctx, l, r);
            },
            [&](DDot const& s) -> Expr const*
            {
                auto* l = rewrite_tree(ctx, s.left, f);
                auto* r = rewrite_tree(ctx, s.right, f);
                return (l == s.left && r == s.right) ? e : make_ddot(ctx, l, r);
            },
            [&](DDotAlt const& s) -> Expr const*
            {
                auto* l = rewrite_tree(ctx, s.left, f);
                auto* r = rewrite_tree(ctx, s.right, f);
                return (l == s.left && r == s.right) ? e :
                                                       make_ddot_alt(ctx, l, r);
            },
            [&](Cross const& s) -> Expr const*
            {
                auto* l = rewrite_tree(ctx, s.left, f);
                auto* r = rewrite_tree(ctx, s.right, f);
                return (l == s.left && r == s.right) ? e :
                                                       make_cross(ctx, l, r);
            },

            // Annotation nodes.
            [&](ExplicitSum const& s) -> Expr const*
            {
                auto* body = rewrite_tree(ctx, s.body, f);
                auto* bound = s.bound ? rewrite_tree(ctx, s.bound, f) : nullptr;
                return (body == s.body && bound == s.bound) ?
                           e :
                           make_explicit_sum(ctx, s.index, body, bound);
            },
            [&](NoSum const& s) -> Expr const*
            {
                auto* body = rewrite_tree(ctx, s.body, f);
                return body == s.body ? e : make_no_sum(ctx, s.index, body);
            },
        },
        *e);

    return f(ctx, rebuilt);
}

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

// Deep structural equality of two expression trees.
// CountableIndex ids must match exactly (free variables are not alpha-renamed).
auto structural_eq(Expr const* a, Expr const* b) -> bool;

auto index_assoc_eq(
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

// Extract (integer coefficient, core expression) from an addend.
// Returns {1, e}       for a plain expression.
// Returns {n, core}    for TensorProduct(ScalarLiteral(n), core).
// Returns {-1, core}   for Negate(core).
auto extract_coeff(Expr const* e) -> std::pair<int64_t, Expr const*>
{
    if (auto const* tp = std::get_if<TensorProduct>(&e->node))
        if (auto const* sl = std::get_if<ScalarLiteral>(&tp->left->node))
            if (sl->value.is_integer())
                return {sl->value.num(), tp->right};
    if (auto const* neg = std::get_if<Negate>(&e->node))
        return {-1, neg->operand};
    return {1, e};
}

} // namespace

namespace steps
{

auto unroll_sums(Context& ctx, Expr const* e) -> Expr const*
{
    return rewrite_tree(
        ctx,
        e,
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
                        // X + (-Y) → X - Y  (cleaner rendering, no "(-(Y))")
                        if (std::holds_alternative<Negate>(s.right->node))
                        {
                            auto const& neg = std::get<Negate>(s.right->node);
                            return make_difference(ctx, s.left, neg.operand);
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
    return rewrite_tree(
        ctx,
        e,
        [](Context& ctx, Expr const* e) -> Expr const*
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

            return make_delta(
                ctx,
                sur1.slot.realm,
                sur1.slot.space,
                sur1.slot.level,
                sur2.slot.level,
                *sur1.index,
                *sur2.index);
        });
}

auto unroll_sums_for(
    Context& ctx,
    Expr const* e,
    std::vector<CountableIndex> const& indices) -> Expr const*
{
    return rewrite_tree(
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
            Expr const* result = nullptr;
            for (int v: vals)
            {
                Expr const* term =
                    substitute(ctx, s->body, s->index.id, ConcreteIndex{v});
                result = result ? make_sum(ctx, result, term) : term;
            }
            return result ? result : make_scalar(ctx, Rational{0});
        });
}

auto has_explicit_sum_for(
    Expr const* e, std::vector<CountableIndex> const& indices) -> bool
{
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
            std::vector<Expr const*> addends;
            collect_addends(e, addends);
            if (addends.size() < 2)
                return e;

            // Group addends by structural equality of core expression.
            // Each group accumulates the integer coefficient total.
            std::vector<bool> used(addends.size(), false);
            std::vector<std::pair<int64_t, Expr const*>> groups;

            for (std::size_t i = 0; i < addends.size(); ++i)
            {
                if (used[i])
                    continue;
                auto [ci, corei] = extract_coeff(addends[i]);
                used[i] = true;

                for (std::size_t j = i + 1; j < addends.size(); ++j)
                {
                    if (used[j])
                        continue;
                    auto [cj, corej] = extract_coeff(addends[j]);
                    if (structural_eq(corei, corej))
                    {
                        ci += cj;
                        used[j] = true;
                    }
                }
                groups.emplace_back(ci, corei);
            }

            if (groups.size() == addends.size())
                return e; // no merging happened

            // Rebuild sum from groups.
            Expr const* result = nullptr;
            for (auto const& [cnt, core]: groups)
            {
                if (cnt == 0)
                    continue;
                Expr const* term;
                if (cnt == 1)
                    term = core;
                else if (cnt == -1)
                    term = make_negate(ctx, core);
                else
                    term = make_tensor_product(
                        ctx, make_scalar(ctx, Rational{cnt}), core);
                result = result ? make_sum(ctx, result, term) : term;
            }
            return result ? result : make_scalar(ctx, Rational{0});
        });
}

} // namespace steps
} // namespace tender
