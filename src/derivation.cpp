#include <tender/derivation.hpp>

#include <mpk/mix/util/overloads.hpp>

#include <functional>

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
                        if (!l || !r)
                            return e;
                        return make_scalar(ctx, l->value + r->value);
                    },
                    [&](Difference const& s) -> Expr const*
                    {
                        auto const* l =
                            std::get_if<ScalarLiteral>(&s.left->node);
                        auto const* r =
                            std::get_if<ScalarLiteral>(&s.right->node);
                        if (!l || !r)
                            return e;
                        return make_scalar(ctx, l->value - r->value);
                    },
                    [&](TensorProduct const& s) -> Expr const*
                    {
                        auto const* l =
                            std::get_if<ScalarLiteral>(&s.left->node);
                        auto const* r =
                            std::get_if<ScalarLiteral>(&s.right->node);
                        if (!l || !r)
                            return e;
                        return make_scalar(ctx, l->value * r->value);
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
                        if (!v)
                            return e;
                        return make_scalar(ctx, -v->value);
                    },
                    [&](auto const&) -> Expr const* { return e; },
                },
                *e);
        });
}

} // namespace steps
} // namespace tender
