#pragma once

#include <mpk/mix/util/overloads.hpp>
#include <tender/expr.hpp>

namespace tender
{

// Rebuild an expression bottom-up: first recurse into each child, then
// apply f(ctx, rebuilt_node).  If f returns the node unchanged, the pointer is
// reused without extra allocation.
//
// Shared by the canonicalizer, the built-in rewrite steps, and the identity
// matcher (apply_identity walks bottom-up with a first-match guard inside f).
template <typename F>
auto rewrite_tree(Context& ctx, Expr const* e, F const& f) -> Expr const*
{
    Expr const* rebuilt = visit(
        mpk::mix::Overloads{
            // Leaf nodes — no children.
            [&](TensorObject const&) -> Expr const* { return e; },
            [&](ScalarLiteral const&) -> Expr const* { return e; },

            // Unary.
            [&](Negate const& n) -> Expr const*
            {
                auto* op = rewrite_tree(ctx, n.operand, f);
                return op == n.operand ? e : make_negate(ctx, op);
            },
            [&](Trace const& u) -> Expr const*
            {
                auto* op = rewrite_tree(ctx, u.operand, f);
                return op == u.operand ? e : make_trace(ctx, op);
            },
            [&](VectorInvariant const& u) -> Expr const*
            {
                auto* op = rewrite_tree(ctx, u.operand, f);
                return op == u.operand ? e : make_vector_invariant(ctx, op);
            },
            [&](Transpose const& u) -> Expr const*
            {
                auto* op = rewrite_tree(ctx, u.operand, f);
                return op == u.operand ? e : make_transpose(ctx, op);
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

            // Scalar fields (vibe 000069).
            [&](ScalarFn const& s) -> Expr const*
            {
                auto* op = rewrite_tree(ctx, s.operand, f);
                return op == s.operand ? e : make_scalar_fn(ctx, s.kind, op);
            },
            [&](Pow const& s) -> Expr const*
            {
                auto* base = rewrite_tree(ctx, s.base, f);
                auto* exp = rewrite_tree(ctx, s.exponent, f);
                return (base == s.base && exp == s.exponent) ?
                           e :
                           make_pow(ctx, base, exp);
            },
        },
        *e);

    return f(ctx, rebuilt);
}

} // namespace tender
