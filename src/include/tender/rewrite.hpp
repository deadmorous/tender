#pragma once

#include <mpk/mix/types/capped_vec.hpp>
#include <mpk/mix/util/overloads.hpp>
#include <tender/expr.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tender
{

// ---- Child navigation (vibe 000054) ------------------------------------

// An expression node holds at most two Expr children (the binary operators and
// Pow; ExplicitSum with a symbolic bound), so a heap-free capped vector keeps
// the whole-tree walks (rewrite_tree, find_occurrences) allocation-free.
using ExprChildren = mpk::mix::CappedVec<Expr const*, 2>;

namespace detail
{
// Build an ExprChildren from a fixed list of child pointers (CappedVec has no
// initializer-list constructor).
template <typename... Ts>
auto expr_children(Ts... cs) -> ExprChildren
{
    ExprChildren v;
    (v.push_back(cs), ...);
    return v;
}
} // namespace detail

// The Expr children of `e`, in selector order — the same order rewrite_tree
// recurses and a Path addresses (vibe 000054): binary `left`=0 / `right`=1;
// unary `operand`=0; `Pow` `base`=0 / `exponent`=1; `ExplicitSum` `body`=0
// (`bound`=1 only when a symbolic bound is present); `NoSum` / `ScalarFn` /
// `Deriv` their single child; the leaves (`TensorObject`, `ScalarLiteral`,
// `Nabla`) none.  Non-Expr fields (a bound index, a ScalarFn kind) are not
// children; `with_children` restores them.
inline auto children(Expr const* e) -> ExprChildren
{
    return visit(
        mpk::mix::Overloads{
            // Leaves.
            [](TensorObject const&) { return ExprChildren{}; },
            [](ScalarLiteral const&) { return ExprChildren{}; },
            [](Nabla const&) { return ExprChildren{}; },

            // Unary.
            [](Negate const& n) { return detail::expr_children(n.operand); },
            [](Trace const& u) { return detail::expr_children(u.operand); },
            [](VectorInvariant const& u)
            { return detail::expr_children(u.operand); },
            [](Transpose const& u) { return detail::expr_children(u.operand); },
            [](ScalarFn const& s) { return detail::expr_children(s.operand); },
            [](Deriv const& s) { return detail::expr_children(s.wrt); },

            // Binary.
            [](Sum const& s) { return detail::expr_children(s.left, s.right); },
            [](Difference const& s)
            { return detail::expr_children(s.left, s.right); },
            [](TensorProduct const& s)
            { return detail::expr_children(s.left, s.right); },
            [](ScalarDiv const& s)
            { return detail::expr_children(s.left, s.right); },
            [](Dot const& s) { return detail::expr_children(s.left, s.right); },
            [](DDot const& s)
            { return detail::expr_children(s.left, s.right); },
            [](DDotAlt const& s)
            { return detail::expr_children(s.left, s.right); },
            [](Cross const& s)
            { return detail::expr_children(s.left, s.right); },
            [](Pow const& s)
            { return detail::expr_children(s.base, s.exponent); },

            // Annotation nodes.  ExplicitSum's optional bound is a child only
            // when present, so `body` stays at selector 0 either way.
            [](ExplicitSum const& s)
            {
                return s.bound ? detail::expr_children(s.body, s.bound) :
                                 detail::expr_children(s.body);
            },
            [](NoSum const& s) { return detail::expr_children(s.body); },
        },
        *e);
}

// Rebuild `e` with new children (vibe 000054), preserving its kind and every
// non-Expr field (a bound index, a ScalarFn kind, the ExplicitSum bound
// arity).  `new_children` must have exactly the arity `children(e)` reports
// (else std::invalid_argument).  Returns `e` unchanged — same pointer — when
// every child is identical, so an identity rewrite allocates nothing; otherwise
// reconstructs via the same `make_*` factory rewrite_tree uses, so any factory
// normalisation (e.g. a scalar leg turning a Dot into a TensorProduct) is
// identical to a full-tree rewrite.
inline auto with_children(Context& ctx, Expr const* e, ExprChildren const& kids)
    -> Expr const*
{
    auto expect = [&](std::size_t n)
    {
        if (kids.size() != n)
            throw std::invalid_argument(
                "with_children: child count does not match the node arity");
    };
    return visit(
        mpk::mix::Overloads{
            // Leaves.
            [&](TensorObject const&) -> Expr const* { return expect(0), e; },
            [&](ScalarLiteral const&) -> Expr const* { return expect(0), e; },
            [&](Nabla const&) -> Expr const* { return expect(0), e; },

            // Unary.
            [&](Negate const& n) -> Expr const*
            {
                expect(1);
                return kids[0] == n.operand ? e : make_negate(ctx, kids[0]);
            },
            [&](Trace const& u) -> Expr const*
            {
                expect(1);
                return kids[0] == u.operand ? e : make_trace(ctx, kids[0]);
            },
            [&](VectorInvariant const& u) -> Expr const*
            {
                expect(1);
                return kids[0] == u.operand ?
                           e :
                           make_vector_invariant(ctx, kids[0]);
            },
            [&](Transpose const& u) -> Expr const*
            {
                expect(1);
                return kids[0] == u.operand ? e : make_transpose(ctx, kids[0]);
            },
            [&](ScalarFn const& s) -> Expr const*
            {
                expect(1);
                return kids[0] == s.operand ?
                           e :
                           make_scalar_fn(ctx, s.kind, kids[0]);
            },
            [&](Deriv const& s) -> Expr const*
            {
                expect(1);
                return kids[0] == s.wrt ? e : make_deriv(ctx, kids[0]);
            },

            // Binary.
            [&](Sum const& s) -> Expr const*
            {
                expect(2);
                return (kids[0] == s.left && kids[1] == s.right) ?
                           e :
                           make_sum(ctx, kids[0], kids[1]);
            },
            [&](Difference const& s) -> Expr const*
            {
                expect(2);
                return (kids[0] == s.left && kids[1] == s.right) ?
                           e :
                           make_difference(ctx, kids[0], kids[1]);
            },
            [&](TensorProduct const& s) -> Expr const*
            {
                expect(2);
                return (kids[0] == s.left && kids[1] == s.right) ?
                           e :
                           make_tensor_product(ctx, kids[0], kids[1]);
            },
            [&](ScalarDiv const& s) -> Expr const*
            {
                expect(2);
                return (kids[0] == s.left && kids[1] == s.right) ?
                           e :
                           make_scalar_div(ctx, kids[0], kids[1]);
            },
            [&](Dot const& s) -> Expr const*
            {
                expect(2);
                return (kids[0] == s.left && kids[1] == s.right) ?
                           e :
                           make_dot(ctx, kids[0], kids[1]);
            },
            [&](DDot const& s) -> Expr const*
            {
                expect(2);
                return (kids[0] == s.left && kids[1] == s.right) ?
                           e :
                           make_ddot(ctx, kids[0], kids[1]);
            },
            [&](DDotAlt const& s) -> Expr const*
            {
                expect(2);
                return (kids[0] == s.left && kids[1] == s.right) ?
                           e :
                           make_ddot_alt(ctx, kids[0], kids[1]);
            },
            [&](Cross const& s) -> Expr const*
            {
                expect(2);
                return (kids[0] == s.left && kids[1] == s.right) ?
                           e :
                           make_cross(ctx, kids[0], kids[1]);
            },
            [&](Pow const& s) -> Expr const*
            {
                expect(2);
                return (kids[0] == s.base && kids[1] == s.exponent) ?
                           e :
                           make_pow(ctx, kids[0], kids[1]);
            },

            // Annotation nodes.
            [&](ExplicitSum const& s) -> Expr const*
            {
                expect(s.bound ? 2 : 1);
                auto* bound = s.bound ? kids[1] : nullptr;
                return (kids[0] == s.body && bound == s.bound) ?
                           e :
                           make_explicit_sum(ctx, s.index, kids[0], bound);
            },
            [&](NoSum const& s) -> Expr const*
            {
                expect(1);
                return kids[0] == s.body ? e :
                                           make_no_sum(ctx, s.index, kids[0]);
            },
        },
        *e);
}

// ---- Whole-tree rewrite ------------------------------------------------

// Rebuild an expression bottom-up: first recurse into each child, then
// apply f(ctx, rebuilt_node).  If f returns the node unchanged, the pointer is
// reused without extra allocation.
//
// Shared by the canonicalizer, the built-in rewrite steps, and the identity
// matcher (apply_identity walks bottom-up with a first-match guard inside f).
// Expressed on children/with_children (vibe 000054) so the per-node-kind child
// structure lives in exactly one place.
template <typename F>
auto rewrite_tree(Context& ctx, Expr const* e, F const& f) -> Expr const*
{
    auto kids = children(e);
    bool changed = false;
    for (auto& c: kids)
    {
        auto* nc = rewrite_tree(ctx, c, f);
        if (nc != c)
        {
            c = nc;
            changed = true;
        }
    }
    Expr const* rebuilt = changed ? with_children(ctx, e, kids) : e;
    return f(ctx, rebuilt);
}

// ---- Positional addressing (vibe 000054) -------------------------------

// A path is a route from the root: one child-selector per level (a Dewey
// address).  Hash-consing makes identical subterms one shared node, so a node
// cannot name its own occurrence; a path is the only handle on *this* one.  A
// path addresses one specific tree — canonicalize reshapes the tree and
// invalidates it, so the workflow is canonicalize → address → rewrite, with no
// canonicalize in between.
using Path = std::vector<int>;

// The subexpression at `path` — navigation / extraction.  The returned node
// shares `e`'s context (arena), so it is a first-class expression any step
// runs on.  An out-of-range selector throws std::out_of_range (never a silent
// nullptr).
inline auto subexpr_at(Expr const* e, Path const& path) -> Expr const*
{
    Expr const* cur = e;
    for (int sel: path)
    {
        auto kids = children(cur);
        if (sel < 0 || static_cast<std::size_t>(sel) >= kids.size())
            throw std::out_of_range("subexpr_at: selector out of range");
        cur = kids[static_cast<std::size_t>(sel)];
    }
    return cur;
}

// Apply `f` to the subexpression at `path`, rebuilding only the spine above the
// target; every off-path node stays shared (its pointer is preserved), so the
// rewrite is cheap.  `f` has the same shape as rewrite_tree's — `(Context&,
// Expr const*) -> Expr const*`.  An out-of-range selector throws
// std::out_of_range.
template <typename F>
auto rewrite_at(
    Context& ctx,
    Expr const* e,
    Path const& path,
    F const& f,
    std::size_t depth = 0) -> Expr const*
{
    if (depth == path.size())
        return f(ctx, e);
    auto kids = children(e);
    int sel = path[depth];
    if (sel < 0 || static_cast<std::size_t>(sel) >= kids.size())
        throw std::out_of_range("rewrite_at: selector out of range");
    auto idx = static_cast<std::size_t>(sel);
    Expr const* nc = rewrite_at(ctx, kids[idx], path, f, depth + 1);
    if (nc == kids[idx])
        return e;
    kids[idx] = nc;
    return with_children(ctx, e, kids);
}

// Replace the subexpression at `path` with `sub` — splice a separately
// worked-on part back in (the round-trip: sub = subexpr_at(e, p); … ;
// replace_at(e, p, sub2)).
inline auto replace_at(
    Context& ctx,
    Expr const* e,
    Path const& path,
    Expr const* sub) -> Expr const*
{
    return rewrite_at(
        ctx, e, path, [sub](Context&, Expr const*) { return sub; });
}

// Every path (pre-order: a node before its children) at which `pred` holds.
// The building block for the `Nth`/predicate selectors — e.g. the k-th
// identity, a tensor by name.  Computed on the tree it will address (see the
// Path note on canon-stability).
template <typename Pred>
void find_occurrences_into(
    Expr const* e, Pred const& pred, Path& cur, std::vector<Path>& out)
{
    if (pred(e))
        out.push_back(cur);
    auto kids = children(e);
    for (std::size_t i = 0; i < kids.size(); ++i)
    {
        cur.push_back(static_cast<int>(i));
        find_occurrences_into(kids[i], pred, cur, out);
        cur.pop_back();
    }
}

template <typename Pred>
auto find_occurrences(Expr const* e, Pred const& pred) -> std::vector<Path>
{
    std::vector<Path> out;
    Path cur;
    find_occurrences_into(e, pred, cur, out);
    return out;
}

// The paths to the top-level addends of `e` — descend the Sum/Difference spine,
// one path per leaf term (a non-Sum/Difference node), so "extract *a term*" is
// a natural selector.  A right addend reached through a Difference is still
// just its own subtree (the sign stays on the spine); a lone non-additive `e`
// yields the single empty path.
inline void addend_paths_into(Expr const* e, Path& cur, std::vector<Path>& out)
{
    if (std::holds_alternative<Sum>(e->node)
        || std::holds_alternative<Difference>(e->node))
    {
        auto kids = children(e);
        for (std::size_t i = 0; i < kids.size(); ++i)
        {
            cur.push_back(static_cast<int>(i));
            addend_paths_into(kids[i], cur, out);
            cur.pop_back();
        }
        return;
    }
    out.push_back(cur);
}

inline auto addend_paths(Expr const* e) -> std::vector<Path>
{
    std::vector<Path> out;
    Path cur;
    addend_paths_into(e, cur, out);
    return out;
}

// ---- Nabla detection ---------------------------------------------------

// True if `e` contains an abstract ∇ (an unlowered `Nabla` operator) anywhere
// in its tree.  A differential operator makes a term positional — no
// contraction / ⊗ reassociation may move a factor across it (mirrors nf
// place_factors) — so this is the barrier both `distribute_contraction` and
// `encapsulate` consult to keep `∇·(∇⊗X)` (any operand, incl. a product `u e`)
// nested through canonicalize (vibes 000085/000086).  A concrete
// post-application `Deriv` is NOT a ∇ and does not trip it.
inline auto contains_nabla(Context& ctx, Expr const* e) -> bool
{
    bool found = false;
    rewrite_tree(
        ctx,
        e,
        [&](Context&, Expr const* n) -> Expr const*
        {
            if (std::holds_alternative<Nabla>(n->node))
                found = true;
            return n;
        });
    return found;
}

} // namespace tender
