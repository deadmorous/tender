#pragma once

// Term-view combinators (vibe 000095, M1 increment 1).
//
// The one home for "walk an expression's additive / term structure" so that
// derivation steps stop re-peeling the raised binary tree by hand (the
// vibe-000092 §2 duplication: `collect_signed_addends`, `expand_unary`'s
// Sum/Difference/Negate/ScalarDiv cascade, `dd_expand`'s shape list).  Two
// levels:
//
//  - *Surface* combinators preserve the expression exactly as written — no
//    canonicalization, no reordering — as required by routes that must not be
//    re-canonicalized mid-way (vibe 000080).  They reuse the input pointer
//    whenever nothing changes, supporting the step no-op contract
//    (vibe 000095 increment 3).
//
//  - The *canonical* combinator exposes the expression as its normal-form
//    term set (`nf::Term`s — signed rational coeff, sorted scalars,
//    positional tensors) for steps whose natural domain is "the terms".

#include <tender/expr.hpp>
#include <tender/nf.hpp>
#include <tender/nf_lower.hpp>

#include <functional>
#include <vector>

namespace tender::view
{

// ---- surface: signed addends -------------------------------------------

// Flatten the outermost additive layer into signed leaf addends: `Sum` keeps
// the sign, `Difference` flips it on the right, `Negate` flips it.  Never
// distributes — an additive node inside a product stays inside its leaf.
// (Delegates to `nf::additive_flatten`; this is the single implementation.)
[[nodiscard]] auto signed_addends(Expr const* e) -> std::vector<nf::SignedExpr>;

// Rebuild a signed-addend list into an expression: the first negative addend
// becomes a leading `Negate`, later ones join with `Sum` / `Difference`.  An
// empty list is the scalar 0.  `sum_of(signed_addends(e))` is `e` up to the
// (+ / −) encoding of signs, and `signed_addends(sum_of(v))` is `v` exactly.
[[nodiscard]] auto sum_of(Context&, std::vector<nf::SignedExpr> const& addends)
    -> Expr const*;

// ---- surface: skeleton-preserving additive map -------------------------

struct AdditiveOptions final
{
    // Descend the dividend of a `ScalarDiv` (a linear op commutes through a
    // scalar divisor: op(X/c) = op(X)/c).  On by default — the behaviour of
    // `expand_unary` and `dd_expand`.
    bool descend_scalar_div = true;
};

// Apply `leaf` to every additive leaf of `e`, descending `Sum`, `Difference`,
// `Negate` (and optionally a `ScalarDiv` dividend) and rebuilding the exact
// skeleton around the results — the shape of the input is preserved
// byte-for-byte in the render.  Reuses input pointers whenever `leaf` returns
// its argument unchanged, so an all-no-op map returns `e` itself.
[[nodiscard]] auto map_additive_leaves(
    Context&,
    Expr const* e,
    std::function<Expr const*(Expr const*)> const& leaf,
    AdditiveOptions = {}) -> Expr const*;

// ---- canonical: normal-form term map -----------------------------------

// Expose `e` as its canonical term set and let `transform` edit it in place
// (drop terms, adjust coefficients, rewrite factors).  The result is
// re-canonicalized (the transform need not keep the set sorted or collected)
// and raised back to an `Expr` in explicit-sum form, exactly like
// `canonicalize`.  If the transform changes nothing, the *input* `e` is
// returned unchanged (no-op pointer contract) — note this means an identity
// transform does NOT canonicalize.
[[nodiscard]] auto map_nf_terms(
    Context&,
    Expr const* e,
    std::function<void(std::vector<nf::Term>&)> const& transform) -> Expr const*;

// ---- fixpoint ----------------------------------------------------------

// Iterate `step` until it returns its input pointer (the step no-op
// contract).  `max_iterations` guards a step that never converges; hitting
// the guard throws.
[[nodiscard]] auto fixpoint(
    Context&,
    Expr const* e,
    std::function<Expr const*(Context&, Expr const*)> const& step,
    int max_iterations = 64) -> Expr const*;

} // namespace tender::view
