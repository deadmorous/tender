#pragma once

#include <tender/expr.hpp>

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tender
{

// An Identity is a directed rewrite rule LHS = RHS over Expr trees (vibe
// 000033).  It is deliberately *not* a Theorem: a theorem is a derivation that
// proves a result and carries its history; an identity is the bare equality
// such a theorem may yield.  The two are separate types — a future Theorem will
// aggregate a Derivation and expose the Identity it proves, never inherit from
// it.
//
// Pattern variables come in two kinds (vibes 000033, 000051):
//
//  - *Index variables* — every free CountableIndex of the LHS matches whatever
//    index sits in the corresponding target slot, consistently across the
//    match. Indices bound by an ExplicitSum/NoSum are local (alpha) variables
//    matched to the target's binder.
//  - *Subtree variables* — a slot-less, non-well-known named TensorObject (e.g.
//    `a`, `A`) matches any whole subtree of compatible rank, consistently. This
//    is what invariant (direct-notation) identities like (a⊗b):(c⊗d)=(a·c)(b·d)
//    need.  Well-known tensors (I, δ, ε) and slotted tensors stay literal.
struct Identity final
{
    std::string name;
    Expr const* lhs;
    Expr const* rhs;
};

// The result of a successful match: pattern index ids paired with the target
// indices they bound, and pattern subtree-variable names paired with the target
// subtrees they bound.
struct MatchBinding final
{
    std::vector<std::pair<int, IndexAssoc>> indices;
    std::vector<std::pair<std::string, Expr const*>> subtrees;

    // The target index bound to pattern index `id`, or nullopt if unbound.
    [[nodiscard]] auto find(int id) const -> std::optional<IndexAssoc>;
    // The target subtree bound to pattern variable `name`, or nullptr.
    [[nodiscard]] auto find_subtree(std::string_view name) const -> Expr const*;
};

// Try to match `pattern` against `target` at the root (no descent into the
// target tree).  Commutative sums and component products are matched modulo
// addend/factor order (bounded backtracking); every other node is matched
// structurally.  Returns the binding on success, nullopt on failure.
//
// For robust matching both `pattern` and `target` should be in canonical form
// (apply_identity arranges this); alpha-normalized ExplicitSum binders then
// line up automatically.
[[nodiscard]] auto match(Expr const* pattern, Expr const* target)
    -> std::optional<MatchBinding>;

// Lower-level matching primitives, exposed for the e-graph matcher (vibe
// 000034).  match() is the fresh-binding wrapper around match_into.
//
// match_into threads an existing binding (extending it in place) and returns
// whether `pattern` matches `target`.  bind_pattern_index binds a pattern index
// id to a target index, or requires consistency with an existing binding — used
// when matching an ExplicitSum/NoSum binder against an e-node's binder id.
[[nodiscard]] auto match_into(
    Expr const* pattern, Expr const* target, MatchBinding& bnd) -> bool;
[[nodiscard]] auto bind_pattern_index(
    MatchBinding& bnd, int pattern_id, IndexAssoc const& target) -> bool;

// Instantiate `rhs` under `binding`: replace each pattern index by the target
// index it was bound to.  Pattern indices absent from the binding are left as
// they are.
[[nodiscard]] auto instantiate(Context&, Expr const* rhs, MatchBinding const&)
    -> Expr const*;

// Canonicalize `e`, walk it bottom-up (deepest first), and at the first subtree
// where `id.lhs` matches, replace that subtree with the instantiated `id.rhs`;
// the result is re-canonicalized.  The return value is always canonical: if
// nothing matched it equals canonicalize(e).
[[nodiscard]] auto apply_identity(Context&, Expr const* e, Identity const& id)
    -> Expr const*;

namespace steps
{

// Derivation-step factory: drv.step(steps::apply_identity(id)).  The derivation
// history thus records one entry per identity application.
[[nodiscard]] auto apply_identity(Identity id)
    -> std::function<Expr const*(Context&, Expr const*)>;

} // namespace steps

} // namespace tender
