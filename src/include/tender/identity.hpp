#pragma once

#include <tender/expr.hpp>

#include <functional>
#include <optional>
#include <string>
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
// Pattern variables are the *free indices* of the LHS: every CountableIndex not
// bound by an ExplicitSum/NoSum inside the LHS matches whatever index sits in
// the corresponding target slot, consistently across the whole match.  Indices
// bound by an ExplicitSum/NoSum are local (alpha) variables, matched to the
// target's binder.  (Slot-less named tensors as whole-subtree variables are not
// supported — see vibe 000033 §4.1; index matching is what the index identities
// actually need.)
struct Identity final
{
    std::string name;
    Expr const* lhs;
    Expr const* rhs;
};

// The result of a successful match: each LHS pattern index id paired with the
// target index it was bound to.
struct MatchBinding final
{
    std::vector<std::pair<int, IndexAssoc>> indices;

    // The target index bound to pattern index `id`, or nullopt if unbound.
    [[nodiscard]] auto find(int id) const -> std::optional<IndexAssoc>;
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
