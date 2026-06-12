#pragma once

#include <map>
#include <vector>

#include <tender/identity.hpp>

namespace tender
{

// ===========================================================================
// Pattern matching
// ===========================================================================

// A binding maps each PatternVar in the identity LHS to the concrete
// sub-expression it was matched to.
using PatternBinding = std::map<PatternVar const*, Expr*>;

// Try to match `pattern` against `expr` given existing `bindings`.
// Returns the (possibly extended) set of bindings on success, or an empty
// vector on failure.  Multiple successful bindings can be returned when the
// same match succeeds in more than one way (rare in practice).
auto match_pattern(Expr* pattern, Expr* expr, PatternBinding const& bindings)
    -> std::vector<PatternBinding>;

// Walk `expr` in pre-order and collect every sub-expression position where
// `id.lhs()` matches.  Returns one binding per match site.
// Throws std::runtime_error if more than `max_nodes` nodes are visited.
auto find_matches(Identity const& id, Expr* expr, int max_nodes = 10'000)
    -> std::vector<PatternBinding>;

// Convenience: apply `id` to `expr` if the LHS matches at the root.
// Throws std::invalid_argument if the root does not match.
// For deep-in-tree matches, use find_matches() + apply_identity() instead.
auto apply_identity_auto(Identity const& id, Expr* expr) -> DerivationStep;

// Walk `root` in pre-order.  At every node where `id.lhs()` matches, compute
// the rewritten root (replacing that node with `id.rhs()` instantiated by the
// binding) and record (new_root, step_name) in the result.
// Intended for BFS-based rewrite search over sub-expressions.
auto find_and_rewrite_all(Identity const& id, ResourceList& rl, Expr* root)
    -> std::vector<std::pair<Expr*, std::string>>;

} // namespace tender
