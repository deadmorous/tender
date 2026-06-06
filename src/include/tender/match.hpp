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
auto find_matches(
    Identity const& id, Expr* expr, int max_nodes = 10'000)
    -> std::vector<PatternBinding>;

// Convenience: apply `id` using the first match found in `expr`.
// Throws std::invalid_argument if no match is found.
auto apply_identity_auto(Identity const& id, Expr* expr, int max_nodes = 10'000)
    -> DerivationStep;

} // namespace tender
