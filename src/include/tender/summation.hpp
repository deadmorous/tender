#pragma once

// Shared index / summation helpers common to the `Expr` canonicalizer
// (derivation.cpp) and the `Nf` lowering (nf_lower.cpp).
//
// These were originally file-local helpers in derivation.cpp.  They are lifted
// here so the `Nf` summation pass (vibe 000058 / C8) can resolve a term's
// bound-index set with the *same* realm-driven Einstein rule and the *same*
// α-renaming the existing `Expr` canon uses — keeping the two IRs in lockstep
// (DRY; the C10 differential harness relies on it).

#include <tender/context.hpp>
#include <tender/expr.hpp>  // Expr, TensorObject
#include <tender/index.hpp> // Level, Realm, CountableIndex

#include <map>
#include <set>
#include <vector>

namespace tender
{

// ---- α-renaming --------------------------------------------------------

// Replace every CountableIndex{old_id} in TensorObject slots with
// CountableIndex{new_id} — used to α-normalize a binder's dummy index.
[[nodiscard]] auto substitute_index_id(
    Context& ctx, Expr const* e, int old_id, int new_id) -> Expr const*;

// Like substitute_index_id but applies a whole id→id map atomically (each slot
// id looked up once), so a remap that permutes ids (e.g. -1↔-2) is correct.
[[nodiscard]] auto substitute_index_ids(
    Context& ctx,
    Expr const* e,
    std::map<int, int> const& remap) -> Expr const*;

// Canonical (reserved, negative) id for a bound index at binder nesting depth
// d.  Free ids are non-negative, so this namespace never collides with them,
// and distinct depths get distinct ids (so Σ_i Σ_j keeps i and j apart).
[[nodiscard]] auto bound_canon_id(int depth) -> int;

// ---- implicit (Einstein) summation detection ---------------------------

// One free occurrence of a CountableIndex within a term.
struct IndexUse final
{
    Level level;
    Realm realm;
};

// A *term*: a pure multilinear expression (no Sum/Difference/ScalarDiv/binder),
// so the whole thing is one Einstein-summation scope.
[[nodiscard]] auto is_term(Expr const* e) -> bool;

// Collect every free CountableIndex occurrence in a term, descending through
// its multilinear structure.  Stops at scope boundaries (a non-term node is
// opaque).  Ids already in `bound` are skipped.
void collect_term_uses(
    Expr const* e,
    std::set<int> const& bound,
    std::map<int, std::vector<IndexUse>>& uses);

// Decide which free ids in `term` are implicitly contracted, per each id's
// realm rule.  Ids already bound by an enclosing ExplicitSum/NoSum are in
// `bound` and excluded — so an explicit override both suppresses contraction
// and silences the error checks.  Throws for ill-formed terms with no such
// override.  Returns the contracted ids ascending (std::map key order).
[[nodiscard]] auto contracted_ids(Expr const* term, std::set<int> const& bound)
    -> std::vector<int>;

} // namespace tender
