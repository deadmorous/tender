#pragma once

#include <tender/expr.hpp>
#include <tender/identity.hpp> // MatchBinding

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace tender
{

class Context;

// An e-class identifier.  Stable for the lifetime of the EGraph; use find() to
// obtain the current canonical representative after merges.
using EClassId = int;

// An equality graph (e-graph) over tender expressions (vibe 000034).
//
// It compactly represents sets of expressions proved equal: e-nodes are
// hash-consed (one per distinct operator + child-classes), e-classes are
// equivalence classes maintained by union-find, and rebuild() restores the
// congruence invariant after merges (if a == b then f(a) == f(b)).
//
// Expressions are canonicalized (vibe 000037) on insertion, so AC ordering and
// α-equivalence of bound indices are handled before hash-consing — two
// algebraically-equal inputs land in the same e-class.
//
// This is the data-structure core: union-find, hash-consing, congruence, and
// cost-based extraction.  The saturation loop and the e-class pattern matcher
// build on top of it.
class EGraph final
{
public:
    explicit EGraph(Context& ctx);
    ~EGraph();
    EGraph(EGraph&&) noexcept;
    auto operator=(EGraph&&) noexcept -> EGraph&;

    // Insert e (canonicalized first) and return the id of its e-class.
    [[nodiscard]] auto add(Expr const* e) -> EClassId;

    // Union the e-classes of a and b; returns the surviving representative.
    // Call rebuild() afterwards to restore congruence before querying.
    auto merge(EClassId a, EClassId b) -> EClassId;

    // Restore the congruence invariant after one or more merges.
    void rebuild();

    // Canonical representative of the class containing id.
    [[nodiscard]] auto find(EClassId id) -> EClassId;

    // Cheapest (smallest-node-count) representative expression of a class.
    [[nodiscard]] auto extract(EClassId id) -> Expr const*;

    // E-matching (vibe 000034): find every (e-class, binding) under which
    // `pattern` — an Identity LHS — matches an expression represented in that
    // class.  `pattern` is canonicalized first; the free indices of the pattern
    // are bound in each returned MatchBinding (feed it to instantiate()).  All
    // e-nodes of a class are searched, not just the extracted representative,
    // so a match is found even when the cheapest form of a class is a
    // different, non-matching expression.
    [[nodiscard]] auto ematch(Expr const* pattern)
        -> std::vector<std::pair<EClassId, MatchBinding>>;

    // Number of distinct e-classes (diagnostics / benchmarks).
    [[nodiscard]] auto class_count() -> std::size_t;

    // Number of distinct e-nodes (diagnostics / benchmarks).
    [[nodiscard]] auto node_count() const -> std::size_t;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tender
