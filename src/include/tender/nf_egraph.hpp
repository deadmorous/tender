#pragma once

// An equality graph over the normal form `Nf` (vibe 000058 / C14d part 2).
//
// This is the Nf-native successor to the `Expr`-structural `EGraph` of vibe
// 000034.  Where that e-graph's e-nodes mirror `Expr` operator nodes (binary
// `Sum` / `TensorProduct` / `Cross` …), this one's e-nodes mirror the `Nf`
// structure: the recursive `Factor` tree (`Atom` / `Contraction` / `Cross` /
// `Paren` / `Unary` / `Div`), the multiplicative `Term`
// (`coeff · scalars · tensors · bound`), and the additive `Sum` of terms.  So
// the e-class matcher can be the flat-form `nf_match` matcher rather than a
// second, divergent structural matcher.
//
// It grows one commit at a time, beside the existing `EGraph`, mirroring the
// parallel-IR strategy of the rest of 000058.  This first commit (data core)
// introduces the e-node representation, union-find, hash-consing, and `add`
// (lowering a canonical `Nf` / `Expr` into the graph); congruence `rebuild`,
// `extract`, `ematch`, and `saturate` arrive in later commits.

#include <tender/expr.hpp>
#include <tender/identity.hpp> // Identity
#include <tender/nf.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace tender::nf
{

// An e-class identifier, stable for the graph's lifetime; use `find()` for the
// current representative after merges.
using EClassId = int;

// Saturation budget (vibe 000096 increment 1; defaults from the vibe
// 000093 decision ledger).  `max_nodes` is checked *between* passes: a
// pass is never started on a graph already at the limit, so one pass may
// overshoot it.
struct SaturateBudget final
{
    int max_passes = 30;
    std::size_t max_nodes = 10'000;
};

// Why saturation stopped.  Only `Saturated` means the rule set was
// exhausted; a budget outcome is *inconclusive* — a caller must never
// read it as "the rules cannot do it".
enum class SaturateOutcome : std::uint8_t
{
    Saturated,  // fixed point: a pass merged nothing new
    PassBudget, // stopped by max_passes while still changing
    NodeBudget, // graph reached max_nodes
    EarlyStop,  // the caller's stop() reported its goal reached
};

// The saturation trace (vibe 000096): what happened, and which rules did
// it.  `fired[i]` counts rule i's graph-changing merges (a rewrite that
// lands in an existing class does not count); `skipped` lists the rules
// that could not be compiled (a multi-term LHS — no `Nf` sub-sum matcher
// yet), so a silently inert rule is visible rather than mysterious.
struct SaturateReport final
{
    SaturateOutcome outcome = SaturateOutcome::Saturated;
    int passes = 0;
    std::size_t nodes = 0;
    std::vector<int> fired = {};
    std::vector<std::size_t> skipped = {};
};

class NfEGraph final
{
public:
    explicit NfEGraph(Context& ctx);
    ~NfEGraph();
    NfEGraph(NfEGraph&&) noexcept;
    auto operator=(NfEGraph&&) noexcept -> NfEGraph&;

    // Insert a canonical `Nf` and return the id of the e-class representing it
    // (its additive `Sum` node).  Equal `Nf`s land in the same class.
    [[nodiscard]] auto add(Nf const* nf) -> EClassId;

    // Convenience: canonicalize `e` to `Nf`
    // (`canonicalize_nf(canonicalize(e))`) and insert it.
    [[nodiscard]] auto add(Expr const* e) -> EClassId;

    // Union the e-classes of a and b; returns the surviving representative.
    // Call rebuild() afterwards to restore congruence before querying.
    auto merge(EClassId a, EClassId b) -> EClassId;

    // Restore the congruence invariant after one or more merges.
    void rebuild();

    // Canonical representative of the class containing id.
    [[nodiscard]] auto find(EClassId id) -> EClassId;

    // Cheapest (smallest node-count) representative `Nf` of a class.  Requires
    // a `Sum`-sort class (an `add`-returned id, or one merged with such).
    [[nodiscard]] auto extract(EClassId id) -> Nf const*;

    // Equality saturation over the `Nf` (vibe 000058 / C14d).  Each
    // single-term identity `lhs = rhs` is fired — via the `nf_match` matcher —
    // on every term of every additive (`Sum`) e-node in the graph: a
    // sub-product match (the LHS sits among extra factors of a term) or a
    // sub-chain match (the LHS is a contiguous run inside a chain factor).  The
    // rewritten `Nf` is canonicalized, inserted, and merged into the matched
    // class.  Passes run to a fixed point within `budget`; `stop`, when given,
    // is polled after each pass so a caller with a goal (`prove_equal`) can
    // finish as soon as it is met.  Afterwards `extract(find(root))` yields
    // the cheapest form.
    auto saturate(
        std::vector<Identity> const& rules,
        SaturateBudget budget = {},
        std::function<bool()> const& stop = {}) -> SaturateReport;

    // Number of distinct e-classes / e-nodes (diagnostics / tests).
    [[nodiscard]] auto class_count() -> std::size_t;
    [[nodiscard]] auto node_count() const -> std::size_t;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tender::nf
