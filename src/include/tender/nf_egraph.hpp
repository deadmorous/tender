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

// What "simplest" means when a saturated e-graph is read back (vibe 000097).
//
// Cheapness is a fine default but it is not always what the user wants: "get
// rid of the two crosses" is a legitimate goal even though the crossed form
// has fewer nodes.  So the extraction cost is a *parameter* — an expression of
// intent — rather than a fixed property of the algebra.
//
// Each weight is added to `node` for the node kinds it names, and a class's
// cost is the sum over its chosen subtree.  Large weights encode a
// lexicographic order (minimize the weighted thing first, then total size),
// which is how the ε preference of vibe 000046 has always worked; keep them
// far apart and far above any realistic node count, and well below overflow.
//
// The cost affects *extraction only*, never which rewrites the engine
// explores — so the same saturated graph can be read out under several
// intents for the price of one saturation.
struct CostModel final
{
    std::size_t node = 1;              // every node: the baseline size term
    std::size_t levi_civita = 1000000; // extra, per ε symbol
    std::size_t cross = 0;             // extra, per × operator in a chain
    std::size_t delta = 0;             // extra, per δ symbol
    std::size_t identity = 0;          // extra, per identity tensor I
    std::size_t unary = 0;             // extra, per tr / vec / transpose
    std::size_t div = 0;               // extra, per division

    // Fewest Levi-Civita symbols, then fewest nodes — the default, and the
    // intent behind the ε-δ identities: contract the ε's away even though the
    // δ-expansion is the larger form (vibe 000046).
    [[nodiscard]] static auto fewest_eps() -> CostModel
    {
        return CostModel{};
    }

    // Plain node count: no thumb on any scale.
    [[nodiscard]] static auto smallest() -> CostModel
    {
        return CostModel{.node = 1, .levi_civita = 1};
    }

    // Fewest cross products, then fewest ε, then fewest nodes — "remove the
    // crosses", which is what turns bac-cab's *expansion* into the preferred
    // reading of the same graph.
    [[nodiscard]] static auto fewest_crosses() -> CostModel
    {
        return CostModel{.node = 1, .levi_civita = 1000, .cross = 1000000};
    }
};

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

    // Cheapest representative `Nf` of a class under `cost` (vibe 000097).
    // Requires a `Sum`-sort class (an `add`-returned id, or one merged with
    // such).  Extracting the same class again under a different `CostModel`
    // costs one more extraction, not another saturation.
    [[nodiscard]] auto extract(EClassId id, CostModel const& cost = {})
        -> Nf const*;

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
