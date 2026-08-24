#pragma once

// The reasoning verbs (vibe 000096, M2): goal-directed entry points that drive
// equality saturation, so a caller states *what* they want rather than which
// rewriting steps to apply in which order — the vibe-000056 usability thesis.
//
// Both verbs run the `NfEGraph` engine under a budget and hand back a report:
// what happened, and which rules did it.  The budget is not an implementation
// detail — a run that stops on budget is *inconclusive*, a different answer
// from a run that exhausted its rules, and the types here keep the two apart.

#include <tender/expr.hpp>
#include <tender/identity.hpp>
#include <tender/nf_egraph.hpp>

#include <vector>

namespace tender::engine
{

using nf::CostModel;
using nf::SaturateBudget;
using nf::SaturateOutcome;
using nf::SaturateReport;

// The outcome of a proof attempt.
//
// Saturation alone is a semi-decision procedure — it can exhibit a proof,
// never a refutation — which left `Exhausted` conflating "this is false" with
// "my rules were too weak", the least useful answer a verb can give.  So a
// second, independent procedure supplies the negative: expanding both sides
// into concrete components *decides* the chart-free algebraic fragment
// (vibe 000097).  `Refuted` is that verdict, and it is a real one.
enum class ProofStatus : std::uint8_t
{
    Proved,    // the two sides landed in one e-class
    Refuted,   // components differ: the statement is false
    Exhausted, // rules ran out and components could not decide either
    Budget,    // stopped on the pass/node budget: nothing is concluded
};

struct ProofResult final
{
    ProofStatus status = ProofStatus::Exhausted;
    SaturateReport report = {};

    // Set when the component check ran and found the two sides *agree*, yet
    // saturation could not prove it: the statement looks true and the rule
    // set is incomplete.  That is a different problem from a false claim, and
    // pointing at the rules rather than the mathematics is the useful thing
    // to say.
    bool components_agree = false;

    // Deliberately explicit rather than an `operator bool`: reading a proof
    // attempt as a plain bool is exactly the mistake that turns "we ran out of
    // budget" into "they are different".
    [[nodiscard]] auto proved() const -> bool
    {
        return status == ProofStatus::Proved;
    }

    // True only for a *decided* negative — never for "we ran out of rules".
    [[nodiscard]] auto refuted() const -> bool
    {
        return status == ProofStatus::Refuted;
    }
};

// What the component decision procedure concluded.
enum class ComponentVerdict : std::uint8_t
{
    Equal,     // the two sides reduce to the same concrete components
    Different, // they do not: the statement is false
    Undecided, // the expansion did not fully reduce (differential / chart
               // -dependent content, an unsupported shape, or it threw)
};

// Decide `lhs == rhs` by expanding both into concrete World-Cartesian
// components (vibe 000097).  This is a genuine decision procedure for the
// chart-free algebraic fragment — the same reduction every L1 challenge test
// performs by hand — and the source of the `Refuted` verdict above.  It is
// deliberately separate from the e-graph: independent machinery, so a bug in
// one does not silently confirm the other.
[[nodiscard]] auto decide_by_components(
    Context&, Expr const* lhs, Expr const* rhs) -> ComponentVerdict;

// Try to prove `lhs == rhs` by equality saturation under `rules`.  Both sides
// are inserted into one e-graph and saturated together, so rules meeting in
// the middle suffice — neither side need be rewritten all the way into the
// other.  Saturation stops as soon as the two roots share an e-class.
[[nodiscard]] auto prove_equal(
    Context&,
    Expr const* lhs,
    Expr const* rhs,
    std::vector<Identity> const& rules,
    SaturateBudget budget = {}) -> ProofResult;

struct SimplifyResult final
{
    Expr const* expr = nullptr; // cheapest extraction (never null)
    SaturateReport report = {};

    // True when the rule set reached a fixed point, so `expr` is the best
    // this rule set can do.  False means the budget cut the search short and
    // `expr` is merely the best form found so far.
    [[nodiscard]] auto complete() const -> bool
    {
        return report.outcome == SaturateOutcome::Saturated;
    }
};

// Saturate `e` under `rules` and extract the best form under `cost` — which
// is the caller's *intent*, not a fixed notion of simplicity (vibe 000097):
// the default minimizes Levi-Civita symbols then size, `CostModel::
// fewest_crosses()` will happily take a larger form to be rid of a `×`.
// The result is returned in the user-facing canonical implicit form.  On a
// budget trip the best form found *is* returned — with `complete() == false`,
// so a caller can report the shortfall rather than pretend to a fixed point.
[[nodiscard]] auto simplify(
    Context&,
    Expr const* e,
    std::vector<Identity> const& rules,
    SaturateBudget budget = {},
    CostModel const& cost = {}) -> SimplifyResult;

} // namespace tender::engine
