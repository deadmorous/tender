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

using nf::SaturateBudget;
using nf::SaturateOutcome;
using nf::SaturateReport;

// The outcome of a proof attempt.  Note what is *absent*: there is no
// "disproved".  Saturation is a semi-decision procedure — it can exhibit a
// proof, never a refutation — so a caller must treat `Exhausted` as "these
// rules did not suffice", not as "the two sides differ".
enum class ProofStatus : std::uint8_t
{
    Proved,    // the two sides landed in one e-class
    Exhausted, // rules ran to a fixed point without joining them
    Budget,    // stopped on the pass/node budget: nothing is concluded
};

struct ProofResult final
{
    ProofStatus status = ProofStatus::Exhausted;
    SaturateReport report = {};

    // Deliberately explicit rather than an `operator bool`: reading a proof
    // attempt as a plain bool is exactly the mistake that turns "we ran out of
    // budget" into "they are different".
    [[nodiscard]] auto proved() const -> bool
    {
        return status == ProofStatus::Proved;
    }
};

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

// Saturate `e` under `rules` and extract the cheapest form (the ε-weighted
// cost of vibe 000046: fewest Levi-Civita symbols first, then fewest nodes).
// The result is returned in the user-facing canonical implicit form.  On a
// budget trip the best form found *is* returned — with `complete() == false`,
// so a caller can report the shortfall rather than pretend to a fixed point.
[[nodiscard]] auto simplify(
    Context&,
    Expr const* e,
    std::vector<Identity> const& rules,
    SaturateBudget budget = {}) -> SimplifyResult;

} // namespace tender::engine
