#pragma once

#include <functional>
#include <string>
#include <vector>

#include <tender/expr.hpp>

namespace tender
{

// ===========================================================================
// State — immutable wrapper around an Expr*
// ===========================================================================

class State
{
public:
    // label is the name of the step that produced this state; empty for the
    // initial state.
    explicit State(Expr* expr, std::string label = {});

    [[nodiscard]] auto expr() const noexcept -> Expr*
    {
        return expr_;
    }
    [[nodiscard]] auto label() const noexcept -> std::string const&
    {
        return label_;
    }
    [[nodiscard]] auto latex() const -> std::string;

private:
    Expr* expr_;
    std::string label_;
};

// ===========================================================================
// DerivationStep — named Expr* → Expr* transformation
// ===========================================================================

class DerivationStep
{
public:
    using Fn = std::function<Expr*(ResourceList&, Expr*)>;

    DerivationStep(std::string name, Fn fn);

    [[nodiscard]] auto name() const noexcept -> std::string const&
    {
        return name_;
    }

    // Apply the step; the returned State carries this step's name as label.
    [[nodiscard]] auto apply(ResourceList& rl, State const& s) const -> State;

private:
    std::string name_;
    Fn fn_;
};

// ===========================================================================
// Derivation — ordered sequence of steps
// ===========================================================================

class Derivation
{
public:
    explicit Derivation(std::vector<DerivationStep> steps);

    // Apply every step in order.  Returns initial + one State per step,
    // so history.size() == steps().size() + 1.
    [[nodiscard]] auto apply(ResourceList& rl, State const& initial) const
        -> std::vector<State>;

    // Concatenate two derivations.
    [[nodiscard]] auto operator+(Derivation const& rhs) const -> Derivation;

    [[nodiscard]] auto steps() const noexcept
        -> std::vector<DerivationStep> const&
    {
        return steps_;
    }

private:
    std::vector<DerivationStep> steps_;
};

// ===========================================================================
// show — render a history produced by Derivation::apply
// ===========================================================================

// Render every state, one per line:  "[label]  latex"
auto show(std::vector<State> const& history) -> std::string;

// Render only the final state's latex.
auto show_final(std::vector<State> const& history) -> std::string;

// ===========================================================================
// Built-in step factories
// ===========================================================================

// d/d(param): apply deriv(rl, param, expr).
auto diff_step(Parameter const* param) -> DerivationStep;

// Expand any PolynomialExpr nodes encountered in the tree to explicit
// sum-of-ring-powers form.
auto expand_poly_step() -> DerivationStep;

// Collapse identity-tensor contractions throughout the tree:
//   Contract(I, e)          → e
//   Contract(e, I)          → e
//   DoubleContract(I, A)    → Trace(A)
//   DoubleContract(A, I)    → Trace(A)
//   DoubleContractReversed  → same rules
auto simplify_identity_step() -> DerivationStep;

// Replace every occurrence of `what` (by pointer identity) with `with_what`.
auto substitute_step(Expr* what, Expr* with_what) -> DerivationStep;

// Replace every occurrence of `what` (by pointer identity) in `root` with
// `with_what`; returns the new root (or `root` unchanged if nothing matched).
auto replace_in_tree(ResourceList& rl, Expr* root, Expr* what, Expr* with_what)
    -> Expr*;

// A step that ignores its input expression and returns `result` unchanged.
// Used by search_apply to encode a sub-expression rewrite as a DerivationStep.
auto capture_step(std::string name, Expr* result) -> DerivationStep;

// Distribute binary operations over Sum, expanding products into sums of
// products:
//   Scale(c, Sum(…))              → Sum(Scale(c, t), …)
//   Contract(Sum(…), r)           → Sum(Contract(t, r), …)
//   Contract(l, Sum(…))           → Sum(Contract(l, t), …)
//   TensorProduct — same pattern
//   DoubleContract / DoubleContractReversed — same pattern
//   Product — same pattern
auto expand_step() -> DerivationStep;

// User-defined step with an explicit name and function.
auto named_step(std::string name, DerivationStep::Fn fn) -> DerivationStep;

} // namespace tender
