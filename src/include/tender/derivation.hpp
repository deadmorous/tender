#pragma once

#include <tender/expr.hpp>

#include <functional>
#include <vector>

namespace tender
{

// A derivation records the expression at each rewriting step.
// history()[0] is the initial expression; history()[k] is the result after
// applying the k-th step.  Calling step() is the only way to advance.
class Derivation final
{
public:
    using Step = std::function<Expr const*(Context&, Expr const*)>;

    explicit Derivation(Context& ctx, Expr const* initial) :
      ctx_(ctx), history_{initial}
    {
    }

    auto step(Step s) -> Derivation&
    {
        history_.push_back(s(ctx_, history_.back()));
        return *this;
    }

    auto history() const -> std::vector<Expr const*> const&
    {
        return history_;
    }
    auto current() const -> Expr const*
    {
        return history_.back();
    }
    auto initial() const -> Expr const*
    {
        return history_.front();
    }

private:
    Context& ctx_;
    std::vector<Expr const*> history_;
};

// ---- Built-in rewriting steps ------------------------------------------

namespace steps
{

// Replace each ExplicitSum whose index has a concrete IndexSpace with a
// binary Sum tree of the summand evaluated at each index value.
// Sums with a symbolic bound (bound != nullptr) are left unchanged.
auto unroll_sums(Context& ctx, Expr const* e) -> Expr const*;

// Replace δ(a, b) where both slots carry ConcreteIndex values with
// ScalarLiteral(1) when a == b and ScalarLiteral(0) otherwise.
auto eval_delta_concrete(Context& ctx, Expr const* e) -> Expr const*;

// Constant-fold arithmetic operations on ScalarLiteral nodes:
//   Sum / Difference / TensorProduct / ScalarDiv / Negate
// applied to scalar literals are reduced to a single ScalarLiteral.
auto fold_arithmetic(Context& ctx, Expr const* e) -> Expr const*;

} // namespace steps
} // namespace tender
