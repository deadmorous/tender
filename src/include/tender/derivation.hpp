#pragma once

#include <tender/expr.hpp>

#include <cstdint>
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

// Distribute TensorProduct over Sum and Difference (expand brackets):
//   (A + B) \, C  →  A \, C + B \, C
//   A \, (B + C)  →  A \, B + A \, C
// Applied bottom-up in one pass, so nested distributions are fully resolved.
auto expand_products(Context& ctx, Expr const* e) -> Expr const*;

// Replace every rank-3 LeviCivita tensor ε_{ijk} with its 6-term cofactor
// (Laplace) expansion in Kronecker deltas over the 3-value IndexSpace.
// Only 3D (|space| == 3) is supported; other ranks are left unchanged.
auto expand_eps(Context& ctx, Expr const* e) -> Expr const*;

// Inverse of unroll_sums: detect a Sum tree whose N addends differ in exactly
// one ConcreteIndex slot value cycling through all N values of some IndexSpace,
// and replace the N addends with a single ExplicitSum over a fresh index.
// Addends that do not form such a pattern are left unchanged.
auto fold_sums(Context& ctx, Expr const* e) -> Expr const*;

// Contract ExplicitSum{m, δ^m_a · δ^m_b} → δ_{ab}.
// Recognises a Sum body that is a TensorProduct of exactly two delta objects
// both carrying the summation CountableIndex m in the same-level slot, and
// replaces the whole ExplicitSum with a single delta over the remaining
// indices.
auto contract_delta(Context& ctx, Expr const* e) -> Expr const*;

// Contract a pair of Levi-Civita symbols sharing p summed indices, directly to
// the generalized Kronecker delta (no concrete WCS unrolling):
//
//   Σ_{i1…ip} ( ε^{… i1…ip} ⊗ ε_{… i1…ip} )
//     →  s · p! · det[ δ^{free_upper_r}_{free_lower_c} ]
//
// where the Σ are nested concrete-bound ExplicitSum nodes (one per shared
// dummy), free_upper / free_lower are the non-contracted slots of the first /
// second ε, the determinant is the (3−p)×(3−p) Kronecker determinant, and
// s = ±1 is the sign of re-ordering each ε to put its contracted slots first.
//
// Examples (3D):
//   Σ_i  ε^{ijk} ε_{iml}  → δ^j_m δ^k_l − δ^j_l δ^k_m   (p=1)
//   Σ_ij ε^{ijk} ε_{ijl}  → 2 δ^k_l                      (p=2)
//
// Only 3D (|space| == 3) and a body that is exactly TensorProduct(ε, ε) are
// supported; anything else is left unchanged.
auto contract_eps_pair(Context& ctx, Expr const* e) -> Expr const*;

// Like unroll_sums but restricted to ExplicitSum nodes whose summation index
// appears in `indices`.  Sums over other indices are left untouched.
auto unroll_sums_for(
    Context& ctx,
    Expr const* e,
    std::vector<CountableIndex> const& indices) -> Expr const*;

// Return true if `e` contains at least one ExplicitSum whose summation index
// appears in `indices`.
auto has_explicit_sum_for(
    Expr const* e, std::vector<CountableIndex> const& indices) -> bool;

// Collect addends from a Sum tree and group those with the same core
// expression (extracted coefficient-core pairs).  Groups with total
// coefficient n > 1 are folded into n·core; n == -1 becomes -core;
// n == 0 is dropped.  No-op if no merging occurs.
auto fold_equal_addends(Context& ctx, Expr const* e) -> Expr const*;

} // namespace steps
} // namespace tender
