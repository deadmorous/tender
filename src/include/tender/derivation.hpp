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

// Replace a Levi-Civita symbol ε whose every slot carries a ConcreteIndex with
// its permutation-symbol value: 0 on any repeated index, else the sign of the
// permutation (+1 even, −1 odd).  A symbol with any symbolic index is left
// unchanged.
auto eval_eps_concrete(Context& ctx, Expr const* e) -> Expr const*;

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

// Contract the identity tensor against a dot product: I · x → x and x · I → x
// (the identity tensor acts as the identity under ·).  Walks the whole tree;
// dots not involving the identity are left unchanged.
auto contract_identity(Context& ctx, Expr const* e) -> Expr const*;

// Distribute a single contraction (· or ×) over the adjacent leg of a tensor
// product (polyad):
//   op(L, A ⊗ B)  →  op(L, A) ⊗ B      (contraction acts on the near leg A)
//   op(A ⊗ B, R)  →  A ⊗ op(B, R)      (contraction acts on the near leg B)
// so e.g. a × (u ⊗ v) → (a × u) ⊗ v and (u ⊗ v) · a → u ⊗ (v · a).  One pass
// (right operand first); apply again for deeper nesting.  Double contractions
// (: , ··) span two factors and are left unchanged.
auto distribute_contraction(Context& ctx, Expr const* e) -> Expr const*;

// Expand a double contraction of dyads by definition:
//   (a⊗b) :  (c⊗d)  →  (a·c)(b·d)      [DDot, the "vertical" double dot]
//   (a⊗b) ·· (c⊗d)  →  (a·d)(b·c)      [DDotAlt, the alternate pairing]
// Scalar coordinate factors are pulled through, and the contraction is
// distributed over sums and pushed through summation binders (fresh-renamed to
// avoid capture), so e.g. (Σ_i e_i⊗e_i):(Σ_j e_j⊗e_j) → Σ_i Σ_j
// (e_i·e_j)(e_i·e_j). A double dot whose sides are not both 2-leg dyads is left
// unchanged.
auto expand_double_dot(Context& ctx, Expr const* e) -> Expr const*;

// Expand the rank-2 invariant operations by their definition on dyads, linear
// over sums and negation:
//   tr(a⊗b)        → a·b
//   vec(a⊗b)       → a×b
//   transpose(a⊗b) → b⊗a   (a symmetric well-known I/δ/g transposes to itself)
// Scalar factors are pulled through.  An operation whose operand is not a dyad
// (or a sum/negation of dyads) is left in place.
auto expand_dyad_ops(Context& ctx, Expr const* e) -> Expr const*;

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

// Rewrite an expression into algebraic normal form (vibe 000037): the local
// AC/α-canonical form for which structural_eq decides theory T0.
//
//   - numerics folded to a single exact Rational coefficient per term;
//   - subtraction/negation carried in the sign of the coefficient (no
//     Difference in the result; Negate only as the -1 term wrapper);
//   - sums flattened, like terms combined, terms sorted by a canonical key;
//   - commutative *component* products (factors that are scalars or
//     fully-indexed coordinates) sorted; invariant (slot-less rank >= 1)
//     factors keep their relative order;
//   - distribution is NOT performed (a*(b+c) stays a product atom).
//
// Two canonical forms compare equal under structural_eq iff the expressions are
// equal in T0.  This is sound but intentionally incomplete for full ring
// equality (distribution, contraction, tensor symmetries are rewrite rules, not
// normalization).
auto canonicalize(Context& ctx, Expr const* e) -> Expr const*;

// Inverse of the implicit-sum convention: strip a null-bound ExplicitSum whose
// index is already an implicit Einstein contraction in its body (so the
// explicit wrapper is redundant), returning the expression in the implicit form
// the user works in.  A symbolic-bound sum, or one over a non-contracted index,
// is kept.
auto implicitize(Context& ctx, Expr const* e) -> Expr const*;

} // namespace steps

// ---- Equality -----------------------------------------------------------

// Deep structural equality of two expression trees: same node shape, same
// names/values, CountableIndex ids matched exactly (free indices are not
// alpha-renamed; ExplicitSum binders compared by id).  This is the equality
// that `canonicalize` turns into a decision procedure for theory T0.
[[nodiscard]] auto structural_eq(Expr const* a, Expr const* b) -> bool;

// Algebraic equality in theory T0: structural_eq of the two canonical forms,
// i.e. structural_eq(canonicalize(a), canonicalize(b)).  Sound and complete for
// T0 (see vibe 000037); intentionally not full ring equality.
[[nodiscard]] auto algebraic_eq(Context&, Expr const* a, Expr const* b) -> bool;

// True iff e denotes a scalar/coordinate (component) value rather than an
// invariant: a scalar, a fully-indexed coordinate tensor, or a combination
// thereof (vibe 000036 coordinate/invariant line).  Component factors commute;
// invariant factors do not.  Used to decide which products are commutative.
[[nodiscard]] auto is_component_valued(Expr const* e) -> bool;

// The (invariant) rank of an expression, inferred from the ranks of its leaves
// and the rank arithmetic of each operator:
//   scalar = 0; Negate / ScalarDiv / ExplicitSum / NoSum keep their operand's
//   rank; Sum / Difference take their operands' shared rank; TensorProduct adds
//   ranks; Dot subtracts 2; DDot / DDotAlt subtract 4; Cross subtracts 1.
// Returns nullopt when a needed leaf rank is unknown (an undeclared tensor) or
// a contraction would drive the rank negative (ill-formed).  A leaf
// TensorObject contributes its declared rank.
[[nodiscard]] auto infer_rank(Expr const* e) -> std::optional<int>;

} // namespace tender
