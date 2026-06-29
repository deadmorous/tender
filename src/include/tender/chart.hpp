#pragma once

#include <tender/basis.hpp>
#include <tender/expr.hpp>

#include <vector>

namespace tender
{

// A coordinate chart (vibe 000069 M4): a coordinate mapping from chart
// coordinates q^i to a reference (orthonormal Cartesian) frame, from which the
// whole orthogonal-curvilinear geometry is *derived* rather than hand-supplied
// (vibe 000069 §3, §5).
//
//   reference — the orthonormal Cartesian Basis the embedding targets (its
//               concrete vectors are the constant i, j, k the geometry is
//               expressed in).
//   coords    — the chart's coordinate variables q^i (make_coordinate atoms),
//               one per chart dimension, in order.  Their `nonneg` domain bit
//               (vibe 000069 M3) licenses √(r²) → r in the scale factors.
//   embedding — the Cartesian components x^a = f^a(q) of the position vector,
//               one scalar Expr per reference direction, in the reference's
//               vector order.
//
// A well-formed chart has embedding.size() == reference.dim() (one Cartesian
// component per reference direction) and a square Jacobian
// (coords.size() == reference.dim()), so it produces a full frame.
struct CoordinateChart final
{
    Basis reference;
    std::vector<Expr const*> coords;
    std::vector<Expr const*> embedding;
};

// The position (radius) vector R = Σ_a f^a(q) ⊗ u_a, the embedding written in
// the reference frame's constant vectors u_a (vibe 000069 step 2).
[[nodiscard]] auto radius_vector(Context& ctx, CoordinateChart const& chart)
    -> Expr const*;

// The i-th holonomic (tangent) basis vector g_i = ∂R/∂q^i (step 3), as a vector
// in the reference frame.  Throws std::out_of_range if i is not a coordinate.
[[nodiscard]] auto tangent_vector(
    Context& ctx, CoordinateChart const& chart, int i) -> Expr const*;

// The metric component g_ij = g_i · g_j (step 4), simplified to a scalar field
// (e.g. polar g_φφ → r², g_rφ → 0).  The reference frame being orthonormal, the
// dot collapses through Kronecker deltas.
[[nodiscard]] auto metric_component(
    Context& ctx, CoordinateChart const& chart, int i, int j) -> Expr const*;

// The i-th scale factor h_i = √(g_ii) (step 4), taken as the positive root
// under the chart's domain convention (decision 2/3: scale factors are
// positive), so √(r²) → r and √(r² sin²θ) → r sin θ.
[[nodiscard]] auto scale_factor(
    Context& ctx, CoordinateChart const& chart, int i) -> Expr const*;

// The derived physical (orthonormal) frame e_i = g_i / h_i (step 5), bridged
// into a tender Basis carrying the chart's coordinate letters as its value
// names (vibe 000067).  For polar this yields e_r = cos φ i + sin φ j,
// e_φ = −sin φ i + cos φ j.
[[nodiscard]] auto physical_basis(Context& ctx, CoordinateChart const& chart)
    -> Basis;

// The derivative ∂_{q^j} e_i of the i-th physical basis vector (step 6), as a
// vector in the constant reference frame: since e_i is written in the constant
// reference vectors, ∂ acts only on its scalar coefficients.  For polar
// ∂_φ e_r = −sin φ i + cos φ j (= e_φ), ∂_φ e_φ = −cos φ i − sin φ j (= −e_r),
// and the ∂_r derivatives vanish.  Throws std::out_of_range on a bad index.
[[nodiscard]] auto basis_derivative(
    Context& ctx, CoordinateChart const& chart, int i, int j) -> Expr const*;

// The physical-basis connection (rotation) coefficients γ^k_{ij} of step 6,
// re-expressing ∂_{q^j} e_i = Σ_k γ^k_{ij} e_k in the local frame: the returned
// vector holds, for k = 0…dim−1, the scalar projection (∂_{q^j} e_i)·e_k (the
// physical frame being orthonormal).  For polar ∂_φ e_r = e_φ gives {0, 1} and
// ∂_φ e_φ = −e_r gives {−1, 0}.  Throws std::out_of_range on a bad index.
[[nodiscard]] auto connection_coefficients(
    Context& ctx,
    CoordinateChart const& chart,
    int i,
    int j) -> std::vector<Expr const*>;

} // namespace tender
