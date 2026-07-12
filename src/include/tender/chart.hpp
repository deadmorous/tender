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

// Validate a chart's shape (vibe 000072 Obs 3): non-empty; embedding size
// equals reference.dim(); every coord is a coordinate() atom; all coords share
// one chart_id and occupy slots 0..n-1 in list order.  Throws
// std::invalid_argument with a specific message otherwise — catching, e.g., a
// stale slot= or a chart_id typo at construction rather than as a confusing
// later result.
void validate_chart(CoordinateChart const& chart);

// The position (radius) vector R = Σ_a f^a(q) ⊗ u_a, the embedding written in
// the reference frame's constant vectors u_a (vibe 000069 step 2).  This is the
// geometry primitive the tangent basis / metric / scale factors derive from, so
// it stays in the reference (WCS) frame; for the intrinsic form see `position`.
[[nodiscard]] auto radius_vector(Context& ctx, CoordinateChart const& chart)
    -> Expr const*;

// The position vector in the chart's own physical frame, e.g. cylindrical
// r e_r + z e_z (vibe 000072 Obs 6): radius_vector projected onto the frame's
// e_i.  Stays intrinsic so grad(position) folds to I without a mixed frame.
[[nodiscard]] auto position(Context& ctx, CoordinateChart const& chart)
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

// Build the chart's physical (orthonormal) frame as before, and additionally
// register its connection table in the Context (vibe 000071): the derivative
// ∂_{q^j} e_i of each frame vector, pre-expressed on the frame's own symbolic
// e_k atoms via the connection γ^k_{ij}.  Thereafter the intrinsic
// differentiator resolves ∂ of a frame-vector atom `basis.direction(i)` through
// that table — no reference-frame expansion.  Returns the frame Basis (its
// `direction(i)` gives the symbolic e_i).  Call once per chart.
[[nodiscard]] auto physical_frame(Context& ctx, CoordinateChart const& chart)
    -> Basis;

// ---- differential operators (vibe 000069 M6) ---------------------------
//
// ∇ is the invariant operator ∇ = Σ_i e_i (1/h_i) ∂_{q^i}, and each operator is
// just ∇ applied by the formal rule
//
//     ∇ ⊙ T = Σ_i (1/h_i) e_i ⊙ ∂_{q^i} T          (⊙ = ⊗, ·, or ×)
//
// to an invariant tensor field T (an `Expr`, written in the constant reference
// frame).  ∂_{q^i} acts on the whole expression by Leibniz, so a field given in
// the moving frame e_j(q) differentiates those e_j too and the connection (∂e)
// terms fall out on their own — no components, no hand-tabulated Christoffels.
// Every operator takes and returns an invariant `Expr`; nothing assumes a
// particular basis for T.

// The invariant ∇ operator itself (vibe 000077), as a first-class expression:
// ∇ = Σ_i (1/h_i) e_i ∂_{q^i}, a sum of terms each a scalar · frame vector ·
// unapplied ∂ operator.  This is the operator grad/div/rot are built from —
// apply it with a product (⊗, ·, ×) — and the raw material for user-defined
// operators (v·∇, the material derivative, …).  Inspectable and renderable
// without applying it to anything.
[[nodiscard]] auto del(Context& ctx, CoordinateChart const& chart)
    -> Expr const*;

// Expand every chart-free ∇ (Nabla) operator in `e` into the *free-index* frame
// form e_i ∂_i — a single implicitly-summed term with a fresh direction index
// per ∇ (an indexed frame vector e_i times a free-index ∂_i tied to it) — then
// apply the operators (vibe 000078).  Unlike `del`'s concrete n-term unrolling,
// this keeps a nested ∇×(∇×ε)ᵀ abstract in ε: it lowers to
// `e_i × (e_j × ∂_i∂_j ε)ᵀ` (i,j summed), the free-index interior the a×B×c
// reduction then folds.  Targets a constant unit-scale (Cartesian/WCS) frame;
// throws for curvilinear charts (use grad/div/rot there).
[[nodiscard]] auto expand_nabla(
    Context& ctx, CoordinateChart const& chart, Expr const* e) -> Expr const*;

// Lower a free-index ∇ expansion (the output of expand_nabla) to concrete
// components (vibe 000078): unroll each summed frame direction index over the
// chart's directions, fixing the frame vector e_i → e_d and the free ∂_i → the
// concrete ∂_{q^d} for each direction d.  A no-op once nothing free remains.
// The bridge that lets the abstract free-index interior be checked, component
// by component, against the brute-force chart operators.
[[nodiscard]] auto componentize_nabla(
    Context& ctx, CoordinateChart const& chart, Expr const* e) -> Expr const*;

// Reassemble a reduced free-index expression back into chart-free ∇ operators
// (vibe 000078 increment 4 — the Phase-2 heart).  The Phase-1 cross reduction
// of a ∇-expression leaves a sum of terms in which every abstract direction
// index pairs a frame vector e_ℓ with a ∂_ℓ mark on the field; this reads each
// pair's role and folds it into a `Nabla`: a `⊗`-adjacent e_ℓ is a gradient leg
// (∇⊗…), a contracted e_ℓ·T a divergence (∇·…), a pair e_ℓ·e_m of two direction
// vectors the Laplacian (Δ = ∇·∇), and `tr` of the marked field its scalar
// invariant.  The inverse of expand_nabla+reduction: e.g. the strain interior's
// Phase-1 sum folds to −∇∇θ + Δθ·I − (∇∇··ε)I − Δε + ∇∇·ε + (∇∇·ε)ᵀ.
[[nodiscard]] auto reassemble_nabla(
    Context& ctx, CoordinateChart const& chart, Expr const* e) -> Expr const*;

// grad T = Σ_i (1/h_i) e_i ⊗ ∂_{q^i} T, raising the rank by one.  For a scalar
// f this is the familiar ∇ = e_r ∂_r + (1/r) e_θ ∂_θ + e_z ∂_z; for the
// position vector R it is the identity tensor ∇R = Σ_i e_i ⊗ e_i = I.
//
// fold_identity (default true): collapse the result's concrete resolution of
// identity Σ_k u_k⊗u_k back to I (vibe 000070 P4) so ∇R = I comes out directly;
// pass false to keep the raw expanded sum of dyads.
[[nodiscard]] auto gradient(
    Context& ctx,
    CoordinateChart const& chart,
    Expr const* f,
    bool fold_identity = true) -> Expr const*;

// div v = ∇·v = Σ_i (1/h_i) e_i · ∂_{q^i} v, lowering the rank by one (a vector
// field → a scalar).  fold_identity as in gradient.
[[nodiscard]] auto divergence(
    Context& ctx,
    CoordinateChart const& chart,
    Expr const* v,
    bool fold_identity = true) -> Expr const*;

// Laplacian Δf = div(grad f) as a scalar field.  fold_identity as in gradient.
[[nodiscard]] auto laplacian(
    Context& ctx,
    CoordinateChart const& chart,
    Expr const* f,
    bool fold_identity = true) -> Expr const*;

// rot v = ∇×v = Σ_i (1/h_i) e_i × ∂_{q^i} v.  3D only (the cross product), and
// the reference frame is taken right-handed (standard for the well-known
// charts); throws std::invalid_argument otherwise.  fold_identity as in
// gradient (so ∇×(R×I) = −2I folds).
[[nodiscard]] auto rot(
    Context& ctx,
    CoordinateChart const& chart,
    Expr const* v,
    bool fold_identity = true) -> Expr const*;

// Evaluate an invariant core-∇ expression in this chart (vibe 000084): walk the
// Expr tree and lower every ∇-combination to the chart operators, inner-first —
// `Dot(∇, X) → div`, `TensorProduct(∇, X) → grad`, `Cross(∇, X) → rot`, so
// `∇·(∇⊗X) → div(grad X) = Δ`; `Sum/Difference/Negate/ScalarDiv/scalar⊗/
// Transpose/Trace/vec/DDot` pass through and recurse; a ∇-free sub-expression
// is an operand, returned untouched.  The result is an invariant in the chart's
// physical frame (like `divergence`/`gradient`); `components` reads off the
// physical components.  This bridges a coordinate-free `t.nabla` expression to
// the curvilinear-correct operators without hand-rewriting via grad/div/rot
// (relaxes the vibe-000081 ∇-first rule).  A bare ∇ (no operand) throws.
[[nodiscard]] auto evaluate(
    Context& ctx, CoordinateChart const& chart, Expr const* e) -> Expr const*;

// The invariant dot u·v and cross u×v reduced in the chart's orthonormal
// reference frame (vibe 000070 P8): distribute, turn frame-vector contractions
// into δ/ε, and simplify.  These expose the reductions the operators use
// internally, so user-built operators — e.g. the directional derivative
// (v·∇)T = v · ∇T — can contract their results.  frame_cross is 3D only.
[[nodiscard]] auto frame_dot(
    Context& ctx,
    CoordinateChart const& chart,
    Expr const* u,
    Expr const* v) -> Expr const*;
[[nodiscard]] auto frame_cross(
    Context& ctx,
    CoordinateChart const& chart,
    Expr const* u,
    Expr const* v) -> Expr const*;

// ---- basis-to-basis expansion (vibe 000071 P4) -------------------------
//
// The intrinsic operators keep results on the chart's own frame; these bring a
// result into another frame on demand — WCS is the common case, expressing e_r
// as cos θ i + sin θ j.  Basis-to-basis expansion is the general primitive of
// which "give me WCS components" is one target.

// Re-express `v` in the reference (WCS) frame: every registered physical-frame
// vector e_i in `v` is replaced by its concrete Cartesian expansion (e.g.
// e_r → cos θ i + sin θ j), then distributed and simplified.  Chart-independent
// (uses the connection registry), so a vector mixing several frames is fully
// expanded; non-frame atoms are left as-is.
[[nodiscard]] auto to_reference(Context& ctx, Expr const* v) -> Expr const*;

// Re-express `v` in `target`'s physical frame, on its symbolic e_k atoms: bring
// `v` to the shared reference frame, then project onto target's orthonormal
// frame (w = Σ_k (w·e_k) e_k).  This is the general change of basis —
// expressing one chart's frame vectors in another chart's frame — with
// `to_reference` the WCS special case.
[[nodiscard]] auto express(
    Context& ctx, CoordinateChart const& target, Expr const* v) -> Expr const*;

// Expand every abstract tensor *field* in `v` into its components on the
// chart's physical frame, Σ T_ij e_i ⊗ e_j, and reduce (vibe 000073).  Unlike
// `express` (which re-expresses concrete vectors between frames), this
// materializes an abstract field that carries no basis vectors of its own.  A
// field derivative ∂_q T is expanded correctly by Leibniz — the base is
// expanded first, then differentiated through the connection ∂_q e_i — which is
// legal here because the chart owns the connection (a bare basis does not;
// expand_in_basis refuses it).  So expand(∂_q T) carries the moving-frame
// (connection) terms that a raw basis expansion would drop.
[[nodiscard]] auto expand(
    Context& ctx, CoordinateChart const& chart, Expr const* v) -> Expr const*;

// The scalar components of a vector `v` on the chart's physical frame, one per
// direction i: c_i = v · e_i, reduced (vibe 000073).  Surfaces an invariant
// vector (e.g. an operator result) as the numbered frame components, so the
// caller need not spell out the project-then-reduce pipeline by hand.  Throws
// std::invalid_argument on a rank-≥2 input (use component_matrix for rank 2).
[[nodiscard]] auto components(
    Context& ctx,
    CoordinateChart const& chart,
    Expr const* v) -> std::vector<Expr const*>;

// The physical component matrix of a rank-2 tensor `v` on the chart's frame:
// out[i][j] = e_i · v · e_j, reduced (vibe 000074).  Expands abstract fields
// first, so component_matrix(T) of a symmetric stress field is the matrix of
// minted components T_ij with the symmetry folded (out[1][0] is T_rθ, not
// T_θr).  The Python `chart.components` dispatches here on a rank-2 input.
[[nodiscard]] auto component_matrix(
    Context& ctx,
    CoordinateChart const& chart,
    Expr const* v) -> std::vector<std::vector<Expr const*>>;

} // namespace tender
