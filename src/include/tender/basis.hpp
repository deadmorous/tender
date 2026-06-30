#pragma once

#include <tender/expr.hpp>
#include <tender/index.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>

#include <optional>
#include <string>
#include <vector>

namespace tender
{

// Orientation of an orthonormal frame, fixing the sign of its cell volume √g
// (and hence the cross product e_i × e_j = √g ε_{ijk} e^k).
enum class Handedness
{
    Right, // √g = +1
    Left,  // √g = -1
};

// A vector basis: the index-emitting bridge between the invariant and
// coordinate layers (vibes 000036, 000049).  It is built from a tuple of
// rank-1 vectors and owns the realm together with the index space those
// vectors' indices range over (cardinality == number of vectors).
//
// A basis with fewer vectors than the ambient dimension is a subspace basis;
// tender does not model the space a vector lives in, so no dimension check is
// made beyond cardinality == #vectors (vibe 000049 §1).
//
// Orthonormal bases have the cobasis e^i coincide with the basis e_i.  An
// oblique basis (covariant/contravariant distinction) derives its contravariant
// cobasis from the covariant vectors via the reciprocal (cross-product) formula
// and distinguishes index level; dotting two same-variance basis vectors yields
// the metric g (vibe 000049).

// How a basis prints (vibe 000067).  All optional; empty/none = the generic
// numeric/`e`-indexed defaults.
//   value_names     — coordinate letter per space value: a concrete index reads
//                     "r" instead of "1" (cylindrical), on coordinates AND on
//                     `e`-indexed vectors.
//   vector_symbols  — a standalone symbol per direction: a concrete basis
//   vector
//                     prints as that symbol (WCS i, j, k) instead of e_x.
//   label           — a frame marker appended to every index of this basis, so
//                     two frames in one term are distinguishable (the primed
//                     index in e_{i'} ⊗ e_i).
struct BasisNaming final
{
    std::vector<IndexName> value_names = {};
    std::vector<TensorName> vector_symbols = {};
    // Free-form LaTeX suffix (e.g. "'" or "(2)"), so it is a std::string rather
    // than a validated IndexName.
    std::optional<std::string> label = {};
};

class Basis final
{
public:
    auto realm() const noexcept -> Realm
    {
        return realm_;
    }
    auto space() const noexcept -> IndexSpace const*
    {
        return space_;
    }
    // This basis's slot tag (vibe 000067): the IndexSlot::basis_id stamped on
    // every index this basis emits (vectors and coordinates).  0 means the
    // basis was constructed but not registered with a Context.
    auto basis_id() const noexcept -> int
    {
        return id_;
    }
    // The coordinate letter for a concrete index `value` (vibe 000067), e.g.
    // value 1 → "r" in a cylindrical frame.  nullopt when this basis has no
    // value names (then a concrete index prints numerically) or `value` is not
    // one of the space's values.  Set by coordinate systems.
    auto value_name(int value) const -> std::optional<IndexName>;
    // The standalone symbol for the basis vector in direction `value` (vibe
    // 000067), e.g. value 1 → "i" in WCS.  nullopt when this basis has no
    // vector symbols (then a concrete basis vector prints as e + value
    // name/number).
    auto vector_symbol_for(int value) const -> std::optional<TensorName>;
    // The frame marker appended to this basis's indices (vibe 000067), or
    // nullopt when the basis is undecorated.
    auto label() const noexcept -> std::optional<std::string> const&
    {
        return naming_.label;
    }
    // Number of basis vectors.
    auto dim() const noexcept -> int
    {
        return static_cast<int>(vectors_.size());
    }
    auto is_orthonormal() const noexcept -> bool
    {
        return realm_ == Realm::Orthonormal;
    }

    // The symbol used for the generic (symbolic, indexed) basis vector.
    auto vector_symbol() const noexcept -> TensorName
    {
        return symbol_;
    }

    // The signed cell volume √g as an invariant scalar.  For an orthonormal
    // basis it is the scalar +1 (right-handed) or -1 (left-handed); for an
    // oblique basis it is the scalar triple product e_0·(e_1×e_2).  It is the
    // weight of the Levi-Civita tensor relative to the symbol (ε_ijk =
    // √g·[ijk]).
    auto volume() const noexcept -> Expr const*
    {
        return volume_;
    }

    // ---- concrete members (for unrolling) ------------------------------

    // i-th covariant basis vector e_i (0-based), rank 1.
    // Throws std::out_of_range if i is not a valid index.
    auto basis(int i) const -> Expr const*;
    // i-th contravariant cobasis vector e^i (0-based), rank 1.
    // Throws std::out_of_range if i is not a valid index.
    auto cobasis(int i) const -> Expr const*;

    // ---- symbolic emission (for expansion) -----------------------------
    //
    // The symbolic basis vectors carry the generic symbol and a CountableIndex
    // over this basis's space/realm, so they Einstein-sum against coordinates.
    // They unroll to the concrete members above (a later step).

    // Symbolic covariant basis vector e_i (rank 1, lower index).
    auto covariant_vector(Context& ctx, CountableIndex index) const
        -> Expr const*;
    // Symbolic contravariant basis vector e^i (rank 1).  For an orthonormal
    // basis the index is spelled lower (upper/lower coincide; the
    // Orthonormal-lower convention of vibe 000047); oblique spells it upper.
    auto contravariant_vector(Context& ctx, CountableIndex index) const
        -> Expr const*;

private:
    Basis(
        Realm realm,
        IndexSpace const* space,
        TensorName symbol,
        std::vector<Expr const*> vectors,
        std::vector<Expr const*> covectors,
        Expr const* volume,
        BasisNaming naming = {});

    friend auto make_orthonormal_basis(
        Context&,
        IndexSpace const*,
        std::vector<Expr const*>,
        TensorName,
        Handedness,
        BasisNaming) -> Basis;
    friend auto make_oblique_basis(
        Context&,
        IndexSpace const*,
        std::vector<Expr const*>,
        TensorName,
        BasisNaming) -> Basis;

    // Internal (vibe 000067): copy `b` into ctx, register it for an id, stamp
    // that id, and return the id-carrying value.  Friend so it can set id_.
    friend auto intern_basis(Context& ctx, Basis b) -> Basis;

    Realm realm_;
    IndexSpace const* space_;
    TensorName symbol_;
    std::vector<Expr const*> vectors_;   // covariant   e_i
    std::vector<Expr const*> covectors_; // contravariant e^i
    Expr const* volume_; // signed √g (scalar ±1, or triple prod)
    int id_ = 0;         // slot tag; set when registered (vibe 000067)
    BasisNaming naming_; // how concrete indices/vectors print (vibe 000067)
};

// Build an orthonormal basis from rank-1 vectors.  The realm is Orthonormal and
// the cobasis coincides with the basis.  The handedness fixes the signed volume
// √g = ±1 (right-handed by default), which orients the cross product.
//
// vector_symbol is the name of the generic symbolic basis vector (default "e").
//
// Preconditions (else std::invalid_argument): space is non-null; at least one
// vector; vectors.size() == space->values().size(); every vector is non-null
// and rank 1 (where its rank is known).
// naming (optional) controls how this basis prints its concrete indices and
// vectors (vibe 000067); the default leaves them numeric / e-indexed.
[[nodiscard]] auto make_orthonormal_basis(
    Context& ctx,
    IndexSpace const* space,
    std::vector<Expr const*> vectors,
    TensorName vector_symbol = make_tensor_name("e"),
    Handedness handedness = Handedness::Right,
    BasisNaming naming = {}) -> Basis;

// Build an oblique basis from its covariant vectors.  The realm is Oblique and
// the contravariant cobasis e^i is derived via the reciprocal (cross-product)
// formula e^0 = (e_1×e_2)/V, e^1 = (e_2×e_0)/V, e^2 = (e_0×e_1)/V with
// V = e_0·(e_1×e_2).  Only 3D bases are supported (the formula is 3D);
// other cardinalities throw std::invalid_argument, as do the same preconditions
// as make_orthonormal_basis.
[[nodiscard]] auto make_oblique_basis(
    Context& ctx,
    IndexSpace const* space,
    std::vector<Expr const*> vectors,
    TensorName vector_symbol = make_tensor_name("e"),
    BasisNaming naming = {}) -> Basis;

// ---- basis-parameterized steps -----------------------------------------

// Which homogeneous polyad an expansion uses (vibe 000049 §3).
//   Covariant     — covariant basis vectors e_i, coordinates A^{i…}
//   Contravariant — contravariant cobasis vectors e^i, coordinates A_{i…}
// For an orthonormal basis the two coincide.
enum class Variance
{
    Covariant,
    Contravariant
};

// Expand every generic invariant tensor in e into its coordinate form in the
// given basis: a slot-less rank-r TensorObject A becomes a coordinate ⊗ polyad,
// with each slot's variance chosen independently — covariant slots use e_i and
// an A^{…i…} coordinate index, contravariant slots use e^i and an A_{…i…}
// index. The r indices are left implicitly Einstein-summed (canonicalize
// materializes the sums).  Mixed variance gives e.g. A^i{}_j e_i e^j (vibe
// 000049 §3).
//
// `variances` holds one Variance per slot; a single entry broadcasts to every
// slot of every expanded tensor, otherwise the count must equal the tensor's
// rank exactly (else std::invalid_argument — no silent misapplication).  For an
// orthonormal basis the two variances coincide, so the choice has no effect.
//
// Walks the whole tree (so operands of products expand in place).  Well-known
// tensors (Identity / Delta / Levi-Civita), whose coordinates are not generic,
// and already-indexed objects are left unchanged.
[[nodiscard]] auto expand_in_basis(
    Context& ctx,
    Expr const* e,
    Basis const& basis,
    std::vector<Variance> variances) -> Expr const*;

// Convenience overload: one variance applied to every slot.
[[nodiscard]] auto expand_in_basis(
    Context& ctx,
    Expr const* e,
    Basis const& basis,
    Variance variance) -> Expr const*;

// Replace each dot product of two basis vectors of `basis` with the
// corresponding Kronecker delta: (s e_i) · (t e_j) → s ⊗ t ⊗ δ_{ij}, pulling
// any component-valued (rank-0 coordinate) factors out of the contraction
// (vibe 000049 §3).  This is the bridge from the invariant dot to the index
// algebra — the resulting δ then feeds the existing contraction machinery.
//
// A dot whose sides are not (optionally coordinate-scaled) basis vectors of
// `basis` is left unchanged.  Walks the whole tree.
[[nodiscard]] auto simplify_basis_dot(
    Context& ctx, Expr const* e, Basis const& basis) -> Expr const*;

// Replace the cross product of two covariant basis vectors with the
// Levi-Civita expansion: (s e_i) × (t e_j) → s ⊗ t ⊗ √g ⊗ ε_{ijk} ⊗ e^k, the
// k index Einstein-summed (vibe 000049).  ε is the rank-3 Levi-Civita symbol
// and √g the basis volume (omitted for an orthonormal basis, where it is 1 and
// e^k = e_k), so orthonormal gives e_i × e_j = ε_{ijk} e_k.
//
// Only the covariant (both-lower) case in a 3D basis is handled; other inputs
// (contravariant or mixed, non-3D) are left unchanged.  Walks the whole tree.
[[nodiscard]] auto simplify_basis_cross(
    Context& ctx, Expr const* e, Basis const& basis) -> Expr const*;

// Fold a coordinate expansion back to its invariant — the inverse of
// expand_in_basis (vibe 000049 §3).  Recognizes a (possibly nested) Einstein
// sum whose body is one coordinate tensor times a polyad of `basis`'s vectors,
// with the summed indices pairing each coordinate slot to one basis vector,
// and replaces it with the slot-less invariant of matching rank and name:
//
//   Σ_i a_i e_i           → a            (rank 1)
//   Σ_i Σ_j A_{ij} e_i e_j → A           (rank 2)
//
// Operates on the canonical (materialized-sum) form.  Anything that is not a
// clean expansion in `basis` is left unchanged (failure is a no-op).  Walks
// the whole tree.
[[nodiscard]] auto reassemble(Context& ctx, Expr const* e, Basis const& basis)
    -> Expr const*;

// Fold the resolution of identity Σ_i e_i ⊗ e^i = I where it appears partially
// contracted — the completeness reassembly that `reassemble` (which needs a
// literal coordinate tensor a_i) cannot do.  In a product term under Σ_i, a
// bare basis vector e_i together with a scalar contraction (X·e_i) over the
// same summed index i (occurring nowhere else) collapses to X in the leg's
// position:
//
//   Σ_i (X·e_i) e_i           → X            (= X·I = X)
//   Σ_i (a·e_i) (b ⊗ e_i)     → b ⊗ a
//
// X must be rank 1 (the dot is scalar).  The sum is distributed over Sum/Negate
// addends by linearity so the fold reaches each term; addends without the
// pattern (e.g. the pure Σ_i e_i⊗e_i = I) are left for `reassemble`.  A no-op
// when no such pattern is present.  Walks the whole tree.
[[nodiscard]] auto reassemble_completeness(
    Context& ctx, Expr const* e, Basis const& basis) -> Expr const*;

// Fold the *concrete, fully-expanded* resolution of identity back to I (vibe
// 000070 P3).  The differential operators emit their results in the chart's
// constant reference frame as a plain sum of concrete dyads — `u_0⊗u_0 +
// u_1⊗u_1 + …`, never the symbolic bound Σ_i e_i⊗e_i that `fold_completeness`
// recognises.  This pass finds, within any Sum, addends `c·u_k⊗u_k` present for
// *every* concrete vector u_k of the orthonormal `basis` sharing one sign and
// coefficient c, and replaces that complete group with a single `c·I`.
//
// `basis` must be the frame the dyads are written in (a chart's reference
// basis); a non-orthonormal basis (where completeness does not hold) is a
// no-op. Operates on the canonical form; anything else is left unchanged. Walks
// the whole tree.
[[nodiscard]] auto fold_resolution_of_identity(
    Context& ctx, Expr const* e, Basis const& basis) -> Expr const*;

// Expand every identity tensor I in `e` into the concrete resolution
// Σ_k u_k⊗u_k over the orthonormal `basis` — the forward direction used by the
// contraction engine so a cross/dot against I reduces leg-by-leg (vibe 000070
// P6).  Throws std::invalid_argument if the basis is not orthonormal (the
// resolution holds only there).  Walks the whole tree.
[[nodiscard]] auto expand_identity(
    Context& ctx, Expr const* e, Basis const& basis) -> Expr const*;

} // namespace tender
