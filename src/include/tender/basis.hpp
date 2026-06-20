#pragma once

#include <tender/expr.hpp>
#include <tender/index.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>

#include <vector>

namespace tender
{

// A vector basis: the index-emitting bridge between the invariant and
// coordinate layers (vibes 000036, 000049).  It is built from a tuple of
// rank-1 vectors and owns the realm together with the index space those
// vectors' indices range over (cardinality == number of vectors).
//
// A basis with fewer vectors than the ambient dimension is a subspace basis;
// tender does not model the space a vector lives in, so no dimension check is
// made beyond cardinality == #vectors (vibe 000049 §1).
//
// This first slice supports the orthonormal flavor only: the cobasis e^i
// coincides with the basis e_i.  The covariant / contravariant (oblique)
// flavors, whose duals are derived through the metric, arrive with the metric.
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
        std::vector<Expr const*> covectors);

    friend auto make_orthonormal_basis(
        IndexSpace const*, std::vector<Expr const*>, TensorName) -> Basis;

    Realm realm_;
    IndexSpace const* space_;
    TensorName symbol_;
    std::vector<Expr const*> vectors_;   // covariant   e_i
    std::vector<Expr const*> covectors_; // contravariant e^i
};

// Build an orthonormal basis from rank-1 vectors.  The realm is Orthonormal and
// the cobasis coincides with the basis.
//
// vector_symbol is the name of the generic symbolic basis vector (default "e").
//
// Preconditions (else std::invalid_argument): space is non-null; at least one
// vector; vectors.size() == space->values().size(); every vector is non-null
// and rank 1 (where its rank is known).
[[nodiscard]] auto make_orthonormal_basis(
    IndexSpace const* space,
    std::vector<Expr const*> vectors,
    TensorName vector_symbol = make_tensor_name("e")) -> Basis;

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

} // namespace tender
