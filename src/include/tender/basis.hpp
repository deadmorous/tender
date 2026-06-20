#pragma once

#include <tender/expr.hpp>
#include <tender/index.hpp>
#include <tender/index_space.hpp>
#include <tender/name.hpp>

#include <string_view>
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
        IndexSpace const*, std::vector<Expr const*>, std::string_view) -> Basis;

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
    std::string_view vector_symbol = "e") -> Basis;

} // namespace tender
