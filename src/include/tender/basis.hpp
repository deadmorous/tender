#pragma once

#include <tender/expr.hpp>
#include <tender/index.hpp>
#include <tender/index_space.hpp>

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

    // i-th covariant basis vector e_i (0-based), rank 1.
    // Throws std::out_of_range if i is not a valid index.
    auto basis(int i) const -> Expr const*;
    // i-th contravariant cobasis vector e^i (0-based), rank 1.
    // Throws std::out_of_range if i is not a valid index.
    auto cobasis(int i) const -> Expr const*;

private:
    Basis(
        Realm realm,
        IndexSpace const* space,
        std::vector<Expr const*> vectors,
        std::vector<Expr const*> covectors);

    friend auto make_orthonormal_basis(
        IndexSpace const*, std::vector<Expr const*>) -> Basis;

    Realm realm_;
    IndexSpace const* space_;
    std::vector<Expr const*> vectors_;   // covariant   e_i
    std::vector<Expr const*> covectors_; // contravariant e^i
};

// Build an orthonormal basis from rank-1 vectors.  The realm is Orthonormal and
// the cobasis coincides with the basis.
//
// Preconditions (else std::invalid_argument): space is non-null; at least one
// vector; vectors.size() == space->values().size(); every vector is non-null
// and rank 1 (where its rank is known).
[[nodiscard]] auto make_orthonormal_basis(
    IndexSpace const* space, std::vector<Expr const*> vectors) -> Basis;

} // namespace tender
