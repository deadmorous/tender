#include <tender/basis.hpp>

#include <stdexcept>
#include <utility>
#include <variant>

namespace tender
{

Basis::Basis(
    Realm realm,
    IndexSpace const* space,
    std::vector<Expr const*> vectors,
    std::vector<Expr const*> covectors) :
  realm_(realm),
  space_(space),
  vectors_(std::move(vectors)),
  covectors_(std::move(covectors))
{
}

auto Basis::basis(int i) const -> Expr const*
{
    return vectors_.at(static_cast<std::size_t>(i));
}

auto Basis::cobasis(int i) const -> Expr const*
{
    return covectors_.at(static_cast<std::size_t>(i));
}

namespace
{

// A provided basis vector must be rank 1 where its rank is known.  Composite
// vectors (not a bare TensorObject) and rank-unknown objects pass; the check
// only rejects an object that is explicitly some other rank.
auto rank_ok(Expr const* v) -> bool
{
    auto const* t = std::get_if<TensorObject>(&v->node);
    if (!t || !t->rank)
        return true;
    return *t->rank == 1;
}

} // namespace

auto make_orthonormal_basis(
    IndexSpace const* space, std::vector<Expr const*> vectors) -> Basis
{
    if (!space)
        throw std::invalid_argument("make_orthonormal_basis: null space");
    if (vectors.empty())
        throw std::invalid_argument(
            "make_orthonormal_basis: at least one vector is required");
    if (vectors.size() != space->values().size())
        throw std::invalid_argument(
            "make_orthonormal_basis: number of vectors must equal the index "
            "space cardinality");
    for (auto const* v: vectors)
    {
        if (!v)
            throw std::invalid_argument(
                "make_orthonormal_basis: null basis vector");
        if (!rank_ok(v))
            throw std::invalid_argument(
                "make_orthonormal_basis: basis vectors must be rank 1");
    }

    // Orthonormal: the cobasis coincides with the basis.
    auto covectors = vectors;
    return Basis{
        Realm::Orthonormal, space, std::move(vectors), std::move(covectors)};
}

} // namespace tender
