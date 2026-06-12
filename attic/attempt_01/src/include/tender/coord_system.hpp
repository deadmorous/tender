#pragma once

#include <tender/expr.hpp>

#include <memory>

namespace tender
{

// Abstract coordinate system. Holds covariant basis vectors g_i, their
// contravariant duals g^i (cobasis), and the metric tensor g_ij.
class CoordSystem
{
public:
    virtual ~CoordSystem() = default;

    virtual auto dim() const noexcept -> int = 0;

    // i-th coordinate Parameter (0-based). Returns nullptr when the CS has no
    // explicit coordinate parameters (e.g. DirectBasisCS).
    virtual auto coord(int i) const -> Parameter* = 0;

    virtual auto basis(int i) const -> Expr* = 0;         // g_i  — rank 1
    virtual auto cobasis(int i) const -> Expr* = 0;       // g^i  — rank 1
    virtual auto metric(int i, int j) const -> Expr* = 0; // g_ij — rank 0

    virtual auto is_orthonormal() const noexcept -> bool = 0;
};

// World Cartesian System — orthonormal frame with basis i, j, k.
// Coordinate Parameters are "x", "y", "z".
auto wcs() -> CoordSystem const&;

// CS defined by explicitly given covariant basis vectors e1, e2, e3 (rank 1).
// Cobasis is derived via the 3-D Levi-Civita formula:
//   g^1 = (e2 × e3) / V,  g^2 = (e3 × e1) / V,  g^3 = (e1 × e2) / V
// where V = e1 · (e2 × e3).
// e1, e2, e3 must outlive the returned CoordSystem.
auto make_direct_basis_cs(Expr* e1, Expr* e2, Expr* e3)
    -> std::unique_ptr<CoordSystem>;

// Cylindrical coordinates (r, theta, z).
// Covariant basis: g_r = e_r, g_theta = r e_theta, g_z = e_z.
// Cobasis:         g^r = e_r, g^theta = (1/r) e_theta, g^z = e_z.
// Metric:          diag(1, r^2, 1).
auto cylindrical_cs() -> CoordSystem const&;

// Spherical coordinates (r, theta, phi).
// Covariant basis: g_r = e_r, g_theta = r e_theta, g_phi = r sin(theta) e_phi.
// Cobasis:         g^r = e_r, g^theta = (1/r) e_theta,
//                  g^phi = (1 / (r sin(theta))) e_phi.
// Metric:          diag(1, r^2, r^2 sin^2(theta)).
auto spherical_cs() -> CoordSystem const&;

// Gradient of a rank-0 scalar field f with respect to coordinate system cs.
// cs must have explicit coordinates (coord(i) != nullptr for all i).
// Returns: sum_i TensorProduct(deriv(f, coord_i), cobasis_i)  — rank 1.
auto grad(ResourceList& rl, Expr* f, CoordSystem const& cs) -> Expr*;

} // namespace tender
