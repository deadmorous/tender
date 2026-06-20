#pragma once

#include <tender/basis.hpp>

namespace tender
{

// Well-known coordinate systems (Stage 4, vibe 000049 §2).
//
// Each returns the orthonormal frame of the system as a Basis (a coordinate
// system *produces* a basis — aggregation, not inheritance).  They are all
// orthonormal: the frame vectors are treated as orthonormal unit vectors at a
// point.  The position dependence of the curvilinear frames (∂e_i/∂x^j), the
// metric/scale factors, and the coordinate parameters are Stage-5 concerns and
// are not modelled here — so a richer CoordSystem object is deferred until
// those exist; today the deliverable is the named frame each system hands back.
//
// Frame vector names are limited to what a TensorName allows (a single letter
// or a LaTeX command), so curvilinear directions are spelled with their
// coordinate letter (e.g. r, \theta, \phi, z).

// World Cartesian System: 3D orthonormal frame i, j, k.
[[nodiscard]] auto wcs(Context& ctx) -> Basis;

// Cylindrical coordinates (r, \theta, z): orthonormal frame r, \theta, z.
[[nodiscard]] auto cylindrical(Context& ctx) -> Basis;

// Spherical coordinates (r, \theta, \phi): orthonormal frame r, \theta, \phi.
[[nodiscard]] auto spherical(Context& ctx) -> Basis;

// 2D polar coordinates (r, \theta): orthonormal frame r, \theta (cardinality
// 2).
[[nodiscard]] auto polar_2d(Context& ctx) -> Basis;

} // namespace tender
