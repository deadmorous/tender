#pragma once

#include <tender/coord_system.hpp>
#include <tender/derivation.hpp>

namespace tender
{

// Replace Contract(e_i, e^j) nodes with their scalar values (0 or 1 for WCS).
// Also pulls rank-0 scalar factors out of contractions so that
// Contract(TensorProduct(s, v), r) becomes make_product(s, Contract(v, r)).
// This step reduces component-form dot products to sums of scalar products.
auto simplify_basis_dot_step(CoordSystem const& cs) -> DerivationStep;

// Rebuild all Sum nodes via make_sum, which removes zero (RationalConst(0))
// terms and unwraps single-element sums.  Call after simplify_basis_dot_step.
auto collect_zero_terms_step() -> DerivationStep;

// Inverse of expand_in_basis.  Recognises a Sum of basis-vector products as a
// named tensor.  Currently handles:
//   rank 2: Sum_i TensorProduct(e_i, e^i) → IdentityTensor
// Other patterns are passed through unchanged.
auto reassemble_from_components_step(CoordSystem const& cs) -> DerivationStep;

} // namespace tender
