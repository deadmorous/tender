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

// Collapse a Sum of dim component-product terms into a compact IndexedSum
// display node.  Recognises:
//   a^1 b_1 + a^2 b_2 + a^3 b_3  →  IndexedSum("a","^","b","_","i")
// The index letter is auto-chosen from {i,j,k,l,m,n,p,q,r,s} unless
// `index_letter` is non-empty, in which case it is validated and used.
// Passes expressions through unchanged if the pattern does not match.
auto collect_repeated_sum_step(
    CoordSystem const& cs, std::string index_letter = "") -> DerivationStep;

// Recognises a Sum of dim TensorProduct(component, basis_vector) terms
// and replaces it with the named rank-1 tensor inferred from the component
// naming convention (e.g. a^1 e_1 + a^2 e_2 + a^3 e_3 → a).
// Recurses bottom-up; passes unrecognised sub-expressions through.
auto reassemble_vector_step(CoordSystem const& cs) -> DerivationStep;

// Recognises Contract(covar_expansion, contravar_expansion) where each
// expansion is a Sum of dim TensorProduct(component, basis_vector) terms,
// and replaces it with the corresponding Contract(a, b).
// Recurses bottom-up; passes unrecognised sub-expressions through.
auto reassemble_dot_step(CoordSystem const& cs) -> DerivationStep;

} // namespace tender
