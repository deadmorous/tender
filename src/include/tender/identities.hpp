#pragma once

#include <tender/identity.hpp>

namespace tender
{

class Context;
class IndexSpace;

// A small curated library of standard index identities (vibe 000033 §3, vibe
// 000034 item #8).  Each factory allocates fresh dummy ids from `ctx`, so the
// returned Identity can be matched against any target built in the same Context
// group.  These are the rule set the e-graph saturates with, and the building
// blocks the invariant layer (vibe 000036) will reduce to.
namespace identities
{

// Σ_p δ^p_a δ^p_b  =  δ_{ab}   (any space)
[[nodiscard]] auto delta_contraction(Context&, IndexSpace const* space)
    -> Identity;

// Σ_p δ^p_p  =  dim(space)   (concrete-cardinality space)
[[nodiscard]] auto delta_trace(Context&, IndexSpace const* space) -> Identity;

// Σ_i ε^{ijk} ε_{ilm}  =  δ^j_l δ^k_m − δ^j_m δ^k_l   (3D)
[[nodiscard]] auto eps_delta_1(Context&) -> Identity;

// Σ_i Σ_j ε^{ijk} ε_{ijl}  =  2 δ^k_l   (3D)
[[nodiscard]] auto eps_delta_2(Context&) -> Identity;

} // namespace identities

} // namespace tender
