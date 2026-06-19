#pragma once

#include <tender/identity.hpp>
#include <tender/index.hpp> // Realm

namespace tender
{

class Context;
class IndexSpace;

// A small curated library of standard index identities (vibe 000033 §3, vibe
// 000034 item #8).  Each factory allocates fresh dummy ids from `ctx`, so the
// returned Identity can be matched against any target built in the same Context
// group.  These are the rule set the e-graph saturates with, and the building
// blocks the invariant layer (vibe 000036) will reduce to.
//
// Every factory is parameterized by `realm` (vibe 000047 decision (a)): an
// expression's indices share one realm in ~all cases, so a rule is built in the
// realm of the target it will be matched against.  Index *order* is handled by
// the symmetry/antisymmetry canonicalization (also vibe 000047), so a rule need
// not enumerate slot orderings.
//
// Index *level* convention: in the Orthonormal realm upper and lower are
// interchangeable, so every Orthonormal index is spelled **lower** (e.g.
// delta_contraction yields δ_pa δ_pb, not δ^p_a δ^p_b).  Match the same lower
// spelling when building Orthonormal targets — matching is level-exact and
// canonicalize does not coerce levels.  Oblique levels are unchanged.
namespace identities
{

// Σ_p δ^p_a δ^p_b  =  δ_{ab}   (any space)
[[nodiscard]] auto delta_contraction(
    Context&, IndexSpace const* space, Realm realm) -> Identity;

// Σ_p δ^p_p  =  dim(space)   (concrete-cardinality space)
[[nodiscard]] auto delta_trace(Context&, IndexSpace const* space, Realm realm)
    -> Identity;

// Σ_i ε^{ijk} ε_{ilm}  =  δ^j_l δ^k_m − δ^j_m δ^k_l   (3D)
[[nodiscard]] auto eps_delta_1(Context&, Realm realm) -> Identity;

// Σ_i Σ_j ε^{ijk} ε_{ijl}  =  2 δ^k_l   (3D)
[[nodiscard]] auto eps_delta_2(Context&, Realm realm) -> Identity;

} // namespace identities

} // namespace tender
