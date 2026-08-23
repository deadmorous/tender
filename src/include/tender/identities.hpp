#pragma once

#include <tender/identity.hpp>
#include <tender/index.hpp> // Realm

#include <string_view>
#include <vector>

namespace tender
{

class Context;
class IndexSpace;

// A curated library of standard identities (vibe 000033 §3, vibe 000034 item
// #8), organised into named *groups* (vibe 000096 M2 increment 2).  Each
// factory allocates fresh dummy ids from `ctx`, so the returned Identity can be
// matched against any target built in the same Context group.  These are the
// rule sets the engine verbs saturate with: a verb call selects the groups its
// problem needs — never "all rules", which is how a saturation blows up.
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
//
// !! Pattern-variable names are load-bearing !!  Canon sorts a *symmetric*
// contraction chain (`:` / `··`, and `·` between rank-1 operands) by tensor
// name, and the matcher compares chain factors positionally.  So a rule whose
// LHS is such a chain matches only targets whose own names happen to sort the
// same way — the rule's variable name silently decides which targets it fires
// on.  Every rule below is fire-tested across the alphabet (identities_test)
// for exactly this reason; a rule that cannot be made name-robust is left OUT
// of the library until AC chain matching lands (vibe 000096 increment 3).
namespace identities
{

// ---- eps_delta group: index-level δ / ε contractions --------------------

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

// ---- cross group: invariant cross-product identities --------------------
//
// These are *invariant* rules: their pattern variables are slot-less abstract
// tensors (vibe 000051 subtree variables), so they match whole factors of any
// shape rather than index expressions.

// a × (b × c)  =  b (a·c) − c (a·b)     — bac-cab, the vector triple product.
// Fires only when the inner cross is a genuine rank-1 triple product: a rank-2
// operand makes the crosses reassociate around the fence (vibe 000055), and
// the identity correctly does not apply there (see `cross_removal`).
[[nodiscard]] auto bac_cab(Context&) -> Identity;

// a × I  =  I × a                       — the cross with the identity commutes.
[[nodiscard]] auto cross_identity(Context&) -> Identity;

// a × (b × I)  =  b ⊗ a − (a·b) I       — cross removal against the identity
// tensor (Zhilin).  The rank-2 companion of bac-cab, and the motivating case
// of vibe 000056: the inner cross is a vector-with-dyad fence, so bac-cab
// itself does not (and must not) fire.
[[nodiscard]] auto cross_removal(Context&) -> Identity;

// (a × b) · (c × d)  =  (a·c)(b·d) − (a·d)(b·c)   — the Lagrange identity.
[[nodiscard]] auto lagrange(Context&) -> Identity;

// ---- dyadic group: rank-2 algebra ---------------------------------------

// tr(A · B)  =  tr(B · A)               — cyclicity of the trace.
[[nodiscard]] auto trace_cyclic(Context&) -> Identity;

// I · a  =  a                           — the identity tensor acts as identity.
[[nodiscard]] auto identity_dot(Context&) -> Identity;

// ---- groups -------------------------------------------------------------

// The names of every group in the library.
[[nodiscard]] auto group_names() -> std::vector<std::string_view>;

// The rules of one named group; throws std::invalid_argument on an unknown
// name.  `realm` / `space` parameterize the index-level groups (`eps_delta`);
// the invariant groups ignore them.  A null `space` means 3-D.
[[nodiscard]] auto group(
    Context&,
    std::string_view name,
    Realm realm = Realm::Oblique,
    IndexSpace const* space = nullptr) -> std::vector<Identity>;

// Every group concatenated.  Provided for exploration and benchmarking —
// prefer naming the groups a problem needs, since rule count is the main
// driver of saturation cost.
[[nodiscard]] auto all_rules(
    Context&,
    Realm realm = Realm::Oblique,
    IndexSpace const* space = nullptr) -> std::vector<Identity>;

} // namespace identities

} // namespace tender
