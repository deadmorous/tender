#pragma once

// Symmetry-orbit canonicalization of a `TensorObject`'s slots (vibe 000047),
// shared between the `Expr` canon (derivation.cpp) and the `Nf` lowering
// (nf_lower.cpp).  Originally file-local in derivation.cpp; lifted here so the
// all-`*` model canonicalizes a symmetric / antisymmetric tensor's slot order
// the same way (DRY; needed for the C13 flip).

#include <tender/expr.hpp> // TensorObject, SlotBinding

#include <utility>
#include <vector>

namespace tender
{

// Total order on slot sequences (the same key as `expr_cmp`'s TensorObject
// arm).  The two arguments are arrangements of one tensor's slots, so they
// share a (realm, space) multiset — only level and index ever differ.
[[nodiscard]] auto slot_seq_cmp(
    std::vector<SlotBinding> const& a,
    std::vector<SlotBinding> const& b) -> int;

// Orbit-minimal slot arrangement of `t` under its declared (anti)symmetry, with
// the sign of the reordering folded out:
//   - sign +1 / −1 with the canonical slot sequence, or
//   - sign 0 when an arrangement is reachable with *both* signs — the object is
//     identically zero (e.g. ε with a repeated index); the returned slots are
//     then unspecified.
// A tensor with no generators (or no traits) returns its slots unchanged, +1.
[[nodiscard]] auto canon_symmetry_slots(TensorObject const& t)
    -> std::pair<std::vector<SlotBinding>, int>;

} // namespace tender
