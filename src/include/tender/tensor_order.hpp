#pragma once

// Shared three-way comparators for the leaf data common to `Expr` and `Nf`:
// tensor names, index spaces, index associations, and whole `TensorObject`s.
//
// These were originally file-local helpers of `expr_cmp` in derivation.cpp.
// They are lifted here so the `Nf` factor/term order (nf.hpp) can sort atoms
// by the *same* key as the existing `Expr` canonical order, keeping the two
// IRs' orderings consistent (DRY; needed by the C10 differential harness).
//
// Each returns a negative / zero / positive int (strcmp convention).

#include <tender/index.hpp>       // IndexAssoc
#include <tender/index_space.hpp> // IndexSpace

#include <optional>
#include <string_view>

namespace tender
{

struct TensorObject; // expr.hpp

// Lexicographic on the raw name text.
[[nodiscard]] auto name_view_cmp(std::string_view a, std::string_view b) -> int;

// By dimension, then by value set, then (same values, distinct instances) by
// pointer for a stable within-run order.
[[nodiscard]] auto space_cmp(IndexSpace const* a, IndexSpace const* b) -> int;

// Empty < filled; among filled, by variant kind then by id / value / name.
[[nodiscard]] auto index_assoc_cmp(
    std::optional<IndexAssoc> const& a,
    std::optional<IndexAssoc> const& b) -> int;

// By name, then rank, then slot count, then per-slot level / realm / space /
// index.  Matches `expr_cmp`'s TensorObject arm exactly.
[[nodiscard]] auto tensor_object_cmp(
    TensorObject const& a, TensorObject const& b) -> int;

} // namespace tender
