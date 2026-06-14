# 000031 — Tensor labels and symmetry

## Context

`TensorObject` currently carries `std::optional<TensorLabel>` where `TensorLabel` is an enum:

```cpp
enum class TensorLabel { Identity, Delta, LeviCivita };
```

These labels let a future derivation/simplification engine (or e-graph saturation pass)
recognise well-known objects without re-proving their properties from scratch:
`I @ a = a` is immediate from `TensorLabel::Identity`; the epsilon-delta identity follows
from the labels on the two tensors involved.

The question is how to extend this as we approach a real simplification engine.

---

## Proposed extension: symmetry as permutation generators

Add a symmetry specification alongside (or subsuming) the well-known-label field.
A symmetry group is finitely generated; storing a small set of generators is compact and
sufficient for a pattern-matching engine.

### Examples walked through

| Object | Symmetry (sign preserved) | Antisymmetry (sign flips) |
|---|---|---|
| Symmetric rank-2 tensor **A** | any swap of same-level slots | — |
| Kronecker delta δ^i_j | swap of slots regardless of level | — |
| Levi-Civita ε_{ijk} | cyclic permutations (even perms) | any transposition (odd perms) |
| Stiffness C_{ijkl} | swap(0,1), swap(2,3), swap(01,23) | — |

---

## Design: `TensorLabel` → `TensorTraits`

"Label" implies a single tag; the struct is a collection of independent properties.
`TensorTraits` names it correctly.  It holds four orthogonal fields:

- **Well-known indicator** — which named tensor this is (Identity / Delta / LeviCivita).
- **Symmetry spec** — slot permutations under which the object is *symmetric* (value unchanged).
- **Antisymmetry spec** — a representative set of slot permutations that flip the sign (see below).
- **Render hints** — display preferences (encoded as `mpk::mix::EnumFlags`, see below).

`SymmetrySpec` and `AntisymmetrySpec` share the same internal structure but carry different
semantics.  They are distinct types so the engine can dispatch without inspecting a flag.
They use **aggregation** of `PermutationSpec` (not inheritance) to keep the class hierarchy flat.

All enums use `uint8_t` as their underlying type — required by `magic_enum` (used inside
`mpk::mix::EnumFlags`) and keeps values small and predictable.

```cpp
enum class WellKnownKind : uint8_t { Identity, Delta, LeviCivita };

enum class RenderHint : uint8_t {
    OmitVoidIndexPlaceholders = 1, // suppress \cdot in mixed upper/lower slots
};

struct SymmetrySpec     final { PermutationSpec generators; };
struct AntisymmetrySpec final { PermutationSpec generators; };

struct TensorTraits {
    std::optional<WellKnownKind>      well_known   = {};
    SymmetrySpec                      symmetry     = {};  // empty → no symmetry generators
    AntisymmetrySpec                  antisymmetry = {};  // empty → no antisymmetry generators
    mpk::mix::EnumFlags<RenderHint>   render_hints = {};
};
```

---

## `PermutationSpec` — class interface and encoding

`PermutationSpec` is a class with private storage and a public interface.  It stores
generators as bit-packed permutation images in an inline `std::array<uint8_t, MaxBytes>`.

### Bit packing

Each slot position is stored in `ceil(log2(rank))` bits — the minimum needed to represent
values 0..rank-1.  A full permutation of rank N occupies `ceil(N × bpp / 8)` bytes, where
`bpp = std::bit_width(rank - 1)` (minimum 1 for rank ≤ 1).

| Rank | bits/pos | bytes/perm | generators in MaxBytes=16 |
|------|----------|------------|--------------------------|
| 2–4  | 2        | 1          | 16                       |
| 5–8  | 3        | 3          | 5                        |
| 9–16 | 4        | 8          | 2                        |

`MaxBytes = 16` supports up to `MaxRank = 25` positions with at least one generator.

### Why `PermutationView` owns its data

With bit packing, a position spans 2-3 bits inside a byte — there is no byte boundary
aligned to each position.  The iterator therefore must **unpack** positions into a
temporary buffer before returning them.  If `PermutationView` were a `std::span` into
the spec's storage, it would reference bit-packed bytes that cannot be directly indexed
(e.g. `image[1]` at bits 2–3 of byte 0 for rank-4 cannot be read as a plain byte).

Furthermore, `operator[]` on the iterator creates a temporary iterator, calls `operator*()`,
and returns a copy.  If the view held a span into the temporary iterator's buffer, the
copy would contain a dangling pointer.

Solution: `PermutationView` owns a fixed-size `std::array<uint8_t, Capacity>` (Capacity =
MaxRank = 25) plus an active-size field `sz`.  The iterator unpacks into its `cached_`
field (a `PermutationView`) on each `operator*()`.  Returning `cached_` by reference is
safe for the caller's lifetime; returning it by value (from `operator[]`) is safe because
the copy owns its bytes.

```cpp
struct PermutationView final {
    static constexpr std::size_t Capacity = 25;
    std::array<uint8_t, Capacity> image = {};
    uint8_t sz = 0;

    auto operator[](std::size_t i) const -> uint8_t;
    auto size()  const -> std::size_t;
    auto rank()  const -> std::size_t;
};
```

### `PermutationSpec` interface

```cpp
class PermutationSpec final {
public:
    static constexpr std::size_t MaxBytes = 16;
    static constexpr std::size_t MaxRank  = 25;

    static constexpr auto bits_per_pos_for(uint8_t rank) -> uint8_t;
    static constexpr auto bytes_per_perm_for(uint8_t rank) -> uint8_t;

    class const_iterator final { /* random-access; yields const PermutationView& */ };

    PermutationSpec() = default;  // empty spec

    // Variadic constructor: all arguments after same_level_only must be
    // Permutation<N> for the same N, deduced from the first permutation.
    template <std::size_t N, std::same_as<Permutation<N>>... Rest>
    explicit PermutationSpec(bool same_level_only, Permutation<N> p0, Rest... ps);

    auto begin()          const -> const_iterator;
    auto end()            const -> const_iterator;
    auto size()           const -> std::size_t;
    auto empty()          const -> bool;
    auto rank()           const -> uint8_t;
    auto same_level_only() const -> bool;
    auto operator==(PermutationSpec const&) const -> bool = default;
};
```

Construction uses `Permutation<N>` — a value type wrapping `std::array<uint8_t, N>`:

```cpp
template <std::size_t N>
struct Permutation final {
    std::array<uint8_t, N> image;
    auto operator==(Permutation const&) const -> bool = default;
};
```

The rank is stored inside `PermutationSpec` (deduced from N at construction), so callers
do not need to pass it again and the spec is self-contained.

### `same_level_only` flag

Controls whether permutation rules fire for mixed-level slot pairs:
- `true` → rule fires only when all permuted slots share the same `Level`
  (correct for user-defined symmetric tensors; A^i_j ≠ A^j_i in oblique basis)
- `false` → rule fires regardless of Level
  (correct for delta: δ^i_j = δ^j_i by definition)

---

## The three distinct kinds of "rank-2 symmetry"

These should NOT all share one label:

| Case | Meaning | Origin |
|---|---|---|
| **A** = **A**^T (tensor level) | Invariant tensor equality | Symmetry spec on the TensorObject |
| A^{ij} = A^{ji} (same-level coords) | Coordinate consequence | Propagated by basis-expansion function |
| δ^i_j = δ^j_i (mixed-level delta) | Follows from **I** = **I**^T | Derivable from `WellKnownKind::Delta`, not a "symmetric tensor" rule |

In particular: a *general* symmetric tensor **A** in an oblique basis does **not** have
symmetric mixed-level coordinates — A^i_j ≠ A^j_i in general.  The `same_level_only`
flag handles this.

---

## Rendering hint vs. semantic label — keep them separate

The `render_hints` field in `TensorTraits` carries display preferences like
`OmitVoidIndexPlaceholders`.  This is set explicitly by factory functions that know
the object warrants it (e.g. `make_delta`), not derived automatically from symmetry.

The simplification engine uses `symmetry`/`antisymmetry` for rewrites.  The renderer
uses `render_hints` for display.  Each layer has one job.

Effect of `OmitVoidIndexPlaceholders`:
- With hint: `\delta^{i}_{j}` (flat band grouping, no `\cdot`)
- Without hint: `\delta^{i\cdot}_{\cdot j}` (positional interleaving)

---

## Factory assignments

```
make_identity:
  TensorTraits{.well_known = Identity}
  (symmetry/antisymmetry to be populated when simplification engine is built)

make_delta:
  TensorTraits{
    .well_known   = Delta,
    .render_hints = OmitVoidIndexPlaceholders
  }

make_levi_civita (rank 3):
  TensorTraits{.well_known = LeviCivita}
  // Cyclic generator (0→1→2→0) → symmetry (preserves sign)
  // Transposition (0↔1)       → antisymmetry (flips sign)
  // To be populated when simplification engine is built.
  //
  // Note: odd permutations do NOT form a group — composing two sign-flipping
  // permutations yields an even permutation (sign preserved).  The antisymmetry
  // spec stores a representative set; the engine derives all sign-flipping
  // permutations by composing them with the symmetry group.

User symmetric tensor A (rank 2):
  TensorTraits{
    .symmetry = {.generators = PermutationSpec(true, Permutation<2>{{1,0}})}
  }

Stiffness C (rank 4), minor + major symmetry:
  TensorTraits{
    .symmetry = {.generators = PermutationSpec(true,
        Permutation<4>{{1,0,2,3}},  // swap slots 0↔1
        Permutation<4>{{0,1,3,2}},  // swap slots 2↔3
        Permutation<4>{{2,3,0,1}})} // swap pairs (01)↔(23)
  }
```

---

## Impact on core engine design

### AST / `TensorObject`

`std::optional<TensorTraits>` replaces `std::optional<TensorLabel>` (which was `optional<enum>`).
All existing code that pattern-matches on the old enum needs updating to check `.well_known`;
factory functions need updating.  This is a contained, mechanical change — done.

### Future simplification engine

The symmetry spec is the primary input for:
- **Canonical slot ordering**: reorder slot bindings into the canonical permutation
  (e.g., alphabetical index name) and track the sign.
- **Contraction simplification**: `δ^i_j * A^j_k = A^i_k` is driven by `WellKnownKind::Delta`.
- **Zero detection**: contracting ε with a symmetric tensor over any two indices → 0.
- **E-graph saturation**: symmetry rewrites are cheap local rules that fire often.

The `well_known` field handles object-specific rules (delta contraction, identity absorption).
The `symmetry` field handles generic permutation rules.  Keeping them separate lets the
engine apply generic rules without special-casing every well-known tensor.
