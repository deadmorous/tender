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

## Design issues to resolve before implementing

### 1. `TensorLabel` → `TensorTraits`

"Label" implies a single tag; the struct is a collection of independent properties.
`TensorTraits` names it correctly.  It holds four orthogonal fields:

- **Well-known indicator** — which named tensor this is (Identity / Delta / LeviCivita).
- **Symmetry spec** — slot permutations under which the object is *symmetric* (value unchanged).
- **Antisymmetry spec** — a representative set of slot permutations that flip the sign (see below).
- **Render hints** — display preferences (encoded as `mpk::mix::EnumFlags`, see below).

`SymmetrySpec` and `AntisymmetrySpec` have the **same internal structure** — packed
permutation images in a fixed-length byte array — but carry different semantics to the
simplification engine.  Keeping them as separate fields (rather than a single spec with
a sign flag) lets the engine dispatch on them independently and makes the intent explicit
at call sites.

Proposed struct:

All enums in the codebase use `uint8_t` as their underlying type — this is required by
`magic_enum` (used inside `mpk::mix::EnumFlags`) and keeps values small and predictable.
We will never need more than 256 values in any tender enum.

```cpp
enum class WellKnownKind : uint8_t { Identity, Delta, LeviCivita };

// Render-time display preferences.  Use mpk::mix::EnumFlags<RenderHint> at
// call sites so the set is type-safe and composable with | and &.
enum class RenderHint : uint8_t {
    SuppressMixedDots = 1 << 0, // omit \cdot placeholders for mixed-level slots
    // (add more flags here as needed)
};

// Symmetry (or antisymmetry) generators stored as permutation images.
// 'same_level_only' controls whether the rule fires for mixed-level slots:
//   true  → only applies when all permuted slot positions share the same Level
//             (correct for user-defined symmetric tensors: A^{ij}=A^{ji}
//              but A^i_j ≠ A^j_i in an oblique basis)
//   false → applies regardless of Level
//             (correct for delta: δ^i_j = δ^j_i)
struct PermutationSpec {
    uint8_t  num_gens        = 0;
    bool     same_level_only = true;
    // Packed permutation images.  Each generator occupies
    // ceil(log2(rank)) × rank bits, rounded to a byte boundary.
    // rank is read from the enclosing TensorObject.
    // N = 16: fits 7 gens for rank ≤ 4 (1 byte/gen) or 5 gens for rank ≤ 8 (3 bytes/gen).
    static constexpr std::size_t MaxBytes = 16;
    std::array<uint8_t, MaxBytes> data{};
};

// SymmetrySpec and AntisymmetrySpec share the same layout but are distinct types
// so the simplification engine can dispatch on them without inspecting a flag.
struct SymmetrySpec     : PermutationSpec {};
struct AntisymmetrySpec : PermutationSpec {};

struct TensorTraits {
    std::optional<WellKnownKind>    well_known;
    std::optional<SymmetrySpec>     symmetry;
    std::optional<AntisymmetrySpec> antisymmetry;
    mpk::mix::EnumFlags<RenderHint> render_hints{};
};
```

**Encoding example — rank 4, 2 bits per position:**

| Object | Generator (image) | Bytes | Notes |
|---|---|---|---|
| Swap slots 0↔1 | `[1,0,2,3]` | `0b01_00_11_10` = 1 byte | transposition |
| Swap slots 2↔3 | `[0,1,3,2]` | `0b10_11_01_00` = 1 byte | transposition |
| Swap pairs (01)↔(23) | `[2,3,0,1]` | `0b00_01_10_11` = 1 byte | pair swap |
| 3-cycle 0→1→2→0 | `[1,2,0,3]` | `0b11_00_10_01` = 1 byte | arbitrary perm |

For rank 6 (3 bits/position): 6 × 3 = 18 bits → 3 bytes per generator.  
For rank 8 (3 bits/position): 8 × 3 = 24 bits → 3 bytes per generator.  
`MaxBytes = 16` handles that comfortably.

**Factory assignments:**

```
make_identity:
  TensorTraits{
    .well_known   = Identity,
    .symmetry     = {num_gens=1, same_level_only=false, [image: 1,0]},
    .render_hints = {}
  }

make_delta:
  TensorTraits{
    .well_known   = Delta,
    .symmetry     = {num_gens=1, same_level_only=false, [image: 1,0]},
    .render_hints = RenderHint::SuppressMixedDots
  }

make_levi_civita (rank 3):
  TensorTraits{
    .well_known   = LeviCivita,
    .symmetry     = {num_gens=1, same_level_only=true, [image: 1,2,0]},  // cyclic 0→1→2→0 preserves sign
    .antisymmetry = {num_gens=1, same_level_only=true, [image: 1,0,2]},  // swap 0↔1 flips sign
    .render_hints = {}
  }
  // The symmetry group of ε is the alternating group A_n (even permutations, generated
  // by the n-cycle).  The antisymmetry spec stores one representative odd permutation;
  // the engine derives all sign-flipping permutations by composing it with the symmetry
  // group.  Note: odd permutations do NOT themselves form a group — composing two
  // sign-flipping permutations yields an even permutation (sign preserved).

User symmetric tensor A (rank 2):
  TensorTraits{
    .symmetry = {num_gens=1, same_level_only=true, [image: 1,0]}
  }

Stiffness C (rank 4), minor + major symmetry:
  TensorTraits{
    .symmetry = {num_gens=3, same_level_only=true,
                 [image: 1,0,2,3],   // swap slots 0↔1
                 [image: 0,1,3,2],   // swap slots 2↔3
                 [image: 2,3,0,1]}   // swap pairs (01)↔(23)
  }
```

### 2. The three distinct kinds of "rank-2 symmetry"

These should NOT all share one label:

| Case | Meaning | Origin |
|---|---|---|
| **A** = **A**^T (tensor level) | Invariant tensor equality | Symmetry spec on the TensorObject |
| A^{ij} = A^{ji} (same-level coords) | Coordinate consequence | Propagated by basis-expansion function |
| δ^i_j = δ^j_i (mixed-level delta) | Follows from **I** = **I**^T | Derivable from `WellKnownKind::Delta`, not a "symmetric tensor" rule |

In particular: a *general* symmetric tensor **A** in an oblique basis does **not** have
symmetric mixed-level coordinates — A^i_j ≠ A^j_i in general.  The engine must not
apply the same-level rule to mixed-level slots.  The `same_level_only` flag on
`PermutationSpec` handles this: `true` for user-defined symmetric tensors, `false`
for delta (whose slot-swap symmetry holds regardless of level).

### 3. Rendering hint vs. semantic label — keep them separate

The proposal to omit void-placeholder dots for symmetric mixed-level objects is
appealing (δ^i_j instead of δ^{i·}_{·j}), but coupling rendering to symmetry labels
creates a layering problem: the renderer would need to inspect semantic metadata to
decide layout.

Preferred approach:
- **Rendering stays purely positional** — always interleave dots for mixed-level slots.
- **A `render_hints` field** (separate, optional) can carry display preferences like
  `suppress_mixed_dots = true`.  This is set explicitly by factory functions that know
  the object warrants it (e.g. `make_delta`), not derived automatically from symmetry.
- The simplification engine uses `symmetry` for rewrites; the renderer uses `render_hints`
  for display.  Each layer has one job.

### 4. Permutation encoding: image array, fixed-length storage

Each generator is stored as its **image array** — where each slot position maps to.
This naturally encodes any permutation, including arbitrary cycles:

- swap 0↔1: image `[1,0,2,3]`
- 3-cycle 0→1→2→0: image `[1,2,0,3]`
- pair-swap (01)↔(23): image `[2,3,0,1]`

Bit width per position: `ceil(log2(rank))` — 2 bits for rank ≤ 4, 3 bits for rank 5–8.
This is tight but exact; no sentinel values are needed (valid images never have
out-of-range values and the number of generators is stored separately).

The `std::array<uint8_t, MaxBytes>` in `SymmetrySpec` provides fixed-size, heap-free
storage that fits inline in `TensorObject`.  `MaxBytes = 16` is the starting point;
it is increased (with a recompile) if ranks beyond 8 or more than ~5 generators
are ever needed.  No API change required when N grows.

---

## Impact on core engine design

### AST / `TensorObject`

`std::optional<TensorTraits>` replaces `std::optional<TensorLabel>` (which was `optional<enum>`).
All existing code that pattern-matches on the old enum (`Identity`, `Delta`, `LeviCivita`)
needs updating to check `.well_known`; factory functions need updating.
This is a contained, mechanical change.

### Factory functions

`make_identity`, `make_delta`, `make_levi_civita` fill in the appropriate `TensorTraits`
fields (`well_known`, `symmetry` or `antisymmetry`, `render_hints`).
`make_tensor_object` (generic) accepts an optional `TensorTraits` from the caller.
Python bindings expose `SymmetrySpec`, `AntisymmetrySpec`, and `RenderHint` so users
can label their own tensors.

### Renderer

Minimal impact.  The renderer checks `rank` for bold/plain.  It will additionally check
`traits.render_hints` — e.g., `SuppressMixedDots` skips dot-interleaving for mixed-level
slots even when both levels are present.  `symmetry` and `antisymmetry` are invisible
to the renderer.

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

---

## Proposed next steps

1. Replace `enum class TensorLabel` with `struct TensorTraits` in `expr.hpp`:
   fields `well_known`, `symmetry`, `antisymmetry`, `render_hints`.
2. Add `enum class RenderHint` and wire `mpk::mix::EnumFlags<RenderHint>` into `TensorTraits`.
3. Update `make_identity`, `make_delta`, `make_levi_civita` with correct `TensorTraits`.
4. Update renderer to check `render_hints` (e.g. `SuppressMixedDots` for delta).
5. Update C++ tests (expr_test, render_test) and Python bindings.
6. Leave the simplification engine for a later session once `TensorTraits` is stable.
