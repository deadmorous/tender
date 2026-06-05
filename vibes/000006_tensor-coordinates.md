# Tensor Coordinates and Co/Contravariance

## Status: Q_coord_extraction partially resolved (parameterized expression concept established); Q_mixed_coords, Q_coord_display open

---

## Core principle

Tensors are invariant (coordinate-free) objects. Co/contravariance is not a
property of a tensor — it is a property of the coordinates you obtain when you
project a tensor onto a chosen basis.

Given a basis **e**_i and its cobasis **e**^i (defined by **e**^i · **e**_j = δ^i_j),
the coordinates of a rank-2 tensor **T** are extracted by explicit contraction:

| Coordinate type | Expression | Index placement |
|---|---|---|
| Contravariant | T^ij = **e**^i · **T** · **e**^j | both upper |
| Covariant | T_ij = **e**_i · **T** · **e**_j | both lower |
| Mixed | T^i_j = **e**^i · **T** · **e**_j | upper i, lower j |
| Mixed (other) | T_i^j = **e**_i · **T** · **e**^j | lower i, upper j |

The same tensor **T** is expressed in full as, e.g.:

```
T = T^ij e_i ⊗ e_j  =  T_ij e^i ⊗ e^j  =  T^i_j e_i ⊗ e^j  =  ...
```

All four representations refer to the same geometric object. tender works with
the invariant tensor; coordinates are extracted on demand by the user.

For orthonormal bases, **e**^i = **e**_i, so all four coordinate types coincide
and the upper/lower distinction can be relaxed (see vibe 000004, Q_index_slots).

---

## Open questions

### Q_coord_extraction — RESOLVED

**Decision: Option B — slots with explicit convolution.**

A parameterized expression carries abstract **slots** (free index positions),
not references to named Index objects. Slot identity is positional, not
name-based. Two expressions with the same slot structure but different display
names for their slots are equivalent (α-equivalence).

#### Where slots live

**Invariant tensors and vectors have no slots.** They are coordinate-free
objects. Operations `@`, `//`, `**`, `trace()` on them are direct geometric
operations — no slot machinery involved.

**Slots appear only on indexed objects:**
- Basis vectors **e**_i: one lower slot
- Cobasis vectors **e**^i: one upper slot
- Polyades **e**_i ⊗ **e**_j (and higher): one slot per factor, lower or upper
  according to whether basis or cobasis vectors are used
- Any expression built from the above (e.g., v^i = **e**^i · **v**): inherits
  slots from the indexed objects it was constructed from

A slot has:
- A **position type**: upper (contravariant) or lower (covariant)
- A **display name**: used only for printing; auto-assigned or user-specified;
  not part of the expression's identity

Operations on slots:
1. **Substitution**: insert a concrete integer → specializes the expression
   (e.g., v^i → v^1)
2. **Index binding**: insert an Index object (with domain and auto-sum flag)
   → gives the slot computational meaning for evaluation and summation
3. **Convolution**: declare that a slot in one expression and a slot in another
   (or the same) expression share the same index → performs the sum if one is
   upper and one is lower (relaxed for orthonormal bases)

Convolution is the fundamental operation on indexed objects; Einstein summation
convention is name-triggered convolution — here it is made explicit.

#### Display name for slots (example 1)

When printing v^i, `i` is the display name of the slot. Options:
- **(b) auto or optionally specify** — chosen.
  Auto-assignment picks a sensible letter (i, j, k, … or α, β for abstract
  slots). User can override by passing an Index object whose name is used:
  `coords(v, basis, index=i_obj)`.

#### Worked examples

**Example 2 — v^i, substitution and binding**

```
v^i                       # parameterized expression; one upper slot, display name 'i'
v^i [i ← 1]              # substitution: scalar, the first component
v^i [i ← idx]            # binding: idx is an Index object with domain and auto-sum
```

Binding alone on `v^i` is not very interesting — the expression remains
parameterized. Its value comes when the bound slot is convolved with another.

**Example 3 — Kronecker delta δ_i^j**

Two slots: one lower (display 'i'), one upper (display 'j').

```
δ_i^j                     # two free slots; 9 components when both substituted
δ_i^j [i ← idx, j ← idx] # same index in both slots → convolution → trace = 3
```

Inserting the same Index into both slots is precisely convolution applied to a
single expression. This confirms convolution is not redundant: it is the
operation that "closes" slots together, triggering summation.

**Example 4 — products of Kronecker deltas**

Multiplying two parameterized expressions yields an expression whose slots are
the union of both factors' slots. Convolution then pairs them up explicitly.

| Expression | Slots before convolution | Convolutions | Result |
|---|---|---|---|
| δ_i^i · δ_j^j | i_low, i_up, j_low, j_up | (i_low↔i_up), (j_low↔j_up) | 3·3 = 9 |
| δ_i^j · δ_j^i | i_low, j_up, j_low, i_up | (j_up↔j_low), then (i_low↔i_up) | δ_i^i = 3 |
| δ_i^j · δ_j^k | i_low, j_up, j_low, k_up | (j_up↔j_low) | δ_i^k — one free lower, one free upper |
| δ_i^j · δ_k^l | i_low, j_up, k_low, l_up | none | 4 free slots; no simplification |

This shows that the number and type of free slots is an intrinsic property of
the expression after convolution, independent of display names.

#### Three-layer model

**Layer 1 — Invariant**: tensors, vectors, scalars, and operations on them
(`@`, `//`, `**`, `trace()`, etc.). No indices, no basis, no co/contravariance.
Results are guaranteed basis-independent.

**Layer 2 — Collection-indexed**: a collection of invariant objects labelled by
an ordinal index. The index has no co/contravariant meaning — it just identifies
*which* object in the collection. The objects themselves remain invariant (a
collection of vectors is still a collection of vectors, not coordinates).

Examples: shape functions φ_i, nodal displacement vectors **u**_i, eigenvectors
**v**_i. Already present in PVW.

Properties of collection indices:
- No implicit summation by default (unlike Einstein convention)
- Auto-sum can be enabled explicitly on the Index object (see vibe 000004,
  Q_index_slots) — e.g., when constructing a FEM approximation Σ_i **u**_i φ_i
- Carry no upper/lower type; `convolve()` does not apply
- `@` etc. apply to individual elements after indexing: `u[i] @ v[j]` is a
  scalar parameterized by collection indices i, j

**Layer 3 — Coordinate-indexed**: basis vectors, cobasis vectors, polyades, and
expressions built from them. Slots carry upper/lower type. `convolve()` applies
with upper↔lower matching required (relaxed for orthonormal bases).

Crossing from Layer 1 to Layer 3 (projecting onto a basis) is explicit.
Crossing from Layer 1 to Layer 2 (labelling a collection of objects) is also
explicit.

**Basis vector indices are Layer 3, not Layer 2.** The index on **e**_i is its
co/contravariant slot — lower for basis vectors, upper for cobasis vectors. It
is simultaneously the identifier of which vector and its slot type. There is no
separate ordinal layer beneath it. A numbered collection of arbitrary vectors
(e.g., eigenvectors **v**_i) would be Layer 2; a set of basis vectors **e**_i
that defines a coordinate system is Layer 3.

### Q_mixed_coords — RESOLVED

Mixed coordinates are user-assembled contractions with a mix of basis and
cobasis vectors: **e**^i · **T** · **e**_j produces a parameterized expression
with one upper and one lower slot. This falls out of the general coordinate
extraction model with no special mechanism required.

### Q_coord_display — RESOLVED

Display is verbatim: whatever form the expression currently has — invariant,
coordinate, dyadic expansion, sum of named sub-tensors — that is what gets
rendered. No automatic switching between representations.

If a different representation is needed, the user applies a derivation to reach
it as a new State, then displays that State. This is consistent with the
State/Derivation model (vibe 000005): display is a pure rendering operation,
not a mathematical one. The State is the representation.
