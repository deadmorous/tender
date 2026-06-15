# 000036 Tensors, coordinates, and bases — where indices come from

## Why this vibe

The eps-pair work (vibe 000035) and a sloppy shorthand ("Σ_i (ε ⊗ ε)") surfaced
that tender had no written account of what an **index actually is**, and how the
coordinate level relates to the invariant tensor.  An earlier framing in that
discussion — "tender uses abstract-index / Penrose notation" — was **wrong** and
is retracted here.  This vibe records the ontology attempt 2 commits to.  It is a
model document; no code changes are made here.

## 1. A tensor is a linear combination of polyads

A rank-2 tensor is

```
A = A^{ij} e_i e_j        (sum over i AND over j)
```

Both indices occur twice — once on the coordinate `A^{ij}`, once on the dyad
`e_i e_j` — and Einstein-sum away.  So **A has no free indices**, and

```
A ≠ A^{ij}.
```

`A^{ij}` is a *coordinate*: a number obtained by contracting the invariant with
cobasis vectors,

```
a^i  = a · e^i,          A^{ij} = e^i · A · e^j,      e_i · e^j = δ_i^j.
```

The index on a coordinate is nothing more than a record of *which (co)basis
vectors were paired with the invariant*.  Consequently `ε^{ijk} ≠ ε`, exactly as
`A^{ij} ≠ A`.

## 2. An indexed object is **not** necessarily a scalar coordinate

This was the missing piece in the first draft of this model.  An index slot does
not imply a scalar value:

- **Basis vectors** `e_i` carry an index but are rank-1.
- If `rank(A) = N > 0`, then `B_i = A · e_i` is a tensor of rank `N − 1` that
  carries the index `i`.

General principle: an index *labels an association with a basis vector*; the
rank of the indexed object is whatever remains after that association.  Scalar
coordinates (`a^i`, `A^{ij}`) are just the special case where nothing remains.

## 3. Indices come from a basis

The unifying mechanism: **a basis emits indices.**

- A **basis** is a tuple of vectors.  Its cardinality defines the index space:
  a 3D basis ranges over 3 values; a 2D basis of a subspace of a 3D space ranges
  over 2 (see vibe 000029 on subspace decomposition).
- The basis fixes the **realm** — oblique vs orthonormal — which governs
  sensitivity to index *level* (upper/lower).  Orthonormal: levels
  interchangeable.  Oblique: the upper/lower distinction matters (and the metric
  is non-trivial).
- Once a basis exists it **propagates** indices: every index it emits inherits
  its `space` and `realm`.  This covers all three cases above uniformly —
  coordinates (via the cobasis), basis members themselves, and the results of
  contracting an invariant with a basis vector.

This is exactly why `IndexSlot` already carries `(level, realm, space)` (vibe
000028): those are the properties a basis stamps onto an index.  Today we pass
`realm`/`space` by hand at construction (`make_levi_civita(ctx, Realm::Oblique,
space, …)`); conceptually they should originate from a `Basis` object and flow
outward from it.

Note: the `Collection` and `Label` realms are non-geometric index sources — an
ordinal within a collection, or a descriptive name — with no metric and no level
sensitivity.  They are degenerate "bases" (an enumeration; a label).  The clean
generalisation is *index source*, of which a vector basis is the geometric case.

## 4. Three objects routinely conflated for ε

| object | indices? | depends on basis? | what it is |
|--------|----------|-------------------|------------|
| `𝛜` (invariant) | none | — | the Levi-Civita *tensor* |
| `ε^{ijk} = √g · [ijk]` | yes (coordinates) | **yes** (metric volume `√g`) | its coordinates in a basis |
| `[ijk]` (symbol) | yes (array index) | **no** | the permutation symbol ±1, 0 |

`expand_eps` operates on the **symbol** `[ijk]` (a determinant of Kronecker
deltas).  The eps-delta derivation (vibe 000032/000035) needs **no basis**
precisely because every array in it — the LC symbol and the Kronecker delta — is
one of the *basis-independent* ones.  The moment one wants the actual coordinates
of `𝛜`, or a general `A^{ij}`, the metric `g_{ij}` and `√g` enter and a basis is
mandatory.

**Category slippage to fix:** a `TensorObject` named `ε` *reads* as the invariant
but is currently *used* as the symbol `[ijk]`.  Three things should stay distinct:
the invariant (`ε`, no indices), its coordinates (`√g[ijk]`, basis-dependent),
and the bare symbol (`[ijk]`, basis-free).  Right now only the third is modelled,
under the name of the first.

## 5. Two layers

- **Invariant / polyadic layer:** tensors as sums of polyads; bases (`e_i`,
  cobasis `e^i`); `a = a^i e_i`; metric `g_{ij}`, `√g`; invariant operations
  (`·`, `×`, `:`, `⊗`).  Objects here have **no free indices**.
- **Coordinate / symbol layer:** arrays carrying coordinate indices; the index
  algebra (contraction, dummy relabeling, Kronecker/LC symbols, the rewrite
  steps in `derivation.cpp`).

The **bridge** between them is the basis: a coordinate is the contraction of an
invariant with (co)basis vectors, and an invariant is reassembled from its
coordinates and the basis polyads.  `unroll_sums` (coordinate index → concrete
value) is one half of the coordinate-side machinery; the invariant↔coordinate
expansion is the half that is missing.

## 6. Where attempt 2 sits — and the direction (answer to "one type or two?")

The reason to build yet another CAS at all: existing systems do **not** treat
tensors as invariant objects.  Attempt 2's purpose is to use invariant (direct)
notation as far as possible.  But the system must *also* work correctly at the
coordinate level — so we start from indices.

So it is **not** "invariant XOR coordinate."  Both layers are first-class:

1. Coordinate / symbol layer — exists today (index algebra, eps-delta, the
   derivation steps).
2. Basis as the index-emitting bridge — the gap.  It owns `(space, realm)` and
   produces indexed objects (its vectors, cobasis, and the coordinates derived
   through them), all inheriting `space`/`realm` from it.  Attempt 1 had this
   (`basis.hpp`, `coord_system.hpp`); not yet ported to attempt 2.
3. Invariant / polyadic layer on top — the differentiator, built once the basis
   bridge exists.

Keep the **symbol** (basis-free arrays: permutation symbol, Kronecker delta)
typed distinctly from **tensor coordinates** (basis-dependent), so the name of an
object stops contradicting its meaning.

## Implications / next steps (not done here)

- Introduce a `Basis` abstraction owning `(space cardinality, realm)` and
  emitting indexed objects; have indices inherit `space`/`realm` from it rather
  than being passed ad hoc at construction.
- In `TensorTraits`, distinguish "symbol" (basis-free) from "tensor coordinate"
  (basis-dependent).  Until the basis layer lands, treat the current
  `LeviCivita` / `Delta` `TensorObject`s honestly as **symbols**.
- Revisit `Realm::Oblique` on the LC/Delta objects: with no metric present, no
  oblique coordinates are actually being computed — it is the bare symbol
  regardless of realm.  This is harmless for the symbol-level identities but must
  not be mistaken for genuine oblique-basis coordinates.
