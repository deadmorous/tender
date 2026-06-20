# 000049 Vector bases, coordinate systems, and the basis bridge

Stage 3 (bases) and Stage 4 (well-known coordinate systems) of the roadmap
(vibe 000038), designed together because Stage 4 is just "named bases." This is
a **model document** in the spirit of vibe 000036 — it commits the ontology and
the operation set; no code is written here. Implementation lands incrementally
afterward (order in the last section).

## 0. The decision that shapes everything

`expand-in-basis` and `fold-back` are **manual, basis-parameterized derivation
steps**, invoked explicitly by the user, each given the `Basis` to work against.
They are *not* e-graph saturation rules and they do *not* require the general
subtree-variable matcher deferred in vibe 000048.

The reason fold-back does not need a subtree variable: the step is **handed the
basis**, so it has the concrete `eᵢ`/`eⁱ` to scan for. Recognition is a
structural match against *known* vectors; whatever multiplies a recognized
polyad is the coordinate, read off positionally — no metavariable binding an
arbitrary subtree. This is what "for rank ≥ 2 the vector basis or cobasis is
always enough" means: the polyads are homogeneous (all-basis or all-cobasis), so
the step never has to discover an unknown mix.

Consequences, all deliberate:

- **expand is generative, kept on-demand.** It multiplies an invariant out into
  a sum over the basis — explosive if run inside saturation, so it stays a step
  the user calls when they want it, not a rule the engine fires automatically.
- **fold-back failure is a no-op.** If the scanned expression is not a clean
  combination of this basis's polyads, the step returns it unchanged (as the
  attempt-1 `reassemble_*` steps did). Composable and safe to over-apply.
- The general matcher / `Theorem` type (vibes 000033, 000048) stays deferred —
  this design does not need it. It returns when a *pattern*-shaped theorem with
  no basis to key on appears (bac-cab, `a×I×b`).

## 1. The `Basis` abstraction (Stage 3)

A `Basis` is a C++ object (not an `Expr` node — a basis is not an expression, it
is the source that *emits* indexed expressions, per vibe 000036 §3). It is built
from a **tuple of vectors** (rank-1 `Expr`s) and owns:

- its **realm** (`Oblique` or `Orthonormal`) — governs level sensitivity;
- the **index space** it emits, whose cardinality = the number of vectors.

A basis with fewer vectors than the ambient dimension is a **subspace basis**;
its emitted index space simply ranges over fewer values. We do **not** check the
count against an ambient dimension, because tender does not yet model the space a
vector lives in (an explicit scope call). Revisit when that notion exists.

Per vibe 000036 §3, every index the basis emits inherits its `(space, realm)` —
this is the long-promised replacement for passing `realm`/`space` by hand at
`make_levi_civita(ctx, Realm::Oblique, space, …)`.

### Three construction flavors

The tuple you pass *is* the basis; the flavor says how to read it and how to get
the dual:

| flavor | given tuple is | dual derived as |
|--------|----------------|-----------------|
| **orthonormal** | unit frame `eᵢ` | `eⁱ = eᵢ` (coincide) |
| **covariant** | `eᵢ` (covariant) | cobasis `eⁱ` from the metric |
| **contravariant** | `eⁱ` (contravariant) | basis `eᵢ` from the metric |

The dual derivation for the oblique flavors is the 3-D cross-product formula
(attic `coord_system.hpp`): `e¹ = (e₂×e₃)/V`, `V = e₁·(e₂×e₃)`, etc., or the
metric inverse in general. It is only exercised once an oblique basis exists; all
*well-known* systems below are orthonormal, so this turn's first slice need only
implement the orthonormal flavor (dual = self).

### What a basis emits

- `basis(i)` → `eᵢ` (covariant member, rank 1);
- `cobasis(i)` → `eⁱ` (contravariant member, rank 1);
- symbolic indexed forms `e` with a fresh `CountableIndex` over its space, for
  building expansions that Einstein-sum;
- (later) the metric `g_{ij}` and volume `√g`. For orthonormal bases `g_{ij}=δ`
  and `√g=1`, so the metric machinery is deferred until an oblique basis needs
  it — at which point the ε-**tensor** vs ε-**symbol** split of vibe 000036 §4
  (`ε^{ijk} = √g·[ijk]`) also becomes real. Today every well-known case has
  `√g = 1`, so the existing `LeviCivita` object stays honest as the symbol.

## 2. Coordinate systems (Stage 4)

A `CoordSystem` **produces** a `Basis` (aggregation, not inheritance —
CLAUDE.md principle 8: a CS *has* a basis, it is not one). Well-known systems:

- **WCS** (world Cartesian) — orthonormal frame `i, j, k`.
- **Cylindrical** `(r, θ, z)` — orthonormal frame `e_r, e_θ, e_z`.
- **Spherical** `(r, θ, φ)` — orthonormal frame `e_r, e_θ, e_φ`.
- **2-D polar** `(r, θ)` — orthonormal frame `e_r, e_θ` (a cardinality-2 basis).

All are returned as **orthonormal** bases: the symbolic algebra treats the frame
vectors as orthonormal unit vectors *at a point*. The fact that the cylindrical/
spherical frames rotate with position (so `∂e_r/∂θ ≠ 0`) is a **Stage 5**
concern — Christoffel symbols and covariant derivatives — and does not enter the
point-wise basis algebra here. Scale factors / coordinate parameters (attic
stored `r`, `θ`, … as `Parameter`s) are likewise deferred to Stage 5; Stage 4's
deliverable is the named orthonormal frame each CS hands back.

## 3. The two operations

### expand-in-basis (generative step)

`expand_in_basis(basis, variance)` rewrites an invariant `A` (slot-less, rank
`r`) into a sum of `coordinate ⊗ polyad`:

```
covariant:      A = Σ  A^{i…}  ⊗  (e_i ⊗ …)        (coords upper, basis lower)
contravariant:  A = Σ  A_{i…}  ⊗  (e^i ⊗ …)        (coords lower, cobasis upper)
```

with `r` dummy indices Einstein-summed (the implicit-summation pass, vibe 000041,
materializes them). The **coordinate** `A^{i…}` is a fully-indexed `TensorObject`
— `is_component_valued` already classifies it as a scalar coordinate
(derivation.cpp:855, the vibe-36 coordinate/invariant line). The **polyad** is a
`TensorProduct` of basis (or cobasis) members. Variance is a single global choice
per expansion — covariant *or* contravariant; per the §0 decision we do **not**
emit mixed-variance polyads, since basis-or-cobasis alone suffices for rank ≥ 2.
(Mixed co/contra coordinates remain possible later if a use appears, but are not
built now.) For an orthonormal basis the two variances coincide.

Special case: a rank-2 well-known like the **identity** expands diagonally,
`I = Σᵢ eᵢ ⊗ eⁱ` (orthonormal: `Σᵢ eᵢ ⊗ eᵢ`) — one sum, not two, because `I` is
`δ`.

### simplify-basis-dot (bridge step)

`eᵢ · eʲ → δᵢʲ` (orthonormal: a Kronecker delta, lower-lower per the
Orthonormal-lower convention of vibe 000047). This is the hinge that turns an
**invariant** product of polyads into an **index**-algebra expression, handing
off to the machinery that already exists — distribution (vibe 000048), δ-
contraction, symmetry-canonicalization (vibe 000047). It also pulls rank-0
coordinate factors out of the contraction so `(s·v)` with scalar `s` becomes
`s·(v-contraction)`.

### fold-back / reassemble (recognition step)

`reassemble(basis, variance)` is the inverse: scan for `Σ (coordinate ⊗ polyad)`
where the polyad is built from *this basis's* members, and fold it to the named
invariant. Because the basis is supplied, the scan compares polyad factors
directly against `basis.basis(i)` / `basis.cobasis(i)`; the coordinate is
whatever remains. **Failure is a no-op** — an unrecognized expression passes
through unchanged. Recognizes at least:

- rank 2 diagonal: `Σᵢ eᵢ ⊗ eⁱ → I`;
- rank 1: `Σᵢ a^i eᵢ → a` (and the contravariant mirror);
- a dot of expansions: `(covariant expansion)·(contravariant expansion) → a·b`.

## 4. Worked examples (feasibility targets)

### (a) Identity round trip

```
I  --expand_in_basis(WCS)-->  Σᵢ eᵢ ⊗ eᵢ  --reassemble(WCS)-->  I
```

The rank-2 round trip; the smallest end-to-end proof that expand and fold-back
are inverse. A natural first integration test.

### (b) `a·b = b·a` from first principles (the attic motivating example)

```
a·b
 = (aⁱ eᵢ) · (bʲ eⱼ)              expand_in_basis(WCS) on a and b
 = aⁱ bʲ (eᵢ · eⱼ)               distribution + pull scalar coordinates out
 = aⁱ bʲ δᵢⱼ                      simplify_basis_dot
 = aⁱ bᵢ                          δ-contraction (existing engine)
 = bᵢ aⁱ                          coordinates are scalars — commute
 = b·a                            reassemble(WCS)
```

Every `=` is one (or a few) steps, and the **middle three** reuse machinery that
already exists. This is the concrete demonstration that the basis bridge plus the
Stage-2 engine composes into an invariant-level identity proof — the Stage-3
payoff.

## 5. Representation decisions

- **No new node types.** Coordinate vs invariant stays the existing slot-fill
  convention (`is_component_valued`): invariant = slot-less rank ≥ 1; coordinate
  = fully-indexed `TensorObject`; the symbol (`δ`, `ε`) is unchanged. Vibe
  000036 §4 wanted symbol ≠ coordinate ≠ invariant *typed* distinctly; we honor
  it operationally without a type split — and the still-missing piece (the
  `√g`-weighted ε-*tensor*) only matters once an oblique metric exists, so it is
  deferred with the metric.
- **Basis identity for fold-back** needs no tagging on the polyad: the step is
  given the basis and compares against its members directly.
- **Coordinates are indexed objects, not string-named.** Attempt 1 minted
  components as named scalars `"a^1"`, `"a^2"`. Attempt 2 uses the indexed
  `TensorObject` `a^i` with a `CountableIndex`, unrolled to concrete values via
  `unroll_sums` only when numbers are actually wanted — consistent with vibe
  000036 and the current engine.

## 6. What this needs from the engine

Nothing new structurally. Implicit Einstein summation materializes the expansion
sums; distribution, δ-contraction, and symmetry-canonicalization carry the middle
of example (b); the steps themselves are basis-parameterized rewrites. The
general subtree matcher / `Theorem` type is explicitly **not** required here —
its absence is the reason the manual-step design is the small, alive move.

## 7. Deferred (not in Stage 3/4)

- General (oblique) metric `g_{ij}`, `√g`, the derived-dual cross-product/metric-
  inverse formula, and the ε-tensor-vs-symbol split (vibe 000036 §4).
- Mixed co/contra polyads and mixed coordinates.
- Position-dependent frames: `∂eᵢ/∂xʲ`, Christoffel symbols, covariant derivative,
  coordinate `Parameter`s — Stage 5.
- The general subtree-variable matcher / `Theorem` machinery — until a
  basis-free pattern theorem (bac-cab) needs it.

## 8. Implementation order (later slices)

1. `Basis` (vector tuple → realm + emitted index space; `basis`/`cobasis`;
   orthonormal flavor first, dual = self).
2. `expand_in_basis` step (covariant + contravariant; orthonormal).
3. `simplify_basis_dot` step (`eᵢ·eʲ → δ`).
4. `reassemble` step (no-op on failure) + the identity round-trip test (4a).
5. Well-known coordinate systems: WCS first, then cylindrical / spherical /
   2-D polar (orthonormal frames).
6. The `a·b = b·a` derivation (4b) as a maintained feasibility example
   (CLAUDE.md principle 5).

## Implemented

All slices landed (orthonormal flavor), each its own commit, on the existing
engine with **no new node types** and without the deferred subtree matcher.

- **`Basis`** (`basis.{hpp,cpp}`) — built from a rank-1 vector tuple via
  `make_orthonormal_basis(space, vectors, symbol="e")`; owns realm + emitted
  index space (cardinality == #vectors, subspace allowed, no ambient check).
  Exposes `basis(i)`/`cobasis(i)` (cobasis = basis for orthonormal) and the
  **symbolic emission** `covariant_vector`/`contravariant_vector` — rank-1
  TensorObjects with the generic symbol and a CountableIndex (orthonormal spells
  both lower).
- **`expand_in_basis`** — a generic invariant (slot-less, rank r, not
  well-known) → `A^{i…} ⊗ (e_i ⊗ …)` with the r indices left as an implicit
  Einstein sum (canonicalize materializes it; coordinate level chosen so the
  shared index contracts). Walks the tree; well-known and already-indexed
  objects untouched.
- **`simplify_basis_dot`** — `(s e_i)·(t e_j) → s ⊗ t ⊗ δ_{ij}`, pulling
  coordinate factors out of the contraction; the δ then feeds the existing
  contraction machinery.
- **`reassemble`** — inverse recognition: peels nested ExplicitSums, splits the
  body into one coordinate + a polyad of the basis's vectors, checks the summed
  indices pair up, and folds to the slot-less invariant; **failure is a no-op**.
  Proven inverse to expand for rank 1 and 2 (covariant + contravariant).
- **Coordinate systems** (`coord_system.{hpp,cpp}`) — `wcs`, `cylindrical`,
  `spherical`, `polar_2d`, each a free factory **producing** an orthonormal
  `Basis` (no metric/coords yet, so no `CoordSystem` class). Frame vectors named
  within the TensorName grammar (`i,j,k`; `r,\theta,\phi,z`).
- **Feasibility** (`basis_feasibility_test.cpp`) — the rank-2 round trip and
  `a·b = b·a` reduced through the basis to `Σ_i a_i b_i` (the δ contracts via
  the concrete `unroll_sums → eval_delta_concrete → fold` path; the existing
  canonicalizer commutes the scalar coordinates).

One refinement vs. the plan: the `a·b` contraction uses the concrete-unroll
path rather than a symbolic δ-substitution, which remains the parametric-RHS
gap (vibes 000033/000040). ~30 C++ tests across `basis_test`,
`coord_system_test`, `basis_feasibility_test`; full suite 443 green.

Still deferred: oblique flavor + metric/√g + ε-tensor-vs-symbol; symbolic
δ-substitution; Stage 5 (position-dependent frames, Christoffel, covariant
derivative).

### Follow-ups landed after the initial slices

- **Python bindings** (`tender.basis`): `Basis`, the coordinate systems, the
  `Variance` enum, and the three steps, mirroring `tender.derivation`; plus the
  `basis_dot_product.{py,ipynb}` example (a·b = b·a).
- **Per-slot variance.** `expand_in_basis` takes a `std::vector<Variance>` (one
  per slot) so a rank-≥2 tensor can get **mixed** coordinates `A^i{}_j e_i e^j`,
  not only all-co/all-contra; a single-element list broadcasts to any rank
  (preserving the uniform `Variance` convenience overload) and a length that is
  neither 1 nor the tensor rank throws (no silent misapplication). Under an
  orthonormal basis the two variances coincide, so the choice is observable only
  once the oblique flavor lands — the API is now the right shape for it.
  `reassemble` matches polyad vectors by symbol (not variance/level), so mixed
  expansions still fold back.
- **Identity expansion round trip.** The identity is no longer skipped: its
  coordinate follows from the defining property `I·a = a`, since
  `I^i_j = e^i·I·e_j = e^i·e_j = δ^i_j`, so `I` resolves to `Σ_i e_i ⊗ e^i`.
  `expand_in_basis` emits that directly for the identity (an intrinsically mixed
  resolution — the variance argument is not consulted; the pure-variance metric
  forms `g_ij` await oblique), and `reassemble` gains the inverse branch (a sum
  of two basis vectors sharing one summed index, no coordinate, → `I`). This is
  the genuine `I → Σ_i e_i ⊗ e^i → I` round trip on orthonormal bases — done
  *without* a hardcoded δ coordinate, which would have needed the symbolic
  δ-substitution we still lack. Supporting pieces: a general `contract_identity`
  step (`I·x → x`, `x·I → x`) and an `Expr.rank` accessor in Python.
