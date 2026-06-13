# Indices

## Index realms

Every index slot on a tensor object belongs to one of four realms.
The realm determines whether upper/lower position matters, when
Einstein summation applies, and whether summation can occur at all.

### Oblique

Used for covariant and contravariant objects (coordinates, basis
vectors) in a general, possibly non-orthonormal basis.

- Upper and lower positions are distinct (contravariant vs. covariant).
- Einstein summation applies when the **same index appears exactly once
  at an upper slot and once at a lower slot** within the same term.
- More than two occurrences of the same index in a term is an error,
  unless summation is explicitly required or denied at that slot.

### Orthonormal

Used when the basis is orthonormal: covariant and contravariant
coincide, so upper/lower distinction carries no algebraic meaning.

- Upper and lower positions are interchangeable; users may choose
  either for stylistic or readability reasons.
- Einstein summation applies when the **same index appears exactly
  twice** within the same term (at any combination of levels).
- More than two occurrences is an error, unless explicitly overridden.

### Collection

Used to number objects within a collection (e.g. FEM shape functions,
nodes, elements).

- May appear at any level.
- **No automatic summation.** Explicit summation must be requested.
- Any number of occurrences of the same collection index is allowed.

### Label

Used to distinguish an object from others by a descriptive word (e.g.
`vol`, `surf`, `ref`).  Carries no algebraic meaning.

- May appear at any level.
- No summation, no occurrence restrictions.
- Only a label-type index (identified by string, not by number) may
  be attached to a label-realm slot.

---

## Index spaces

An index space defines the set of values that an index ranges over
when a summation is performed.  Index spaces apply to the three
countable realms (Oblique, Orthonormal, Collection); label slots have
no space.

Two kinds of index space exist:

- **Well-known (global) spaces** — predefined singletons importable by
  name, e.g. `Space3D` (Orthonormal, `{1,2,3}`), `Space2D`
  (Orthonormal, `{1,2}`), `Space4D` (Oblique, `{0,1,2,3}`).
  These are the common case and require no user construction.

- **User-defined spaces** — constructed explicitly for a specific
  problem, e.g. a FEM problem with N shape functions.  Each
  construction produces a distinct instance.

**Compatibility rule:** a given index instance may only be attached to
slots whose realm and index space instance both match.  Attaching the
same index to a slot in `Space3D` and a slot in `Space2D` is an error,
even if both are Orthonormal.

An index space also owns the **naming schema** used to generate dummy
index names at render time (see §Index naming below).  For example,
a 3D spatial Orthonormal space uses Latin letters `i, j, k, l, …`;
a 4D Oblique space uses Greek letters `μ, ν, ρ, σ, …`.  Keeping the
naming policy in the space (rather than in the renderer) means the
renderer is policy-free.

### Parametric cardinality

Some index spaces have an abstract, symbolically-specified cardinality.
The canonical example is a Ritz/FEM problem with N shape functions,
where N is unknown at derivation time:

```
u = Σ_{i=1}^{N} uᵢ φᵢ
```

Here `i` is a Collection index over a space whose value set is
`{1, 2, …, N}` for some symbolic scalar `N` that will only be
instantiated when a concrete problem is solved.

**Design decision (Option A):** `IndexSpace` always stores a concrete,
explicit value set (`std::vector<int>`).  It does not carry a symbolic
count.  For a parametric space:

- The `IndexSpace` stores only the **naming schema** and carries an
  **empty value set**, signalling "cardinality is not known concretely".
- The actual symbolic upper bound `N` is a scalar expression in the
  **expression tree** — specifically, it is the second argument of the
  `ExplicitSum` node (see §Explicit override below).
- Whenever the derivation reaches a step requiring concrete evaluation
  (sum expansion, numerical substitution), the user substitutes a
  concrete value for `N` in the expression tree.  The `IndexSpace` object
  itself never changes.

This keeps `IndexSpace` structurally simple and avoids any circular
dependency between index spaces and expression nodes.  The expression
tree is the single source of truth for all symbolic algebra; index spaces
are identity tokens and naming policies, nothing more.

An `IndexSpace` with an empty value set may carry an optional
**display hint** — a short string such as `"N"` — used by the renderer
to emit `Σ_{i=1}^{N}` when no explicit bound expression is available.
This hint is cosmetic only and has no algebraic role.

---

## Index slots on tensor objects

A tensor object carries an ordered list of **index slots** at each
level (upper and lower).  Two kinds of slot exist:

### Void slot

A positional cell that holds no index and cannot have one attached.
Its only purpose is to fix the positions of other slots across levels.
Example: `A_{\cdot j k}^{i \cdot \cdot}` has void slots at lower
position 1 and upper positions 2 and 3.

### Index slot

A positional cell that **must** have an index (or a concrete value)
attached before the expression is considered valid.  Every index slot
carries:

- **Realm** — one of Oblique, Orthonormal, Collection, Label.
- **Space** — the index space for the slot (absent for Label realm).

An expression is valid only when every index slot in every tensor
object in that expression has an associated index or concrete value.

---

## Index character: dummy, free, contracted

All countable (non-label) indices are **dummy**: renaming an index
everywhere it appears in a term yields an identical expression.  This
holds whether the index is free or contracted.

- **Free index** — appears exactly once in a term; not summed over
  within that term.  It remains as an open slot in the term's result.
- **Contracted index** — appears twice in a term and satisfies the
  Einstein rule of its realm; summed over within that term.

The free/contracted distinction is local to a single `Term` node.  The
same index id may appear as contracted in one term of a sum and free in
another — but doing so is a rank mismatch error (see §Cross-term
occurrences below).

---

## Indices and concrete values

An index slot may be associated with either:

1. **A countable index** — a symbolic placeholder identified by a
   numerical id (not a name; names are assigned only at render time).
   Summation rules apply according to the slot's realm.

2. **A concrete index value** — a specific integer drawn from the slot's
   index space.  No summation applies to a slot carrying a concrete
   value.

---

## Index identification

| Realm                            | Identity carrier            |
|----------------------------------|-----------------------------|
| Oblique, Orthonormal, Collection | Numerical id (integer)      |
| Label                            | String name (the label itself) |

Two index instances are the **same index** when they share the same
numerical id (for countable realms) or the same string (for Label).
The numerical id is opaque to users; it is an implementation detail
of the expression tree.

---

## Cross-term index occurrences

Within a single `Term` node the Einstein rules apply directly.  When
the same index id appears across multiple `Term` nodes in a sum, the
following rules apply.

**Sum validity — free-slot structure must match.**  For a sum of terms
to be valid, every term must have the same **structural pattern** of
free index slots: the same count at each level, and the same realm and
index space at each positional slot.  The specific index ids assigned
to corresponding positions may differ — all countable indices are dummy
and can be renamed.

This is precisely what is needed to express, for example, the
symmetrized tensor:

```
S = (A + transpose(A)) / 2
S[i,j] = (A[i,j] + A[j,i]) / 2
```

Both terms are rank-2 with two free lower slots sharing the same realm
and space.  The first term puts id `i` in slot 1 and `j` in slot 2; the
second puts `j` in slot 1 and `i` in slot 2.  The sum is valid; at each
component position `(i,j)` it evaluates to `A_{ij} + A_{ji}`.

**Case table:**

| Case | Example | Verdict |
|------|---------|---------|
| Free in all terms, same structural pattern, ids may vary | `A[i,j] + A[j,i]` | **Valid** |
| Contracted independently in multiple terms | `A[i]*B[i] + C[i]*D[i]` | **Valid** — independent summations; id reuse is harmless |
| Free in all terms, structure matches, both ids identical | `A[i] + B[i]` | **Valid** |
| Free in one term, contracted in another | `A[i] + B[i]*C[i]` | **Error** — rank mismatch (rank-1 + rank-0) |
| Free-slot structures differ (count or realm/space mismatch) | `A[i] + B[j,k]` | **Error** — rank or type mismatch |

### α-equivalence and canonical form

Because contracted indices are term-local and all countable indices are
dummy, two expressions that differ only in the choice of contracted
index ids within terms are **semantically identical**.  Example:

```
A[i,i] + B[i,i]   ≡   A[i,i] + B[j,j]
```

In the first expression the user happened to reuse id `i` for the
contraction in the second term; in the second they chose a fresh id `j`.
Both express `tr(A) + tr(B)`.

**Consequence for equality, simplification, and pattern matching:**
expressions must never be compared by raw id values.  Instead a
**canonical form** is computed first:

1. Walk the expression tree in a fixed deterministic order.
2. For each `Term`, rename its contracted indices to fresh canonical ids
   (`c0, c1, …`) in order of first appearance within that term,
   discarding the original ids.
3. Free indices are shared across terms of a sum; rename them
   consistently across the entire expression, again by first-appearance
   order.

After canonicalisation both expressions above become `A[c0,c0] + B[c0,c0]`.

**Design constraint:** the C++ index id is an **opaque internal handle**
whose numeric value carries no meaning outside the expression tree.
Renaming (alpha-conversion) must always be possible, and no code outside
the canonical-form pass should compare ids for equality directly.

---

## Index naming

Dummy indices carry no human-readable name internally.  At the
**pre-rendering enrichment stage** (see vibe 000020, phase 13.7),
each dummy index is assigned a display name drawn from the naming
schema of its index space.  Rules:

- Names are assigned consistently: if two slots in an expression share
  the same dummy index id, they receive the same display name.
- The schema is defined per index space (e.g. `i, j, k, …` for 3D
  spatial).
- Label-realm indices carry their string directly as a display name;
  no enrichment step is needed for them.

---

## Summation rules

### Implicit rule (Einstein convention)

Every index occurrence in a term is either **contracted** (summed over)
or **free** (an open slot of the result) — there is no unspecified
state.  The implicit rule decides which:

| Realm        | Auto-contraction trigger               | Otherwise      | Error (no override) |
|--------------|----------------------------------------|----------------|---------------------|
| Oblique      | Same id: exactly one upper + one lower | Free           | Same id same level; ≥ 3 total |
| Orthonormal  | Same id: exactly two slots (any level) | Free           | ≥ 3 occurrences     |
| Collection   | Never                                  | Always free    | —                   |
| Label        | Never                                  | Always free    | —                   |

### Explicit override

Explicit annotations always take precedence over the implicit rule.
Two annotations are available:

- **`ExplicitSum(index, body)`** — force summation over `index` in
  `body` regardless of realm or occurrence count.  Required whenever
  the implicit rule cannot apply: Collection realm, or ≥ 3 occurrences
  of the same id in a term.  The summation range is taken from the
  concrete value set of the index's space.

- **`ExplicitSum(index, body, N_expr)`** — same, but the summation
  range is `{1, …, N_expr}` where `N_expr` is a symbolic scalar
  expression.  Required when the index belongs to a parametric-
  cardinality space (empty value set).  `N_expr` is an ordinary node
  in the expression tree and participates in all symbolic operations
  (substitution, simplification, etc.).

- **`NoSum(index)`** — suppress summation for this index even when the
  implicit rule would contract it.  Makes the index free.

### Example: spectral decomposition

A symmetric tensor expressed in its eigenbasis:

```
A = ExplicitSum(i,  a[i] * v[i] * v[i] )
```

Here `i` is a Collection index (ordinal of eigenvalue/eigenvector)
over a space with a concrete value set (say `{1, 2, 3}` for a 3-mode
truncation).  It appears three times in the term; Collection realm has
no implicit summation.  `ExplicitSum` both enables summation and
resolves the three-occurrence case.

When the number of eigenmodes is symbolic (e.g. the first `N` modes):

```
A = ExplicitSum(i,  a[i] * v[i] * v[i],  N )
```

The index space for `i` has an empty value set; `N` is the symbolic
scalar expression giving the upper bound.

To select a single eigenmode without summing:

```
A_k = NoSum(i,  a[i] * v[i] * v[i] )  with i bound to concrete value k
```

or equivalently, substitute a concrete index value for `i` directly
(concrete substitution suppresses summation automatically — see
§Indices and concrete values).

---

## C++ type system sketch (first iteration)

The full design will be refined in later iterations.  High-level
structure:

```cpp
enum class Realm { Oblique, Orthonormal, Collection, Label };
enum class Level { Upper, Lower };

// Index space: value range + naming schema.
// Concrete type to be designed; holds the value set and a letter
// sequence for dummy name generation.
// Identity is by instance pointer (well-known spaces are singletons;
// user-defined spaces are separate instances even if their value sets
// happen to coincide).
struct IndexSpace { /* … */ };

// A slot on a tensor object.
struct VoidSlot { Level level; };

struct IndexSlot {
    Level      level;
    Realm      realm;
    IndexSpace space;   // absent / sentinel for Label realm
};

using Slot = std::variant<VoidSlot, IndexSlot>;

// An index associated with an IndexSlot.
struct CountableIndex { int id; };        // dummy; free or contracted depending on context
struct ConcreteIndex  { int value; };     // no summation
struct LabelIndex     { IndexName name; };// label realm only

using IndexAssoc = std::variant<CountableIndex, ConcreteIndex, LabelIndex>;
```

Strong types from vibe 000027 (`TensorName`, `IndexName`) are used
wherever names appear.  Further iterations will address: how slots are
stored on a `TensorObject` node, how associations are stored in the
expression tree, and how the summation-detection pass walks the tree.

---

## Sum expansion as the foundational evaluation mechanism

### The question

How is an identity like `δ_{ii} = 3` (in 3D) established?  Does it
require unrolling the sum and performing rational arithmetic, or are
there better alternatives?

### Sum expansion

Unrolling is a universal, first-principles capability:

1. Recognise `i` is contracted over {1, 2, 3}.
2. Expand: `δ_{11} + δ_{22} + δ_{33}`.
3. Apply the definition of δ at each concrete index pair: `1 + 1 + 1`.
4. Rational arithmetic: `3`.

Any tensor with a computable definition (δ, ε, metric components, …)
over a finite index space can be fully evaluated this way, with no
special-case knowledge about any particular tensor.  The engine only
needs to know how to expand a contracted sum and evaluate a tensor at
concrete index values.  This produces human-readable derivation steps
and is auditable at every stage.

### Supplementary theorem library

For expressions with many contracted indices, unrolling can be
impractical as a display strategy even when it is computationally
feasible.  `ε_{ijk} ε_{ijk} = 6` visits 3³ = 27 index combinations;
more complex contractions are worse.  Additionally, sum expansion
requires a concrete finite index space — it cannot prove `δ_{ii} = n`
for symbolic dimension `n`.

A theorem library provides named single-step rewrites for established
results, making derivations readable at the mathematical level.

### The synthesis: two complementary layers

Sum expansion and the theorem library are not competing approaches;
they occupy different layers:

- **Axiomatic layer (sum expansion):** proves any concrete identity
  from first principles.  This is how theorems in the library are
  *established* — `δ_{ii} = 3` is proved once by expansion, then
  recorded.

- **Efficiency layer (theorem library):** provides single-step rewrites
  for named results.  Each entry carries a proof trace that is itself
  a sum-expansion derivation (or a chain of previously proved
  theorems), making the entire system auditable.

Sum expansion must therefore be a first-class, general-purpose
capability in the engine — not a last resort.  The workflow is: prove
by expansion, record as a theorem, quote freely in subsequent
derivations.
