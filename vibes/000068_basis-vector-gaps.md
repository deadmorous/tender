# 000068 — Basis-vector gaps found by experiment

Collecting problems found while experimenting with basis vectors / the basis
layer (vibe 000067 having just landed).  Each problem is a short sequence that
replaces `# STEPS` in this single-cell notebook harness:

```python
import tender
import tender.basis as tb
import tender.derivation as td
from IPython.display import Math, display

def disp(*exprs):
    for e in exprs:
        display(Math(e.latex()))

ctx = tender.Context()

I = tender.identity(ctx)
cs = tb.wcs(ctx)

x = I
# STEPS
disp(x)
```

Status: **collecting** — recording problems first, addressing later.

## Problems

<!-- Each problem: the STEPS sequence, what happens, what's expected, notes. -->

### P1 — `simplify_basis_dot` no-op when dotting an expansion with a concrete frame vector `cs.basis(k)`

```python
x = tb.expand_in_basis(x, cs, tb.Variance.Covariant)   # I → e_i e_i  (dyad)
x = x @ cs.basis(0)                                     # · the 0th frame vector
x = tb.simplify_basis_dot(x, cs)                        # expected to reduce
```

**Actual:** the last step is a no-op.  After the dot the term is
`e_i e_i · 𝐢` (canonicalizes to `Σ_i (e_i · 𝐢) e_i`) and stays that way.

**Expected:** `e^i · e_1 → δ^i_1`, giving `e_i δ_{i1}` (which then sums to `e_1`).

**Diagnosis.** `cs.basis(0)` returns the *concrete frame vector* — a bare
`TensorObject` literally named `i` (the WCS vector symbol), no index slot,
`basis_id 0`.  That representation is disconnected from the symbolic `e`/index
algebra:

- `simplify_basis_dot`'s `as_basis_vector` matcher requires `name ==
  cs.vector_symbol()` (`"e"`), exactly one slot, and a **`CountableIndex`**
  tagged with the basis id.  The frame vector `𝐢` matches none of these, so the
  dot `e^i · 𝐢` is never recognized as a basis–basis dot.
- Even if `cs.basis(0)` instead returned `e` with a `ConcreteIndex{1}` tagged
  with the basis (the vibe-000067-consistent form that renders as `𝐢` via
  `vector_symbols`), `as_basis_vector` still rejects it — it special-cases
  `CountableIndex` and ignores `ConcreteIndex`.  So the dot-against-a-concrete-
  direction `e^i · e_1 → δ^i_1` path does not exist at all.

Two coupled gaps: (a) the concrete basis vectors `cs.basis(k)` are a separate
named representation, not `e` + concrete-index-tagged-with-basis_id, so nothing
bridges them to the contraction machinery; (b) the basis-dot machinery handles
only dummy (`CountableIndex`) basis vectors, not concrete (`ConcreteIndex`)
ones, so a mixed dummy/concrete `δ^i_1` is never produced.

### P2 — `reassemble` doesn't do the completeness fold; you must reach for `reassemble_completeness`

```python
x = tb.expand_in_basis(x, cs, tb.Variance.Covariant)   # I → e_i e_i
x = x @ cs.basis(0)                                     # · e_1   (= I·e_1 = e_1)
x = tb.reassemble(x, cs)                                # expected to fold
```

**Actual:** `reassemble` only self-preps — it strips the `Σ` and reorders to
`(e_i · 𝐢) e_i`, no fold.

**Expected:** `I · e_1 = e_1`, i.e. the term reduces to `𝐢` (= `cs.basis(0)`).

**Diagnosis.** The pattern here is the resolution of identity *with a
contraction*, `Σ_i (X · e_i) e_i → X` (here `X = 𝐢`).  That fold is implemented
only in **`reassemble_completeness`**, a separate public step — and indeed
`tb.reassemble_completeness(x, cs)` returns `𝐢` correctly.  `reassemble` itself
handles only the coordinate-carrier folds (`Σ a_i e_i → a`, trace, bilinear,
…) and never invokes the completeness fold, so the plain `reassemble` call a
user naturally reaches for leaves the term unfolded.

This is an API/composition gap (two reassembly entry points; the obvious one is
incomplete), not a representation gap — the concrete frame vector `𝐢` rides
through fine as the opaque `X`.  Candidate fix: have `reassemble` also run the
completeness fold (or fold a single "reassemble everything" entry point), so one
call finishes.

### P3 — `simplify_basis_dot` is a no-op on unrolled (concrete-index) basis vectors

```python
x = tb.expand_in_basis(x, cs, tb.Variance.Covariant)   # I → e_i e_i
x = x @ cs.basis(0)                                     # · e_1
x = td.unroll_sums(x)                                   # → 𝐢(𝐢·𝐢) + 𝐣(𝐣·𝐢) + 𝐤(𝐤·𝐢)
x = tb.simplify_basis_dot(x, cs)                        # expected to reduce dots
```

**Actual:** no-op.  The unrolled term `𝐢 𝐢·𝐢 + 𝐣 𝐣·𝐢 + 𝐤 𝐤·𝐢` is unchanged.

**Expected:** `𝐢·𝐢 = 1`, `𝐣·𝐢 = 𝐤·𝐢 = 0`, so the whole thing collapses to `𝐢`
(= `e_1`).

**Diagnosis.** After `unroll_sums` the basis vectors carry **`ConcreteIndex`**
(1, 2, 3, rendered `𝐢, 𝐣, 𝐤` via `vector_symbols`).  `simplify_basis_dot`'s
`as_basis_vector` only accepts `CountableIndex` (dummy) basis vectors, so it
matches none of them — the concrete-direction dot path doesn't exist (same root
as [[P1]]'s gap (b)).  Compounding it, the right operand is the frame vector
`cs.basis(0)` (name `i`, `basis_id 0`, no slot, [[P1]] gap (a)): so a dot like
`𝐢·𝐢` is actually `e_1(concrete, tagged cs, symbol "e")  ·  𝐢(frame vector,
name "i", untagged)` — **two different objects that render identically**.  Even
a concrete-aware dot rule keyed on structural equality would not see them as the
same direction without comparing the underlying value+basis.

There is a concrete-evaluation path (`eval_delta_concrete` / `eval_eps_concrete`)
but it needs the dots turned into δ's first — which is exactly what the missing
concrete `simplify_basis_dot` would produce (`e_1 · e_1 → δ_{11}` → 1).

## Triage / resolution

### The common theme

A basis direction has **three representations** and the machinery only speaks
one of them:

1. symbolic dummy `e_i` — `e` + `CountableIndex`, tagged with `basis_id`
   (what `expand_in_basis` / `covariant_vector` emit);
2. symbolic concrete `e_1` — `e` + `ConcreteIndex`, tagged with `basis_id`
   (what `unroll_sums` emits; renders `𝐢` via `vector_symbols`);
3. the frame vector `cs.basis(k)` — the user's actual rank-1 vector (for WCS the
   bare tensor `𝐢`), untagged (`basis_id 0`), no index slot.

`simplify_basis_dot`/`_cross` recognise only (1).  `reassemble` and
`reassemble_completeness` are split.  Plan:

### Step 1 — make `reassemble` finish on its own (P2)

`reassemble` also runs the completeness fold `Σ_i (X·e_i) e_i → X`, iterating
with its existing coordinate-carrier folds to a fixpoint.  Keep
`reassemble_completeness` as the focused primitive (like
`fold_equal_addends_structural`).  One `reassemble` call then reduces the P2
case to `𝐢`.  (User's larger idea — a single "do everything" reassembly and a
generalised completeness operation — is noted under Deferred.)

### Step 2 — teach the basis-vector recogniser all three forms (P1, P3)

`as_basis_vector(e, b)` is the shared chokepoint (feeds `as_vec_side`, hence
both `simplify_basis_dot` and `simplify_basis_cross`).  Generalise it to return
`(IndexAssoc, Level)` instead of `(CountableIndex, Level)`:

- **concrete on the `e`-form:** accept a `ConcreteIndex` (not only
  `CountableIndex`) on a slot whose `name == b.vector_symbol()` and
  `basis_id == b.basis_id()` — covers representation (2).
- **reverse-lookup of frame vectors:** if `e` is `structural_eq` to `b`'s k-th
  covariant vector `basis(k)` → `(ConcreteIndex{values[k]}, Lower)`, or its k-th
  cobasis `cobasis(k)` → `(ConcreteIndex{values[k]}, Upper)` — covers
  representation (3).  Identity is by structural match against *this* basis's
  vectors, so it is inherently single-frame (no `basis_id` needed, no cross-frame
  leak).

`VecSide.index` becomes `IndexAssoc`; `simplify_basis_dot` then emits
`δ`/`g` with the resulting assocs — `e^i·e_1 → δ^i_1`, `e_1·e_1 → δ_{11}` — and
`simplify_basis_cross` gains concrete `ε` for free through the shared path.

### Step 3 — close the δ/ε flow for concrete indices

Verify (and extend if needed) that the emitted deltas reduce:

- `δ_{11}`/`δ_{12}` (concrete–concrete) → 1/0 via `eval_delta_concrete`;
- `δ^i_1` (summed dummy `i` + concrete `1`) → substitutes `i := 1`, so
  `Σ_i e_i δ^i_1 → e_1`, via `unroll_sums` + `eval_delta_concrete` + folding, or
  a direct `contract_delta` extension if that chain doesn't already close.

The three example sequences must then reach their expected results
(P1 → `e_i δ_{i1}`, then `e_1`; P2 → `e_1`; P3 → `e_1`).

### Tests

- The three notebook sequences as regression tests (C++ + Python).
- Units: `as_basis_vector` recognising concrete-`e` and reverse-looked-up frame
  vectors; `simplify_basis_dot` producing `δ^i_1` / `δ_{11}`; `reassemble`
  doing the completeness fold.

### Deferred

- A deeper generalisation of `reassemble_completeness` and/or a single
  "reassemble everything" entry point (user's note).
- Broad concrete-index support across the other basis steps if more cases need
  it.
