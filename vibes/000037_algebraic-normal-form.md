# 000037 Algebraic normal form (ANF)

## Goal

A function `canonicalize(Expr) -> Expr` such that, for a **declared** equational
theory T₀,

```
algebraic_eq(a, b)  :=  structural_eq(canonicalize(a), canonicalize(b))
```

decides T₀-equality. This unblocks like-term collection, pattern matching, and
e-node deduplication for the e-graph (vibe 000034).

`structural_eq` (derivation.cpp) is positional and exact — no AC, no
α-equivalence, `Difference ≠ Sum+Negate`, bound indices compared by id. That is
exactly what we want: it is the *equality of canonical forms* primitive, and the
ANF's job is to feed it canonical forms. `Polynomial` (polynomial.hpp) is the
precedent for the mechanism (sorted terms, drop-zero, accumulate duplicates),
generalised from univariate to tensor monomials.

## What "canonical" means here — and what it does NOT decide

"Canonical" is always relative to a theory T: `canon_T(a)=canon_T(b) ⟺ a ≡_T b`.
Choosing the form is choosing T. We deliberately choose a **weaker** theory than
the full commutative ring:

**T₀ (what the ANF normalises):**
- AC of `+`, identity `0`.
- subtraction/negation folded into a signed ℚ coefficient — `Difference` never
  appears in canonical form; `Negate` survives only as the `-1` coefficient
  wrapper at term level (renders cleanly as `-X`).
- AC of the **component product** (commutative *iff* all factors are
  scalar/coordinate-valued — see gate predicate), identity `1`, `0` annihilator.
- exact ℚ numeric folding (`ScalarLiteral`, `ScalarDiv` of literals, `Negate` of
  a literal).
- like-term collection within a flat sum (`X + X → 2X`, `X - X → 0`).
- α-renaming of bound indices (`ExplicitSum`/`NoSum`).
- `Dot` of two rank-1 operands: commutative.  `Cross` of two rank-1 operands:
  anticommutative (`a×b → -(b×a)`).

**Not in T₀ (deliberately):**
- **distributivity** — `a(b+c)` and `ab+ac` are *different* canonical forms.
- contraction identities (`contract_delta`, `contract_eps_pair`, …).
- tensor index-symmetries (symmetric/antisymmetric slot reorder, δ symmetry, ε
  antisymmetry) — trait-driven, a later refinement.
- higher-rank `Dot`/`DDot`/`DDotAlt`/`Cross` reordering (`A·B ≠ B·A`).

Consequence: `algebraic_eq` is **sound but incomplete** for full ring equality —
if it reports equal they are; if unequal they may still be ring-equal via a law
outside T₀. Full ring equality is decided by *delegating* the missing laws:

- **on-demand** — run `expand` (distribute) + `canonicalize` only when a
  ring-equality verdict is actually needed (not as the stored form); or
- **e-graph** — distributivity, contraction, symmetry are *rewrite rules*;
  `a(b+c)` and `ab+ac` land in the same e-class after saturation. Equality =
  "same e-class," not "same canonical form." Here the ANF's role shrinks to
  **e-node dedup** (so `a+b` and `b+a` hash identically) — exactly the cheap
  noise an e-graph needs killed at the node level.

This is why the e-graph world never stores a global expanded normal form: it
blows up (the eps-delta 12 terms) and fights the compact symbolic forms we built
(`contract_eps_pair`'s 2-term answer would be re-expanded to 12 just to compare).

## Canonical shape — distributed sum of terms (not expanded)

Bottom-up, every canonical expression is:

- a **Sum** of **terms**, each `coefficient · monomial`, coefficient ∈ ℚ\{0},
  terms sorted by monomial key, like monomials combined, zero terms dropped;
- a **monomial** is a sorted (by key) product of **atoms** (empty ⇒ the term is a
  bare scalar; `coeff == 1` ⇒ the coefficient is omitted);
- an **atom** is indivisible: a `TensorObject` (coordinate or invariant), an
  opaque composite (a `Dot`/`Cross`/`DDot`/`DDotAlt` subtree with canonicalised
  operands), or an `ExplicitSum`/`NoSum` with α-normalised index and canonical
  body.

`(A+B)·C` therefore stays a product atom over the canonical sum `(A+B)`; it is
*not* expanded into `A·C + B·C`. Repeated identical factors are kept (no power
collapsing in v1 — there is no `Power` node).

## Decisions (ratified)

1. **Distributivity: delegated.** Local AC/α form; distribution is a rewrite rule
   (e-graph) or an on-demand `expand`, never baked into the canonical form.
2. **Component-product commutativity: gate on operands.** A `TensorProduct`
   commutes (its factors are sorted) iff every factor is *scalar/coordinate-
   valued*; otherwise (a dyad of invariants) order is preserved. No AST change.
   The node-split (a distinct commutative `ComponentProduct` vs non-commutative
   `OuterProduct`) is the cleaner eventual fix but deferred.
3. **Dot/Cross: included for rank-1.** `a·b ≡ b·a` (sort); `a×b ≡ -(b×a)` (sort
   with sign), gated on both operands being rank 1. Higher-rank stays opaque.

### Gate predicate `is_component_valued(e)`

The vibe-000036 coordinate/invariant line, applied to commutativity:

- `ScalarLiteral` → true.
- `TensorObject` → true iff `rank == 0` **or** (slots non-empty and *every* slot
  carries an index) — i.e. a scalar or a fully-indexed coordinate. A slot-less
  rank ≥ 1 object (an invariant) → false. Partially-indexed → false
  (conservative).
- `Negate`, `Sum`, `Difference`, `TensorProduct`, `ScalarDiv` → all children
  component-valued.
- `ExplicitSum`/`NoSum` → body component-valued.
- `Dot`/`DDot`/`DDotAlt`/`Cross` → false in v1 (conservative; a scalar `a·b`
  factor will simply not enable commuting its enclosing product — safe, just a
  missed normalisation; refine later).

Conservative falses only *miss* canonicalisations; they never wrongly commute a
dyad.

### Total order `compare(a, b)`

Drives the commutative sorts. Node-kind ordinal first, then per-kind structural
compare with children (already canonical, bottom-up) compared lexicographically.
For `TensorObject`: name, rank, then per slot (level, realm, **stable space
key**, index assoc). The space key must NOT be a raw `IndexSpace*` address
(non-deterministic across runs); within-run pointer order suffices for dedup and
matching but would make rendered golden tests flaky, so spaces should expose a
stable id. Flagged as an implementation prerequisite.

### α-normalisation

Canonicalise the body first; then walk it in a fixed deterministic pre-order and
relabel bound indices (`ExplicitSum`/`NoSum`) to a reserved canonical namespace
in order of first occurrence, leaving free indices untouched. Single-index
binders make this simple; the reserved namespace must not collide with free ids
(de Bruijn levels are the rigorous general version if nesting deepens).

## Relationship to existing steps

- Subsumes `fold_arithmetic`'s numeric folding and `fold_equal_addends`'s
  like-term collection (now with sorting + α).
- **Drops** `fold_arithmetic`'s `Sum(A,Negate(B)) → Difference` rewrite — that is
  a *presentation* concern the renderer already handles. Canonical form is the
  *algebra* normal form (signed sums); the renderer is the *presentation* normal
  form (differences). Two opposing normal forms, kept apart (this is the split we
  found in the fold_equal_addends work).
- `expand_products` stays a separate rule (distribution), not part of canon.
- The hand-orderable `fold_*` steps remain for pedagogical derivations;
  `canonicalize` is the workhorse for matching and the e-graph.

## Non-goals (v1)

Distribution, contraction, tensor index-symmetries, higher-rank Dot/Cross
reordering, power collapsing, the TensorProduct node split. Each is either a
rewrite rule (distribution, contraction) or a later ANF refinement (symmetries,
node split).

## Staging / implementation plan

1. Stable `IndexSpace` key + `compare(a,b)` total order.
2. `is_component_valued(e)` predicate.
3. Core `canonicalize`: bottom-up; numeric fold → signed-term collection (reuse
   `collect_signed_addends`, `extract_coeff`) → monomial flatten + coeff-pull +
   sort → like-term combine + sort + rebuild.
4. α-normalisation for `ExplicitSum`/`NoSum`.
5. Rank-1 `Dot` commutativity, `Cross` antisymmetry.
6. Tests: AC of `+`; AC of component product; `X+X→2X`, `X-X→0`; `a⊗b` NOT
   reordered vs `δ^a_b δ^c_d` reordered; `Σ_i f(i) ≡ Σ_p f(p)`; `a·b ≡ b·a`;
   `a×b ≡ -(b×a)`; `algebraic_eq` pairs.
7. Expose `td.canonicalize`; C++/Python bindings and tests.
8. (Later) hook into the pattern matcher and e-node construction.
