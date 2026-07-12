# 000088 — structural reassembly of ∇-expanded products of fields

Status: **DONE** (two-field shapes folded correctly; ≥3-field bails safe).

## Implemented

- **`try_reassemble_structural`** (src/chart.cpp) replaces the vibe-87
  `try_reassemble_bilinear` pre-check. It reads a term's frame δ-pairs / free
  legs / operands and folds two focused shapes:
  - **(i) two separate ⊗ operand factors** (each a *simple* marked field), one
    mark each, joined cross ⇒ `(∇⊗A)ᵀ·(∇⊗B)` (ᵀ dropped when `∇⊗A` is rank 1).
  - **(ii) one contracted operand `Dot(X,Y)`** (X,Y simple fields): both marks on
    one side ⇒ a **scoped** Laplacian (`X·ΔY` / `ΔX·Y`); split across the two ⇒
    the **double contraction** `∇X:∇Y` (both leg-pairs — the frame-dot and the
    original `X·Y` — meet).
- **`expand_double_dot` operator barrier** (src/derivation.cpp): `contains_nabla`
  guard so `(∇⊗u):(∇⊗v)` is not dyad-expanded to `(∇·∇)(u·v)` (the `:` analogue
  of the vibe-85 distribute_contraction float).
- **Safety valve** (`reassemble_term`): a genuinely multi-field term the
  structural path did NOT fold — ≥2 ∂-marked factors, or a field·field
  contraction operand — is left **un-reassembled** (returns the term) rather than
  reaching the single-operand classifier that would emit silently-wrong output.
  Shapes (i)/(ii) require *simple*-field operands, so a triple product (an
  operand that is itself a contraction) bails here instead of mis-folding.

Result: `Δ(u·v) = (Δu)·v + 2∇u:∇v + u·Δv` and `Δ(u⊗v)`, `Δ(f v)` all
`algebraic_eq` to their textbook RHS. A triple `Δ(f(u·v))` folds its two-field
subterms and leaves the genuine 3-field terms (`2∇f·∇(u·v)`) as verbatim
interior — correct but unfolded (an N-field generalisation is future work). 821
C++ + 284 Python pass; navier_lame + strain_compatibility unchanged. Tests:
`Chart.ReassembleNablaFoldsContractedDotProduct`,
`ExpandDoubleDot.AbstractNablaDoubleDotIsNotFloated`,
`test_reassemble_second_order_leibniz_dot_product`.

## Original plan (as written)

## Problem

`reassemble_nabla` returns **silently wrong** results for a Laplacian (or any
∇-operator) of a *dot-product* of fields. `Δ(u·v)` reassembles to `4·Δ(u·v)`
(the render is faithful — the tree is wrong), where the correct second-order
Leibniz is

```
Δ(u·v) = (Δu)·v + 2 ∇u:∇v + u·Δv .
```

The `expand_nabla` interior is *correct* (four terms `(Δu)·v`, two `∇u:∇v`,
`u·Δv`); `reassemble_term` mis-folds every one to `Δ(u·v)`.

## Root cause — the monolithic-operand model (again)

`reassemble_term` assumes a term is `(coefficients) × (operators over ONE
field blob)`. It classifies top-level ⊗ factors, counts δ-pairs `eᵢ·eⱼ` as
"Laplacians", picks ONE field-carrying factor as the operand, folds it
(`fold_divergences` strips its ∂-marks), then wraps the *whole* blob in the
counted ∇/Δ. This breaks for a *structured* operand:

- **Mis-scoped Laplacian.** For `(eᵢ·eⱼ)(u·∂ᵢ∂ⱼv)` the operand is the whole
  `Dot(u, ∂ᵢ∂ⱼv)`; stripping marks gives `u·v`, and the δ-pair Laplacian wraps
  it → `Δ(u·v)`. But `i,j` are `v`'s marks, so it must be `u·Δv` — the operator
  belongs to the *sub-field carrying those marks*, not the whole dot.
- **No double contraction.** `(eᵢ·eⱼ)(∂ᵢu·∂ⱼv)` is `∇u:∇v` — the two gradient
  legs contract (`eᵢ·eⱼ`) *and* the operand legs contract (the original `u·v`).
  `reassemble_term` cannot emit a `DDot` (`:`) at all.

This is the same limitation the vibe-000087 bilinear fix chipped at (two
top-level ⊗ operands), now general: **operators must be scoped to the specific
∂-marked sub-fields and their contraction structure**, not to one monolithic
blob. `Δ(u⊗v)` worked only because the operands were separable ⊗ factors; a
`·`/`:` structure entangles them.

## The real fix — a mark-graph reassembly engine

Reassemble structurally from the ∂-mark ↔ frame-direction linkage:

1. **Nodes = ∂-marks.** Each free `DerivMark.link` id on a field = one `∇`
   applied *to that field at its position* in the term's product/contraction
   tree.
2. **Edges = frame-direction roles.** For each mark index `a`, find its frame
   vector `e_a`:
   - `e_a` *free* (a ⊗ leg) ⇒ a **gradient** leg `∇⊗` on that field.
   - `e_a` inside a dot `e_a·e_b` ⇒ the `∇_a` leg **contracts** with `e_b`'s:
     - `b` a mark on the **same** field ⇒ a **Laplacian** (`∇·∇`) on it.
     - `b` a mark on a **different** field ⇒ a **bilinear** contraction of the
       two gradients (`∇A·∇B` or, when the fields themselves are already
       contracted, `∇A:∇B`).
3. **Emit** the invariant by threading these ∇'s into the operand structure —
   `A·(∇∇ scoped to B)`, `∇A:∇B`, etc. — reusing `fold_divergences` per field.

The vibe-000087 `try_reassemble_bilinear` is the two-node special case; this
generalises it to N marked sub-fields inside an arbitrary product/contraction
operand, and adds the `DDot` output. Cross-check against the vibe-000063 general
reassembly contraction engine ([[anf-design-in-progress]]) — the ∇-agnostic
bilinear/trace/rank-2 folds there may host or share this leg-graph logic (DRY)
rather than growing it all inside `reassemble_term`.

## Interim safety valve (decide with the user)

Until the engine lands, `reassemble_term` should **not emit silently-wrong**
output. Detect the unsupported shape — a δ-pair Laplacian over an operand that is
a *field·field* contraction (not the `frame·field` divergence folds, which stay
supported: `∇·(∇·ε)` has no δ-pair), or ≥2 marked sub-fields inside a
contraction that the bilinear pre-check doesn't cover — and **leave that term
un-reassembled** (return the frame-expanded form). Correct-but-unfolded beats a
wrong `4·Δ`. Risk: must not trip the working single-field / double-divergence /
⊗-bilinear paths — gate precisely and lock with the existing reassemble tests.

## Tests (when built)

- `Δ(u·v) = (Δu)·v + 2∇u:∇v + u·Δv`, `Δ(u⊗v)` (regression), a mixed
  `Δ((u·v) w)`, and `∇·((∇u)·v)` style.
- Regression: every existing `Chart.ReassembleNabla*` test, navier_lame, strain.

## Note on observing Leibniz

There is no chart-free abstract-∇ Leibniz; the round-trip is
`chart.expand_nabla → reassemble_nabla` (Cartesian; endpoint frame-independent).
A `chart.evaluate`-level convenience (vibe 000084) could wrap it. This engine is
what makes that convenience *correct* for products.

See [[reassemble-nabla-bilinear-cross-term]] (vibe 000087 — the 2-operand
special case), [[express-invariant-nabla-in-chart-plan]] (vibe 000084 — the
consumer), [[route-b-curvilinear-derivations]], [[anf-design-in-progress]]
(vibe 000063 general reassembly engine).
