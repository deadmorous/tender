# 000053 Completeness reassembly

`reassemble_completeness` — a basis-layer step that folds the **resolution of
identity** `Σ_i e_i ⊗ e^i = I` where it appears *partially contracted*.  It is
the invariant-level fold-back that closes coordinate derivations back into pure
direct notation, complementing `reassemble` (vibe 000049 §3).

## Why a new step

`reassemble` reconstructs a *named* tensor from its literal coordinate
components (`Σ_i a_i e_i → a`, reading the name `a` off the coordinate tensor
`a_i`).  But a coordinate derivation often produces the component as an
**invariant dot** `a·e_i`, not a coordinate tensor `a_i` — e.g. proving
`a × I × b` via the pattern matcher (vibe 000050 #3, second proof) leaves
`Σ_i (a·e_i)(b ⊗ e_i)`.  `reassemble` does not recognize `a·e_i` (a `Dot`) as a
coordinate, so it no-ops.

There are two ways to fold `Σ_i (a·e_i) e_i`:

1. **component route** — `a·e_i = a_i`, then `Σ_i a_i e_i = a`.  Needs to
   materialize `a` into components and a *symbolic* δ-substitution
   `Σ_j δ_{ij} a_j → a_i` (the deferred parametric-RHS gap, vibes 000033 §6 /
   000040).  Avoided.
2. **resolution route** — `Σ_i (a·e_i) e_i = a · (Σ_i e_i⊗e_i) = a·I = a`.
   Purely structural, no component materialization.  **This is what the step
   does.**

## What it folds

In a product term under `Σ_i`, over a summed index `i` that occurs nowhere else:

- **shape A** (contraction): one bare basis vector `e_i` together with a *scalar*
  contraction `(X·e_i)` (so `X` is rank 1) collapses to `X` in the leg's
  position:
  ```
  Σ_i (X·e_i) e_i        → X
  Σ_i (a·e_i) (b ⊗ e_i)  → b ⊗ a      -- a lands on the leg e_i held (the RIGHT leg)
  ```
- **shape B** (resolution → I): two bare basis vectors `e_i` with only rank-0
  (scalar) other factors fold to `I`, the scalars passing through:
  ```
  Σ_i (a·b) e_i ⊗ e_i    → (a·b) I
  Σ_i e_i ⊗ e_i          → I          -- empty coefficient
  ```

The sum is distributed over `Sum`/`Negate` addends by linearity, but **only when
a fold actually fires below**, so the step stays a no-op on anything else
(matching `reassemble`'s contract).  Addends that are *not* a completeness
pattern are left for `reassemble`.

## Implementation notes (`src/basis.cpp`)

- `as_completeness_dot` — a `Dot` of rank 0 (scalar) with one operand a bare
  basis vector `e_i`; returns the index and the other operand `X`.
- `fold_completeness_term` — classifies a flattened product term's factors per
  summed index into bare legs / completeness dots / other (rejecting any other
  use of `i`); applies shape A or B.  Shape A substitutes `X` **in place** of the
  leg (preserving non-commuting dyad-leg order — same discipline that the
  ε-pair transpose bug, vibe 000050 #3, taught); shape B emits the scalar
  coefficients first then `I` (the conventional coefficient·tensor order, which
  the canonicalizer preserves because both are invariants).
- `fold_completeness` — recursive driver: peel binders, distribute over
  `Sum`/`Negate`, fold products; `nullptr` signals "nothing folded" so the
  public `rewrite_tree` pass is a clean no-op when no pattern is present.

Reuses `as_basis_vector` / `flatten_product` from `reassemble`; `infer_rank`
gates the scalar checks.

## Status

Proves `a × I × b = b ⊗ a − (a·b) I` a **second way** (pattern matcher +
completeness, distinct from the ε-pair contraction of the first proof), reading
out structurally as `b ⊗ a − (a·b) I`.  Tests: `ReassembleCompleteness.*`
(unit, both shapes + no-ops), `BasisFeasibility.CrossIdentityCrossViaReassembly`
(C++), `test_cross_identity_cross_via_reassembly` (Python).  Python binding
`tb.reassemble_completeness`.

## Future

- **Higher-rank shape A**: `Σ_i (T·e_i) ⊗ e_i = T` for `T` rank ≥ 2 (the dot is
  then rank ≥ 1, not scalar) — currently restricted to rank-1 `X`.  The general
  `T·I = T` holds; the leg/slot bookkeeping for non-commuting reattachment is
  the only reason it is deferred.
- Folding a completeness pattern that straddles a `Sum`/`Difference` *factor*
  (un-distributed) — today the caller distributes (`expand_products`) first, the
  same boundary as vibe 000052.
