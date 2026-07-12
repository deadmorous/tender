# 000087 — reassemble_nabla must fold a two-field bilinear cross term

Status: **DONE** (prerequisite (b) for vibe 000084).

## Implemented

`try_reassemble_bilinear` (src/chart.cpp), a pre-check run by `reassemble_term`
before the single-operand classifier (which is left untouched — no regression).
It matches the focused bilinear shape — two ∂-marked operands, one free gradient
leg each, one frame-dot joining those two legs, plus plain scalar coefficients —
and folds it to `(∇⊗A)ᵀ·(∇⊗B)` (the ᵀ dropped when `∇⊗A` is rank 1, i.e. a scalar
operand's `∇u`), contracting the two **gradient** legs. Leg ownership is resolved
by matching each frame-direction's index id (`frame_dir_index`) against each
operand's free `DerivMark.link` ids (`free_mark_ids`): both on one operand ⇒ a
same-field Laplacian δ-pair (single path); on different operands ⇒ the bilinear.

`Δ(u e)` now reassembles to `(Δu)e + 2(∇u)·(∇⊗e) + u Δe` — the exact second-order
Leibniz rule (`algebraic_eq` to the textbook RHS). Tests: C++
`Chart.ReassembleNablaFoldsBilinearCrossTerm`, Python
`test_reassemble_second_order_leibniz_bilinear`. 819 C++ + 282 Python pass;
navier_lame + strain_compatibility unchanged.

## Coefficient-order fix (also DONE — a correctness bug the vector case exposed)

`Δ(a⊗b)` for two *vector* fields exposed a second, orthogonal bug:
`reassemble_term` reattached every non-operand factor as a **left** coefficient
(`make_tensor_product(coef, cur)`), fine for a scalar (commutes) but wrong for a
rank-≥1 undifferentiated factor — the single-operand term `(Δa)⊗b` came out
`b⊗Δa`, a *different tensor* (not a render glitch: the tree itself was mis-built).
Fix: track each coefficient's **position** and reattach on the correct side of the
operand — left of the operand ⇒ left (`λ Δu`, `a⊗Δb`), right ⇒ right (`(Δa)⊗b`);
scalars are order-immaterial (canonicalize pools them); no operand ⇒ all left
(historical). Now `Δ(a⊗b)` reassembles to `(Δa)⊗b + 2(∇a)ᵀ·(∇⊗b) + a⊗(Δb)`
(`algebraic_eq` to the textbook RHS). navier/strain unchanged (their non-operand
factors are scalars λ/μ + the identity). Test
`test_reassemble_second_order_leibniz_vector_dyad`.

## Observing the Leibniz rule (the "how")

There is no *chart-free* abstract-∇ Leibniz — `∇` needs a frame to lower
(`∇ = eᵢ∂ᵢ`), so `expand_dyad_ops` (tr/vec/transpose of dyads only) is a no-op on
`t.laplacian(u v)`. Observe it via a Cartesian chart round-trip — `expand_nabla`
(lowers ∇, applies Leibniz on the concrete ∂'s) then `reassemble_nabla` (folds
back to invariant ∇/Δ); the endpoint is frame-independent. (A convenience wrapper
for this invariant Leibniz could live in vibe 84's `chart.evaluate` surface.)

## Original plan (as written)

## Problem

The second-order Leibniz rule for a scalar×vector product,

```
Δ(u e) = (Δu) e + 2 (∇u)·(∇⊗e) + u (Δe) ,
```

is derived *correctly* by `expand_nabla` + Leibniz (verified: the expanded
interior is exactly `(Δu)e + [(∂_j u) e_i·e_j ∂_i e + (∂_j u) e_j·e_i ∂_i e] +
u Δe`, the two bracketed copies being the factor of 2). But `reassemble_nabla`
then mis-folds the **cross term** `2 (∇u)·(∇⊗e)`, producing garbage like
`(u + 2) Δe` — it drops the `∇u` field and invents a Laplacian of `e`.

This blocks vibe 000084: realistic invariant expressions are built from
*products* of fields, whose ∇-expansions generate exactly these bilinear
cross terms; the invariant round-trip (and any verification through it) must
fold them faithfully.

## Root cause (`reassemble_term`, src/chart.cpp ~L1050)

`reassemble_term` assumes **exactly one** field-carrying operand blob per term,
plus δ-pairs (`e_ℓ·e_m`) that each mean a Laplacian of that one operand. The
cross term `(∂_j u)(e_j·e_i)(∂_i e)` violates both:

1. **The inter-gradient dot is misread as a Laplacian.** The classifier counts
   *any* `Dot(frame_dir, frame_dir)` as `++laplacians` (L1088). But here
   `e_j·e_i` contracts the gradient index `j` of `∂_j u` against the gradient
   leg `i` of `∂_i e` — it is a contraction *between two different gradients*,
   `(∇u)·(∇⊗e)`, not the two legs of one field's second derivative.
2. **The second field is dropped.** Both `∂_j u` and `∂_i e` satisfy
   `carries_field`, so the single-`operand` loop (L1112) keeps only the last
   (`∂_i e`) and silently discards `∂_j u`.

`fold_divergences` (L981) likewise only threads δ-pairs into *one* operand's
nested divergences; it has no notion of a bilinear leg between two operands.

## Approach (to refine at implementation)

Teach `reassemble_term` to recognise a **bilinear** term: two (or more)
field-carrying, ∂-marked factors whose gradient legs are joined by a frame-dot.

- **Distinguish a bilinear dot from a Laplacian δ-pair.** A δ-pair is
  `e_ℓ·e_m` where *both* directions index the ∂'s of the **same** operand
  (the second-derivative legs). A bilinear dot connects the ∂-index of operand
  A to the gradient leg of operand B. Track which ∂-index / free frame leg each
  frame-dir belongs to (the ∂-marks already carry the linking index ids) and
  classify by whether the two sides sit on the same operand or different ones.
- **Fold to a dot of gradients.** For a bilinear pair, emit
  `(∇⊗A)·(∇⊗B)` (with the correct leg/transpose bookkeeping) instead of a
  single-operand Laplacian — reusing `fold_divergences` per operand and then
  contracting the two results at the linked legs.
- **Generalise the single-`operand` assumption.** Collect *all* field-carrying
  factors and their ∂/leg indices, not just the last; drive reassembly off the
  contraction graph (which frame-dirs pair which legs), so N operands and their
  inter-gradient dots reassemble together.

## Reuse / relation

This is the ∇-flavoured case of the **general reassembly contraction engine**
(vibe 000063 — trace / bilinear / rank-2-transpose / composite-leg /
tensor-tensor folds; [[anf-design-in-progress]]). Check first whether the cross
term can be routed through that engine (it already folds bilinears in the
coordinate→invariant direction) rather than duplicating leg-tracking inside
`reassemble_term` — DRY. `try_contract_eps_pair` / `contract_eps_pair`
(src/derivation.cpp ~L2649) are the existing precedent for index-linked pair
contraction to mirror.

## Tests

- `Δ(u e)` reassembles to `(Δu) e + 2 (∇u)·(∇⊗e) + u Δe` (`algebraic_eq` to the
  textbook RHS, after `reduce_field` unifies `e_i·e_j = e_j·e_i`).
- A rank-1·rank-1 bilinear (`∇a · ∇b` style) and a `∇(a·b)`-derived term.
- Regression: navier_lame + strain_compatibility single-field reassembly
  unchanged (they must not be perturbed by the multi-operand path).

## Scope note

Substantial (leg-tracking / contraction-graph work), unlike (a). Sequence:
land (a) [vibe 000086] first, then this, then build vibe 000084 on top — the
`chart.evaluate` dispatcher itself routes `Dot(∇,·)`→div etc. and does **not**
go through `reassemble_nabla`, but vibe-84 examples and any invariant round-trip
need this fold to be correct. See [[express-invariant-nabla-in-chart-plan]]
(vibe 000084), [[route-b-curvilinear-derivations]] (reassembly workflow),
[[canonicalize-preserves-nabla-fence]] (vibe 000085).
