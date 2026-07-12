# 000085 — canonicalize must not float an abstract-∇ operator fence

Status: **DONE** (implemented + tested).

## The problem (user)

Vibe 000084 carried a load-bearing caveat — *"interpret BEFORE canonicalize"* —
because `canonicalize` rewrites `∇·(∇⊗X)` into `(∇·∇)⊗X`, floating the operand
across the operator (the I2/B3 wall). The user pushed back, correctly:

> The user cannot really control when canonicalization is done — it can be a
> prerequisite of an operation, done automatically at the start. Canonicalization
> should never create such an inability to expand differential ops, at least not
> in an irreversible way.

That is exactly right. A silent `canonicalize` upstream of a chart expansion
would irreversibly corrupt `ΔX` (`∇·(∇⊗X)` → `(∇·∇)⊗X`), and the user has no
hook to prevent it. The fix belongs in `canonicalize`, not in a "sequence things
carefully" dance around it.

## Root cause (one site)

`distribute_contraction` (`src/derivation.cpp`) applies the tensor identity

```
a·(b⊗c) = (a·b) c      (contract the contractor with the near fence leg)
```

to `∇·(∇⊗X)`. But **that identity holds only when the operands are ordinary
vectors.** `∇` is a differential *operator*: it must differentiate everything to
its right. Folding the two `∇`s together *first* strands `X` — the inner `∇` is
pulled off the field it was applied to (`∇⊗X` = grad X). The result reads as
`ΔX` only under the "operator acts right" convention in a *constant* Cartesian
frame; in a moving frame it is genuinely wrong (you cannot contract `∇·∇` before
the inner `∇` has hit `X` **and the frame vectors**, `∂_i e_j = γ^k_{ij} e_k`).
And it is irreversible, so any auto-`canonicalize` corrupts the expression.

`place_factors` already refuses to commute across an operator (*"once a term
contains one, no factor may be commuted"*, vibe 000077). The fence-float simply
never got the same rule.

## Why the float existed (the real coupling)

It is **not** gratuitous. The nf normal-form model has **no `TensorProduct`
factor** — a `⊗` lives only at the *term* level joining a term's factors, never
nested inside a factor. So `encapsulate` *requires* every `⊗` fence inside a
contraction operand to have been distributed away first; a survivor hits
`throw "a nested ⊗ inside an operand awaits fence distribution"`. The float was
how the nf model kept its "flat factors" invariant. Fixing the float therefore
needs a **matched pair** of changes.

## The fix (two coupled edits + a render follow-up)

1. **Barrier the float (`distribute_contraction`).** Key on a fence *leg* being
   an abstract `Nabla`: in `op(L, A⊗B)` skip the reassociation when `A` or `B`
   is a `∇`; symmetrically for `op(A⊗B, R)`. Keying on a *leg* (not the
   contractor) guarantees every fence left un-distributed provably contains a
   `∇` at an immediate leg — so (2) can carry it and the plain-dyad `throw`
   still bites genuine bugs (a plain `a·(u⊗v)` still distributes: no `∇` leg).
2. **Carry the operator fence (`encapsulate`, `src/nf_lower.cpp`).** A `∇⊗X`
   fence reaching `encapsulate` is wrapped as a `Paren` over its canonical
   sub-`Nf` — exactly like a genuine `Sum` operand (the model's existing opaque
   escape hatch). It round-trips: `raise(Paren)` rebuilds the `⊗`, so
   `∇·(∇⊗X)` lowers to nf and raises back to `Dot(∇, ∇⊗X)`, which the `Dot`
   render arm shows as `ΔX`. A fence *without* an operator still can't appear
   here (it was distributed) → the `throw` is preserved as the bug net.
3. **Render the coefficient case (`src/render.cpp`).** The nested form
   `c ⊗ ∇·(∇⊗X)` = `TensorProduct(c, Dot(…))` would parenthesise the `Dot`
   child (`c (ΔX)`). New helper `nested_laplacian_operand` + a `TensorProduct`
   arm: a scalar-only left chain times a nested Laplacian renders the clean
   `c Δ X`. (Complements the vibe-000083 *floated*-form recogniser, still kept
   as a defensive path for hand-built `(∇·∇)⊗X`.)

## Effect

`canonicalize(∇·(∇⊗X))` now **preserves** the nesting, is **idempotent**, and
renders `ΔX`; with a coefficient, `μ ∇·(∇⊗X)` → `μ Δ X` (no float, no parens).
Canonicalize is thus **non-destructive on ∇-nesting** — safe to run before a
chart expansion. This directly **removes vibe 000084's "interpret BEFORE
canonicalize" risk**: the interpreter can canonicalize freely.

The concrete `expand_nabla` path is unaffected — it lowers `∇` to `e_i ∂_i`
(concrete `Deriv`, no abstract `Nabla`) *before* `distribute_contraction`, so
the leg barrier never fires there; the frame-vector dyad distributions in
`reduce_field` still run. navier_lame + strain_compatibility both still reduce
and verify (Cartesian + cylindrical).

## Tests

- C++ `DistributeContraction.AbstractNablaFenceIsNotFloated` (float barred;
  canon preserves nesting; idempotent), `…PlainDyadDivergenceStillDistributes`
  (no over-reach). 818 C++ tests pass.
- Python `test_canonicalize_preserves_nabla_laplacian_nesting` (structural
  preservation + idempotence + `μΔX`/`2μΔX` render). 281 Python tests pass.

## Note for the operations-revisit vibe

This resolves the *irreversible* face of the I2/B3 wall for the ∇-fence case.
The general principle — **a contraction/⊗ reassociation is unsafe across any
differential operator** — is now enforced for abstract `∇`. A concrete `Deriv`
is deliberately still distributable (it is an ordinary factor once *applied*).
If a future case needs the same barrier for an *applied* operator context, it
is the same one-site guard.

See [[vibe81-explicit-basis-operator-route]] (I2/B3 scalar-float wall),
[[vibe80-notebook-gaps-sprint]] (basis-expand-first is a correctness trap),
[[express-invariant-nabla-in-chart-plan]] (vibe 000084 — this removes its main
risk), [[notation-no-nabla-squared]] (Δ is `∇·∇`, never `∇²`).
