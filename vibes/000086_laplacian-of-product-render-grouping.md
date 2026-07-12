# 000086 — Δ of a compound operand must group (render)

Status: **DONE** (prerequisite (a) for vibe 000084).

## Implementation note (what a product operand revealed)

Implementing the render grouping surfaced a deeper gap: `∇·(∇⊗(u e))` (a
*product* operand) was **floated** by canonicalize to `∇·∇ u e`, because the
vibe-000085 barrier keyed on the *immediate* fence leg being a bare `∇` — but a
scalar `u` makes canonicalize re-associate the operand to `(∇⊗u)⊗e`, so the `∇`
is no longer the immediate leg. Two coupled fixes were needed on top of the pure
render change:

1. **Deepen the vibe-85 barrier** (`distribute_contraction` + `encapsulate`):
   key on **"contains an abstract ∇ anywhere"** (new shared `contains_nabla` in
   `rewrite.hpp`), not just an immediate leg. Now a contraction node holding a
   `∇` stays fully positional (mirrors nf `place_factors`), so
   `∇·(∇⊗(u e))` is *preserved* as `Dot(∇, (∇⊗u)⊗e)` — the Laplacian nesting is
   intact (verified: canon == built Laplacian, idempotent, `L−C=0` on expansion).
2. **Recognise a ∇-led ⊗-chain in the render** (`grad_chain_rest`): the Δ arms
   flatten the divergence operand and match when the *leftmost* factor is `∇`,
   taking the remaining factors as the operand `X` — so a re-associated
   `(∇⊗u)⊗e` still renders `Δ(u e)`. `laplacian_chain_str` groups a multi-factor
   operand.

## The pure render change (as planned)

## Problem

`∇·(∇⊗(u e))` (with `u` scalar, `e` vector — i.e. `Δ(u e)`) renders as

```
\Delta u \, \mathbf{e}      →   Δ u e
```

which a reader parses as `(Δu) e`, but it *means* `Δ(u e)`. The `Δ` prefix does
not scope its operand: a compound operand (a `⊗`-product, a sum, a contraction)
is emitted without grouping, so `Δ` visually binds only to the first token. This
is a silent miscommunication — the rendered LaTeX asserts a different expression
than the tree. It will bite vibe 000084 immediately: users will feed invariant
`∇` expressions built from *products* of fields and read back wrong-looking Δ's.

## Root cause

All three Laplacian render sites emit the operand at `TENSOR_PREC`, which does
not parenthesise a `TensorProduct`/`Sum` operand (`prec(child) < req` is false at
equal precedence). In `src/render.cpp`:

1. `Dot` arm — `∇·(∇⊗X)` → `"\\Delta " + sub(*tp->right, TENSOR_PREC)`.
2. Nested-form `TensorProduct` arm (vibe 000085) — `c ⊗ ∇·(∇⊗X)` →
   `… + "\\Delta " + sub(*operand, TENSOR_PREC)`.
3. Floated-form `TensorProduct` arm (vibe 000083) — `(c ∇·∇)⊗X` →
   `… + "\\Delta " + sub(*p.right, TENSOR_PREC)`.

## Fix

`Δ` binds tighter than `⊗`/contraction but must *group* any non-atomic operand.
Render the operand requiring `ATOM_PREC`, so anything with lower precedence
(TensorProduct=3, Sum/Difference=1, Dot=2, Negate=4) is parenthesised, while the
atom-like operands stay bare (bare tensor / `ScalarLiteral` / `Trace` /
`VectorInvariant` / `Transpose` all report `ATOM_PREC`).

- New DRY helper `laplacian_operand_str(Expr const& e)` = `sub(e, ATOM_PREC)`,
  used at all three sites, so the grouping rule lives in one place.
- Effect: `Δ(u e)`, `Δ(a + b)`, `Δ(u 𝐞)` group; `Δε`, `Δ tr(ε)`, `Δ𝐗`, `μ Δ𝐗`,
  `2 μ Δ𝐗` (existing tests) are **unchanged** (their operands are atom-like).
- `ScalarDiv` operand (`\frac{}{}`, self-delimiting) would get redundant-but-
  harmless parens; acceptable, or special-case if it looks noisy in practice.

## Tests (`tests/render_test.cpp`)

- `Δ(u e)` → `\Delta (u \, \mathbf{e})` (regression for the reported bug).
- `Δ(a + b)` → grouped; coefficient case `μ Δ(u e)` → `\mu \, \Delta (u \, …)`.
- Re-assert the atom-operand forms are untouched (`Δε`, `Δ tr(ε)`, `Δ𝐗`, `μΔ𝐗`).

Small and independent of (b) / the node decision. Constraints: clang-format;
≥90% coverage. See [[canonicalize-preserves-nabla-fence]] (vibe 000085, the
nested-Δ render arm this generalises), [[express-invariant-nabla-in-chart-plan]]
(vibe 000084, the goal), [[notation-no-nabla-squared]] (Δ is `∇·∇`, never `∇²`).
