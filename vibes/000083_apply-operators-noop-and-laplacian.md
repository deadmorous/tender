# 000083 — apply_operators no-op guard + invariant Laplacian constructor

Status: **PLANNED** (to implement).

Two small, independent fixes surfaced by a user example on the strain `reass`:

```python
# #1 (correct): tr of the scalar Hessian ∇∇θ, θ = tr ε
a = nabla*(nabla*eps.tr()); a = t.tr(a); a = td.expand_dyad_ops(a)   # → Δθ  ✓
# #2 (corrupt): same, but with apply_operators inserted
a = nabla*(nabla*eps.tr()); a = t.tr(a)
a = td.apply_operators(a)        # → ∇·∇ tr(ε)  (θ DETACHED from a bare ∇·∇)
a = td.expand_dyad_ops(a)        # can't recover — stays ∇·∇ tr(ε)
```

Root cause: `apply_operators` has **nothing to apply** in #2 — the ∇'s are
*abstract* `Nabla` operators over a scalar (no concrete `Deriv`, no expanded
basis), so its operator-application step is a no-op. But it still runs
`canon_tolerant` (canonicalize), and canonicalize **floats the scalar θ off the
operator**: `∇·(∇⊗θ)` → `θ·(∇·∇)` — the I2/B3 operator-position wall (vibe
000081). `expand_dyad_ops` alone (#1) never canonicalizes, so θ stays attached
and `∇·(∇⊗θ)` renders/reduces as the Laplacian `Δθ`.

## Part A — `apply_operators` is a genuine no-op when there is nothing to apply

**Problem:** `apply_operators` canonicalizes even when it applies no operator,
and that canonicalize corrupts bare-∇ forms (scalar Hessian → detached
`θ·(∇·∇)`). Its *contract* is "apply the first-class ∂ operators"; the
canonicalization is only incidental cleanup after applying.

**Design:**
- `apply_operators` only ever applies concrete **`Deriv`** operators (the frame
  ∂'s from `expand_nabla`); it never applies an abstract `Nabla` (that is
  `expand_nabla`'s job) — confirmed in vibe 000081 (abstract ∇ + canon = the
  corruption).
- So: at the top of `apply_operators`, after the existing
  `abstract_nabla_over_expanded_basis` guard (KEEP it — case 2 `tr(∇ u_i e_i)`
  must still raise), add: **if `e` contains no `Deriv` node, return `e`
  unchanged** (skip `canon_tolerant` entirely). New helper `contains_deriv(e)`.
- Order matters: guard first (raise on abstract-∇-over-basis), then the
  no-Deriv no-op, then `canon_tolerant(apply_operators_impl(e))`.

**Effect:** #2 leaves `tr(∇∇θ)` untouched, so the following `expand_dyad_ops`
still yields `Δθ`. `tr(reass)` etc. are unaffected (users already use
`expand_dyad_ops`, not `apply_operators`, on reassembled forms). The concrete
post-`expand_nabla` flows (which DO have `Deriv` nodes) are unchanged.

**Risk / verify:** some flow may call `apply_operators` on a Deriv-free
expression *relying on* the incidental canonicalization. Run the full suite; if
a caller depends on it, canonicalize explicitly there instead. `expand_nabla`
internally calls `apply_operators` on output that DOES contain `Deriv` (the ∂'s
it just emitted) — unaffected.

Guard test: `apply_operators(tr(∇⊗∇⊗θ))` is a structural no-op (== input);
`expand_dyad_ops` of the result is `Δθ` = `∇·(∇⊗θ)`.

## Part B — an official invariant Laplacian constructor

**Problem:** the only way to write ΔX today is `nabla @ (nabla * X)`
(`∇·(∇⊗X)`), which is correct and already renders as `ΔX` (render-recognition,
vibe 000080 Inc 3) — but non-obvious. And `nabla @ nabla` (bare `∇·∇`, no
operand) is *not* a Laplacian (it has nothing to act on); the detached
`θ·(∇·∇)` from #2 is the corruption, not a form to convert back.

**Design:** add a thin constructor (NO new node — `Δ` stays a *rendering* of
`∇·(∇⊗·)`):
- `t.laplacian(operand, ctx=None)` → `∇·(∇⊗operand)` =
  `make_dot(nabla, make_tensor_product(nabla, operand))`, with `∇ = t.nabla(ctx)`
  (ctx from operand's context / the arg). Works for any rank (scalar → Δθ,
  rank-2 → Δε), and renders `Δ operand`.
- Keep the chart's concrete `cs.laplacian(f)` as-is (it evaluates on the chart);
  `t.laplacian` is the abstract invariant form (the operator applied, unexpanded).
- Optional (decide at impl): a tiny C++ `make_laplacian(ctx, operand)` so C++
  callers/tests share it; or Python-only. Lean: Python-only wrapper is enough,
  but a C++ helper keeps one definition — decide by whether any C++ path wants it.

Guard test: `t.laplacian(eps.tr()).latex() == "Δ tr(ε)"` and it is structurally
`nabla @ (nabla * theta)`; `t.laplacian(eps)` renders `Δε`.

## Increments

1. **Part A** — `contains_deriv` + no-op short-circuit in `apply_operators`
   (after the abstract-∇ guard). Tests + full suite.
2. **Part B** — `t.laplacian(operand, ctx=None)` constructor (Python binding,
   optional C++ helper). Tests; update the example/docs to prefer `t.laplacian`.

Constraints: buildable/tested per increment; ≥90% coverage; clang-format;
strip notebooks. Both are small and independent — either can land first.

See [[vibe81-explicit-basis-operator-route]] (the I2/B3 scalar-float wall and the
`expand_dyad_ops` vs `apply_operators` split), [[differential-operators-and-strain-compat]],
[[notation-no-nabla-squared]] (never write ∇²; Δ is `∇·∇`). This split
(`apply_operators` for ∇/∂ vs `expand_dyad_ops` for tr/vec/ᵀ) is a discoverability
wart for the **operations-revisit vibe** to reconcile later.
