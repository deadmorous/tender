# 000079 — expand_nabla div-of-grad-scalar bug (∇·∇f → rank 2)

**Status: OPEN — next task.** Discovered during vibe 000078 increment 4 (the
`reassemble_nabla` round-trip unit tests): `reassemble_nabla(expand_nabla(∇·∇f))`
did not round-trip because `expand_nabla` itself returns the wrong thing for a
scalar Laplacian. Not a reassembly bug — the reassembler correctly folded the
(wrong) input it was given — and it does **not** arise in the strain-compat
reduction, so vibe 000078 is complete; this is a separate follow-up.

## The bug

`expand_nabla(∇·(∇f))` for a **scalar field `f`** returns **rank 2** instead of
the scalar `Δf` (rank 0). `∇·(∇f)` is chart-free rank 0 (correct), but the
free-index expansion produces a rank-2 dyad — the outer `∇·`'s frame vector fails
to contract with the inner grad's frame vector; the `·` degrades to a `⊗`.

## Exact repro

```python
import tender as t
ws = t.Workspace()
x, y, z = ws.coords("x", "y", "z")
cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
f = ws.field("f", 0)                 # scalar field
v = ws.field("v", 1)                 # vector field
nab = t.nabla(ctx=ws.ctx)

cart.expand_nabla(nab @ (nab * f)).rank   # ==> 2   WRONG (want 0, Δf)
#   renders  Σ_i Σ_j (∂_j ∂_i f) e_j e_i      — a dyad, not δ_ij ∂_i∂_j f
cart.expand_nabla(nab @ (nab * v)).rank   # ==> 1   correct (Δv)
#   renders  Σ_j Σ_i e_i·e_j (∂_i∂_j v)      — the ·/δ IS present here
cart.expand_nabla(nab * f).rank           # ==> 1   correct (∇f)
```

Chart operators are unaffected: `cart.laplacian(f).rank == 0` (correct).

## Diagnostic — scalar fails, vector works

The decisive contrast: `∇·(∇v)` for a **vector** expands correctly to
`Σ e_i·e_j ∂_i∂_j v` (the `e_i·e_j` δ-contraction is there), but `∇·(∇f)` for a
**scalar** expands to `Σ ∂_j∂_i f e_j e_i` — the same `e_i·e_j` has become
`e_j ⊗ e_i`. So the failure is specific to the operand being a **scalar**.

`∇⊗f` for scalar `f` = `e_j ⊗ (∂_j f)` = `(∂_j f) e_j` — a scalar coefficient
times a frame vector. The outer `∇·` is `e_i · ∂_i(that)` = `e_i · (e_j ∂_i∂_j f)`,
which should fold to `(e_i·e_j) ∂_i∂_j f`. Instead the dot between `e_i` and the
scalar-scaled `e_j` did not fire.

## Suspects / where to look

- **`make_dot` scalar-redirect (vibe 000078 cause-2, `0fe8655`).** `make_dot`
  redirects to `make_tensor_product` when a *known* rank-0 operand is present. In
  `e_i · (∂∂f · e_j)`, the intermediate `∂∂f` (a scalar) sits next to `e_j`; if
  the distribution/apply order ever forms `make_dot(scalar, …)` or
  `make_dot(e_i, scalar)` transiently, the redirect turns the `·` into `⊗`. This
  is exactly the interaction the user flagged in
  `../tender-sandbox/ERR_scalar-dot-cross_rank.txt` (throwing there would have
  surfaced this instead of silently making a ⊗). **Prime suspect — start here.**
- `apply_operators` / `apply_product_operators` distribution of `∇·(∇⊗f)` when
  the innermost operand is a scalar (the ⊗ leg is `scalar ⊗ e_j`, which may
  reorder so the dot meets the scalar first).
- `expand_nabla`'s post-apply cleanup (`nf::fold_forced_zeros`) — unlikely, but
  confirm it isn't collapsing/reordering the contraction.

## Verify the fix

- `expand_nabla(∇·∇f).rank == 0` and, componentized, equals `cart.laplacian(f)`.
- Re-enable the scalar-Laplacian round-trip: `reassemble_nabla(expand_nabla(∇·∇f))
  == ∇·∇f` (a `ReassembleNabla…` C++ test, currently only vector/rank-2 cases
  round-trip — see `tests/chart_test.cpp`).
- Full suite stays green; watch the vector `∇·∇v` (Δv) and grad-div paths, which
  currently work, for regressions.

See also [[differential-operators-and-strain-compat]] and the vibe 000078
increment-4 "Known limitation" note.
