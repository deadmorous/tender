# 000090 — coordinates of charts sharing a reference frame are related

Status: **PLANNED / PROBLEM STATEMENT**.

## The problem (user)

Two charts built over the same reference (WCS) — a Cartesian `cart` and a
cylindrical `cyl` — share a manifold: their coordinates are related by the
embeddings (`x = r cosθ`, `y = r sinθ`, `z = z`). But tender treats each chart's
coordinates as **independent variables**, so a quantity written in one chart
evaluates to `0` (not its true value) under the other chart's operators:

```python
a = nabla * cart.position()   # ∇⊗R,  R = x i + y j + z k
cart.evaluate(a)  # I     ✓ (∇R = I)
cyl.evaluate(a)   # 0     ✗  — should be I; ∇R = I is chart-independent

b = nabla * cyl.position()    # ∇⊗R,  R = r e_r + z e_z
cart.evaluate(b)  # 0     ✗  — should be I
cyl.evaluate(b)   # I     ✓
```

`∇R = I` is coordinate-free, so **every** chart must give `I`. The foreign-chart
answers being `0` is a correctness bug rooted in a missing feature: *cross-chart
coordinate dependencies*.

## Root cause

`cyl.grad(f)` differentiates w.r.t. the cyl coordinates via `diff`/`partial`,
whose elementary rule is `∂_{q} q' = 1` iff `q'` **is** `q` (same coordinate
object), else `0`. `x` (a `cart` coordinate) and `r` (a `cyl` coordinate) are
distinct coordinate objects, so `∂_r x = 0` — verified: `cyl.grad(x) == 0`,
`cart.grad(r) == 0`. tender never consults the relationship `x = r cosθ`, even
though it is **latent in the embeddings**: `cyl.radius_vector()` already yields
`r cosθ i + r sinθ j + z k` — the WCS Cartesian coordinates as functions of the
cyl coordinates.

## What is available vs missing

- **Forward embedding (have):** each chart C knows `WCS_coord = C.embedding(q_C)`
  (its `radius_vector` / `embedding`). For the identity Cartesian chart the
  embedding is trivial (`x_WCS = x`).
- **Inverse embedding (missing):** `q_C = C⁻¹(WCS_coord)` — e.g. `r = √(x²+y²)`,
  `θ = atan2(y, x)`. Not stored, and in general not closed-form.

The two directions differ sharply:

- **WCS-quantity → curvilinear chart (tractable).** `cyl.evaluate(a)`: substitute
  each WCS-Cartesian coordinate by the target chart's forward embedding
  (`x → r cosθ`, from `cyl.radius_vector`), then differentiate. Worked by hand:
  `∇⊗(r cosθ i + r sinθ j + z k)` in cyl folds to `e_r⊗e_r + e_θ⊗e_θ + e_z⊗e_z =
  I`. Needs only the **forward** embedding of the *evaluating* chart.
- **Curvilinear-quantity → other chart (needs the inverse).** `cart.evaluate(b)`:
  `r e_r + z e_z` must become a function of `x, y, z`, which requires the inverse
  embedding of the *source* chart (`r = √(x²+y²)`, …) — or recognising the
  forward images (`r cosθ ≡ x`) by matching. Harder; the "missing functionality"
  the user noted.

## Candidate approaches

- **A — coordinate reprojection in `evaluate`/`expand` (forward direction).**
  Before differentiating in chart C, rewrite every foreign WCS-Cartesian
  coordinate `x_a` in the expression by `C.radius_vector`'s a-th component (a
  function of C's coords). Then `diff` sees only C's own coords + constant WCS
  vectors and differentiates correctly. Fixes `cyl.evaluate(a)`. Smallest correct
  core; leaves the inverse direction unsupported (clear error).
- **B — give a chart an optional inverse embedding** `q_C = C⁻¹(WCS)` (stored,
  user-supplied or derived for the well-known charts cyl/spherical). Cross-chart
  reprojection is then `C.embedding ∘ D⁻¹` (D-coords → WCS → C-coords), covering
  **both** directions and curvilinear↔curvilinear. Larger; the inverse of a
  general user embedding is the open problem (require it as input, or derive for
  known charts only).
- **C — WCS pivot + matching.** Always express a quantity in the WCS frame with
  WCS coords (via `to_reference` + forward-embedding *matching* `r cosθ ≡ x`),
  then reproject into the target. Avoids a symbolic inverse for the specific
  polynomial embeddings, but the matching (`r cosθ`, `r sinθ` → `x`, `y`) is
  itself an inversion in disguise and brittle.
- **D — Jacobian/chain-rule between charts.** Register `∂_{q_C} q_D` (the
  chart-to-chart Jacobian) so `diff` chain-rules through foreign coords. Same
  inverse-embedding requirement as B, expressed differentially.

## Recommendation (to decide with the user)

Ship **A** first — it makes `cyl.evaluate(a)` (a Cartesian/WCS-expressed quantity
evaluated in a curvilinear chart) correct, reusing `radius_vector`, with no
inversion. Then decide **B**: store an inverse embedding per chart (user-supplied,
or derived for the built-in cyl/spherical) to cover the reverse and
curvilinear↔curvilinear directions. `evaluate`/`expand`/`components` all route
through the same reprojection.

## Obstacles / open decisions

1. **Identifying "foreign" coordinates.** A coordinate carries a `chart_id`
   (`CoordinateRef`); the evaluating chart knows its own coord `chart_id`s. Any
   coordinate in the expression not belonging to C, but to a chart sharing C's
   reference, is reprojected. (Shared coords like `z` are already the same object
   — verified `structural_eq(cart.z, cyl.z)` — so they need no reprojection.)
2. **Where to hook.** A pre-pass in `evaluate` (and `expand`?) that substitutes
   foreign coords before the operators differentiate — vs teaching `diff` the
   cross-chart Jacobian directly (D). A substitution pre-pass is the least
   invasive.
3. **Inverse embeddings.** Require user-supplied for a custom chart; auto-derive
   for the well-known cyl/spherical (their inverses are standard: `r=√(x²+y²)`,
   `θ=atan2`, …). Needs `atan2`/`√` as scalar functions and the domain bits.
4. **Unsupported policy.** Until B lands, `evaluate` of a curvilinear-expressed
   quantity in a *different* chart should **error clearly** ("needs the inverse
   embedding of chart D"), not silently return `0`.
5. **Interaction with `simplify_scalars`/frame reduction.** After substitution
   the folds must recognise `e_r = cosθ i + sinθ j` etc. so the result collapses
   to `I` — the existing `to_reference` / `simplify_basis_dot` machinery, now
   with distribution (vibe 000089) in place.

Constraints (unchanged): buildable/tested per increment; ≥90% coverage;
clang-format; never write `∇²`. See [[express-invariant-nabla-in-chart-plan]]
(vibe 000084 — `chart.evaluate`), [[differential-foundations-plan]] (charts,
`radius_vector`, `to_reference`), [[route-b-curvilinear-derivations]] (frame
reduction), [[basis-aware-indices-plan]] (per-chart ids).
