# 000090 — coordinates of charts sharing a reference frame are related

Status: **DONE** (approach A — forward direction — shipped).

## Implemented

- **`ws.wcs()` is memoised** (workspace.py): the world frame is unique per
  workspace, so both charts share *one* reference basis — the precondition for
  relating their coordinates. (Previously each call minted a fresh basis, so the
  two charts were genuinely independent frames.)
- **Chart-embedding registry** (`Context::ChartEmbedding` +
  `register_chart(ctx, chart)`): at construction, record `coord_chart_id →
  {reference basis_id, is_identity}` (`is_identity` ⇔ `embedding[a] ≡ coords[a]`).
  Registered in the `PyChart` constructor and (idempotently) in `evaluate`.
- **`reproject_coords` pre-pass in `evaluate`**: rewrite each coordinate not
  belonging to chart `C`. An identity-chart coord over the same reference is a
  WCS coord `x_a` → substitute `C.embedding[a]` (`x = r cosθ`). A curvilinear
  source coord, a foreign-reference coord, or an unregistered coord → **clear
  error** (the reverse direction needs the inverse embedding, approach B).
- **Re-express the result** in `C`'s frame when reprojection fired, so the
  reprojected quantity's WCS operand vectors `i,j,k` fold into `C`'s frame
  (`cosθ e_r⊗i + … → e_r⊗e_r + … → I`). Skipped for native quantities.

Result — every chart agrees `∇R = I`:
```
cart.evaluate(∇⊗cart.position()) = I
cyl.evaluate(∇⊗cart.position())  = I   (was 0; forward reproject)
cyl.evaluate(∇⊗cyl.position())   = I
cart.evaluate(∇⊗cyl.position())  = clear error (reverse — approach B)
```
824 C++ + 293 Python pass; navier_lame + strain_compatibility verify. Tests:
`Chart.EvaluateReprojectsForeignWcsCoordinates` (C++),
`test_chart_evaluate_cross_chart_position_gradient`,
`test_chart_evaluate_cross_chart_reverse_direction_errors`,
`test_workspace_wcs_is_memoised` (Python).

## Deferred (approach B — the reverse direction)

Curvilinear-quantity → other chart needs the **inverse embedding** `q_C =
C⁻¹(WCS)`; **derive for the built-in charts** (cyl/spherical/polar; inverses are
standard), custom charts user-supplied or forward-only. Until then `evaluate`
errors clearly instead of returning `0`. `expand`/`components` could route through
the same reproject later.

---

Status (original): **PLANNED / PROBLEM STATEMENT**.

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

## Decisions (locked with the user)

- **Direction: forward-only (A) now.** WCS/Cartesian-quantity → curvilinear chart
  via `radius_vector` substitution (`cyl.evaluate(a) → I`). The reverse
  (curvilinear-quantity → other chart, e.g. `cart.evaluate(b)`) **errors clearly**
  — no silent `0`.
- **Inverse embeddings (for the future reverse direction, B): derive for the
  built-in charts** (cylindrical, spherical, polar); a *custom* chart would be
  user-supplied or stay forward-only. Not implemented now.

## Implementation design (approach A)

A chart must recognise a *WCS-identity* coordinate to reproject it. Nothing today
lets `cyl` discover that `x,y,z` are the reference's Cartesian coords (the
`ChartFrame` registry stores only `basis_id`; `structural_eq` ignores the
coordinate trait; `ws.coords` mints a fresh `chart_id` per set, so even the
"shared" `z` is a *distinct* coordinate object of a different chart). So:

1. **A minimal chart registry** on `Context`: at chart construction, register
   `coord_chart_id → { reference basis_id, embedding, coords, is_identity }`
   (a chart's coords share one `chart_id`; `is_identity` ⇔ `embedding[a] ≡
   coords[a]` for all `a`). Populate in the `PyChart` constructor (Python path)
   and a C++ `register_chart(ctx, chart)` for tests.
2. **Reproject pre-pass in `evaluate`.** Before lowering `expr` in chart `C`,
   walk it; for each coordinate `q` not belonging to `C`:
   - look up `q`'s chart in the registry;
   - *identity + same reference* ⇒ `q` is WCS coord slot `q.slot` ⇒ substitute
     `C.embedding[q.slot]` (a function of `C`'s coords);
   - *non-identity* (curvilinear source) ⇒ **error**: "needs the inverse
     embedding of chart D (vibe 000090 approach B)";
   - *unregistered* ⇒ error (can't reproject an unknown coordinate).
3. Then lower as today. The frame folds (`to_reference` / `simplify_basis_dot`,
   now with vibe-000089 distribution) collapse `e_r⊗e_r + e_θ⊗e_θ + e_z⊗e_z → I`.

Verified by hand: `cyl.evaluate(∇⊗(x i + y j + z k))` with `x→r cosθ, y→r sinθ,
z→z` grads to `I`. `expand`/`components` can route through the same reproject
later; start with `evaluate`.

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
