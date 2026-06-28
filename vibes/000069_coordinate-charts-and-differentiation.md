# 000069 — Coordinate charts, scalar fields, and differentiation (foundations for ∇)

Planning the layer that must exist *before* differential operators (roadmap
000038 Stage 5).  Today we have the invariant/coordinate algebra and bases
(vibes 000049, 000067), but **nothing** for: scalar fields (coordinate
variables, elementary functions), taking derivatives, specifying a coordinate
mapping, or deriving the geometry (metric, scale factors, ∂ of basis vectors)
from that mapping.  Without these, ∇ / div / rot / Laplacian can only be
hand-typed, not derived, and won't compose with the rest.

This vibe is a **plan only** — no implementation yet.

## Driving example (the spec): polar coordinates

Everything below must make this derivation mechanical, for *any* concrete
mapping:

1. **Mapping:** `x = r cos φ`, `y = r sin φ`  (chart coords `r, φ` → reference
   Cartesian `i, j`).
2. **Radius vector:** `R = x i + y j = r cos φ · i + r sin φ · j`.
3. **Tangent (holonomic) basis:** `g_i = ∂R/∂q^i`
   - `g_r = ∂_r R = cos φ · i + sin φ · j`
   - `g_φ = ∂_φ R = −r sin φ · i + r cos φ · j`
4. **Metric / scale factors:** `g_{ij} = g_i·g_j` → `g_rr = 1`, `g_φφ = r²`,
   `g_rφ = 0` (needs `cos²+sin² = 1`); `h_r = 1`, `h_φ = √(r²) = r` (needs `r ≥ 0`).
5. **Physical orthonormal basis:** `e_i = g_i / h_i`
   - `e_r = cos φ · i + sin φ · j`
   - `e_φ = −sin φ · i + cos φ · j`
6. **Derivative formulas:** differentiate the e_i (in the *constant* reference
   frame) and re-express in the local basis:
   - `∂_r e_r = 0`, `∂_r e_φ = 0`
   - `∂_φ e_r = −sin φ · i + cos φ · j = e_φ`
   - `∂_φ e_φ = −cos φ · i − sin φ · j = −e_r`
7. **∇ (and later Δ):** `∇ = e_i (1/h_i) ∂_{q^i}`, then div/rot/Laplacian follow,
   the curvilinear `1/r` factors and the `∂e` formulas falling out of 3–6.

## What we have vs what's missing

Have: scalar arithmetic on rank-0 (`Sum`/`Difference`/`TensorProduct`/
`ScalarDiv`/`Negate`), bases + dot/cross + reassemble, canonicalize/e-graph.

Missing (in dependency order):

1. **Scalar fields.** Coordinate *variables* (`r, φ`) as first-class symbols the
   differentiator recognises, and **elementary functions** (`sin, cos, sqrt,
   pow`, later `exp, log`).  Today the only scalars are `Rational` literals.
2. **A differentiation engine** `∂_{q}`: linearity, Leibniz over `⊗`/scalar·,
   quotient rule, chain rule with a derivative table; `∂_{q^i} q^j = δ_ij`;
   `∂` of a *constant* (reference basis vector, parameter, literal) `= 0`.
3. **Enough scalar simplification** for the geometry: Pythagorean
   `cos²+sin² → 1`, `√(r²) → r` (under `r ≥ 0`), power/trig folding.  This is the
   riskiest piece (a slice of scalar CAS).
4. **A coordinate chart**: `{reference Basis, coords q^i, embedding x^a = f^a(q)}`.
5. **The geometry pipeline**: `R → g_i → g_{ij} → h_i → e_i (+ cobasis) →
   Christoffel Γ → ∂_j e_i`, and a **bridge** so the derived physical frame *is*
   a tender `Basis` (carrying the chart's coordinate names from vibe 000067).
6. **Field specification**: a convention for "field over a chart" (scalar
   `f(q)`, vector `v = v^i(q) e_i`) so operators know what they act on.
7. **Differential operators**: `∇`, grad, div, rot, Laplacian on 1–6.

## Proposed building blocks (representations — proposals, open to change)

- **Coordinate variable:** a rank-0 atom tagged as chart coordinate `i` (a
  `TensorObject` rank-0 with a `Coordinate{chart_id, slot}` trait, or a small new
  node).  `∂` matches it; renders as its letter (`r, φ`), reusing vibe-000067
  naming.
- **Elementary function:** a unary `ScalarFn{kind, operand}` node (`sin, cos,
  sqrt, …`) plus `Pow{base, exponent}`; arithmetic reuses existing nodes.  Each
  `kind` has a derivative-table entry for the chain rule.
- **Differentiation:** a derivation step `partial(ctx, expr, coord)` returning an
  `Expr`.  Constancy is decided via the chart: reference basis vectors and
  non-coordinate symbols are constant; only the chart's coordinates vary.
- **Chart:** `CoordinateChart{ Basis reference; vector<Coordinate> coords;
  vector<Expr> embedding /* one f^a(q) per reference direction */ }`.
- **Geometry:** functions producing `g_i`, `g_{ij}`, `h_i`, the physical `Basis`,
  `Γ^k_{ij}`, and the `∂_j e_i` table; built on `partial` + dot + simplification.
- **Field:** lightweight — a field is just an `Expr` over a chart's coords/basis;
  operators interpret it.  Avoid a heavy type system initially.

## Milestones (each buildable + tested, per CLAUDE.md #1)

- **M1 — Scalar fields.** Coordinate-variable atom + `ScalarFn`/`Pow` nodes +
  rendering; interplay with existing scalar arithmetic.  *Alive:* can write
  `r cos φ`, render it.
- **M2 — Differentiation `∂_q`.** All rules + derivative table; `∂` of
  coordinates and of constant reference vectors.  *Alive:* `∂_r R`, `∂_φ R` give
  the step-3 tangent vectors.
- **M3 — Targeted scalar simplifier + per-coordinate domains.** The specific
  trig/`√(square)`/power identities the geometry needs, using the chart's
  coordinate domains where required (`r ≥ 0` ⇒ `√(r²) → r`).  *Alive:*
  `cos²+sin² → 1`, `√(r²) → r`.
- **M4 — Chart + tangent basis + metric + physical basis.** Chart spec → `g_i`,
  `g_{ij}`, `h_i`, orthonormal `e_i`/`e^i`; bridge to `Basis`.  *Alive:* build
  polar/cylindrical/spherical from their mappings; validate the derived `Basis`
  against the hand-written `coord_system` ones.
- **M5 — `∂_j e_i` / Christoffel.** Differentiate `e_i`, re-express in the local
  basis.  *Alive:* reproduce the step-6 polar formulas (and cylindrical/
  spherical).
- **M6 — Differential operators.** `∇`, grad, div, rot, Laplacian using the
  chart's `∂`, scale factors, and `∂e` table.  *Alive:* the cylindrical
  `∇ = e_r ∂_r + (1/r) e_φ ∂_φ + e_z ∂_z` and the curvilinear div/Laplacian.

## Decisions (settled)

0. **Naming: `rot`, not `curl`.**  The rotation operator is `rot` everywhere
   (more familiar); `curl` is not used.  (No code exists yet — this is the
   forward convention; the archived `attic/` is left as-is.)
1. **Scalar simplification = a targeted scalar simplifier**, starting small:
   the specific identities the orthogonal-curvilinear pipeline needs
   (Pythagorean `cos²+sin² → 1`, `√(square) → value`, power/product folding),
   extended on demand.  Promote rules into the e-graph/identity library later if
   they generalise.  Never require the user to hand-supply a simplified metric.
2. **Assumptions = per-coordinate domains, carried on the chart.**  Each
   coordinate has a known domain (e.g. cylindrical `r ≥ 0`, `φ ∈ (−π, π]`,
   `z ∈ ℝ`), and the simplifier *relies on the domain when needed* — e.g.
   `√(r²) → r` because `r ≥ 0`, scale factors taken positive.  Well-known charts
   ship with their domains; a custom chart supplies them.
3. **Physical (orthonormal) basis is the default** the operators use (matches
   textbook curvilinear ∇/div/rot/Δ); the holonomic `g_i` and the metric live
   underneath and are available.

## Remaining notes

- **IR footprint.** Keep new nodes minimal and additive (coordinate atom +
  `ScalarFn`/`Pow`); thread them uniformly through canonicalize / structural_eq /
  hash / render (as `basis_id` was threaded in vibe 000067).
- **Reuse vs new.** The embedding targets an existing orthonormal reference
  `Basis` (e.g. `wcs`); `partial` of vectors leans on the existing dot/reassemble
  for the "re-express in local basis" step.

## Relation to the roadmap

This fills the gap roadmap 000038 glossed over: Stage 4 named "metric,
Christoffel, √g" but never said *how* — the answer is "derive them from a
coordinate mapping", which needs the scalar-field + differentiation foundations
above.  Stage 5 (operators) then sits on M6.  Builds on [[basis-aware-indices-plan]]
(vibe 000067) for the frame + naming and on the basis layer (vibe 000049).

Status: **plan; scope confirmed** (decisions 0–3 settled).  Ready to start at
**M1** (scalar fields).
