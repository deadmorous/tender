# 000084 — express an invariant ∇-expression in any coordinate system

Status: **DONE** — approach A shipped (`chart.evaluate`).

## Implemented (approach A)

`chart.evaluate(expr)` (src/chart.cpp `evaluate`, bound in `_core.cpp`, C++
`tender::evaluate`) — a core-Expr interpreter that lowers an invariant `t.nabla`
expression to the chart operators, inner-first:
`Dot(∇, X) → divergence(eval X)`, `TensorProduct(∇, X) → gradient(eval X)`,
`Cross(∇, X) → rot(eval X)`; so `∇·(∇⊗X)` composes to `div(grad) = Δ` with no
special case (`laplacian` *is* `div(grad)`). `Sum / Difference / Negate /
ScalarDiv / scalar⊗ / Transpose / Trace / vec / DDot` pass through and recurse;
a ∇-free sub-expression (via `contains_nabla`) is an operand returned untouched;
a non-`∇`-leading `Dot`/`Cross` of ∇-containing sub-results uses
`frame_dot`/`frame_cross`. A bare `∇` (no operand) throws.

**Decisions settled:** surface = `chart.evaluate(expr)` (mirrors the DSL
`.evaluate(chart)`); output = an invariant in the chart's physical frame (like
`divergence`/`gradient`), `chart.components` reads the physical components;
unsupported-pattern policy = hard error with a clear message. Approach A now, D
(unify the two ∇s) later — A can be subsumed.

**Verified:** on a *cylindrical* chart, `evaluate(∇·u)=div`, `∇⊗u=grad`,
`∇×u=rot`, `∇·(∇⊗u)=Δ`, `μΔu`, and the **full Navier–Lamé `∇·T`** — written
coordinate-free with `u` abstract — match the operator-built endpoint component
by component (Cartesian + cylindrical). A *canonicalized* invariant evaluates the
same as the raw one (vibe 000085 preservation pays off — the interpreter may
canonicalize freely). Tests: `Chart.EvaluateLowersNablaToChartOperators` (C++,
curvilinear + bare-∇ throw), `test_chart_evaluate_lowers_invariant_nabla_cylindrical`
+ `test_chart_evaluate_bare_nabla_raises` (Python); showcase in
`examples/navier_lame.py` §6.

This **supersedes the vibe-000081 "∇-first is the only supported order" rule**:
write it invariant, `chart.evaluate` it.

### Robustness pass (evaluating a *reduced* invariant)

A user's expression is often the *reduced* endpoint a derivation leaves, not the
raw `∇·T`. Evaluating `factor_common(collect_terms(reass))` — the Navier–Lamé
`μΔu + (λ+μ)∇(∇·u)` — surfaced three shapes the first cut mishandled (obscure
"bare ∇" errors / a wrong z-component). Fixed:

1. **Floated Laplacian.** `(∇·∇)⊗X` (renders `ΔX` but is `Dot(∇,∇)⊗X`, the
   vibe-000083 canonical form) → `laplacian`. `evaluate` now flattens the
   ⊗-chain and treats a bare `∇·∇` factor as the Laplacian operator.
2. **Operator-left-normalised gradient.** `X⊗∇` (∇ on the *right*, vibe-000080
   `X⊗∇ = (∇⊗X)ᵀ`) → `gradient` (transposed for a rank-≥1 operand). The
   flatten-and-find-operator handles the operator at the FRONT (`∇⊗X`, floated
   `(c ∇·∇)⊗X`) or the BACK (`X⊗∇`).
3. **Constant hoisting.** A diff-constant scalar coefficient *inside* an operator
   (`grad((λ+μ) div u)`, as `factor_common` leaves it) is value-correct but hits
   a **chart-operator index-collision** when summed with another operator term
   (`grad(c·div u) + div(grad u)` mis-reduces — a *pre-existing* operator bug,
   independent of `evaluate`). `evaluate` sidesteps it by hoisting diff-constant
   scalars OUT of the operator (`∇(cX)=c∇X`, via a local `is_diff_constant`).

Error messages clarified: a bare `∇` or a bare `∇·∇` (Laplacian operator with no
operand) now name the problem and how to fix it.

**KNOWN follow-up (separate bug):** the chart operators mis-reduce
`chart.grad(c · chart.div(u)) + chart.div(chart.grad(u))` (a scalar coefficient
inside one gradient, summed with another operator result) — an index-hygiene
issue in the operators/`components`, not `evaluate`. `evaluate` avoids it by
hoisting constants; a direct user hitting it would see the same. Worth its own
vibe.

(Other follow-ups: N-field structural reassembly beyond two-field is still vibe
000088's; a `Dot`/`DDot` of two frame-reduced sub-results uses
`frame_dot`/`make_ddot` best-effort — fine for the operator patterns, revisit if
a genuine bilinear-of-evaluated-operands case needs full frame reduction.)

---

Status (original): **PLANNED / PROBLEM STATEMENT**. No code yet.

## The problem (user pain)

A user writes an invariant coordinate-free expression with `t.nabla` and abstract
tensor fields — e.g. `∇·u`, `Δu = ∇·(∇⊗u)`, the Navier–Lamé `∇·T` — for display
and for the coordinate-free record. To then **evaluate it in a chart** (esp. a
curvilinear one like cylindrical) they must *rewrite the whole thing* with the
chart operators `cyl.grad / div / rot / laplacian`:

```python
expr = nabla @ (nabla * u)        # what the user has (Δu, coordinate-free)
cyl.div(cyl.grad(u))              # what they are forced to re-type to evaluate
```

This **doubles the work** (write the invariant, then hand-translate it), is hard
to teach ("why can't I just evaluate the thing I wrote?"), and is exactly the
sharp edge behind the vibe-000081 "∇-first is the only supported order" rule.
The user's ask: **there must be a clean `expr → in this chart` path**, and the
tender-side obstacles "must be resolvable." They are — see below.

## Current state: two ∇ worlds, two lowering routes

**Two `∇` objects:**
- **Core `t.nabla(ctx)`** — a raw `Expr` `Nabla` node. `nabla @ u` / `nabla * u`
  / `nabla % u` build ordinary core `Dot` / `TensorProduct` / `Cross` trees with a
  `Nabla` leaf. This is what every example and the user use. Renders (∇·u, Δu…).
- **DSL `tender.operators.nabla`** — a `Nabla` wrapper whose `@ * %` build a
  `DifferentialExpr("div"/"grad"/"rot", operand)`, composable and **`.evaluate(chart)`-able on ANY chart** (verified: `(operators.nabla @ (nabla*u)).evaluate(cyl)`
  gives the correct cylindrical Δu components).

**Two lowering routes for a core-`∇` expression:**
- `chart.expand_nabla(e)` — the *free-index* route: replaces each `∇` by
  `e_i ∂_i` as a **single implicitly-summed term**, keeping the operand abstract
  (so `∇×(∇×ε)ᵀ` stays symbolic in ε). **Cartesian-only** — it asserts every
  scale factor `h_i == 1` and throws otherwise (src/chart.cpp ~L769).
- `chart.grad/div/rot/laplacian(f)` — the *concrete-component* route
  (`del_apply`, src/chart.cpp ~L686): **expand the field into frame components
  first** (`Σ T_ij e_i e_j`), then differentiate the explicit sum, **unrolling i
  concretely** so each `1/h_i` is a per-term scalar and `∂_i e_j = γ^k_{ij} e_k`
  (the connection) falls out by Leibniz. Curvilinear-correct. Composes
  (`cyl.div(cyl.grad(u))` works).

## The real obstacles (why `expand_nabla` is Cartesian-only)

The free-index route's whole value — keeping the operand **abstract** — is exactly
what blocks curvilinear:
1. **Per-direction scale factors on a summed index.** `∇ = Σ_i (1/h_i) e_i ∂_i`
   is kept as ONE summed term; a moving frame's `1/h_i` differs per i (e.g.
   `h_θ = r`) and cannot ride a single Einstein-summed index as one scalar.
2. **Moving-frame connection.** `∂_i e_j ≠ 0`; the derivative must hit the frame
   vectors too (`γ^k_{ij}`). The free-index route never materialises the operand's
   components, so there is nothing for the connection to act on.

**Key finding:** the concrete route (`del_apply`) already solves BOTH by
expand-then-differentiate + concrete unroll — "the only correct order in a moving
frame." And the DSL `DifferentialExpr.evaluate(chart)` is **already the clean
dispatcher** that lowers a whole `∇`-tree to `chart.grad/div/rot/laplacian`,
inner-first, on any chart. **The only gap is that it consumes DSL wrapper objects,
not core `t.nabla` Expr trees.** So the "clean way" is a bridge from a core-`∇`
invariant to that dispatch — NOT a new lowering algorithm.

## Candidate approaches (to decide next session)

- **A (recommended) — `chart.evaluate(expr)` core-Expr interpreter.** Walk the
  core tree; lower the `∇`-combinations to chart operators, recursing inner-first:
  `Dot(∇, X) → chart.div(eval X)`, `TensorProduct(∇, X) → chart.grad(eval X)`,
  `Cross(∇, X) → chart.rot(eval X)`, `Dot(∇, TensorProduct(∇, X)) → chart.laplacian`/`div(grad)`;
  pass `Sum / Difference / Negate / ScalarDiv / scalar⊗ / Transpose / I / DDot`
  through and recurse; a `∇`-free leaf is returned (optionally field-expanded).
  Returns an invariant in the chart's frame; `chart.components(...)` then gives
  physical components. Reuses the proven chart operators — smallest correct core.
- **B — core-Expr → `DifferentialExpr` converter,** then `.evaluate(chart)`.
  Same dispatch, staged through the DSL. More moving parts; core Expr is richer
  than the DSL, so a faithful converter is ~as much work as interpreting directly.
- **C — generalise the free-index `expand_nabla` to curvilinear.** Hard, and
  self-defeating: to carry `1/h_i` and the connection you must unroll i and
  materialise the operand — i.e. you rebuild the concrete route and LOSE the
  abstract-operand property that is `expand_nabla`'s only reason to exist.
- **D — unify the two `∇`s** so a single `nabla` both renders coordinate-free AND
  is `.evaluate(chart)`-able (fold the DSL onto core `t.nabla`, or vice-versa).
  Best long-term UX; largest blast radius. A can ship first and D can subsume it.

## Obstacles/risks for approach A (the design work)

1. **Interpret BEFORE canonicalize — RESOLVED by vibe 000085.** This *was* the
   main risk: `canonicalize` floated the operand across `∇` (`∇·(∇⊗X) →
   (∇·∇)⊗X`, the I2/B3 wall) irreversibly, so the interpreter had to run first.
   Vibe 000085 barred that float at its one site (`distribute_contraction`) and
   taught the nf model to carry the `∇⊗X` operator fence (a `Paren`), so
   `canonicalize` now **preserves** `∇·(∇⊗X)` and is idempotent. **The
   interpreter may canonicalize freely.** (The other reorderings the interpreter
   cares about — `Dot`/`⊗` grouping — are stable under the preserved nesting.)
2. **Operator precedence / association.** `nabla @ nabla * u` follows Python
   precedence; the interpreter follows whatever core tree that built. Document the
   parenthesisation, and recognise both `div(grad)` and the `(∇·∇)⊗` Laplacian.
3. **Composite terms.** Transpose `(∇u)ᵀ → chart.grad(u).transpose()`; sums and
   scalar coefficients (λ, μ); the identity `I`; `DDot`; nested compositions —
   each must pass through and recurse. Verify chart ops accept an already-lowered
   invariant operand (they do: `cyl.div(cyl.grad(u))`).
4. **Bare `∇` / `∇·∇` with no operand.** Not meaningful to evaluate → clear error.
5. **Output shape.** `chart.evaluate` returns the invariant-in-chart (like
   `chart.div`), and `chart.components` reads off frame components — or offer a
   one-shot `chart.components(chart.evaluate(expr))`. Decide the default surface.
6. **Rank / operator-kind inference** already handled by the chart operators
   (they infer the operand rank); the interpreter only routes `· / ⊗ / ×`.

## Open decisions (for tomorrow)

- Surface & name: `chart.evaluate(expr)` vs `expr.in_chart(chart)` vs a
  `td`/step. Symmetry with the DSL `.evaluate(chart)`.
- A now, D later? (bridge first, unify the two `∇`s afterwards.)
- Default output: invariant-in-chart vs components; scalar results.
- Unsupported-pattern policy: hard error vs leave-abstract.
- Supersede the vibe-000081 "∇-first is the only supported order" rule with
  "write it invariant, `chart.evaluate` it" once A lands.

Constraints (unchanged): buildable/tested per increment; ≥90% coverage;
clang-format; strip notebooks; never write `∇²`. See
[[vibe81-explicit-basis-operator-route]] (the ∇-first rule this relaxes),
[[differential-operators-and-strain-compat]] (Deriv/apply_operators, chart.nabla),
[[route-b-curvilinear-derivations]] (expand-then-differentiate, connection on the
chart), [[differential-foundations-plan]] (M4–M6 chart operators),
[[vibe80-notebook-gaps-sprint]] (basis-expand-first is a correctness trap).
