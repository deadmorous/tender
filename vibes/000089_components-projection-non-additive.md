# 000089 — chart.components mis-projects a sum (non-additive projection)

Status: **PLANNED** (found in vibe 000084; `evaluate` already sidesteps it, but a
direct operator user hits it).

## Symptom

`chart.components(X + Y)` can return the **wrong** physical components even though
the invariant `X + Y` is mathematically correct. Concretely, summing a gradient
of a *Sum*-coefficient with another operator result:

```python
A   = mu * cyl.div(cyl.grad(u))          # μ Δu
Bin = cyl.grad((lam + mu) * cyl.div(u))  # ∇((λ+μ)∇·u), coeff INSIDE the gradient
# A + Bin is the correct Navier–Lamé vector, but:
cyl.components(A + Bin)                   # WRONG in the r and z components
```

`evaluate` (vibe 000084) avoids this by hoisting diff-constant scalars OUT of the
operator, but a user calling the chart operators directly — or writing
`grad(c·div u) + div(grad u)` — gets a wrong answer with no warning.

## Root cause — the projection is non-additive (not the operators / canonicalize)

Pinned by bisection on the cylindrical case:

- **The invariant is correct.** `expand(canonicalize(A+Bin)) == expand(A+Bin)`
  and `A+Bin == A+Bout` (coeff outside) both hold — canonicalize does *not*
  corrupt the sum, and the operators produce the right invariant.
- **`components` is non-additive.** `components(A+Bin)[i] ≠ components(A)[i] +
  components(Bin)[i]` for i = r and z (θ is additive). Since `A+Bin == A+Bout`
  as invariants but `components(A+Bin) ≠ components(A+Bout)`, the projection of
  `A+Bin` is simply **wrong**.

So the fault is in `components` → `reduce_dot` (src/chart.cpp ~L203), which
projects `v` onto each `e_i` via `distribute_bilinear` → `simplify_basis_dot` →
`canonicalize` → `eval_delta_concrete` → `fold_arithmetic` → `canonicalize` →
`simplify_scalars`. One of these scalar/basis-reduction steps is **not additive**
across the summed terms.

**Why r and z, not θ:** `grad((λ+μ)·div u)` yields per-direction coefficients
split across a **fraction and a non-fraction** term on the *same* `e_i`
(e.g. the `e_r` coefficient is `\frac{…}{r²} + μ(∂_r∂_z u_z)`), because in
cylindrical `div u` carries `1/r`, `1/r²` weights. The θ-component happens to be
single-form, so it stays additive. The mis-combination therefore lives in the
mixed-denominator handling of `fold_arithmetic` / `simplify_scalars` /
`canonicalize` *inside* `reduce_dot` when the two operands' `e_i` coefficients
meet — most likely a **dummy-index collision** (two terms reuse an implicit-sum
id) or a **fraction-combination** slip surfaced only when both operands' terms
are present.

## Plan (to refine at implementation)

1. **Sharpen the mechanism.** Reduce further: build two hand-made concrete
   vectors whose `e_r` coefficients are `a/r² + b` and `c/r²` (fraction +
   non-fraction), sum, and check `components` additivity — does it reproduce
   *without* the operators? If yes ⇒ a pure `reduce_dot`/scalar-arithmetic bug;
   if no ⇒ a dummy-index collision from the operator results. Instrument
   `reduce_dot` to print `e` after each step and see which step first differs
   between `project(A+Bin)` and `project(A)+project(Bin)`.
2. **Fix the offending step.** Either
   - make the scalar reduction additive (a `fold_arithmetic` / `simplify_scalars`
     fraction-combination fix), or
   - restore dummy-index hygiene (rename implicit-sum ids per term before the
     reducing `canonicalize`, mirroring the `implicitize` / unbounded-dummy work
     in vibe 000064), or
   - project **term-by-term**: distribute the outer `+` before `reduce_dot` so
     each additive term is reduced independently, then sum (guarantees
     additivity by construction — likely the smallest safe fix).
3. **Lock it with an additivity invariant test.** `components(X+Y)[i] ==
   components(X)[i] + components(Y)[i]` for the Navier–Lamé `A`, `Bin`, and a
   couple of hand-built curvilinear vectors — in both cyl and spherical.

## Notes

- Whichever fix lands, keep `evaluate`'s constant-hoisting (vibe 000084): it is
  independently nice output (`(λ+μ)∇(∇·u)` reads better than `∇((λ+μ)∇·u)`), and
  it keeps `evaluate` robust even if a residual projection edge remains.
- Check `component_matrix` (rank-2) for the same non-additivity — it uses the
  same `reduce_dot`, so the fix should cover it; add a rank-2 additivity test.

See [[express-invariant-nabla-in-chart-plan]] (vibe 000084 — where this surfaced;
`evaluate` sidesteps it by hoisting constants), [[route-b-curvilinear-derivations]]
(reduce_dot / simplify_basis_dot), [[anf-design-in-progress]] (dummy-index /
implicitize hygiene, vibe 000064).
