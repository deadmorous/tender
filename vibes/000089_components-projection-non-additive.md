# 000089 — simplify_scalars must distribute a product over a sum

Status: **DONE** (root cause was mis-diagnosed at first — see below).

## The correction (what this really was)

The symptom, from vibe 000084, looked like a projection bug:
`chart.components(A + Bin)` (with `Bin = grad((λ+μ)·div u)`, coefficient *inside*
the gradient) seemed to give a wrong z-component, and `components` looked
*non-additive*. **That diagnosis was wrong.** Bisecting with a *stronger* zero
check exposed the truth:

- The invariant is correct: `A+Bin == A+Bout` via `expand`, and
  `components(A+Bin)[i] == components(A+Bout)[i]` under a strong check.
- The projection **is** additive and correct.
- The failing comparisons used `simplify_scalars(...) == "0"`, and the residual
  was `r·(∂_z u_r + …) − r·∂_z u_r − …` — **algebraically zero**, but
  `simplify_scalars` left it as `−a r − b r + r(a+b)` (not reduced).

So the operators, `components`, and `evaluate` were **all correct all along**.
The real bug is small and general: **`simplify_scalars` did not distribute a
scalar product over a sum** (`r·(a+b)`), so a polynomial's factored and expanded
forms never reconciled — every downstream `is_zero` / equality check (tests,
examples, the user's component comparisons) gave false negatives.

## Root cause

`normalize_scalar` decomposes a scalar into a monomial `FactorBag` (coeff +
bases^powers) and treats a `Sum` factor like `(a+b)` as **one atomic base**. So
`r(a+b)` stays a single monomial `{r:1, (a+b):1}` and never combines with the
distributed monomials `ra`, `rb`. `combine_fractions` (common-denominator) has
the same blind spot in its numerators.

## Fix

`distribute_scalar_sum` (src/derivation.cpp) — a rank-0-only rewrite added to the
`simplify_scalars` fixed-point step: `r(a+b) → ra + rb` (mirrored for a left sum
and for `Difference`). canonicalize then collects the like monomials, so
`r(a+b) − ra − rb → 0` and factored/expanded forms normalise equal. **Rank-0
only** — a dyad/vector sum `(a+b)⊗c` is left factored (that is `expand_products`'
job, not scalar simplification).

Effect: robust scalar equality / `is_zero`; the curvilinear component displays
come out as flat, fully-expanded polynomials over the common denominator (longer
but standard). 823 C++ + 290 Python pass; navier_lame + strain_compatibility
verify. Tests: `SimplifyScalars.DistributesProductOverSum` (C++),
`test_simplify_distributes_scalar_product_over_sum` +
`test_simplify_leaves_tensor_sum_factored` (Python).

## Fallout / cleanup

- The vibe-000084 `evaluate` **constant-hoisting** (`∇(cX)=c∇X`) was originally
  justified as dodging an "index collision" — that reasoning was this same
  simplify_scalars gap, now fixed. The hoisting is kept purely as a cleanliness
  normalisation (`(λ+μ)∇(∇·u)` reads better than `∇((λ+μ)∇·u)`); its comment is
  corrected.
- No projection / operator change was needed — the earlier plan (project
  term-by-term / dummy-index hygiene) was based on the wrong diagnosis and is
  dropped.

## Lesson

A weak canonicaliser turns a *representation* gap into a phantom *correctness*
bug. When "X and Y differ" but both look right, check equality with a stronger
normaliser (here `expand_products` first) before blaming the producer.

See [[express-invariant-nabla-in-chart-plan]] (vibe 000084 — where the phantom
surfaced), [[route-b-curvilinear-derivations]] (reduce_dot / simplify_scalars).
