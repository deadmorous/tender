# Polynomial Extensions

## Status: design resolved — PolynomialExpr implemented

---

## Context

The Phase 5 `Polynomial` class stores exact `Rational` coefficients and
integer exponents for a single anonymous variable.  It supports arithmetic,
formal differentiation, rational-point evaluation, and LaTeX/Python output.
A key limitation was identified: `Polynomial` is self-contained and cannot
be embedded in an `Expr` tree, making it unusable inside expressions.

---

## Core insight — ring/field abstraction

A polynomial over an associative ring is evaluable at any ring element.
The ring structure determines what "power" means:

| Ring element          | Product operation   | Unit element  | Negative powers? |
|-----------------------|---------------------|---------------|-----------------|
| Scalar (`Expr`, rank 0) | `make_scale` / `make_product` | `RationalConst(1)` | Yes — via existing `Pow` node |
| Rank-2 tensor         | Single contraction `make_contract` | `IdentityTensor` | Not yet — needs tensor inverse |

Two types now exist with complementary roles:

- `Polynomial` — standalone value type (no `ResourceList`): exact-rational
  coefficient arithmetic, formal differentiation, evaluation at `Rational`.
  Remains useful wherever a ResourceList-free computation is needed.

- `PolynomialExpr : public Expr` — embeds a `Polynomial` plus a `var`
  expression into the tree.  `rank()` equals `var->rank()`.  Provides
  `expand(ResourceList&) -> Expr*` which materialises the sum of ring-power
  terms.

---

## PolynomialExpr design

### Construction

```cpp
PolynomialExpr(Polynomial poly, Expr* var)
```

- `var->rank()` must be 0 or 2 (only supported rings).
- `Polynomial` already rejects negative exponents; rank-2 negative exponents
  would additionally require a tensor inverse (not yet implemented — deferred
  to a future phase).

### Power semantics

Implemented by a private `ring_power(rl, var, n)` helper:

```
rank 0, n ≥ 0: make_pow(rl, var, n)        → existing Pow node
rank 0, n < 0: make_pow(rl, var, n)        → existing Pow node (field)
rank 2, n = 0: make_identity(rl)           → I
rank 2, n = 1: var
rank 2, n ≥ 2: make_contract(rl, var, var^{n-1})   → repeated ·
```

### expand(rl) → Expr*

Builds `∑ coeff_i * ring_power(var, exp_i)`.  For a zero polynomial, returns
`RationalConst(0)`.

### Rendering

LaTeX — rank 0: delegates to `Polynomial::to_latex(var->latex())` with
parenthesisation around `Sum` variables.  Rank 2: same but the constant term
(exponent 0) renders as `coeff \mathbf{I}` instead of bare `coeff`.

Python — rank 0: delegates to `Polynomial::to_python(var->python())`.
Rank 2: serialises as `polynomial_expr(var, [(coeff, exp), …])`.

### Dependency tracking and differentiation

`depends_on(p, poly_expr)` delegates to `depends_on(p, poly_expr->var())`.

`deriv(rl, p, poly_expr)` applies the chain rule for rank-0 variables:

```
d/dp p(v) = p'(v) · dv/dp
```

where `p'` is `poly.diff()` as a `Polynomial` (coefficient-only, no
ResourceList), wrapped in a fresh `PolynomialExpr` node, and multiplied by
`deriv(rl, p, var)` via `make_product`.  Rank-2 variables reach the zero
shortcut before this case because `NamedTensor` always reports no parameter
dependencies.

---

## Status of original directions

| Direction | Status |
|---|---|
| Symbolic coefficients (`Expr*`) | Still deferred — wait for concrete use case |
| `eval(rl, Expr*)` scalar | **Subsumed by `PolynomialExpr` + `expand()`** |
| Matrix polynomial | **Subsumed by `PolynomialExpr` with rank-2 var** |
| Multivariate polynomial | Still deferred — separate `MultivariatePolynomial` class |

---

## Summary table

| Feature | Blocking dependency | Status |
|---|---|---|
| `PolynomialExpr` with scalar var | None | **Implemented** |
| `PolynomialExpr` with rank-2 var | None | **Implemented** |
| Negative exponents for rank-2 | Tensor inverse primitive | Deferred |
| Symbolic coefficients | Concrete use case | Deferred |
| Multivariate polynomial | Design decision on storage | Deferred |
