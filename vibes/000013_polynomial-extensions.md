# Polynomial Extensions

## Status: open questions

---

## Context

The Phase 5 `Polynomial` class stores exact `Rational` coefficients and
integer exponents for a single anonymous variable.  It supports arithmetic,
formal differentiation, rational-point evaluation, and LaTeX/Python output.
Three independent directions for future generalisation were identified; none
has been implemented yet.

---

## Direction 1 — Symbolic (Expr*) coefficients

**Motivation.** Coefficients that are themselves symbolic expressions arise
naturally: e.g., a polynomial whose leading coefficient is a material constant
`E`, or a Lagrange basis polynomial whose nodes are symbolic parameters.

**Issue.** The current API is entirely self-contained — arithmetic requires no
`ResourceList`.  Changing coefficients to `Expr*` makes every operation
(`+`, `*`, `diff`) produce new `Expr*` objects, which requires a `ResourceList`
to be threaded through the entire arithmetic interface.  That is a large,
breaking API change.

**Preferred path.** Keep the exact-rational `Polynomial` as-is.  Introduce a
parallel type (working name `ExprPolynomial`) when a concrete use case arises,
rather than generalising prematurely.  The two types can share the same
rendering logic and formal-derivative structure.

---

## Direction 2 — Evaluating polynomials over a ring (including tensors)

**Motivation.** A polynomial over a ring can be evaluated at any element of
that ring.  Two sub-cases are relevant:

1. **Scalar `Expr*` argument.**  Evaluating `p` at a symbolic scalar `x`
   produces an `Expr*` tree built from `make_pow`, `make_scale`, and `make_sum`.
   This is a straightforward addition alongside the existing `eval(Rational)`:
   a new overload `eval(ResourceList& rl, Expr* arg) -> Expr*` that works today
   without any new infrastructure.

2. **Rank-2 tensor argument (matrix polynomial).**  Evaluating `p` at a matrix
   **A** requires interpreting `A^n` as repeated single contraction
   **A**·**A**·…·**A**, not as `make_pow(A, n)` (which is undefined for
   non-scalar arguments).  This needs a dedicated "tensor power" primitive
   that does not exist yet.  Applications include the Cayley-Hamilton theorem,
   matrix exponentials approximated by truncated series, and spectral
   projectors in structural mechanics.

**Preferred path.** Add `eval(ResourceList& rl, Expr* arg) -> Expr*` as a
near-term convenience for the scalar case.  The matrix-polynomial evaluation
is a later-phase feature gated on a tensor-power primitive.

---

## Direction 3 — Multivariate polynomials

**Motivation.** Many practical applications require polynomials in more than
one variable:

- **FEM shape functions**: e.g., bilinear quad shape functions are polynomials
  in two coordinates (ξ, η), such as `N₁ = (1−ξ)(1−η)/4`.
- **Polynomial bases for approximation**: Legendre, Bernstein, or monomial
  bases in 2-D or 3-D.
- **Tensor-valued polynomial fields**: displacement fields in isoparametric
  elements are vector-valued multivariate polynomials.

**Design options.**

| Option | Description | Trade-off |
|---|---|---|
| Sparse monomial map | `map<vector<int>, Rational>` keyed by multi-index | General; expensive for dense low-degree cases |
| Tensor-product form | Store as a product of univariate polynomials | Efficient for quad/hex elements; not general |
| Recursive nesting | `Polynomial<Polynomial<Rational>>` | Clean algebra; awkward for > 2 variables |

The sparse monomial representation is the most general and the natural fit for
FEM: shape function libraries typically enumerate monomials up to a given total
degree.  The number of variables and the variable names (or indices) would be
fixed at construction time.

**Current gap.** The univariate `Polynomial` provides no path to multivariate
extension without a redesign of the storage and arithmetic.  The `eval` and
`diff` APIs would need to become indexed (`eval(variable_index, value)`,
`diff(variable_index)`), or take maps from variable name to value/argument.

**Preferred path.** Design a separate `MultivariatePolynomial` class rather
than retrofitting `Polynomial`.  The univariate class remains useful as a
building block (e.g., as the coefficients of a multivariate polynomial in
one distinguished variable) and for cases where the single-variable assumption
genuinely holds.

---

## Summary table

| Extension | Blocking dependency | Priority driver |
|---|---|---|
| `eval(rl, Expr*)` for scalars | None — ready to add | Symbolic evaluation, parametric FEM |
| Symbolic coefficients (`Expr*`) | Concrete use case | Parametric coefficients |
| Matrix polynomial (`eval` at rank-2) | Tensor-power primitive | Cayley-Hamilton, matrix functions |
| Multivariate polynomial | Design decision on storage | FEM shape functions |
