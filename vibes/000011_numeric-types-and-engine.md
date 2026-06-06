# Numeric Types and the Scalar Number Engine

## Status: open questions

---

## What we know so far

Vibe 000004 mentions "arithmetic on known constants (rational arithmetic)" as
an always-on simplification, but no numeric engine was chosen and no tier
assignment for number types was made.

---

## Number types by tier

### Tier 1 — Core (needed from day one)

| Type | Examples | Notes |
|---|---|---|
| Integers | 0, 1, −3, 42 | Exact; needed for indices, coefficients, exponents |
| Rationals | 1/2, −3/7 | Exact; needed for Rodrigues formula, normalisation constants |
| Named constants | π, e | Symbolic; appear in trig identities, Rodrigues, etc. |

Named constants like π and e are symbolic atoms — they participate in
simplification rules (sin(π) = 0, exp(1) = e, etc.) but are never converted
to floating-point unless the user requests numerical evaluation.

### Tier 2a — Scalar CAS extensions

| Type | Examples | Notes |
|---|---|---|
| Algebraic numbers | √2, ∛5, roots of polynomials | Exact; arise naturally from eigenvalue problems |
| Symbolic parameters | λ, μ, E, ν | Already in Tier 1 as named scalars; exact arithmetic here means simplification, not evaluation |

### Tier 3

| Type | Notes |
|---|---|
| Floating-point | For numerical evaluation; IEEE 754 double initially |
| Arbitrary precision | If needed for ill-conditioned symbolic evaluation |
| Complex numbers | As decided in vibe 000009 |

---

## Open questions

### Q_rational_engine — RESOLVED

**Decision: stable API over a swappable backend; start with (a), upgrade to
(c) if overflow is ever observed in practice.**

**Dependency philosophy**: large dependencies trade short-term convenience for
long-term support burden. Small header-only libraries are acceptable; large
ones (Boost, GMP) are avoided unless unavoidable. Own projects are privileged
dependencies. Re-inventing wheels is acceptable for codebase stability.

#### Interface-first design

The integer and rational types expose a fixed API (arithmetic operators,
comparison, conversion, GCD) that does not reveal the underlying
representation. No abstract classes or virtual functions — just a header
with a concrete type whose internals can be swapped. Callers never depend on
whether the backend is fixed-width or arbitrary-precision.

#### Backend progression

**(a) `int64_t`-backed rational (start here)**: ~100 lines, trivial to
implement, fast. Overflow triggers a **fatal error** — not silent corruption.
If overflow is ever hit in practice, that is the signal to upgrade.

**(c) Home-made arbitrary-precision backend (upgrade path)**: ~400–600 lines,
correct for all inputs, no external dependency. Swapped in behind the same API
when (a) proves insufficient.

**(b) GMP**: possible future option behind the same API if (c) becomes a
performance bottleneck. Not anticipated.

### Q_named_constants — RESOLVED

Named constants (π, e, and domain-specific ones) are named objects from the
named object library (vibe 000010), exactly like named tensors. They carry
pre-registered simplification rules (sin(π) = 0, exp(1) = e, etc.) which live
in the identity library.

In identity **RHS**: named constants appear freely as literal atoms.

In identity **LHS** (pattern matching): a pattern variable can be constrained
to match only a specific named constant, but this is a very tight requirement
that narrows the match to essentially just that value. This use case is rare;
usually named constants appear in RHS, not LHS.

**Symbolic variables** (scalars like λ, μ, E, ν) are opaque — no built-in
simplification rules apply beyond algebra. They can however carry **traits**
via the same constraint mechanism as tensor pattern variables (vibe 000007):

```python
lam = scalar('lambda').non_negative()
n   = scalar('n').integer().positive()
```

Traits like `non_negative`, `positive`, `integer_valued`, `unit` (= 1 in
magnitude) serve as requirements for identity matching and can enable
simplifications (e.g., √(λ²) = λ if λ is non-negative).

### Q_algebraic_numbers — RESOLVED

Not in Tier 1. Deferred to Tier 2a. In Tier 1, `sqrt(2)` remains a symbolic
expression — an unevaluated `pow(2, 1/2)` atom — rather than an exact algebraic
number type. Eigenvalues introduced via the spectral theorem (vibe 000007) are
named symbolic scalars, not computed algebraic numbers.
