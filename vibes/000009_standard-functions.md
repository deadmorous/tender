# Standard Mathematical Functions

## Status: open questions

---

## Scope

Tender needs symbolic awareness of standard mathematical functions — not just
as opaque named nodes, but with enough structure to support differentiation
(chain rule), simplification (e.g., sin²+cos²=1), and LaTeX rendering.

## The complex exponential as the root

All classical transcendental functions derive from the complex exponential via
Euler's formula: e^(iθ) = cos θ + i sin θ. The exponential `exp(z)` is the
fundamental object; sin, cos, sinh, cosh, tan, etc. are derived. This is worth
keeping in mind for the internal representation — the system may treat `exp`
as primitive and the others as aliases or special evaluations.

## Function inventory

### Always needed
- `exp(x)` — exponential
- `log(x)` — natural logarithm (inverse of exp; defined for complex x)
- `sin(x)`, `cos(x)`, `tan(x)` — trigonometric
- `asin(x)`, `acos(x)`, `atan(x)`, `atan2(y, x)` — inverse trigonometric
- `sinh(x)`, `cosh(x)`, `tanh(x)` — hyperbolic
- `sqrt(x)` — square root (special case of power)
- `pow(x, n)` — power function x^n; n may be integer, rational, or symbolic

### Polynomials
A dedicated `Polynomial` type is included — it covers the most common
power-like use cases and is easier to reason about than a symbolic `pow(x, n)`.
See Q_symbolic_pow.

## Application domain: scalars only (for now)

These functions apply to scalar expressions. Applying them to tensors (e.g.,
matrix exponential exp(**A**)) is a separate, more advanced topic deferred to
a later tier. For now, if a user passes a non-scalar to `exp`, it is an error.

## Differentiation rules

Each function carries its derivative rule, enabling the chain rule to fire
automatically during symbolic differentiation:

```
d/dx exp(f)  =  exp(f) · f'
d/dx log(f)  =  f' / f
d/dx sin(f)  =  cos(f) · f'
d/dx cos(f)  = -sin(f) · f'
d/dx pow(f,n) = n · pow(f, n-1) · f'
```

These are built-in; users do not define them.

---

## Open questions

### Q_complex_numbers — RESOLVED

Already decided in vibe 000003: complex analysis is Tier 3. Standard functions
in Tier 1 are real-valued only. Inputs that would produce complex results (log
of a negative, asin outside [-1,1]) are out of scope for Tier 1; complex
support arrives as a first-class concept at Tier 3.

### Q_symbolic_pow — RESOLVED

A dedicated `Polynomial` type handles the common case of integer exponents,
with coefficient access, arithmetic, and differentiation. `pow(x, r)` is
retained for numeric non-integer exponents (rational, floating-point — e.g.,
`sqrt` = `pow(x, 1/2)`, `pow(x, 2/3)`). Fully symbolic exponents (arbitrary
expression as the power) are deferred; they are difficult to simplify and rarely
needed in Tier 1 mechanics.

### Q_known_identities — RESOLVED

Standard function identities are part of the identity library (vibe 000007).
They include at minimum:

- sin²x + cos²x = 1, and derived forms (1 + tan²x = sec²x, etc.)
- exp(a + b) = exp(a)·exp(b), exp(0) = 1, exp(log x) = x
- log(a·b) = log(a) + log(b), log(a/b) = log(a) − log(b), log(1) = 0
- sinh/cosh analogues of the trig identities
- pow(x, m)·pow(x, n) = pow(x, m+n), pow(pow(x, m), n) = pow(x, m·n)

These live in the standard functions submodule of the identity library.
