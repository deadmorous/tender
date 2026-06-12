# Parameters, Time, and Parametric Derivatives

## Status: open questions

---

## Parameters as a general concept

Fields in tender are not only functions of spatial position — they can depend on
arbitrary scalar parameters. Spatial coordinates are a special case of parameters
that carry geometric meaning (linked to a CS). General parameters carry no
geometric meaning; they are just scalars the field depends on.

```python
t = Parameter('t')           # time
lam = Parameter('lambda')    # load parameter, arc-length, etc.
u = field('u', depends_on=[x, y, z, t])   # displacement field
```

The `deriv(param, expr)` function (introduced in vibe 000005) applies to any
parameter, spatial coordinate or otherwise:

```python
deriv(t, u)      # ∂u/∂t — time rate of change
deriv(lam, u)    # ∂u/∂λ — parametric sensitivity
deriv(x, u)      # ∂u/∂x — spatial partial derivative
```

---

## Time as a distinguished parameter

Time deserves special status: as soon as a derivation involves dynamics,
time appears everywhere. Tender is aware of time as a concept:

- A default `t` parameter is available as `from tender import t`
- Partial time derivative shortcuts: `dt(u)` = ∂u/∂t, `ddt(u)` = ∂²u/∂t²
- The overdot notation (ṙ, ü) denotes the full material time derivative D/Dt,
  which is distinct from ∂/∂t in continuum mechanics (see Q_material_derivative)
- Time appears in the expression tree as a named Parameter, not a special node,
  so all general parameter machinery applies to it

---

## Open questions

### Q_dot_notation — RESOLVED

`dt(u)` and `ddt(u)` denote the **partial** time derivatives ∂u/∂t and
∂²u/∂t².

The overdot notation (ṙ, ü — Newton's fluxion notation) denotes the **full
(material) time derivative** D/Dt, which in continuum mechanics is not the
same as ∂/∂t. This connects directly to Q_material_derivative below.

### Q_material_derivative — RESOLVED

`material_deriv(v, f)` is a named operation: D f/Dt = ∂f/∂t + **v**·∇f.
It is ergonomic, recognisable by the system (e.g., for identity matching and
simplification), and its expansion into `dt(f) + dot(v, grad(f))` is available
as an explicit derivation step when needed.

### Q_parameter_dependence_tracking — RESOLVED

Tender tracks parameter dependence on every object. Differentiating with
respect to a parameter the object does not depend on yields the zero tensor of
the corresponding rank — no error, no warning, just the correct mathematical
result.

```python
u = field('u', depends_on=[x, y, z, t])   # spatial field, time-dependent
A = tensor('A')                            # constant tensor, no dependencies

deriv(t, u)    # ∂u/∂t  — non-trivial
deriv(t, A)    # zero rank-2 tensor  — A does not depend on t
deriv(lam, u)  # zero rank-1 tensor  — u does not depend on λ
```

Dependence tracking also enables the system to short-circuit evaluation early
(no need to expand an expression if the result is known to be zero) and to
give informative derivation history entries ("A is independent of t → 0").
