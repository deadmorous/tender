# Architecture, tiers, and third-party strategy

_First discussed: 2026-06-02_

## Key terms

**CAS — Computer Algebra System.** Software that manipulates mathematical expressions
symbolically rather than numerically. Examples: Mathematica, Maple, SymPy (Python),
GiNaC (C++), SymEngine (C++). A CAS can expand polynomials, compute indefinite integrals,
factor expressions, solve equations symbolically, and so on. tcalc is, in essence,
a domain-specific CAS for tensor algebra and continuum mechanics derivations.

**UFL — Unified Form Language.** A domain-specific language embedded in Python, used by
the FEniCS finite element framework to express variational forms (weak forms of PDEs).
A UFL expression describes something like ∫_V σ:ε(v) dV in a form that FEniCS can
automatically discretize and assemble into finite element matrices. It is one example
of what the output of a tcalc derivation might be fed into for numerical solving.

---

## The containment observation

Tensor algebra contains scalar and matrix algebra as special cases:
- Rank-0 tensors are scalars
- Rank-2 tensors with a chosen basis are matrices
- Scalar fields on ℝⁿ are the domain of classical analysis

The expression tree and rewriting engine designed for tensors should degrade gracefully
to scalar CAS behavior without artificial exclusion. Much of Tier 2 comes for free
if the Tier 1 design is not needlessly tensor-specific.

---

## Tiered architecture

### Tier 1 — Core (primary focus)

Symbolic tensor algebra and PVW derivation assistant:
- Expression trees for tensor expressions of all ranks
- Algebraic simplification and rewriting rules
- Coordinate systems (3D volumes + embedded manifolds)
- Tensor analysis: ∇, divergence, curl
- Integral expressions with domain types (Volume, Surface, Curve, Point)
- Divergence theorem and integration by parts as user-invoked rewriting rules
- Localization as a named logical inference step
- Numerical evaluation as a terminal pass

### Tier 2a — Scalar CAS extensions

Falls naturally out of Tier 1 infrastructure applied to rank-0:
- Taylor series expansions
- Symbolic indefinite integration
- Series analysis, residual term estimation

### Tier 2b — ODE/PDE symbolic analysis

Symbolic manipulation of systems that emerge after PVW localization:
- Characteristic polynomials of differential operators
- Stability analysis of numerical schemes (roots of stability polynomials)
- Formal ODE/PDE manipulation

### Tier 3 — Specialized domains

- **Complex analysis**: holomorphic functions as a field type, Cauchy-Riemann as a rewriting
  rule; needed for 2D elasticity via Kolosov-Muskhelishvili potentials (complex potential
  functions Φ(z), Ψ(z) express stress and displacement fields in closed form for 2D problems)
- **Numerical ODE integration**: design and analysis of time-stepping schemes for Cauchy problems
- **PDE solving**: delegate entirely to external solvers via bridges (see below)

---

## Third-party strategy: build from scratch, bridge optionally

**Core principle: build from scratch. No exceptions.**

Embedding a third-party CAS (GiNaC, SymEngine, or similar) as the foundation would
create a structural dependency that constrains every design decision — expression tree
shape, simplification strategy, extension points — to what that library allows.
The risk of sinking into someone else's architecture outweighs the short-term convenience.

**Third-party systems are accessed through dedicated, optional bridge interfaces only:**

- The bridge is a thin adapter layer; the core has no knowledge of what is on the other side
- Bridges are optional — the system is fully functional without any bridge installed
- Examples of useful bridges:
  - **Symbolic integration bridge**: delegate indefinite integrals to an external CAS
    (GiNaC, SymEngine, or even Mathematica/Maple via IPC) when Tier 2a integration is needed
  - **FEM bridge**: emit the localized weak form as UFL (FEniCS) or equivalent,
    for downstream numerical solving by deal.II, FEniCS, FreeFEM, etc.
  - **ODE solver bridge**: emit evaluated ODEs to SUNDIALS, scipy, or similar
  - **Linear algebra bridge**: emit evaluated matrix expressions to Eigen or similar

**The user's own general-purpose C++ library is an unconditional dependency** (not a
third-party bridge — it is part of the project's foundation). It provides:
strong types, logging, JSON & YAML serialization, and other general-purpose utilities.
It is orthogonal to the tensor algebra domain and will be used throughout.

---

## Design implications of the scratch-build decision

1. **Expression tree is ours**: its node types, visitor interface, and extension points
   are designed for tensor algebra first, scalar CAS second — not inherited from elsewhere.

2. **Simplification engine is ours**: rewriting rules, normal form strategy, and
   termination guarantees are under our control.

3. **Bridge interfaces must be narrow and stable**: they translate our expression tree
   to external formats, never the reverse. External representations do not leak inward.

4. **Scalar CAS capabilities grow organically**: we implement what we need, when we need it,
   rather than pulling in a full CAS and using 10% of it.
