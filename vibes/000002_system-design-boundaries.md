# System design and boundaries

_First discussed: 2026-06-01_

## Driving use case: Virtual Work Principle (PVW)

The primary workflow the system must support is PVW-based derivation of balance equations
and natural boundary conditions for mechanical systems — discrete, continual (3D), and
structural (rods, shells, plates as embedded manifolds).

PVW statement (continuum, d'Alembert form):

∫_V **σ**:δ**ε** dV = ∫_V **f**·δ**u** dV + ∫_∂V **t**·δ**u** dS − ∫_V ρ**ü**·δ**u** dV

for all virtual displacements δ**u** compatible with constraints.

---

## Two classes of derivation steps

### Mechanical steps — system handles automatically

- Linearity of integrals
- Algebraic simplification of integrands (contraction identities, symmetry, ε-identities)
- Applying the divergence theorem and integration by parts as rewriting rules
- Collecting/grouping terms by virtual quantity (all δ**u** together, all δ**ε** together)

### Creative steps — user drives explicitly

- Positing a kinematic hypothesis (e.g. plane cross-sections, Kirchhoff-Love, Reissner-Mindlin)
- Choosing strain measures for rods/shells (non-trivial; reflects physical insight, not algebra)
- Declaring which terms vanish by assumption (thin-wall, inextensibility, small strain, etc.)
- Invoking **localization**: the logical step from "integral = 0 for all admissible δ**u**"
  to "integrand = 0 pointwise" (fundamental lemma of calculus of variations — a theorem,
  not an algebraic identity; treated as a named, explicit inference step)

The rod strain measure problem is a concrete example of a creative step: choosing
**ε**(s) = **r**'·**t** − 1 vs. alternatives encodes physical judgment the system cannot supply.

---

## Mental model: derivation assistant, not automaton

The system is a **tensor algebra derivation assistant**:
- The user drives strategy at creative decision points
- The system handles all tensor algebra in between
- Analogous to a CAS handling polynomial manipulation while the mathematician directs the proof

---

## Expression tree: extended with integrals

A new node type `Integral(integrand, domain)` where domain carries geometric character:

| Domain type     | Geometry              | Has                        |
|-----------------|-----------------------|----------------------------|
| `Volume(V)`     | 3D region             | outward normal **n** on ∂V |
| `Surface(S, n)` | 2D manifold in 3D     | boundary curve ∂S          |
| `Curve(C, t)`   | 1D manifold in 3D     | endpoints, tangent **t**   |
| `Point`         | 0D (discrete loads)   | —                          |

### Integral rewriting rules

- **Linearity** — automatic
- **Divergence theorem**: ∫_V ∇·**A** dV → ∫_∂V **A**·**n** dS (user-invoked; tensor generalizations included)
- **Integration by parts** (derived from divergence theorem, named for convenience):
  ∫_V **A**·(∇**B**) dV = ∫_∂V (**A**⊗**n**):**B** dS − ∫_V (∇·**A**)·**B** dV
- **Localization** — user-invoked logical inference step, not algebraic rewriting

---

## Scope summary

| Capability                                   | Status        |
|----------------------------------------------|---------------|
| Symbolic tensor expressions (all ranks)      | In scope      |
| Algebraic simplification / rewriting rules   | In scope, core|
| Coordinate systems (3D + embedded manifolds) | In scope      |
| Tensor analysis (∇, div, curl)               | In scope      |
| Integral expressions with domain types       | In scope      |
| Divergence theorem / integration by parts    | In scope      |
| Localization (named inference step)          | In scope      |
| Kinematic hypotheses / strain measure choice | User-supplied |
| Numerical evaluation (terminal pass)         | In scope      |
| FEM shape function evaluation                | In scope (terminal)|
| Automated proof search / strategy            | Out of scope  |
