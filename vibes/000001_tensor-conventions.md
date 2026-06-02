# Tensor algebra conventions and scope

_First discussed: 2026-06-01_

## Context

Designing a C++ tensor algebra library for analytical work in computational mechanics.
The system is primarily a **symbolic computation engine with simplification**, not a numerical array library.
Numeric evaluation (against concrete numbers, spatial fields, FEM shape functions) is a late-stage pass.

The user works in direct (coordinate-free) tensor notation: vectors as directed lines in Euclidean 3D space,
rank-2 tensors built from dyads **a**⊗**b**, higher-rank tensors from polyads.

---

## Conventions established

### 1. Gradient convention

∇ = **g**ⁱ∂ᵢ (contravariant basis vector times partial derivative, Einstein summation).

∇**v** = **g**ⁱ⊗(∂ᵢ**v**), so component (∇**v**)ᵢⱼ = ∂vⱼ/∂xⁱ — ∇-index first, field-index second.

In curvilinear coordinates ∂ᵢ becomes the covariant derivative ∇ᵢ, absorbing Christoffel symbols.
The tensor ∇**v** is invariant; only its coordinate expression changes with the basis.

**Design implication:** a component array is never a sufficient representation — it always needs
its basis/metric context. All numerical work happens in a fixed frame; basis transformations are explicit.

### 2. Single contraction ( · )

Contracts the rightmost index of the left operand with the leftmost index of the right operand
("innermost vectors of adjacent polyads face each other").

### 3. Double contraction — two operators

- **A**:**B** = AᵢⱼBᵢⱼ  — same-order contraction (Frobenius); equivalent to tr(**A**ᵀ·**B**)
- **A**··**B** = AᵢⱼBⱼᵢ — reversed-order contraction; equivalent to tr(**A**·**B**)

Relation: **A**:**B** = **A**··**B**ᵀ. Coincide for symmetric tensors.

For higher-rank tensors both operators consume the two innermost (adjacent) index pairs —
the rightmost pair of the left operand and the leftmost pair of the right operand.

### 4. Cross product

**a**×**b** = aᵢbⱼεᵢⱼₖ **e**ₖ = −**a**·**ε**·**b**

where **ε** = εᵢⱼₖ **e**ᵢ⊗**e**ⱼ⊗**e**ₖ is the Levi-Civita tensor.

Extends uniformly to tensor operands via the same innermost-index rule:
- **T**×**v** = −**T**·**ε**·**v**
- **v**×**T** = −**v**·**ε**·**T**

In curvilinear coordinates: εᵢⱼₖ = √g·[ijk], where g = det(gᵢⱼ).

---

## Scope of tensor ranks

All ranks needed. Rank-4 arises from the elasticity stiffness tensor **C** with the following
symmetry classes (all required):

| Class                  | Independent constants |
|------------------------|----------------------|
| Full anisotropy        | 21                   |
| Orthotropy             | 9                    |
| Transverse isotropy    | 5                    |
| Isotropy               | 2 (λ, μ)             |

Symmetries of **C**: minor (Cᵢⱼₖₗ = Cⱼᵢₖₗ = Cᵢⱼₗₖ) and major (Cᵢⱼₖₗ = Cₖₗᵢⱼ), reducing 81→21.

### Special tensors needed

- Rank-2 identity: **I** = δᵢⱼ **e**ᵢ⊗**e**ⱼ
- Levi-Civita: **ε** = εᵢⱼₖ **e**ᵢ⊗**e**ⱼ⊗**e**ₖ
- Isotropic rank-4 basis (three tensors without symmetry constraints):
  - **P** with components δᵢⱼδₖₗ  (= **I**⊗**I**)
  - **Q** with components δᵢₖδⱼₗ
  - **R** with components δᵢₗδⱼₖ
  - With minor+major symmetry, 2D basis suffices: **I**⊗**I** and **Is** = ½(**Q**+**R**)
  - Isotropic stiffness: **C** = λ**I**⊗**I** + 2μ**Is**

---

## System character: symbolic with simplification

The library is **not** a numerical array library. It is a symbolic tensor algebra engine.

- Tensors are **expression trees**: linear combinations of weighted polyads, named special objects,
  or results of operations on other expressions.
- Operations (·, :, ··, ×, ∇) are **expression rewriting rules**, not array computations.
- **Simplification is the core product** — without it expressions grow unboundedly and become useless.
- Numeric evaluation is a **late-stage interpretation pass** on a finalized symbolic expression,
  applicable to concrete numbers, spatial field functions, or FEM shape functions.

Key simplification identities needed (non-exhaustive):
- **I**·**a** = **a**, **I**:**A** = tr(**A**)
- εᵢⱼₖεᵢⱼₗ = 2δₖₗ, and the full ε⊗ε contraction identity (generalized Kronecker delta)
- Cayley-Hamilton reductions for rank-2 tensors
- Symmetry/skew decomposition identities

Normal form strategy is non-trivial — tensor algebra lacks an obvious canonical form like polynomial rings.
Design will need a mix of eager structural rewrites and user-directed derivation steps.

---

## Field representation and coordinate systems

### Coordinate systems (CS)

A CS object carries:
- Dimension: 1 (curve), 2 (surface), or 3 (volume)
- Embedding map: coordinates (q¹, …, qⁿ) → position **x** in a World Cartesian System (WCS)
- Covariant basis vectors **g**ᵢ = ∂**x**/∂qⁱ (functions of position)
- Metric tensor gᵢⱼ = **g**ᵢ·**g**ⱼ and its inverse gⁱʲ
- Christoffel symbols (derivable from the metric)

Standard CS types: Cartesian (WCS reference), cylindrical, spherical, and arbitrary user-defined.

### Fields

- **Scalar field**: symbolic expression or lambda in CS coordinates
- **Vector/tensor field**: scalar coefficient functions × CS basis vectors, e.g. **v** = vⁱ(q)**g**ᵢ
- Fields carry a reference to their CS so ∇ knows which operator to apply

### Gradient operator

- On a 3D CS field: full covariant derivative (with Christoffel symbols)
- On a curve/surface CS field: tangential (intrinsic) derivative on the embedded manifold

### Embedded manifold case (rods, shells, plates)

Example: elastic rod axis **r**(s), s = arc length.
- **r**(s) maps ℝ → ℝ³; the Frenet-Serret frame {**t**, **n**, **b**} = {**r**', …} is the induced basis
- Intrinsic operators (d/ds, curvature κ, torsion τ) live in the reduced 1D space
- The CS has dimension 1 with embedding into ambient 3D WCS
- Balance equations (rod/shell theories) are naturally expressed in this framework
