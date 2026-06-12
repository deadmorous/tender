# Libraries and Rotation Tensors

## Status: open questions

---

## The bootstrap philosophy

Rotation tensors are a good first test case for a general principle:

> Use tender itself to derive what the library should contain, then
> incorporate the results into the library.

**Phase 1**: build a version of tender capable of deriving the core properties
of rotation tensors (orthogonality, determinant, angular velocity, rigid body
kinematics). No special C++ support is expected to be needed for rotations — the
core expression machinery suffices.

**Phase 2**: run those derivations. The results become library entries — named
identities, theorems, named objects — that future users can invoke without
re-deriving.

**Phase 3**: repeat for other domains (thin-walled structures, continuum
mechanics, etc.) as the library grows.

This creates a living, self-documenting library whose entries carry proofs by
construction.

---

## Rotation tensors: what belongs in the library

A rotation tensor **R** is a rank-2 orthogonal tensor with det(**R**) = +1
(the special orthogonal group SO(3)). The core results to derive and store:

### Fundamental properties (derivable from definition)
- **R**^T · **R** = **R** · **R**^T = **I**
- det(**R**) = +1
- **R** is invertible; **R**^{-1} = **R**^T
- Eigenvalues: one real (= +1), two complex conjugates on the unit circle

### Kinematics
- Angular velocity (spin) tensor: **Ω** = **Ṙ** · **R**^T (skew-symmetric)
- **Ω** = **ω** × **I** where **ω** = vect(**Ω**) is the axial (angular velocity) vector
- Inverse relation: **Ṙ** = **Ω** · **R**

### Action on tensors
- Rotation of a vector: **R** · **v**
- Rotation of a rank-2 tensor: **R** · **A** · **R**^T (push-forward)
- Composition: **R**₁ · **R**₂ is also a rotation

### Rodrigues formula
**R** = **I** + sin(θ) **N** + (1 − cos θ) **N**²,
where **n** is the unit rotation axis vector and **N** = **n** × **I** is its
associated skew-symmetric tensor. This follows directly from the cross-product
convention (vibe 000001): (**n** × **I**) · **v** = **n** × **v** for all **v**.

---

## Library architecture

The rotation tensor question surfaces a broader design question: is there one
library (the identity library from vibe 000007), or should we consider multiple
specialized libraries? And regardless of structure, how are they documented?

### Q_library_structure — RESOLVED

**Decision: (b) multiple distinct library kinds, with good submodule structure.**

A single library kind imposes today's assumptions on future growth. Keeping
library kinds distinct preserves the ability to add new kinds as tender evolves,
without retrofitting the existing ones.

Current distinct library kinds:
- **Identity library**: algebraic rewrite rules (LHS → RHS with pattern variables)
- **Theorem library**: existence assertions (introduce named objects with constraints)
- **Named object library**: pre-declared named tensors/objects with constraints
  attached (see Q_named_object_library)

Future kinds may emerge (e.g., numerical evaluation recipes, code generation
templates, CS-specific formulas) and can be added without disturbing existing kinds.

Within each kind, submodule organization is still desirable:
`tender.lib.identities.rotation`, `tender.lib.theorems.spectral`,
`tender.lib.objects.rotation`, `tender.lib.objects.cs.polar`, etc.
Submodules are packaging; library kinds are distinct types in the object model.

### Q_named_object_library — RESOLVED

Named objects with pre-declared constraints belong in the same library as
identities and theorems — the distinction is *type*, not *location*. Libraries
are domain-oriented; within a library, entries are typed (Identity, Theorem,
NamedObject, etc.). Dependencies between domain libraries require careful
management; hierarchy organization needs dedicated attention before
implementation.

#### Singletons vs. prototypes

Two fundamentally different kinds of named object:

**Singletons** — there is exactly one such object (up to dimension). Use as-is;
no cloning needed. Examples: **ε** (Levi-Civita tensor), **I** (identity tensor),
**g** (metric tensor of a given CS).

```python
from tender.lib.objects import eps, I   # singletons — use directly
```

Note: what is often called the "metric tensor" g_{ij} is not a separate invariant
object — it is the covariant coordinate representation of **I** in a given basis:
g_{ij} = **e**_i · **I** · **e**_j = **e**_i · **e**_j. There is no singleton **g**;
there is only **I**, and its coordinate matrices are computed from the basis when needed.

**Prototypes** — there are many instances of this kind of object (different
bodies, different frames, different contexts). The library entry is a template
carrying pre-declared constraints; the user instantiates it with a specific name
before use. Instantiation creates a fresh named object with the same constraints
but its own identity.

```python
from tender.lib.rotation import R_proto
R1 = R_proto.instance('R_1')   # a specific rotation tensor, constraints inherited
R2 = R_proto.instance('R_2')   # another one, independent of R1
```

`instance(name)` is the cloning mechanism: same constraint cache as the
prototype, new identity, new display name. The prototype itself is never used
directly in a derivation.

### Q_library_documentation — RESOLVED

Each library entry is self-documenting. A `doc()` call on any entry renders its
full specification in a Jupyter notebook:

- **Identities**: name, LHS, RHS (as rendered LaTeX), pattern variable
  constraints, proof derivation (if derived via tender) or external reference
- **Theorems**: name, statement, introduced objects and their constraints,
  external reference
- **Named objects / prototypes**: name, type, pre-declared constraints, relevant
  identities that apply to it

```python
doc(bac_cab)           # renders identity spec + proof
doc(R_proto)           # renders rotation tensor constraints
doc(format="plain")    # plain-text fallback for non-Jupyter contexts
```

Documentation is auto-generated from the objects themselves — no separate
documentation files needed.

#### Python as the primary layer

Documentation, library tooling, derivation notebooks, and any future
DSL/GUI/interactive layer all sit in Python. C++ is responsible solely for
performance-critical core machinery (expression tree, evaluation, code
generation). This is consistent with vibe 000004 (Q_ui_cppapi) and extends it:
if a richer interface is ever added — a domain-specific language, a graphical
derivation tool, a web frontend — it is built on top of the Python layer, not
on C++ directly. The Python layer is the stable integration point.

### Q_vect_operation — RESOLVED

`vect` is a first-class operation in the core API alongside `trace`, `det`,
`transpose`. It extracts the axial vector of a skew-symmetric rank-2 tensor:
`vect(Ω)` = **ω** such that **Ω** = **ω** × **I**.

This also simplifies the angular velocity definition in the kinematics section:

- **Ω** = **ω** × **I** (spin tensor from axial vector, same pattern as N in Rodrigues)
- **ω** = vect(**Ω**) (inverse)

The relation **Ω** · **v** = **ω** × **v** follows immediately from the
cross-product convention, just as for N (vibe 000001).

In coordinates: ω_k = −(1/2) ε_{ijk} Ω_{ij} — derivable as a contraction with
**ε**, but common enough in mechanics to warrant a named function rather than
requiring the user to assemble it each time.
