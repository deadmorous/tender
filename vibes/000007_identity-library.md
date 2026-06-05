# Tensor Identity Library

## Status: open questions

---

## Motivation

Many derivations in tensor mechanics rely on well-known algebraic identities
that transform one expression into an equivalent form. Examples:

```
a % (b % c)  =  b * (a @ c) - c * (a @ b)          # BAC-CAB rule
a % I % b    =  b * a - (b @ a) * I                 # cross-identity with rank-2
```

If these identities are first-class objects in tender, they can be:
- Stored in a reusable library (standard identities + user-defined)
- Invoked by name as derivation steps (consistent with Q_rule_repr in vibe 000004)
- Applied to sub-expressions during a derivation

---

## Identities as directed rewrite rules

An identity has a **left-hand side** (pattern) and a **right-hand side**
(replacement). Application is directed: the user chooses which side to rewrite
toward. Bidirectionality is achieved by registering both directions or by
explicit `reverse=True` on invocation.

The LHS is a pattern containing **pattern variables** — placeholders that match
any sub-expression of the appropriate rank. Pattern variables are distinct from
named tensors (**I**, **ε**, **nabla**, etc.), which match only themselves.

```python
a, b, c = pattern_vars('a', 'b', 'c')   # rank-1 pattern variables (vectors)
bac_cab = Identity(
    name  = 'bac_cab',
    lhs   = a % (b % c),
    rhs   = b * (a @ c) - c * (a @ b),
)
```

---

## Integration with derivations

An identity application is a named derivation step (see vibe 000005,
State/Derivation model). The user invokes it explicitly; tender never applies
identities automatically.

```python
step = apply_identity(bac_cab, target=some_subexpr)
history = derivation([step]).apply(state)
```

The step records which identity was applied and to which sub-expression,
so the derivation history is fully auditable.

---

## Two kinds of reusable results

### Identities (algebraic rewrites)

An identity transforms one expression into an equivalent one. Both sides are
explicit. A proof is a derivation from LHS to RHS (or can be stated as an
external reference). Application is a syntactic substitution step.

### Theorems (existence assertions)

Some results assert that certain objects *exist* with certain properties, without
providing an explicit formula. The spectral theorem is a clear example:

> For any symmetric rank-2 tensor **A**, there exist real numbers A_i and an
> orthonormal basis of vectors **v**_i such that **A** = Σ_i A_i **v**_i ⊗ **v**_i.

Applying this theorem in a derivation is not a rewrite — it is an
**introduction step**: new named objects (A_i, **v**_i) are introduced into the
state, carrying constraints derived from the theorem:

- **v**_i orthonormal: **v**_i · **v**_j = δ_ij
- A_i real scalars
- **A** = Σ_i A_i **v**_i ⊗ **v**_i (the decomposition itself, as an equation in the state)

Subsequent derivation steps can exploit these constraints (e.g., the
orthonormality of **v**_i immediately simplifies many contractions).

The proof of the theorem lives outside tender — it is a reference (a name, a
citation) not a derivation. The system records which theorem was invoked and
what objects it introduced, keeping the derivation history auditable even when
the proof is external.

**The "∃" structure is general.** Whenever a derivation step says "there exist
objects satisfying these constraints, let us name them and proceed", that is a
theorem application. The introduced objects become first-class named expressions
in the state, with their constraints available for subsequent steps.

---

## Identities promoted from derivations

An identity can be the *result* of a derivation — BAC-CAB, for example, is
provable by expanding via Levi-Civita contractions. Tender should make it easy
to promote a completed derivation to a named identity so it can be reused:

```python
# prove it once
history = prove_derivation.apply(State(a % (b % c)))

# register the result as a reusable identity
bac_cab = Identity.from_derivation('bac_cab', history, vars=[a, b, c])
```

`Identity.from_derivation` takes the first and last states of the history as
LHS and RHS respectively, and the declared pattern variables. The derivation
steps are preserved as the identity's proof, which can be displayed on demand.

This creates a closed loop: derive → register → reuse in future derivations.

---

## Basis vector derivatives in curvilinear coordinate systems

Computing gradients in curvilinear CS requires derivatives of basis vectors.
The general formula is:

```
∂e_i/∂x^j = Γ^k_ij · e_k
```

where Γ^k_ij are Christoffel symbols (three-index). In a general CS this is
heavy machinery; in specific well-known CSs it simplifies greatly. For example,
in polar coordinates:

```
∂e_r/∂θ = e_θ       ∂e_θ/∂θ = −e_r
∂e_r/∂r = 0         ∂e_θ/∂r = 0
```

**Design decision**: each well-known CS provides its basis vector derivatives
as the simplest possible form — pre-simplified identities, not the full
Christoffel expansion. These live in the standard identity library under the
CS's namespace, e.g., `tender.cs.polar.basis_derivatives`.

For user-defined CSs (constructed via embedding map — see vibe 000001), the
system should be able to derive the Christoffel symbols automatically from the
map using Γ^k_ij = **e**^k · ∂**e**_i/∂x^j, and from these generate the basis
derivative identities. Whether this auto-derivation uses tender's own
derivation machinery (closing the loop) or a separate computation path is an
open implementation question.

---

## Constraint verification

### The problem

Identities and theorems impose requirements on their inputs — symmetry,
skew-symmetry, positive definiteness, orthogonality, rank, isotropy, minor/major
symmetry (for rank-4), etc. Before applying an identity the system must
establish whether the requirement is satisfied. This is non-trivial in general.

### Catalogue of common requirements

| Requirement | Invariant definition | Notes |
|---|---|---|
| Rank n | expression is rank-n | Always verifiable structurally |
| Symmetric | **A** = **A**^T | A_{ij} = A_{ji} |
| Skew-symmetric | **A** = −**A**^T | A_{ij} = −A_{ji} |
| Traceless | tr(**A**) = 0 | |
| Orthogonal | **A**^T·**A** = **I** | |
| Idempotent | **A**²= **A** | projection tensors |
| Positive definite | **v**·**A**·**v** > 0 ∀**v**≠0 | hard to verify symbolically |
| Isotropic (rank-2) | **A** = λ**I** | |
| Minor symmetry (rank-4) | C_{ijkl}=C_{jikl}=C_{ijlk} | elasticity |
| Major symmetry (rank-4) | C_{ijkl}=C_{klij} | elasticity |

This list will grow as new identities are added to the library.

### Three verification strategies

**1 — Structural (syntactic):** the system determines the property directly from
the expression tree, with no computation. Examples:
- **a**⊗**b** + **b**⊗**a** is visibly symmetric
- A rank-1 expression trivially satisfies `rank=1`
- A scalar times **I** is visibly isotropic

**2 — Algebraic:** the system computes a witness expression and checks if it
reduces to zero (or the expected value). Example: compute **A** − **A**^T and
attempt to simplify; if the result is zero, **A** is symmetric. This may fail
if the simplifier is not strong enough — it gives a definitive yes but not a
definitive no.

**3 — User declaration:** the user asserts the constraint explicitly. The system
trusts the declaration and records it in the derivation history for auditability.

```python
A = tensor('A').declare(symmetric=True)
# or as a separate step:
state = state.assume(A, symmetric=True)
```

The system tries strategies 1 and 2 in order. If both fail, it either requires
a user declaration or raises an informative error explaining what could not be
verified.

### Caching verified constraints

Since expression objects are immutable, the result of any constraint check can
be cached directly on the object as an attribute — set at most once, just like
`name`. This means a constraint is never verified more than once for the same
expression, regardless of how many identities or derivation steps query it.

User declarations write to the same cache, making structural verification and
user declaration uniform at the storage level: the attribute is either absent
(unknown), `True` (verified or declared), or `False` (verified false — useful
to rule out spurious identity matches quickly).

### Representation-dependence: the mixed-coordinates pitfall

Some requirements that are simple to state in terms of covariant or contravariant
components become non-obvious in mixed coordinates. **Symmetry** is the canonical
example.

A = a **i**⊗**i** + b **j**⊗**j** + c **k**⊗**k** is clearly symmetric.
Expressed as A^i_j **e**^i⊗**e**_j in a non-orthonormal custom basis
{**e**_1=**i**, **e**_2=2**j**+**k**, **e**_3=2**i**+**j**+3**k**},
the mixed coordinate matrix A^i_j is:

```
[[ a,          4(b−c)/5,    (10a+2b−12c)/5 ],
 [ 0,          (6b−c)/5,      3(b−c)/5     ],
 [ 0,          2(c−b)/5,      (6c−b)/5     ]]
```

This matrix is not symmetric, yet the tensor is. The covariant matrix A_{ij}
is symmetric; the mixed matrix A^i_j = g^{ik} A_{kj} is not, because the
metric g^{ij} deforms the matrix in general.

**General principle**: any requirement stated as a condition on a coordinate
matrix is only reliable when using covariant OR contravariant coordinates — not
mixed. The system must know the coordinate type of an expression before applying
a matrix-level check. For mixed coordinates, the system must first lower (or
raise) both indices, then check.

This generalises beyond symmetry: any matrix identity that holds for covariant
components (A_{ij} = A_{ji}, A_{ij} = −A_{ji}, A_{ii} = 0, etc.) may fail
as a matrix identity in mixed coordinates. The invariant definition is always
the ground truth; coordinate representations are checked by converting to
covariant/contravariant form first.

---

## Open questions

### Q_pattern_matching — RESOLVED

#### Constraints on pattern variables

Rank alone is insufficient — pattern variables carry general **constraints**,
which are predicates that a candidate sub-expression must satisfy. Examples:

- No constraint (matches any expression)
- Rank only: `rank=1`, `rank=2`
- Rank + symmetry: `rank=2, symmetric=True`
- Rank + skew-symmetry: `rank=2, skew_symmetric=True`
- Arbitrary user-supplied predicate for advanced cases

Constraints serve two purposes: they gate whether a candidate matches, and they
can license simplifications that only hold under those conditions (e.g., vector
invariant of a symmetric rank-2 tensor is zero; trace of a skew-symmetric
rank-2 tensor is zero). See also Q_identity_conditions.

#### Targeting: manual and automatic

**Manual targeting**: user supplies an explicit mapping from each pattern
variable to the sub-expression it should match. Unambiguous; always works;
preferred when the user knows the structure.

```python
apply_identity(bac_cab, mapping={a: expr_a, b: expr_b, c: expr_c})
```

**Automatic targeting**: the system searches for a valid mapping by
combinatorial enumeration of candidate sub-expressions. Bounded by a
configurable iteration budget to avoid runaway search. Returns the first
valid mapping found, or all valid mappings if requested.

```python
apply_identity(bac_cab)                    # auto, first match
apply_identity(bac_cab, all_matches=True)  # auto, all matches
apply_identity(bac_cab, budget=500)        # explicit budget override
```

**Interactivity**: highly desirable but not implemented initially. For now,
`all_matches=True` in a Jupyter notebook lets the user inspect candidates and
re-invoke with a manual mapping for the chosen one. A richer interactive picker
is deferred.

#### Partial match

Best-effort only, kept minimal. When automatic targeting finds no full match,
the system reports which pattern variables it could bind and which it could not,
so the user can diagnose the mismatch without extra implementation cost.

### Q_pattern_variables — RESOLVED

**Decision: (a) explicit declaration.** Pattern variables are a distinct type,
not tensors. Constraints are specified by chaining methods on the returned
objects — a natural fluent API that avoids a separate constraint argument:

```python
a, b, c = pattern_vars('a', 'b', 'c')        # unconstrained
A = pattern_var('A').rank(2).symmetric()      # rank-2 symmetric tensor
B = pattern_var('B').rank(2).skew_symmetric() # rank-2 skew-symmetric tensor
T = pattern_var('T').rank(2)                  # rank-2, no symmetry constraint
v = pattern_var('v').rank(1)                  # vector
```

`pattern_vars` is the multi-name convenience form; `pattern_var` is the
single-variable form used when constraints need to be chained.

This keeps identity definitions unambiguous regardless of what names the user
has chosen for their actual tensors.

### Q_identity_library — RESOLVED

A standard identity library ships as a tender submodule. Coordinate system and
dimensionality assumptions (e.g., 3D-only identities) are expressed as
constraints on pattern variables using the constraint mechanism established in
Q_pattern_variables. The constraint vocabulary will be refined as concrete
identities are added to the library.

### Q_identity_conditions — RESOLVED

Preconditions are expressed as constraints on pattern variables (Q_pattern_variables).
An identity that requires, e.g., a symmetric rank-2 tensor simply declares its
pattern variable with `.rank(2).symmetric()` — the matcher enforces the condition
at application time.
