# Subspace decomposition and the 2D nature of projected tensors

_First discussed: 2026-06-13_

---

## Motivation: thin-walled structures

In the mechanics of shells, plates, and similar structures it is common to
decompose a 3D rank-2 tensor **A** as

```
A = A_flat + A₃₃ k⊗k + a_left⊗k + k⊗a_right
```

where:

- **A_flat** is the "in-plane" part:
  `A_flat = A₁₁ i⊗i + A₁₂ i⊗j + A₂₁ j⊗i + A₂₂ j⊗j`
- **a_left**, **a_right** are in-plane vectors:
  `a_left = A₁₃ i + A₂₃ j`,  `a_right = A₃₁ i + A₃₂ j`
- **k** is the through-thickness unit normal
- **i**, **j** are in-plane basis vectors

The objects **A_flat**, **a_left**, **a_right** are "2D" in the sense that they
can be fully expanded using only the in-plane basis vectors.  The scalar A₃₃
is just a component.

---

## Key invariance principle

Tensors in direct (coordinate-free) notation are **invariant objects**.
**A**, **A_flat**, **a_left**, **k** are all abstract tensors — none of them
carries index slots.  Slots belong exclusively to **basis vectors** and their
polyads (tensor products), because those are the objects that *reference* an
index space when constructing a component expansion.

Therefore **A_flat** cannot be declared "2D" by assigning slot types to it —
it has no slots.  A different mechanism is needed.

---

## Where index spaces do live

Index spaces belong on `IndexSlot`s of basis vectors (and cobasis vectors):

```
g_α  →  IndexSlot{ space_2d, … }
g_i  →  IndexSlot{ space_3d, … }
```

When **A** is expanded as `A^{ij} g_i ⊗ g_j`, the summation range `{1,2,3}`
comes from `space_3d` attached to the slots of `g_i` and `g_j` — not from
**A** itself.  When **A_flat** is expanded as `A_flat^{αβ} g_α ⊗ g_β`, the
range `{1,2}` comes from `space_2d` on the slots of `g_α` and `g_β`.

This part of the design is already correct and needs no change.

---

## Labelling a tensor as 2D: home-space annotation

To let the system know that **A_flat** should be expanded using `space_2d`
(not `space_3d`), each abstract tensor object can optionally carry a
**home-space** annotation — a pointer to an `IndexSpace` that defines the
natural summation range for basis expansion.

| Tensor | Home space |
|--------|-----------|
| **A** | `space_3d` |
| **A_flat** | `space_2d` |
| **a_left**, **a_right** | `space_2d` |
| **k** | `space_3d` (or none — it is a fixed vector) |
| **A₃₃** | — (scalar) |

The annotation is distinct from a slot: it is a property of the tensor
*object*, not of any positional cell.  When the simplifier is asked to expand
a tensor in a basis, it reads the home-space to select the correct basis
vectors and summation range.  Tensors without a home-space annotation inherit
the ambient space of the expression (e.g. the coordinate system attached to
the enclosing integral or gradient).

---

## Subspace relationships

Home-space alone is not sufficient for operations that cut across spaces.  To
perform the decomposition automatically — or to contract an in-plane tensor
with a 3D quantity — the system needs to know that `space_2d` sits inside
`space_3d` and which directions it occupies.

This is captured by an **IndexSpaceEmbedding**:

```
embedding: space_2d → space_3d
           maps {1,2} (2D indices) → {1,2} ⊂ {1,2,3} (3D indices)
normal:    index 3 of space_3d corresponds to k
```

An embedding is a named, explicit object created by the user when a subspace
decomposition is intended.  It is *not* inferred automatically from the index
spaces themselves, because the same `space_2d` could be embedded in `space_3d`
in multiple ways (different choice of in-plane directions).

---

## Planned operation: `decompose_by_subspace`

Given the embedding above, a named simplification step

```python
decompose_by_subspace(A, embedding)
```

rewrites **A** (expanded in the 3D basis) by grouping terms into:

1. The `{1,2} × {1,2}` block → **A_flat** with home space `space_2d`
2. The `{3} × {3}` scalar → A₃₃
3. The cross blocks → **a_left**, **a_right** with home space `space_2d`
4. The normal dyads → assembled from **k**

The result is the decomposition at the top of this vibe.  This operation is
consistent with the "named operations, user-composed" principle (vibe 4): the
system handles the algebra; the user supplies the embedding and invokes the
step explicitly.

---

## Relation to `ExplicitSum` and `NoSum`

`ExplicitSum(body, index, space)` already names the space explicitly in the
AST.  The home-space annotation on a tensor has the same spirit: it is an
explicit declaration that prevents the system from silently using the wrong
range.  The two mechanisms are complementary:

- Home space: declares the natural range of a *tensor object* for basis
  expansion.
- `ExplicitSum`: overrides the summation range for a *specific repeated index*
  in an expression, regardless of the tensor's home space.

---

## Summary of design decisions

| Question | Decision |
|----------|----------|
| Do abstract tensors carry index slots? | No — tensors are invariant; slots belong to basis vectors |
| Where does `space_2d` live for **A_flat**? | Home-space annotation on the tensor object |
| Is `space_2d` embedded inside `space_3d` automatically? | No — an explicit `IndexSpaceEmbedding` is required |
| When is `IndexSpaceEmbedding` created? | By the user, when a subspace decomposition is needed |
| How is the decomposition performed? | `decompose_by_subspace(expr, embedding)` — a named step |
| Does space-in-slot for basis vectors need to change? | No — that design is correct as-is |
