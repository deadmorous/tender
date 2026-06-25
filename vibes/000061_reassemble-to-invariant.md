# 000061 — Reassembling component form back to invariant notation

Generalising `reassemble` so a basis-expanded expression with **several**
coordinate factors returns to coordinate-free form.  Driven by the `(a×I)×b`
worked example: after the cross is removed component-wise, Route A lands on

```
-a_i b_i (e_j ⊗ e_j)  +  a_i b_j (e_j ⊗ e_i)
```

and the old `reassemble` only handled a *single* coordinate tensor per term, so
this stayed in components.

## What it now recognises

The fold runs per summed index, each independently — no special dyad/triad
assembly (user's steer: "recognize a and b individually, and as soon as we do so
we have an invariant dyad"):

| group | shape | → |
|-------|-------|---|
| vector | `c_i e_i` (one coord vector + its basis vector) | `c` (basis vector replaced in place by the invariant) |
| dot    | `c_i d_i` (two coord components, no basis vector) | `c · d` (scalar invariant) |
| identity | `e_i e_i` (two basis vectors, no coord) | `I` (resolution of identity) |

So `a_i b_j e_j e_i` folds `a` (from `a_i e_i`) and `b` (from `b_j e_j`)
individually, the basis vectors' positional order giving `b ⊗ a` for free; the
same mechanism reassembles `u⊗v⊗w` etc.  Result for the example:
`-(a·b) I + b ⊗ a`.

Indices whose two occurrences are not one of these clean shapes (e.g. a slot of
a higher-rank coordinate `A_ij`) are left bound and handled by the pre-existing
whole-term path, which `fold_reassembly_groups` falls through to.

## The coordinate-vs-basis discriminator

A coordinate component and a (possibly foreign) basis vector are both a named
tensor with one CountableIndex slot — indistinguishable by shape.  The
discriminator: `expand_in_basis` emits the coordinate at **rank 0** (a scalar
component, `basis.cpp` ~258) while a basis vector is **rank 1**.  So
`as_coord_vector` requires `rank == 0`.  Without it, `reassemble(Σ a_i e_i,
g_basis)` would read the *e*-basis vector `e_i` as a coordinate and fold
`a_i e_i → a·e` — the regression `Reassemble.ForeignBasisUnchanged` caught
exactly this.

Also: peel a leading `Negate` (a subtracted term's sign) before flattening, or
the sign hides the product from `flatten_product` and the negative term doesn't
fold — the same blind spot fixed in `contract_delta` (vibe 000060).

## Tests

`Reassemble.TwoVectorsFoldIndividuallyIntoDyad` (`u⊗v` round-trip) and
`Reassemble.ContractedCoordsFoldToDot` (`u·v` via component form).  All prior
`Reassemble.*` round-trips (vector, rank-2, contravariant, identity, foreign,
no-op) still pass.

## Still open

- Route A needs a manual `canonicalize` before `reassemble` (to materialise the
  implicit sums) — that is the self-prepare follow-up: every step should
  materialise/distribute internally so `canonicalize`/`expand_products` are never
  required by the caller (see [[steps-self-prepare]]).  Deferred to its own step.
- Dot reassembly assumes an orthonormal frame (`Σ u_i v_i = u·v`); the oblique
  metric form is out of scope here.
