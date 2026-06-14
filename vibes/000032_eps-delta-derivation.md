# 000032 eps-delta derivation

## Problem

Derive the identity `ε_{ijk} ε_{iml} = δ_{jm} δ_{kl} - δ_{jl} δ_{km}`
(contraction of two Levi-Civita symbols over one shared index, with four
Kronecker deltas on the right-hand side) using the tender derivation system.

## Strategy

The derivation proceeds by three new rewriting steps, layered on top of the
existing `unroll_sums` / `eval_delta_concrete` / `fold_arithmetic` pipeline.

### Step 1: `expand_eps`

Replace every rank-3 `LeviCivita` tensor `ε_{ijk}` with its cofactor
(Laplace) expansion along the first "row" — the WCS dimension label.  In 3D:

```
ε_{ijk} = + δ^1{}_i  δ^2{}_j  δ^3{}_k
          - δ^1{}_i  δ^3{}_j  δ^2{}_k
          - δ^2{}_i  δ^1{}_j  δ^3{}_k
          + δ^2{}_i  δ^3{}_j  δ^1{}_k
          + δ^3{}_i  δ^1{}_j  δ^2{}_k
          - δ^3{}_i  δ^2{}_j  δ^1{}_k
```

Each `δ^a{}_b` is a Kronecker delta with the concrete WCS-component `a` in
the Upper slot and the tensor-index `b` (still symbolic) in the original
slot's level.  The six terms are all 3! permutations of `{1,2,3}` weighted by
their sign.

Implementation: `rewrite_tree` + detection of `WellKnownKind::LeviCivita`
with `slots.size() == 3` and a concrete (3-value) `IndexSpace`.  Hardcoded
for 3D; a future version can generalise.

### Step 2: existing steps

After `expand_eps`, applying `unroll_sums` + `eval_delta_concrete` +
`fold_arithmetic` to the expression `Σ_i ε_{ijk} ε_{iml}` fully evaluates
the concrete WCS contractions and produces a 12-term sum of Kronecker-delta
products with symbolic indices j, k, m, l.

### Step 3: `fold_sums`

The inverse of `unroll_sums`.  Detects a binary `Sum` tree whose addends form
a cyclic family `{f(v₁), f(v₂), …, f(vₙ)}` where:

- n = |IndexSpace values|,
- the expressions differ in exactly one `ConcreteIndex` slot value that
  cycles through all n space values,
- all other slot assignments are identical.

When such a family is found the addends are replaced by a single
`ExplicitSum{m, f(m)}` over a fresh `CountableIndex m`.

Algorithm:

1. Collect all addends of the `Sum` tree (recursively flatten the binary tree).
2. Find an `IndexSpace` via the first `ConcreteIndex`-bearing slot in addend 0.
3. For each candidate "template value" `v₀` present in addend 0:
   a. Build template `T` = `substitute_concrete(addend[0], v₀, m)` (replace
      every occurrence of `ConcreteIndex{v₀}` with a fresh `CountableIndex m`).
   b. For each remaining addend `eₖ`, verify there is a unique unvisited
      space value `vₖ` such that `substitute(T, m, vₖ)` is structurally equal
      to `eₖ`.
   c. If all n addends match and the matched values cover the whole space: fold.

Helper `structural_eq` compares two expression trees node-by-node (same
node types, same `CountableIndex` ids, same `ConcreteIndex` values, same slot
layouts and space pointers).

Helper `substitute_concrete` is the reverse of `substitute`: replaces every
`ConcreteIndex{old_val}` with `CountableIndex{new_idx}` in TensorObject slots.

### Step 4: `contract_delta`

Recognises `ExplicitSum{m, TensorProduct(δ^m{}_j, δ^m{}_k)}` (a product of
two deltas that share the summation `CountableIndex m` in the same-level slot)
and replaces it with `make_delta(ctx, realm, space, level_j, level_k, j, k)`.

In the specific case above both surviving slots are `Lower`, so the result is
`δ_{jk}` (both-lower Kronecker delta).  The step generalises to any level
combination by reading the surviving slot's level directly from the two delta
objects.

## Note on full eps-delta derivation

After `expand_eps` + `unroll_sums` + `eval_delta_concrete` + `fold_arithmetic`
the expression `Σ_i ε_{ijk} ε_{iml}` becomes a 12-term sum with all concrete
WCS indices evaluated.  The individual 12 addends do *not* split into
disjoint groups of exactly 3 that cycle cleanly (because the double permutation
structure of the two ε symbols produces `Σ_{a≠b}` patterns rather than
`Σ_a Σ_b`).

`fold_sums` handles clean N-addend cycles and is therefore useful for simpler
expressions such as `δ^1{}_k δ^1{}_l + δ^2{}_k δ^2{}_l + δ^3{}_k δ^3{}_l`
→ `Σ_a δ^a{}_k δ^a{}_l`.  A dedicated algebraic simplification step for
the full eps-delta identity is left to a future iteration.

## Implementation files

| File | Role |
|------|------|
| `src/include/tender/derivation.hpp` | Declare `expand_eps`, `fold_sums`, `contract_delta` |
| `src/derivation.cpp` | Implement: helpers `structural_eq`, `substitute_concrete`, `collect_addends`, `find_space_from_concrete`; three new steps |
| `tests/derivation_test.cpp` | Unit tests for each new step |
| `python/_core.cpp` | C++ → Python bindings for the three new steps |
| `python/tender/derivation.py` | Public wrappers `expand_eps`, `fold_sums`, `contract_delta` |
| `python/tests/test_derivation.py` | Python-level tests |
| `examples/eps_delta.py` | End-to-end example script |
| `examples/eps_delta.ipynb` | Notebook version |
| `examples/Makefile` | Add `eps_delta` to the example list |
