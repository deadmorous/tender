# 000035 ε-pair contraction and the "creative step"

## Context

Vibe 32 set up the one-index ε-δ derivation and noted it stalls: after
`expand_eps → unroll_sums → expand_products → eval_delta_concrete →
fold_arithmetic` we are left with **12 concrete Kronecker-delta products**
(6 positive, 6 negative), of the form `± δ^x_j δ^y_k δ^z_m δ^w_l` with `x,y,z,w
∈ {1,2,3}`, which do **not** fold back into a sum without re-introducing terms
that arithmetic folding cancelled.

Going from those 12 terms to the answer `δ^j_m δ^k_l − δ^j_l δ^k_m` by the
`fold_sums`/`contract_delta` route needs a **creative step**: add-and-subtract
the diagonal terms (`3 3 3 3`, `2 2 2 2`, `1 1 1 1`) so that partial
contraction cycles become complete and re-foldable into `Σ_i`, then contract.

A second, independent hand derivation (`vibes/notes/bac-cab.tex`, traced from
`vibes/images/bac-cab.png`) takes a different creative step: write each ε as a
determinant of δ's, then use `det A · det B = det(AᵀB)` to merge the two
determinants before expanding.

The two creative steps look different but are the **same insight**: *the
product of two Levi-Civita symbols is a generalized Kronecker delta.* This vibe
records the diagnosis and the design decision that follows.

## Diagnosis: where the creative step comes from

`expand_eps` expands each ε **individually** into the concrete reference frame
`{1,2,3}` (the cofactor determinant in the eps_delta.py preamble). Then
`unroll_sums` + `eval_delta_concrete` erase the summed index `i` entirely. The
target `δ^j_m δ^k_l − δ^j_l δ^k_m` has **no summed index left**, so `fold_sums`
has no target to fold into — you must first *manufacture* a sum (the
add/subtract) and then contract it away. That round-trip is pure waste motion
caused by destroying structure too early.

Concretely: terms 5 (`1 3 1 3`) and 1 (`2 3 2 3`) fold into `Σ_i δ^i_j δ^3_k
δ^i_m δ^3_l` only if the `i=3` member `3 3 3 3` is present — but ε's
antisymmetry killed exactly that diagonal term during folding. The creative
step *recovers information that was thrown away*.

### Principle adopted

> **`unroll_sums` / `eval_delta_concrete` are *evaluation* tactics, not
> *simplification* tactics.** Use them when the answer is itself concrete
> (e.g. `ε(1,2,3) = +1`). For a *symbolic* identity, contract symbolically and
> never unroll. Symbolic structure that is needed in the answer must survive
> the whole derivation.

## Lengthening ≠ non-terminating

The worry (raised at the crossroads): the creative step *lengthens* the
expression, so allowing it during e-graph saturation makes convergence "a
separate problem with no trivial solution."

The crucial distinction is **bounded vs. unbounded**, not shorter vs. longer.
Equality saturation tolerates size-increasing rules fine; it only diverges on
rules that can fire *unboundedly often*, generating infinitely many e-nodes.

- "Add and subtract an arbitrary `T`" — unbounded (any `T`). This is the
  truly creative, non-terminating move. **Never put it in the rule set.**
- "Complete a partial contraction cycle" — the missing members are
  *determined* by the cycle already present (`{1313, 2323}` forces exactly
  `3333`). Bounded, hence terminating, even though it lengthens.

So the derivation-1 creative step *can* be mechanised as goal-directed
cycle-completion — but it should not be, because there is a strictly *shortening*
route to the same normal form (below). This refines vibe 34's "all productive
rules are size-reducing": the real requirement is *boundedness*; size-reduction
is the easy sufficient condition.

## Decision: ε-pair contraction (generalized Kronecker delta)

Add one symbolic primitive that contracts a **pair** of Levi-Civita symbols
sharing `p` summed indices, directly to the generalized Kronecker delta — never
introducing concrete `1,2,3`, never going uphill:

```
contract_eps_pair :  Σ_{i₁…i_p} ( ε^{… i₁…i_p} ⊗ ε_{… i₁…i_p} )
                       →  s · p! · det[ δ^{free_upper_r}_{free_lower_c} ]
```

where the `Σ` is the contraction over the shared dummies (in direct notation, a
single dot `ε·ε`; in this codebase, one nested `ExplicitSum` per shared dummy),
`free_upper` are the non-contracted slots of the first ε, `free_lower` those of
the second, `q = 3 − p`, the determinant is the `q×q` Kronecker determinant
(the generalized Kronecker delta), and `s = parity_a · parity_b` is the sign of
re-ordering each ε so its contracted slots come first.

One rule, uniform in `p` and dimension `n`:

| case                       | `p` | `q` | result                         |
|----------------------------|-----|-----|--------------------------------|
| `Σ_i ε^{ijk} ε_{iml}`      | 1   | 2   | `δ^j_m δ^k_l − δ^j_l δ^k_m`     |
| `Σ_{ij} ε^{ijk} ε_{ijl}`   | 2   | 1   | `2 δ^k_l`                       |
| `Σ_{ijk} ε^{ijk} ε_{ijk}`  | 3   | 0   | `6`                             |

It is strictly ε-count-reducing → trivially terminating, no concrete numbers,
no add/subtract. Derivation 4 collapses to a single `contract_eps_pair`;
derivation 3 collapses from its 10 hand-ordered steps to one.

This is strictly more general than the fixed `eps-delta-1` / `eps-delta-2`
theorems sketched in vibe 33: it is one *computational* rule covering all `p`,
rather than a separate `lhs=rhs` pair per case. The creativity is paid **once**,
by us, encoded as the determinant identity, and reused forever — exactly how
`bac_cab` would be a named lemma in a proof assistant.

### Route A — direct rule (implemented now)

Pattern-match the nested `ExplicitSum`s wrapping `TensorProduct(ε, ε)`, read off
the contracted vs. free slots, compute `s · p!`, and emit the expanded `q×q`
Kronecker determinant directly (sum of signed δ-products, built with
`make_sum`/`make_negate` like `expand_eps`). Small, closes derivation 4 today,
reproduces 3. The lemma is baked in.

Algorithm:

1. Peel consecutive concrete-bound `ExplicitSum` nodes; collect their index ids
   as the summed set `S` (`p = |S|`).
2. Require the body to be `TensorProduct(εa, εb)`, both rank-3 `LeviCivita`,
   all slots carrying `CountableIndex`.
3. Partition each ε's slots into *contracted* (id ∈ `S`) and *free*, preserving
   order. Require each id in `S` to appear exactly once in each ε.
4. Order contracted ids by appearance in `εa`; compute `parity_a`, `parity_b`
   as the sign of re-ordering each ε to `[contracted-in-that-order, free]`.
5. Build `D[r][c] = δ` connecting `free_a[r]` and `free_b[c]` (levels read from
   the respective slots). Expand `det(D) = Σ_σ sgn(σ) Π_r D[r][σ(r)]`.
6. Return `s · p! · det(D)` (scalar factor only when `p! ≠ 1`; sign via negate).

Limitations (acceptable for now): 3D only; body must be exactly a product of two
ε's (no extra factors); shared indices must be genuine contractions (appear once
in each ε). A bigger product (`ε · ε · X`) or 2D/ND is future work.

### Route B — determinant algebra (future, more faithful)

Mirror `bac-cab.tex` exactly with reusable machinery rather than a baked-in
lemma:

- A `Det` node (or a matrix abstraction over index slots).
- `det A · det B → det(A·B)` with the matrix product contracting the shared
  index.
- `Σ_i δ^i_i → n` (delta-trace; the `δ_ii = n` open limitation from vibe 33)
  and `Σ_i δ^a_i δ^i_b → δ^a_b` (delta-substitution).
- Cofactor `expand_det` deferred to the very end.

No creative step anywhere — the whole derivation is a chain of strictly
simplifying, terminating rewrites. The determinant algebra is reusable
(Binet–Cauchy, Jacobi, etc.). Route A is the `p`-specialised collapse of Route B;
when B exists, A becomes a fast path / can be derived from B's rules.

## E-graph implications (refines vibe 34)

- Put only **bounded** rules in the saturation set: `ε·ε → det δ` (this rule),
  delta-substitution, delta-trace, repeated-index → 0, arithmetic folding.
  Convergence follows from a lexicographic measure (ε-count, then
  summed-index-count, then term-count).
- Keep `unroll_sums` / `eval_delta_concrete` **out** of saturation — they are
  extraction-time *evaluation*, not simplification, and they destroy the
  structure the symbolic rules need.
- The add/subtract creative step never arises, because the e-graph never
  commits to the concrete 12-term form: the un-contracted and contracted forms
  coexist, and extraction simply picks the cheapest (2-term) representative.

## Implementation files (Route A)

| File | Role |
|------|------|
| `src/include/tender/derivation.hpp` | declare `steps::contract_eps_pair` |
| `src/derivation.cpp` | helpers (`is_levi_civita`, parity, det builder) + step |
| `tests/derivation_test.cpp` | unit tests: p=1, p=2, sign, non-match no-ops |
| `python/_core.cpp` | bind `_contract_eps_pair` |
| `python/tender/derivation.py` | wrapper `contract_eps_pair` |
| `python/tests/test_derivation.py` | Python-level tests |
| `examples/eps_delta.py` | close derivation 4 (and simplify 3) with the new step |
