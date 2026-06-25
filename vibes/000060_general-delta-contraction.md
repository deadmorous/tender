# 000060 — General Kronecker-delta contraction

Generalising `steps::contract_delta` from the narrow `δ·δ → δ` recogniser to
contracting a Kronecker δ against *any* factor it shares a summed index with.
Driven by user feedback: the basis-expansion pipeline left δ's stranded.

## The trigger

Expanding `(a × I) × b` component-wise (basis vectors `e_i`) and simplifying
basis dots yields terms like

```
-a_i b_k δ_ik (e_j ⊗ e_j)   +   a_k b_j δ_ki (e_j ⊗ e_i)
```

`contract_delta` was a no-op here: it only recognised an `ExplicitSum` whose
body is a `TensorProduct` of **exactly two** δ objects (`Σ_m δ^m_a δ^m_b → δ_ab`).
A δ contracting a dummy index carried by an ordinary tensor (`a_i δ_ij → a_j`)
had no code path.

## The rule

At a summation binder `Σ_m` whose body holds a Kronecker δ with `m` in one slot
and partner index `n` in the other: δ identifies `m` with `n`, so the sum
collapses — **drop the δ, substitute `m := n` through the rest, and shed the
`Σ_m` binder**.  Bottom-up `rewrite_tree`, one δ per firing.

This subsumes the old `δ·δ → δ` (there the "rest" is itself a δ: dropping `δ^m_a`
and substituting `m := a` turns `δ^m_b` into `δ^a_b = δ_ab`).  So the special
case was deleted, not kept alongside.

## Soundness guards

Two checks keep it from firing where the substitution is invalid:

1. **The δ must be a genuine Kronecker, not a metric in disguise.**  An
   `Oblique` δ with its two slots at the *same* level is really `g_mn`, and
   `Σ_m g_mn X^m` is index *lowering*, not a Kronecker substitution.  So require
   `realm == Orthonormal` (upper/lower interchangeable) **or** the δ's two slots
   at opposite levels.

2. **`m` must have a real contraction partner.**  After dropping the δ, `m` must
   still occur in the remaining body, at a slot with **matching realm and
   space** (levels need not match — δ identifies its indices regardless of which
   is up or down).  This rejects:
   - `Σ_m A δ^m_k` with `A` free of `m` (`Σ_m δ^m_k = 1`, no partner — not ours
     to collapse here);
   - `Σ_m δ^m_k(Oblique) δ^m_l(Orthonormal)` (a contraction across mismatched
     realms is ill-typed).

The δ-removal walks through `TensorProduct`, `ExplicitSum`/`NoSum`, **and
`Negate`** (the sign of a subtracted term sits inside the binders — the first
cut missed it, so the negative term kept a stray `δ_ii`).

## Result on the motivating case

```
contract_delta:  -a_i b_i (e_j ⊗ e_j)  +  a_i b_j (e_j ⊗ e_i)
              =  -(a·b) I  +  b ⊗ a
```

## Test changes

- New `ContractDelta.ContractsDeltaAgainstTensor` (`a_i δ_ij → a_j`).
- `ContractDelta.NoContractWithMismatchedLevels` → renamed
  `ContractsMixedLevelDeltas`: `Σ_m δ^m_k δ_m^l` has the summed index at
  *opposite* levels, which is a *valid* oblique contraction (`→ δ^l_k`).  The old
  test asserted no-fire — a limitation of the narrow rule, not a correctness
  property — so it now asserts the contraction.
- `NonDeltaInProductUnchanged`, `MismatchedRealmUnchanged`,
  `SumIndexNotInEitherDeltaUnchanged` still pass unchanged (guard 2 / no δ
  carrying `m`).

## Implementation note

Added `substitute_index(ctx, e, from_id, IndexAssoc to)` — the index→index
sibling of the existing `substitute` (which maps an index to a concrete value).
