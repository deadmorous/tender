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

## Single-term confinement (correctness)

The substitution `m := n` must stay inside **one multiplicative term**.  The
ε-pair contraction leaves the generalized-Kronecker determinant as a *distributed*
sum under shared binders:

```
Σ_k Σ_j Σ_i Σ_l  -a_i b_l e_k e_j (δ_kj δ_il − δ_kl δ_ij)
```

Here `k` pairs with `j` in the first addend but with `l` in the second.  A naive
"find a δ carrying k, drop it, substitute k everywhere" crosses the `−` and
identifies indices across addends — collapsing everything to a single index with
`δ_ii` self-traces (it briefly did exactly that).  The old narrow rule was immune
only because it required the body to be a single `TensorProduct`.

Guard: peel the binders and sign to the multiplicative core, `flatten_factors`,
and if any factor is a `Sum`/`Difference`, **bail** — leave it for a distribution
step.  Regression test `ContractDelta.DistributedSumUnchanged`.  So the recipe
for the ε route is: contract the ε-pair, **distribute** (`expand_products` then
`canonicalize` to float binders per-term), *then* `contract_delta`.

## Sums must be explicit

`contract_delta` materialises implicit Einstein sums itself, but its **siblings
do not**: `reassemble_completeness` (via `fold_completeness`) and
`contract_eps_pair` (via `try_contract_eps_pair`) only recognise their patterns
under explicit `ExplicitSum` binders — they peel binders and bail when none are
present.  The basis steps (`expand_in_basis`, `simplify_basis_*`) and
`contract_delta` all emit *implicit* form, so a bare hand-off no-ops.  Insert a
`canonicalize` (which materialises) before those two steps, exactly as the C++
tests do (`reassemble_completeness(ctx, steps::canonicalize(ctx, term), b)`).
Folding implicit sums into them is a separate follow-up.

## Implementation note

Added `substitute_index(ctx, e, from_id, IndexAssoc to)` — the index→index
sibling of the existing `substitute` (which maps an index to a concrete value).
