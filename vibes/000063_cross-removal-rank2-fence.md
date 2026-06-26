# 000063 — Removing crosses in a × B × c (B rank 2)

`a × B × c` with `a, c` vectors and `B` rank 2.  The fence rule (vibe 000055)
makes the bracketing immaterial — `(a×B)×c = a×(B×c)` — and the result *is*
expressible cross-free, yet the Levi-Civita route dead-ends.  This note records
why, the "creative" decomposition that unblocks it, what the system can already
do, and the two concrete gaps.

## The dead end (confirmed)

`expand_in_basis → simplify_basis_cross` gives

```
c_k B_nj a_m  ε_{mnl} \, e_l  ε_{jki} \, e_i
```

The two ε's share **no** index (`{m,n,l}` vs `{j,k,i}`), so the ε-δ identity has
nothing to contract — `contract_eps_pair` correctly no-ops.  Unlike `a × I × c`
(vibe 000060/061), there is no `I` in the middle to supply a shared dummy.

## The creative step: a dyad identity that inserts an I

For a single dyad `p ⊗ q` (verified by hand; `ε_{ajm} ε_{jbn} = δ_an δ_mb −
δ_ab δ_mn`):

```
q × I × p  =  p ⊗ q − (p·q) I        ⟹        p ⊗ q  =  q × I × p + (p·q) I
```

Expand `B` in a basis (`B = B_jk e_j ⊗ e_k`, a sum of dyads) and rewrite each
dyad with this identity.  Every `q × I × p` now carries an `I` in the middle —
**which is exactly the shared dummy the dead-end was missing.**  So:

```
a × B × c  =  Σ_dyads [ a×(q×I×p)×c  +  (p·q) a×I×c ]
```

- the `(p·q) a×I×c` term is the already-solved `a×I×c = c⊗a − (a·c)I` route;
- the `a×(q×I×p)×c` term still has crosses, but now reducible (below).

## What works today

Tested on the live system:

- **The dyad identity fires through `apply_identity`.**  As a `td.Identity`
  `p⊗q = (q%I%p) + (p@q)*I`, it rewrites an abstract dyad
  `u⊗v → (u·v)I + v×(I×u)`, *and* it fires on the basis dyads of
  `expand_in_basis(B)` → `B_ji(e_j·e_i)I + B_ij(e_j×(I×e_i))` (first term →
  `tr(B) I` after `simplify_basis_dot`/`contract_delta`).
- **The inserted I does create shared ε-indices.**  `a×(v×I×u)×c` expands to
  `ε_{rlq} ε_{pqn} … ε_{lmj} ε_{jki}` — pairs sharing `q` and `j`.

So the essential "creative" move is fully expressible; nothing here needs new
notation.

## The two gaps

1. **`contract_eps_pair` handles exactly two ε's.**  The reduced
   `a×(v×I×u)×c` has **four** ε's; `try_contract_eps_pair` bails on "more than
   two ε's — not handled".  Generalising it to contract *any* pair sharing a
   summed index, iterating to a fixpoint, would grind the whole thing to
   cross-free form with no further creativity.  **(Recommended — highest
   leverage; helps any multi-cross expression.)**

2. **The fence-over-contraction regroup is not automated:**
   `a × (X · Y) × c  =  (a×X) · (Y×c)` (the contraction `·` fences the two
   crosses onto the outer legs, the rank-≥2 generalisation of vibe 000055).
   With it, `a×(q×I×p)×c = [a×(q×I)] · [(I×p)×c]` and each factor is a *two*-ε
   removal the current `contract_eps_pair` already does
   (`a×(q×I) = (a·q)I − q⊗a`, `(I×p)×c = c⊗p − (p·c)I`).  This is the user's
   route; it sidesteps gap 1 by keeping every contraction at two ε's, but needs
   a new structural rule (a canonicalisation, like the fence re-association).

   **Gap 2 wants sub-expression addressing (vibe 000054).**  The regroup is not a
   global canon — it must hit *one* chosen contraction in a larger expression
   (`a × (X · Y) × c`, picking *this* `·` to fence), not every `·` in the tree.
   That is exactly the positional / targeted-rewrite primitive vibe 000054 is
   about (`rewrite_at` / `find_occurrences`, or a targeted `apply_identity`).
   Without it, gap 2 as a blunt whole-tree rule would fence every contraction it
   sees.  So gap 2 is best deferred until 000054's sub-expression selection
   lands, at which point it becomes a clean targeted rewrite the user invokes by
   address.  Gap 1 needs none of this and stands alone.

Either gap, closed, completes the derivation.  Gap 1 is the more general fix and
makes the *existing* basis route handle `a×B×c` directly after the dyad
identity; gap 2 is the more "structured" route and is reusable wherever a cross
chain brackets a contraction.

## Other paths

- **Compact identity** `a × B × c = (a×I) · B · (I×c)` (verified:
  `(a×B×c)_pn = ε_{imp} a_i B_mj ε_{jln} c_l`).  Elegant, but `(a×I)` and `(I×c)`
  still hold crosses; it only becomes fully cross-free once `B` is dyad-expanded,
  i.e. it collapses back to the dyad path.  Still possibly useful as a tidy
  intermediate "creative" rewrite to offer the user.
- **Stay symbolic** (keep the two non-sharing ε's): no ε-δ reduction is possible
  without a shared dummy, so there is no closed cross-free form along this path —
  it confirms the I-insertion is *necessary*, not merely convenient.

## Recommendation

Implement **gap 1** (N-ε pairwise `contract_eps_pair`).  Combined with the dyad
identity (already works) and the established `a×I×c` route, it closes `a×B×c`
end-to-end and generalises to deeper cross chains.  Keep gap 2 (the
fence-over-contraction regroup) as a follow-up — it is the cleaner "by hand"
story and reusable, but strictly optional once gap 1 lands.  See
[[steps-self-prepare]]; both new capabilities should self-prepare per vibe 000062.

## Gap 1 done — and what the full run revealed (a third gap)

`contract_eps_pair` now picks the **first ε-pair (in factor order) sharing a
summed index** instead of demanding exactly two ε's, and the public step
**iterates the walk to a fixpoint** so a product of N ε's collapses pair-by-pair.
The 4-ε product the dyad identity produces is contracted directly.

Driven end-to-end on `a × B × c` (orthonormal frame), the reduction is:

```
e = a % B % c
e = expand_in_basis(e, frame)            # B → B_ij e_i e_j
e = apply_identity(dyad)(e)              # e_i e_j → e_j×I×e_i + (e_i·e_j) I
e = expand_in_basis(e, frame)            # expand the inserted I's
e = simplify_basis_cross(e, frame)       # crosses → ε's  (a 2-ε and a 4-ε term)
e = simplify_basis_dot(e, frame)         # basis dots → δ's
repeat until ε-free:                     # TWO rounds here
    e = contract_eps_pair(e)             #   N-ε contraction (this gap)
    e = simplify_basis_dot(e, frame)
    e = contract_delta(e)                #   collapses δ's AND redistributes the
                                         #   binders per-term, so the next
                                         #   contract_eps_pair sees local sums
e = fold_arithmetic(e)
e = reassemble(e, frame)                 # folds the pure-vector parts only
```

The **iteration is essential, not incidental**: after one ε-pair fires it leaves
a δ-determinant *Sum* under the summation binders; only `contract_delta`'s
`expand_products`+`canonicalize` floats those binders into each term so the
*next* `contract_eps_pair` finds two ε's sharing a *local* summed index.  One
pass can't see across an undistributed Sum.

The ε-free coordinate result was checked numerically against direct `(a×B)×c`
and matches exactly:

```
a×B×c = (a·c) Bᵀ + tr(B) c⊗a − c⊗(B·a) − (Bᵀ·c)⊗a
        + [ c·B·a − tr(B)(a·c) ] I
```

**Gap 3 — `reassemble` cannot yet recognise the invariants buried in the
B-bearing coordinates.**  It folds the pure-vector parts (`a_i c_i → a·c`,
`c_i a_j e_i e_j → c⊗a`, `e_i e_i → I`) but stalls on every term carrying `B`'s
components.  Four missing folds, all "recognise the invariant inside the
coordinates" (the rank-2 generalisation of vibe 000061's *recognise a and b
individually*):

1. **Trace** — `B_kk → tr(B)`.
2. **Bilinear contraction** — `B_ij a_i c_j → a·B·c` (a named rank-2 saturated by
   two coordinate vectors).
3. **Named rank-2 reassembly** — `B_ij e_i e_j → B`, `B_ij e_j e_i → Bᵀ`.
4. **Composite dyad legs** — a coordinate vector that is itself a contraction,
   `(B·a)_k e_k → B·a`, so `c_? (B·a)_k … → c ⊗ (B·a)`.  Today `reassemble`'s
   vector-fold only recognises a *bare* component `a_i e_i`, not `B_ij a_j e_i`.

So `a×B×c` reaches **correct ε-free coordinate form automatically**, but the
last mile to the boxed invariant needs these `reassemble` extensions.  That is
the next implementation chunk; gaps 1 (done) and 3 (open) are independent of
gap 2 (the targeted regroup, deferred behind vibe 000054).

## Gap 3 done — a general coordinate→invariant reassembly engine

`reassemble`'s per-term fold was rebuilt as a small **contraction engine** that
folds each recognisable invariant independently, *leaving every unrelated factor
in place* — so the folds fire even when the pattern is one factor of a larger
product (the key requirement: applicable inside bigger terms).  The design rests
on one structural fact: **coordinate components are rank-0 scalars and commute
freely, while basis vectors carry the non-commuting tensor order** — so a
reassembled invariant lands at the *position of the basis vector(s)* it pairs
with, and the order of those basis vectors fixes the result's slot order (hence
the transpose).

Each summed index is classified by where its two occurrences sit, and carriers
(an invariant value + the index riding each slot) are contracted within a blob:

| index occurrences | fold |
|---|---|
| carrier–basis | leg realization: `c_i e_i → c`, `B_ij e_i e_j → B`, `B_ij e_j e_i → Bᵀ` |
| carrier–carrier | contraction: `u_i v_i → u·v`, `B_ij a_j → B·a`, `B_ij D_jk → B·D` |
| same carrier twice | trace: `B_ii → tr B` |
| basis–basis | resolution of identity `e_i e_i → I` |

Chained within a blob these compose: `B_ki a_i c_k → c·(Bᵀ·a)` (a bilinear
scalar), `c (B·a)_k e_k → c ⊗ (B·a)` (a composite dyad leg).  A blob that cannot
be fully expressed (rank ≥ 3 leg ordering, a partial trace, a middle-slot
contraction, an index also carried by a foreign factor) is left **entirely
untouched**, its indices still bound — never a wrong fold.

The whole `a×B×c` derivation now runs end-to-end to the boxed invariant
(integration test `BasisFeasibility.CrossTensorCross`, checked against the closed
form at the coordinate level; numerically equal to direct `(a×B)×c`).  Unit
tests pin each new fold, including inside bigger terms (`Reassemble.TraceFold`,
`TraceFoldInBiggerTerm`, `BilinearFold`, `Rank2TransposeRoundTrip`,
`CompositeDyadLegFold`, `TensorTensorContraction`).

Note the engine handles `tr` arising as a *coordinate* repeated index (`B_kk`,
which is how the ε-contraction leaves it); expanding an explicit `tr(B)` leaves a
basis-dot `e_j·e_i` that `simplify_basis_dot` does not currently reduce — a
separate, pre-existing trace-expansion gap, not part of this work.

Remaining: rank ≥ 3 leg-ordering (needs a permutation operator) and gap 2 (the
targeted regroup, behind vibe 000054).  See [[steps-self-prepare]].
