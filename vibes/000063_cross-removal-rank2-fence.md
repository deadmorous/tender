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
