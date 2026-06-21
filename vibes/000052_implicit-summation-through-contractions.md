# 000052 Implicit summation through contractions (term scope)

A correctness improvement to the implicit Einstein-summation analysis, plus a
careful record of the one case the user anticipated may still go wrong:
`a_i (b^i + c^i)`.

## The change

`materialize` (the pass at the front of `canonicalize` that turns implicit
Einstein contractions into explicit `ExplicitSum`, vibe 000028/000041) used to
detect contraction only among the **direct tensor factors** of a product (and
the slots of a lone tensor).  An index buried inside a *contraction* factor —
e.g. the `i` in `(e_i·e_j)(e_i·e_j)` — was not counted, so it was left free
(under-summed).

Now the analysis works on a **term**: a *pure multilinear* tree of tensors —
products and contractions (`⊗ · : ·· ×`) and the linear unary ops
(`Negate`/`tr`/`vec`/`transpose`), with **no** `Sum`/`Difference`/`ScalarDiv`/
binder.  The whole term is one Einstein-summation scope, and repeated free
indices are counted **across the entire term, descending through the
contractions** (`collect_term_uses`).  So `(e_i·e_j)(e_i·e_j)` correctly sums
both `i` and `j`.

- `is_term` + `collect_term_uses` replace the old `is_polyad` /
  `materialize_product`; `contracted_ids` now takes the term `Expr` and uses the
  deep collector.  `materialize` wraps a term once and otherwise descends.
- This made `implicitize` (the inverse — strip a redundant explicit sum) agree
  with `materialize`, so results **round-trip**: `apply_identity` returns the
  implicit form `(e_i·e_j)(e_i·e_j)`, and `canonicalize` re-derives the double
  sum.  It is the fix that let `apply_identity` return implicit sums (vibe
  000051 follow-up).
- All existing behavior preserved (explicit/bound indices, the eps-δ identities):
  full suite green.  Side benefit: `a_i·b_i`-style contractions inside a `Dot`
  now sum correctly everywhere, not only in the motivating case.

## The boundary that remains: `Sum`/`Difference` are opaque

A term stops at a `Sum`/`Difference`/`ScalarDiv`/binder — those are separate
scopes.  This is deliberate (an un-distributed sum cannot be summed factor-wise
without distributing first), but it is exactly where the user expects trouble.

### `a_i (b^i + c^i)` — verified behavior

```
a_i (b^i + c^i)                         -- canonicalize: i is NOT summed (left as-is)
  --expand_products-->  a_i b^i + a_i c^i
  --canonicalize-->     Σ_i a_i b^i + Σ_i a_i c^i      -- now each term sums i
```

So the implicit sum over `i` in `a_i (b^i + c^i)` is **not established until the
product is distributed over the sum**.  The `i` repeats once outside the
parenthesis and once inside each addend; at the term level the analysis sees
only the outside `a_i` (one occurrence) and stops at the `Sum`, so `i` reads as a
free index rather than a contraction.  The workaround today is to
`expand_products` first; after distribution each addend is a pure term and the
sum is detected per term.

This is the long-standing "deferred with distribution" limitation (vibe 000041),
now isolated to exactly the `Sum`/`Difference`-factor case.

### Why not just descend into `Sum` factors too

`(Σ over the term) ⊗ (b^i + c^i)` is genuinely `Σ_i a_i b^i + Σ_i a_i c^i` by
linearity, so one *could* count `i` through the addends and contract.  But it is
subtle and easy to get wrong:

- The repeated index must appear in **every** addend with the matching
  complementary level/realm, or the contraction is ill-formed for some addend;
  the analysis would have to validate that per addend.
- A `Difference` flips signs; a nested sum compounds the bookkeeping.
- Distribution (`expand_products`) already turns this into the clean, unambiguous
  per-term form — so the principled move is "distribute, then sum", not "sum
  across an un-distributed bracket."

If we ever want `a_i (b^i + c^i)` to sum **without** an explicit
`expand_products`, the cleanest implementation is to have `materialize` (or a
small pre-step) distribute products over sums on the index-carrying paths first,
then run the term analysis — i.e. fold a targeted distribution into the
implicit-sum pass, not to teach `collect_term_uses` to reach across a `Sum`.

## If something goes wrong here later

Symptoms to recognise this is the culprit: an index that should be summed stays
**free** (renders without a `Σ`, and a later `unroll_sums`/contraction no-ops on
it) specifically when it straddles a `Sum`/`Difference`/`ScalarDiv` — i.e. it
appears outside a bracket and inside the bracketed addends.  First remedy:
`expand_products` (distribute) before the index step.  Proper fix: the
distribute-then-term approach above.
