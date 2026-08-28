# 000103 Index-directed folding — the missing mechanical step

Vibe 000100 catalogued *nine* instances of one failure and offered four
directions.  This note answers the sharper question the user asked: **what is
the small, mechanical thinking step a human performs that tender does not?** —
and turns the answer into a single targeted feature.

## How a person actually folds `a_j b_i e^j·e^i → a·b`

Introspecting on the real thing, rather than describing the result:

1. **Look at the dummy indices.**  There are two, `i` and `j`.
2. **For each, ask what it connects.**  `j` joins `a_j` to `e^j`; `i` joins
   `b_i` to `e^i`.
3. **Ask whether that pairing is a known form.**  "Component times its basis
   vector, summed" is one — it *is* the vector.
4. **Fold it, ignoring everything else in the expression.**

The whole discovery is step 2, and it is not creative: **the index tells you
where to look.**  There is no search over subexpressions, no guessing.  The
number of dummy indices is small, and each points directly at its own
endpoints.

Two things make this cheap for a person and expensive for tender:

- We do not store a canonical factor order, so "bring `a_j` next to `e^j`" is
  free.  Canon deliberately sorts, destroying exactly that adjacency.
- We read an index as a **pointer**.  tender reads a term as a **shape**.

## The diagnosis, stated precisely

`reassemble` asks a question about the **whole term**:

> does this term have the form (coordinate tensor) × (polyad of basis vectors)?

Any extra factor changes the answer, which is why a neighbour is enough to
defeat it — measured in vibe 000100: `a_i e^i` folds, `(a_i e^i)·b` does not,
*even in an orthonormal basis*.

The right question is **local**:

> index `j` connects `a_j` and `e^j` — is that pair foldable?

## The evidence: tender already does this, twice, and those steps never fail

`contract_delta` and `contract_eps_pair` are **index-directed**.  Measured:
`contract_delta` fires on

```
δ_kj a_k b_j c_i e_i        →        a_j b_j c_i e_i
```

— a term carrying two extra factors (`c_i`, `e_i`) of exactly the kind that
defeats `reassemble`.  It does not care, because it is looking at what `j`
connects, not at the term's silhouette.

And these two steps are conspicuously **absent from vibe 000100's catalogue of
nine failures**.  The steps that work are index-directed; the ones that get
stuck are shape-directed.  That is the whole diagnosis, and it is testable
rather than aesthetic.

## The feature: a local fold table, keyed by what an index connects

One pass, not one patch per case:

1. Build the **index-incidence** structure of a term: for each dummy index,
   the factors carrying it and the slots involved.
2. For each index (or small group of indices), consult a **fold table**.
3. Apply the fold **locally** — replace those factors, leave the rest of the
   term untouched.
4. Repeat to a fixed point.

The table is short, and its entries are exactly the outstanding gaps:

| what the index connects | folds to | today |
|---|---|---|
| coordinate slot of `T` ↔ basis vector `e^i` | the invariant `T` | `reassemble`, whole-term only |
| `δ` ↔ any slot | index substitution | **`contract_delta`** ✓ |
| `ε` ↔ `ε` | δδ − δδ | **`contract_eps_pair`** ✓ |
| `ε` ↔ two vector slots | a cross product | missing — challenge 000017 |
| `g` ↔ coordinate slot | raise / lower | missing — challenge 000018 |
| coefficient `c_i` ↔ `∂_i` mark | the operator | missing — challenge 000024 |

Four gaps, one mechanism.  The two rows that already work are the proof that
the mechanism is right; the four that do not are what it would deliver.

**It is bounded, not a search.**  The candidate folds are enumerated by the
dummy indices present — a handful — and each names its own operands.  Nothing
backtracks; nothing explores.  That is what makes this the *mechanical* kind of
thinking rather than the creative kind, and why it belongs in the library
rather than in a rule set.

## Why this beats the alternatives in vibe 000100

- **(B) more rules**: a rule is a shape, so every new context needs another
  rule.  This is the treadmill vibe 000100 documented.
- **(C) `td.at`**: makes the *user* find the position; and after
  canonicalization the position may not exist.
- **(A) index-directed folding**: the position is *derived from the index*, so
  there is nothing to find.

Vibe 000100 listed (A) as "the narrowest fix and probably the correct one for
reassembly specifically".  The fold table generalises it: not narrow at all —
four of the outstanding gaps are one table with four rows.

## Note on the operator case

The `∂_i` mark already carries a `link` — "the abstract-direction tie of a
free-index ∂, so this `∂_i` and the frame vector `e_i` that carries the same id
contract" (vibe 000078).  The index-as-pointer idea is therefore already
*in the representation* for operators; what is missing is a pass that reads it.
That is a good sign for the design: it was arrived at once before, under
pressure, for one case.

## Status

Design proposal, unscheduled.  It supersedes vibe 000100's option (A) by
generalising it, and it is the blocker for finishing (D) (vibe 000102), for
challenge 000017's ε-reassembly, for challenge 000018's L2, and for challenge
000024's L2.
