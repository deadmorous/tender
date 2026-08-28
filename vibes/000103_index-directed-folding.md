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

## Correction from the user's examples: the unit is a *cluster*, and the fold is *counted*

The table above was written with one shape in mind — a component slot against
its basis vector.  The user's two examples are a different class, and they
break the "pair of factors, look up a pattern" framing in a way that improves
it.

### (a) The five dot-product spellings are *one* row, not five

```
    a_i b_i             (orthonormal)
    a_i b^i   a^i b_i   (mixed variance)
    a_i b_j g^{ij}      a^i b^j g_{ij}
```

All five are the same cluster: **two coordinate slots joined through a metric**,
where the orthonormal and mixed-variance spellings are the degenerate cases in
which the metric is δ and has been elided.  Writing them as five rows would be
the vibe-000100 treadmill in miniature; writing them as one row with a guard is
the point of the design.  The guard is already available — `Basis::is_orthonormal()`
and the basis `Realm` — so the pass can *check* rather than assume, which
matters because `a_i b_i = a·b` is simply false in an oblique basis.

Note this row has no basis vector in it at all.  The existing rows fold a
component against `e^i` and yield an **invariant tensor**; this one folds a
component against another component and yields a **contraction of invariants**.
Two different outputs, same mechanism.

### (b) `C_{ijkl} e_{lk}` shows the fold is decided by *count and order*

```
    C_{ijkl} e_{kl}  =  C : e        [DDot,    (a⊗b):(c⊗d) → (a·c)(b·d)]
    C_{ijkl} e_{lk}  =  C ·· e       [DDotAlt, (a⊗b)··(c⊗d) → (a·d)(b·c)]
```

Both are well-formed, both fold, and they are **different tensors**.  So the
fold is not selected by a stored pattern at all — it is computed from two facts
the incidence structure already carries:

- **how many indices the two clusters share** — one gives `·`, two give a
  double dot, and in general *n* gives an *n*-fold contraction;
- **in what order** — the permutation matching the shared index sequence
  against slot order picks `:` from `··`.

That is *more* mechanical than a pattern table, and it subsumes rows rather
than adding them: `a_i b_i` is the n=1 case of the same rule that gives
`C·· e` at n=2.  The dot-product row and the stiffness row are one row.

### The hazard this creates, and the rule for it

A wrong permutation is **silent** — it yields a well-formed expression that is
simply not equal to the input.  So the pass must **refuse rather than guess**:
when the shared indices sit in an order the surface language cannot name
(`C_{ikjl} e_{lk}`, which needs a transpose interposed), the fold does not
fire.  Declining to fold leaves the expression correct; folding to the wrong
pairing does not.  This is the one place in the design where an extra
capability is worse than a missing one.

### Revised table

| what the index cluster connects | folds to | today |
|---|---|---|
| coordinate slot ↔ basis vector `e^i` | the invariant | `reassemble`, whole-term only |
| `δ` ↔ any slot | index substitution | ✓ `contract_delta` |
| `ε` ↔ `ε` | δδ − δδ | ✓ `contract_eps_pair` |
| `ε` ↔ two vector slots | a cross product | gap — 000017 |
| *n* coordinate slots ↔ *n* coordinate slots, through a metric | an *n*-fold contraction, pairing fixed by the index order | gap — 000018 (n=1), stiffness (n=2) |
| coefficient `c_i` ↔ `∂_i` mark | the operator | gap — 000024 |

Six rows, of which the fifth is now a *family* parameterised by n and a
permutation — which is where most of the reach is.

## Status

Design proposal, unscheduled.  It supersedes vibe 000100's option (A) by
generalising it, and it is the blocker for finishing (D) (vibe 000102), for
challenge 000017's ε-reassembly, for challenge 000018's L2, and for challenge
000024's L2.
