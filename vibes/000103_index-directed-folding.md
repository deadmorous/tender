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
| `ε` ↔ two vector slots (+ a leg, or a third) | a cross, or the triple product | ✓ challenge 000017 |
| *n* coordinate slots ↔ *n* coordinate slots, through a metric | an *n*-fold contraction, pairing fixed by the index order | ✓ orthonormal (`reassemble`); oblique via `contract_metric`, challenge 000018 |
| coefficient `c_k` ↔ `∂_k` mark | the operator | ✓ `fold_operator`, challenge 000024 |

Six rows, of which the fifth is now a *family* parameterised by n and a
permutation — which is where most of the reach is.

## Implemented

Two increments, both inside `fold_reassembly_groups` (`src/basis.cpp`) — which
turned out to *already be* the index-directed engine this note argues for:
union-find over shared summed indices, each blob folded independently.  What was
missing was not the mechanism but its reach.

### 1. Basis sites are addressed by path, not by factor position

The engine keyed every basis vector by its position in the flattened factor
list, so `Σ_i a_i (e_i·b)` hid `e_i` inside a `Dot` operand where no position
could name it, and the fold stalled.  A `Site` is now `{factor, path}`, paths
are built from `children` (the same accessor `replace_at` navigates), and a
realized invariant is spliced in at its path with the rest of the contraction
left alone.

The `blocked` test sharpened with it: it used to be "does any non-basis factor
mention this id", a blanket veto.  It is now "does the id occur in this factor
more times than the basis vectors we collected from it" — which admits the
foldable leg and still rejects a second carrier hiding in the same operand.

Placement is restricted to **rank-1** invariants, and that restriction is
principled rather than provisional: a vector occupies exactly the one slot the
basis vector did, so every enclosing contraction keeps its meaning.  A rank-2
value spliced at a nested site would silently take the wrong slot orientation —
`e_i`'s position says "first slot", but a `Dot` contracts the last — so rank ≥ 2
with any nested site refuses and leaves the blob untouched.

Measured, against the boundary table in vibe 000100:

| expression | before | after |
|---|---|---|
| `a_i e_i` | `a` | `a` |
| `(a_i e_i)·b` | *stalls* | `a·b` |
| `b·(a_i e_i)` | *stalls* | `b·a` |
| `(a_i e_i)·(b_j e_j)` | *stalls* | `a·b` |
| `(a_i e_i)×b` | *stalls* | `a×b` |
| `(a_i e_i)·B` | *stalls* | `a·B` |
| `s (a_i e_i)·b` | *stalls* | `s (a·b)` |
| `(A_ij e_i e_j)·b` | *stalls* | *refuses* (rank 2, nested) |

### 2. Carrier contraction is counted and ordered

`contract_carriers` contracted over one shared index at a time and gave up above
rank 2, so a pair sharing *two* indices could never fold — taking them one at a
time asks for an intermediate rank the notation cannot express.
`contract_carriers_n` contracts all the shared ids in one move, and reads the
result off the index structure rather than a stored pattern: the ids on the left
carrier's last two slots, matched against the right carrier's first two, give
`:` in order and `··` reversed.

Verified both ways round, through the full component pipeline:
`A_ij B_ij → A:B` and `A_ji B_ij → A··B` — different tensors, told apart by
index order alone.

The rank-4 case that motivated this (`C_ijkl e_lk → C··e`) folds by the same
code, but cannot yet be *reached*: `expand_double_dot` handles only 2-leg dyads,
so a rank-4 double dot never reduces to components in the first place.  That is
a separate, upstream gap.

### What this did and did not unblock

- **Challenge 000024** (operator round-trip) stays red, and the change tells us
  why with more precision than before: the blame on "reassemble does not descend
  into a contraction" was wrong.  It descends now.  The real blocker is the
  table's *operator row* — the index there joins a coefficient `c_i` to a `∂_i`
  mark, and nothing reads that pairing.  The test's stated reason was corrected.
- **Challenge 000017** (ε → cross) stays red: a different row, untouched.
- **Challenge 000018** stays red *by choice*.  The round-trip route this work
  would have opened is the one `meta/l2-route.md` rejects as evasion — both
  component forms were built from `a·b`, so folding one back proves nothing
  about the metric.  It still needs `tender.metric` and the inverse-metric axiom.

## The remaining two rows, since built

**ε (challenge 000017 → L2).**  ε is well-known, so it is not a coordinate
carrier and the classifier used to let it block its own three indices.  It is
now set aside and matched against what those indices connect: two on rank-1
carriers plus one on a basis vector is a cross realized at that leg; all three
on carriers is the scalar triple product.  The slot *order* fixes the result, as
it did for the double dots — ε is totally antisymmetric, so rotating the leg
index to the front is sign-free (ε_abc = ε_bca = ε_cab), and the remaining two
in that rotated order are the operands.

It refuses rather than guesses in four places: non-orthonormal or left-handed
frames (where the ε weight would have to come back too), an index shared by
*two* ε's (that is the ε-pair contraction's business, so `(a×b)×c` is left
alone), a repeated index inside one ε, and a carrier of rank ≥ 2 (ε does not say
which slot would be the operand).

`Carrier` gained a `folds` field — ids it has absorbed that no later step will
see.  They are released, their Σ binders dropped, only if the carrier is
actually realized, so a blob that fails still leaves them bound.

**The operator row (challenge 000024 → L2).**  This one folds differently from
the rest, and the difference is the interesting part.  The other rows fold *one
index cluster* inside a term.  Here the summed direction is spread across a
whole **group of addends** — `f ∂ₓg i + f ∂_y g j` is one operator application
wearing two terms — so the unit is a complete group, and the argument for
folding it is completeness, exactly as in `fold_resolution_of_identity`
(`Σ_k e_k⊗e_k = I`).  `steps::fold_operator(e, op)` is the inverse
`apply_operators` never had.

The caller supplies `op`.  That is not a shortcut: nothing in the library knows
that `i∂ₓ + j∂_y` deserves to be read back as one thing, and in challenge 000024
the operator has no name at all — it is assembled in the test.  This is
vibe 000102's Q2 answered the way it was posed, with the user declaring the
equivalence and owning it.  ∇, which the library *does* name, keeps its own
route: `reassemble_nabla` folds the frame-vector/∂-mark pairing back to the
symbol.

Refusals again: an incomplete group, members disagreeing on the operand, sign,
or company, a non-scalar factor alongside (where the folded operator belongs in
the product order would be a guess), and an `op` that is not a sum of at least
two distinct concrete directions.

**The metric row (challenge 000018 → L2).**  This one did not become a
`reassemble` fold at all, and the reason is worth recording: the round-trip it
would have enabled — fold `g^{ij} a_i b_j` back to `a·b`, re-expand covariantly
— is the one the challenge's own route doc rejects as evasion.  Both component
forms were *produced* from `a·b`, so folding one back proves nothing about the
metric.  The content of the claim is the index gymnastics, so the derivation had
to stay in components.

What it needed was `tender.metric` on the surface (now bound beside `delta`) and
one operation, not the two planned: `contract_metric` spends a metric to move an
index — the survivor is g's *other* index at g's *other* level.  Read three ways
that is raising, lowering, and the inverse pair, since raising `g_{pk}`'s lower
index gives `g^i{}_k`, and a g whose slots straddle the divide *is* δ
(`g^i{}_j = e^i·e_j`).  So `metric-inverse` never needed postulating as a
separate axiom; it falls out of what raising does.  `insert_metric(level)` runs
it backwards, paying a metric to move an index the other way.

Modelled on `contract_delta`, which is the precedent: a step for a derivation to
use, with a DAG node reserved for a rule the e-graph fires.  Extracting the
shared mechanics — the distributed-sum guard, the factor drop, the partner
search — made `contract_delta` shorter rather than longer.

## What is left

All six rows work.

## Status

Design proposal, unscheduled.  It supersedes vibe 000100's option (A) by
generalising it, and it is the blocker for finishing (D) (vibe 000102), for
challenge 000017's ε-reassembly, for challenge 000018's L2, and for challenge
000024's L2.
