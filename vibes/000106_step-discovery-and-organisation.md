# 000106 Steps: discovery, and a step set worth searching over

Vibe 000105 proposed named routes as the answer to "which steps do I need?".
The user's objection, and it is correct: **a route prescribes a path, but the
problem is discovery.**  A canned route helps only when it happens to match; it
cannot help a derivation that needs its own sequence, and it turns the library's
inability to *find* a sequence into the user's problem.

**Scope.**  This vibe covers only the prerequisites: the discovery probe, the
step set it reports over, and the organisation of that set.  Path search and the
specification of a derivation's *goal* are deliberately **out of scope** and are
the next milestone — see the last section.  The prerequisites are worth doing on
their own merits; the search is what they unlock.

## 1. The probe, and the reason it can exist

> a function taking an expression and printing names of all steps that are not
> no-ops for that expression

This is implementable because of a contract this project has enforced and
repeatedly tightened: **a step that does nothing returns its input unchanged**.
That makes "did it bite?" a one-line test — `structural_eq(step(e), e)` — needing
no per-step cooperation.  A library whose steps returned canonicalized-but-
unchanged output (as `apply_identity` once did, vibe 000056) could not have this
feature at all.

Measured surface: of the callables on `td` / `tb` / `tender.steps`, **21 take
just `(expr)`**.  The rest take exactly one more thing, drawn from a short and
*closed* list of kinds — a frame/basis, a coordinate, a rule set, a level, an
index list, an identity, an operator.  That is the "dictionary of what we could
throw in there", and its smallness is what makes the probe practical.

### The refinement the prototype forced

Run naively on `a_j e_j · b_i e_i`, **13 steps fire**, and seven are
`canonicalize`-class: they change the tree without changing its content.  A raw
"what fires" list is mostly noise and would train the user to ignore it.

**"Not a no-op" is the wrong filter; "changed the content" is the right one.**

With a first-cut fingerprint (node count, Σ / δ / ε / basis-vector counts) the
list collapses to four real options, each annotated with what it does:

```
tb.simplify_basis_dot   nodes−2, deltas+1, basisvecs−2   → b_j a_i δ_ij
tb.reassemble           nodes−4, basisvecs−2             → a·b
td.unroll_sums          nodes+64, basisvecs−2
td.canonicalize         nodes+2, sums+2
…7 others               reordered only
```

Swap the dot for a cross and the useful list changes (`simplify_basis_cross`
appears, `…_dot` drops to "reordered only"), so the probe reads the expression
rather than reciting a menu.

### The fingerprint stays open

That fingerprint is a **first cut, not a decision**.  It counts rendered LaTeX
substrings, which is expedient for a prototype and wrong for a library.  Likely
additions once real cases are in hand: **index-slot layout** (which slots carry
which dummy, at which level) and **rank**, which together would distinguish
moves that today look alike — a raise and a lower both "change a level", but
they are not the same event.

The criterion for evolving it is empirical, not aesthetic: *does the probe's
output, on the corpus's real derivations, put the step a human would take near
the top?*  That is measurable against the nine hand-written pipelines, so the
fingerprint should be tuned against them rather than argued about.

## 2. The step set is probably wrong, and here is the evidence

The user's instinct — that today's steps are not a proper user-facing basis —
is measurable, and it holds.  Across the 24 challenges, 12 examples and the
core ∇ test suite, **ten steps are never used alone**:

| uses | solo | step |
|---|---|---|
| 30 | 0 | `canonicalize` |
| 13 | 0 | `expand_in_basis` |
| 11 | 0 | `simplify_basis_dot` |
| 9 | 0 | `simplify_basis_cross` |
| 5 | 0 | `unroll_sums` |
| 4 | 0 | `eval_delta_concrete`, `fold_arithmetic` |
| 3 | 0 | `eval_eps_concrete`, `contract_delta`, `simplify` |

**A step used 30 times and never once on its own is not an operation a user
wants — it is punctuation the user must recite to make the next step work.**
`canonicalize` is the extreme case: it is the single most-called name in the
corpus, it appears *twice in a row* three times, and it follows `simplify_scalars`
seven times.  Nobody's intent is "canonicalize this".

The adjacent-pair counts show the clusters directly:

```
expand_in_basis → simplify_basis_cross → simplify_basis_dot → canonicalize
canonicalize → unroll_sums → eval_eps_concrete → eval_delta_concrete
    → fold_arithmetic → canonicalize
simplify_basis_dot → contract_delta → canonicalize
```

### The pattern to generalise already exists here, twice

- **Self-preparation** (vibes 000060/000061): a step must not require the caller
  to canonicalize or distribute first.  Applied piecemeal — `reassemble`,
  `contract_delta`, `contract_metric`, `fold_equal_addends` all self-prepare
  today.  Generalised, it removes `canonicalize` from user code almost entirely.
- **`target=`**: `contract_metric(expr, target="b")` and
  `insert_metric(expr, level, target="a")` (vibe 000104) are exactly the shape
  the user describes — do the whole job by default, narrow it on request.

The user's own example fits: there are four reassembly-ish entry points today
(`reassemble`, `reassemble_completeness`, `fold_resolution_of_identity`,
`reassemble_nabla`), and one `reassemble(expr, frame, target=None)` that tries
all of them is plainly the better surface.

### Method for the investigation

1. For each never-solo step, find the maximal cluster it appears in.
2. Name the cluster's **intent**, not its mechanism.  If it has no nameable
   intent, that is a finding too.
3. Ask whether a `target=` narrows it, so one name covers the general and the
   specific case.
4. Re-measure the corpus: do the 36 derivations get *shorter and more legible*?
   That is the acceptance test, and it is a number.
5. Only then decide what stays user-facing and what is demoted.

The risk to watch: a higher-level step that bundles too much becomes the very
thing vibe 000105 got wrong — a prescribed route with a shorter name.  The
`target=` escape hatch is what keeps it from becoming that, and step 4's
legibility measure is what detects it.

### Investigation results (2026-08-30)

Re-measured on **real derivations only** — the 24 challenges and 12 examples,
excluding `python/tests`, where a unit test calls a step alone *by design* and so
is not evidence of anything.  The result is sharper than the first count:

> **14 of the 15 steps used in real derivations are never used alone.**
> The exception is `simplify_scalars`, once.

Every real derivation is a pipeline.  So the question is not *whether* to raise
the level but *where* the joints are.  Three candidate bridge steps, each a
fixpoint of today's moves, were prototyped and checked against the corpus's own
hand-written pipelines:

```python
reduce_frame(e, frame)   # fixpoint: simplify_basis_cross, simplify_basis_dot,
                         #           canonicalize, contract_delta
to_concrete(e, frame)    # fixpoint: unroll_sums, eval_eps_concrete,
                         #           eval_delta_concrete, fold_arithmetic
reassemble(e, frame)     # fixpoint: reassemble, reassemble_completeness,
                         #           fold_resolution_of_identity
```

Measured:

| corpus pipeline | today | proposed | result |
|---|---|---|---|
| 000004 `a·b` → symbolic components | 5 moves | 2 | match |
| 000005 `a×I` → symbolic components | 3 moves | 2 | match |
| 000001 `a×(b×c)` → concrete | 10 moves | 3 | match |
| 000014 `(a×b)·(c×d)` → concrete | 8 moves | 3 | match |
| 000017 round-trip battery, 10 shapes | — | 3 | 10/10 |

`canonicalize` — 19 uses, never alone — **disappears from user code entirely** in
all of them.  That is the self-preparation principle finally applied where it was
always going to pay.

### The proposed bridge set: 12 → 4

| keep | why |
|---|---|
| `expand_in_basis(e, frame, variance)` | choosing to go to components, and on which frame, is a real decision |
| `reduce_frame(e, frame)` | reduce whatever the frame licenses |
| `to_concrete(e, frame)` | evaluate over the frame's directions |
| `reassemble(e, frame, target=None)` | come back to direct notation |

Demoted but still importable: `simplify_basis_dot`, `simplify_basis_cross`,
`contract_delta`, `canonicalize`, `unroll_sums`, `eval_eps_concrete`,
`eval_delta_concrete`, `fold_arithmetic`, `reassemble_completeness`,
`fold_resolution_of_identity`.

Note what is *not* absorbed: `contract_eps_pair`, `contract_metric`,
`insert_metric`, `contract_identity` stay user-facing, because each is a distinct
mathematical event a derivation chooses deliberately — not punctuation.  That
line (an event the mathematician names vs. a move the machine needs) is the one
worth holding.

### What each step does, concretely

Real intermediates, not a description:

```
a·b        invariant        a·b
           expand_in_basis  a_j e_j · b_i e_i        ← raw: the dot is still between basis vectors
           reduce_frame     a_i b_i                  ← no basis vector survives; the dot consumed both
           to_concrete      a_x b_x + a_y b_y + a_z b_z
           reassemble       a·b

a×b        expand_in_basis  a_j e_j × b_i e_i
           reduce_frame     ε_ikj a_k b_j e_i        ← one *free leg* e_i survives: the result is a vector
           to_concrete      a_y b_z i − a_z b_y i − …
           reassemble       a×b

A·b        expand_in_basis  A_kj e_k e_j · b_i e_i
           reduce_frame     A_ij b_j e_i
           reassemble       A·b
```

Output is in **implicit (Einstein) form**: `a_i b_i`, not `Σ_i a_i b_i`.  The
steps materialise the Σ binders internally — the contractions need to see them —
and must put them back before returning.  A leaked binder is a bug, and one the
prototype had.

So the exit conditions, which is what makes the joints crisp:

| step | does | exits when |
|---|---|---|
| `expand_in_basis` | choose a frame and a variance; go to components | — (one shot) |
| `reduce_frame` | everything the frame *alone* licenses: `e·e → δ`, `e×e → ε`, contract δ | no basis-vector product remains |
| `to_concrete` | replace symbolic indices by the frame's directions, evaluate | no symbolic index remains |
| `reassemble` | fold components back to invariants | nothing further folds |

Each exit condition is **checkable**, which is not incidental: it is exactly the
goal predicate a later search would use, and exactly what `applicable`'s stall
diagnosis would report.

### Where `reduce_frame` stops — and why that is the point

`a×(b×c)` shows the boundary:

```
expand_in_basis   a_k e_k × (b_j e_j × c_i e_i)
reduce_frame      −ε_iml ε_mkj a_l b_k c_j e_i   ← stops: two ε's, nothing more the *frame* can say
contract_eps_pair δ_ij δ_lk a_l b_j c_k e_i      ← a mathematical event, the user's choice
reduce_frame      a_j b_i c_j e_i − a_j b_j c_i e_i   ← re-entered
reassemble        (a·c) b − (a·b) c               ← bac-cab, derived
```

`reduce_frame` does **not** apply `contract_eps_pair`, and `reassemble` correctly
refuses on the un-contracted form.  The user interleaves their own mathematical
decisions with the bookkeeping steps — which is the whole difference from vibe
000105's routes, and it is why this is a step vocabulary rather than a pipeline
with a shorter name.

It also settles the merge question: **`reduce_frame` is re-entrant, and
`expand_in_basis` is not.**  In the derivation above `reduce_frame` runs twice
against one `expand_in_basis`, so folding them into a single `to_components`
would make the second entry inexpressible.  They stay separate.

### Open calls before this lands

1. ~~Names.~~ **Settled (user): `reduce_frame` and `to_concrete`.**
2. ~~Does `expand_in_basis` stay separate?~~ **Settled: yes** — `reduce_frame`
   is re-entrant and `expand_in_basis` is not; see the bac-cab derivation above.
3. ~~What does `target=` mean?~~ **Settled (user): a tensor name**, as
   `contract_metric` uses.  Richer selectors can come later; a name will not
   block them.

## 3. Organisation

The categories a user needs are **not** the current module split (`td` / `tb` /
`tender.steps`), which is by implementation history:

| category | what it does to an expression |
|---|---|
| **normalise** | reshapes, preserves content |
| **bridge** | invariant ⇄ components |
| **index algebra** | contracts / moves indices |
| **operators** | ∇ and ∂ |
| **engine** | goal-directed |

Someone asking "how do I get back to direct notation?" wants the **bridge** row
and does not care that half of it lives in `tb` because it takes a basis.

**The mechanism already exists**: `tender.identities` gives every identity `tags`
and exposes `rules("cross", "leibniz")` / `group_names()`.  Steps carrying the
same tags gives sorted probe output, `steps_in("bridge")`, and a documentation
order — one idea, three payoffs, consistent with a pattern the project chose.

This also gives the vibe-000092 namespace complaint somewhere to land, and fixes
a regression: vibe 92 objected to ~30 exported steps, M3 got
`tender.derivation.__all__` to 36, it is **41** today.

## 4. Other feedback forms

1. **`applicable(expr, **context)`** — §1.  Content-changing first, "reordered
   only" collapsed to a count.
2. **`why_not(expr, step)`** — the inverse, higher value when the user *has* an
   expectation: "`contract_delta` did nothing: no δ carries a summed index".
   Needs per-step precondition reporting, so: incrementally, for the most-used
   steps, not all 41.
3. **`explain(before, after)`** — what a step changed, rendered; the probe needs
   it internally and `tender.render.labeled` already shows it positionally.

## 5. Order of work

The step set comes **first**: `applicable` reports *over* a set of steps, so
building it before §2 settles that set would mean building it twice, and tuning
its fingerprint against steps that are about to be replaced.

1. **Step-set investigation** (§2), ending in a proposed user-facing set.
2. **Categories/tags** (§3), reusing the identities mechanism.
3. **`applicable`** with content-change classification, over the settled set and
   sorted by category; fingerprint tuned against the corpus.
4. **`why_not`** for the most-used steps.

## 6. Out of scope — the next milestone

**Path search and goal specification.**  The prerequisites above are what make
it possible: `applicable` is the successor function, and hash-consing gives a
visited set for free.  A first measurement — branching factor 2–12 at depth 8
along challenge 000001's pipeline, with a crude over-counting classifier — says
the space is small enough to be worth searching.

But that measurement was taken over *today's* step set, and §2 expects that set
to change.  So the search must be designed and assessed against the **redesigned**
steps, not these; and it needs its own answer to the harder question of how a
user *states the goal* of a derivation.  Both belong in the next vibe.

Vibe 000105's nine hand-written pipelines are then not a design but **test data**:
what a search must be able to rediscover.

### Does search duplicate the e-graph?  Measured: no, and here is the boundary

A fair concern — the project already has equality saturation, and two engines
would cut against "a single way to do a thing".  The decisive experiment:

```python
prove_equal(a·b, a_i b_i, <every shipped rule group>)
    → ProofResult(proved=False, status='refuted', passes=1)
```

It does not merely fail — it **refutes**, in one pass, and correctly.  `Refuted`
means "components differ: the statement is false", and as *chart-free*
expressions those two are not equal: the equality `a·b = a_i b_i` holds only
**relative to a chosen basis**.  None of the 16 shipped identities mentions a
basis, frame or chart, and that is not an oversight:

- **Rules are not parameterized.**  `expand_in_basis(e, frame, variance)` depends
  on a runtime object and a variance choice.  There is no LHS→RHS pattern for
  "…in this basis"; you would need a rule per basis.
- **Rules cannot mint indices.**  Every shipped identity's RHS indices are bound
  by its LHS.  `a → Σ_i a_i e_i` introduces a fresh unbound `i` — the classic
  e-graph blow-up shape, unboundedly many e-nodes.
- **Some bridge moves are not local.**  `unroll_sums` multiplies term count by
  dim^k (measured: nodes +64 on a two-vector dot).  In an e-graph that lands in
  the class permanently and is then re-matched by everything.

So the two operate in different worlds: **saturation reasons *within* a
representation; the bridge moves *between* representations.**  The engine's own
verdict is the proof — it classifies the far side of the bridge as a different
thing.

Sizing them: in the corpus the bridge steps account for roughly 86 calls against
roughly 30 for the engine surface.  Search would target the larger, currently
un-automated half.

**Where the concern *is* valid**, and the rule that follows: for a purely
algebraic derivation the e-graph already does this, and better — it explores all
rule orders at once, where a path search must commit and backtrack.  Search must
therefore not re-implement equational reasoning.  The clean division:

> **the engine is one of the steps the search can take.**

Search navigates representations; inside one, it calls saturation.  That also
gives goal specification a natural form — "reach a state where `prove_equal`
finishes" is a goal the library can already evaluate.

Risk to watch: if §2's redesign makes steps higher-level, some may come to
*contain* engine calls, blurring this boundary.  It should stay explicit.

## Status

Design, revised twice — after the routes objection, and after the user's scoping
call.  `applicable` is prototyped in scratch and the numbers here are measured
from it; nothing is committed to the library.
