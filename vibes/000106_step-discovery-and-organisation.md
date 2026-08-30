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
`tender.derivation.__all__` to 36, it reached **41**.

**Done (2026-08-30).**  `tender.steps` is now the catalogue: every step with its
category, a one-line summary, and — the part `applicable` needs — the *kinds* of
argument it wants beyond the expression (`needs` for required, `options` for
accepted).  Measured: **34 of 39 steps run from a basis alone**; the other five
announce exactly what they lack (`rules`, `op`, `level`, `coord`).  The
"dictionary of what we could throw in there" is therefore data, not lore.

`__all__` — the *advertised* surface, not the reachable one — shrank from 41 to
**30** on `tender.derivation` and 18 to 13 on `tender.basis`, with every demoted
name still importable and working.  `expand_in_basis` gained a `Covariant`
default for `variance`, which is what the corpus passes almost everywhere and
what an orthonormal frame makes moot.

A reconciliation test mirrors the identity DAG's: an advertised step that nobody
catalogued fails the suite, because a step nobody can find is worse than no
catalogue.

## 4. Other feedback forms — **all three built (2026-08-30)**

`ts.applicable(expr, **context)`, `ts.why_not(expr, step, **context)`,
`ts.explain(before, after)`.  Three notes from building them:

**The fingerprint moved to the IR, and grew.**  It counts nodes, Σ binders,
index slots, δ/ε/I/g, ∂-marks, unapplied ∂s, ∇s, rank — and, following the
user's suggestion, distinguishes **coordinates** (`a_i`, rank-0 with a basis
tag) from **basis vectors** (`e_i`, rank ≥ 1 with one).  That last split is what
makes a bridge step's effect legible: `reduce_frame` reads as
`basis_vectors−2, index_slots−2, nodes−4`.

**Preconditions are data, which is what makes `why_not` cheap.**  Each step
records `wants` — minimum fingerprint counts, `{"deltas": 1}` — so the answer is
"deltas (needs 1, has 0)" rather than silence.  19 of 39 steps carry one; the
rest fall back to "it ran and changed nothing (no precondition recorded)", which
is honest and improvable step by step, as the vibe proposed.

**The probe immediately found a bug in the library.**  `sym` and `skew` appeared
as content-changing options on a *scalar*: `sym(a·b)` produced ½(a·b + (a·b)ᵀ)
and never collapsed, because transpose had no rank-0 case.  A tool that reports
what applies is also a tool that notices what applies *and should not*.

Fixed in two places, and the second was the real one.  `nf_lower` gained the
rank-0 transpose (`sᵀ = s`), which made the result *simplifiable*; but `sym`
still built the formula, so it still appeared as an option.  The builders now
decide by rank:

- **rank 0** — `sym(s) = s`, `skew(s) = 0`.  Worth stating because they
  degenerate *differently*: the identity and the annihilator, not both the
  identity.  `A = sym + skew` still holds.
- **rank 2** — unchanged, the case they exist for.
- **rank 1 or ≥ 3** — refused.  Transpose swaps *two* slots: a vector has none,
  and for a rank-3 "which pair?" is exactly the missing information.  Building
  ½(T ± Tᵀ) there reads as if it meant something.

Afterwards `sym` correctly vanishes from the report on a scalar, while `skew`
correctly *stays* — it does apply, and gives zero.  The report was not wrong to
list them; the steps were wrong to accept them.

## 4b. Steps report for themselves — the interface change (user's call)

The `wants` preconditions above infer a reason **from outside**, and the user's
objection is right: that is orthogonal to what the fingerprint is for, and it
does not scale.  A fingerprint answers "did the expression change, and how",
after the fact.  It cannot answer *why* a step declined, because the reason is
internal — `contract_delta` knows it found a δ whose partner index sits in a
foreign factor, and no amount of counting the result recovers that.  As steps
grow more capable the gap widens, and each new reason would need its own
fingerprint extension.

So the step says it itself.

**Shape.**  C++ steps take an optional out-parameter, `StepReport* = nullptr`,
rather than changing their return type: existing call sites need no change, a
step with nothing to say needs no change, and the steps that *do* have something
gain it one at a time — which keeps the tree building at every commit.  The
Python catalogue carries the real interface, `Step.run(expr, **ctx) ->
StepResult(expr, fired, reason)`, since the bare `td.*` functions are the
plain-`Expr` convenience and (the user) "we'll probably not be relying on bare
steps anyway".  `StepResult` is uniform from day one: a step that explains
itself supplies the reason, one that does not gets the synthesized `wants`
message, so `why_not` improves step by step with no flag day.

**What it buys, immediately.**  Same step, same "nothing happened", two causes
that point at different next moves:

```
contract_delta on  a·b        → "there is no summation for a δ to contract over"
contract_delta on  a_j e_j·b_i e_i → "no δ in this term carries a summed index"

reduce_frame on    a·b        → "no frame vectors here — expand_in_basis first"
reduce_frame on    ε ε a b c e → "the frame has nothing further to say; what
                                  remains needs a step it cannot justify (an
                                  ε-pair contraction, a metric move, an identity)"
```

**And it resolved a tension that had been distorting the steps.**  Before the
report, `fired` had to be *inferred from the return value*, which forced a step
either to lie about its output (return the un-normalized input so it looked
untouched) or to lie about its effect (return the normalized form and be read as
having worked).  `reduce_frame` was caught in exactly that, and a first attempt
to fix the report by changing what it returns broke two tests.  With `fired`
carried separately, both can be honest: **the return value is the normalized
result; the report says whether any work was done.**

That is the third appearance of one distinction — "did it change?" versus "did
it do work?" — after composing the self-preparing folds in `reassemble` and
after `applicable`'s reordering noise.  Three independent encounters make it a
property of the step contract rather than a local quirk, and the report is where
it now lives.

**Done:** `contract_delta`, `reduce_frame`, `reassemble`.

`reassemble` was the richest, and the exercise confirmed the suspicion that made
it worth doing: **every refusal was already written out as a comment.**  Teaching
it to report was mostly moving prose from a comment to a string — the knowledge
was in the code, just unreachable from outside it.  Eleven refusal points now
speak, among them:

```
an invariant                 → "there is nothing here in component form to fold back"
a×(b×c) after reduce_frame   → "this index is shared by two ε's — that is the
                                ε-pair contraction's business, not this fold's"
a rank-2 leg nested in a dot → "…e_i's position says 'first slot' but a dot
                                contracts the last, and a wrong orientation is
                                undetectable afterwards"
reassemble(…, target="z")    → "the coordinates here are not the one you named"
n ≥ 3 shared indices         → "no double dot expresses this pairing (it would
                                need a transpose interposed) — and guessing
                                would be silent"
```

Two things worth recording from the work:

- **First reason wins, not last.**  An early refusal is more specific than a
  later blob finding nothing, so the note sink keeps the first and ignores the
  rest.
- **A reason can mislead if it does not know why it was reached.**  With a
  `target` set, non-target coordinates fall through to the *foreign factor*
  branch, which reported "a summed index is also carried by a factor this fold
  does not read" — true of the mechanism, useless to the reader.  The branch now
  distinguishes "held back by your target" from "genuinely unreadable".  A
  reporting step has to know not just that it declined but *which decline this
  is*, which is a slightly higher bar than it first appears.

### Does the report make the fingerprint obsolete?

No — and building both made the division sharper than it was.  They answer
different questions, and three uses of the fingerprint have no report-based
substitute:

- **Comparing options.**  `applicable` ranks steps by *what they change*
  (`basis_vectors−2, deltas+1`), which is how a reader tells "this moves me
  toward components" from "this moves me back".  `fired: bool` cannot rank.
- **`explain(before, after)`.**  It compares two arbitrary expressions.  There
  is no step in the picture to ask.
- **Policing the reports.**  The report is what a step *says*; the fingerprint
  is what the expression *shows*.  Keeping both makes a disagreement detectable
  — and it is not hypothetical: `reduce_frame` briefly reported `fired=True` on
  a term it had only reordered.  There is now a test asserting, for every
  reporting step over a battery of expressions, that `fired` matches the
  fingerprint and that a reason appears exactly when it did not fire.  That test
  is only possible because the two measures are independent.

And a fourth use, for the milestone after this one: a *goal* like "no basis
vector remains" is a fingerprint query, and ranking successors by how much they
reduce needs a measure rather than a flag.

**What the report does obsolete is `wants` as a source of reasons.**  Once a
step explains itself, its `wants` entry is no longer consulted — two sources of
truth for one thing, and the step's own is better.  But `wants` earns its keep
differently: as a **cheap pre-filter**.  `applicable` currently runs all 39 steps
in ~12 ms, and `wants` can skip 9 of them without running them at all, on the
shape alone.  That is the cost driver for path search, so `wants` should stay —
reframed from "the reason" to "a necessary condition, checked without working".

## 4a. The original list

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

1. ~~Step-set investigation~~ **done** — 14 of 15 steps in real derivations
   were never used alone; the bridge went 12 → 4 (`dec51b6`, `c8492e3`).
2. ~~Categories/tags~~ **done** — `tender.steps` is the catalogue; advertised
   surface 41 → 30 and 18 → 13, everything still importable (`f27c4f0`).
3. ~~`applicable`~~ **done**, with the fingerprint moved to the IR (`dfbf711`).
4. ~~`why_not`~~ **done**, plus `explain`, in the same commit.

## 6. Out of scope — the next milestone

**Path search and goal specification.**  The prerequisites above are what make
it possible: `applicable` *is* the successor function, and hash-consing gives a
visited set for free.

The measurement, **re-taken over the redesigned set** now that it exists.  Along
challenge 000001's derivation (`a×(b×c)` out to components, through the ε-pair,
and back to bac-cab):

| | branching | depth |
|---|---|---|
| old set, crude LaTeX classifier | 2–12 | 8 |
| new set, all content-changing steps | 4–10 | 5 |
| **new set, primaries only** | **2–5** | **5** |

The redesign roughly halved both, and restricting to primaries halves it again —
because the punctuation steps that used to pad every state are no longer things
a derivation reaches for.  A worst case of 5⁵ before dedup is not a search
problem so much as an enumeration.

What is *not* settled, and is the real work of the next milestone: how a user
**states the goal**.  §2 gave every bridge step a checkable exit condition ("no
basis-vector product remains", "no symbolic index remains"), which is a start —
those are goals a search can evaluate — and §6's boundary note gives another
("reach a state where `prove_equal` finishes").  Whether that vocabulary is
enough for a real derivation is the open question.

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

**Implemented** (2026-08-30), in four commits: `dec51b6` `reduce_frame` /
`to_concrete`, `c8492e3` the unified `reassemble` with `target=`, `f27c4f0` the
catalogue, `dfbf711` `applicable` / `why_not` / `explain`.

Every number in this note is measured, not estimated — from the corpus for the
step-set claims, and from the shipped code for the rest.

Two things the work turned up that were not in the design:

- **Composing self-preparing steps needs a "did it do work?" test**, not just
  "did it change?".  Each fold canonicalizes on entry, so a fold that did
  nothing still returns a reordered expression and would silently undo its
  predecessor's answer (`y·a` came back as `a·y`).  This bit twice while
  unifying `reassemble`, and it is the same distinction `applicable` rests on —
  which suggests it is a property of the step contract, not a local quirk.
- **A tool that reports what applies also notices what applies and should
  not.**  `sym`/`skew` surfaced as options on a scalar, because transpose had no
  rank-0 case.  Fixed.

Left for the next milestone: path search and goal specification (§6).  Also
open, and now better shaped than "add 20 preconditions": teach the remaining
steps to report for themselves (§4b), richest first — `reassemble`,
`contract_metric`/`insert_metric`, `fold_operator`, `apply_operators`.  A
handful of steps (`simplify`, `canonicalize`, `simplify_scalars`) genuinely have
no precondition and should keep the fallback; and `sym`/`skew` want fixing
rather than explaining, since `sym(s) = s` and `skew(s) = 0` for a scalar.
