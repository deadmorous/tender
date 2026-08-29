# 000106 Steps: discovery first, organisation second, routes last

Vibe 000105 proposed named routes as the answer to "which steps do I need?".
The user's objection, and it is correct: **a route prescribes a path, but the
problem is discovery.**  A canned route helps only when it happens to match; it
cannot help with the derivation that needs its own sequence, and it makes the
library's inability to *find* a sequence into the user's problem.

This note keeps routes — in a smaller role, and at the end — and puts the user's
counter-proposal first: **feedback**.  It was prototyped before being written up.

## The probe works, and the reason it can is recent

> a function taking an expression and printing names of all steps that are not
> no-ops for that expression

This is implementable *because* of a contract this project has been enforcing
all along and tightened repeatedly this month: **a step that does nothing
returns its input unchanged**.  That makes "did it bite?" a one-line test —
`structural_eq(step(e), e)` — with no per-step cooperation needed.  A library
whose steps returned canonicalized-but-unchanged output (as `apply_identity`
once did, vibe 000056) could not have this feature at all.

Measured surface: of the callables on `td` / `tb` / `tender.steps`, **21 take
just `(expr)`**.  The rest take exactly one more thing, drawn from a short list
of *kinds* — a frame/basis, a coordinate, a rule set, a level, an index list, an
identity, an operator.  That is the "dictionary of what we could throw in there":
it is small, and it is closed.

## The refinement the prototype forced: most firing steps are noise

Run naively on `a_j e_j · b_i e_i`, **13 steps fire**.  Seven of them are
`canonicalize`-class: they change the tree without changing its content — factors
commuted into canonical order, nothing else.  A raw "what fires" list is
therefore mostly noise, and would train the user to ignore it.

Classifying the change fixes it.  A cheap structural fingerprint (node count,
Σ count, δ/ε count, basis-vector count) splits the list cleanly:

```
expand_in_basis(a·b)             a_j e_j · b_i e_i
  td.canonicalize                nodes+2, sums+2
  td.unroll_sums                 nodes+64, basisvecs−2
  tb.reassemble                  nodes−4, basisvecs−2      → a·b
  tb.simplify_basis_dot          nodes−2, deltas+1, basisvecs−2   → b_j a_i δ_ij
  …7 others                      reordered only
```

Four real options, each annotated with *what it does to the expression*, instead
of 41 names to remember.  For the same expression with a cross, the useful list
is different and equally short (`simplify_basis_cross` appears, `…_dot` drops to
"reordered only") — so the probe is genuinely reading the expression, not
reciting a menu.

**"Not a no-op" is the wrong filter; "changed the content" is the right one.**
That distinction is the whole difference between a useful tool and a noisy one.

## Organisation: the categories are already in the codebase

The user also wants the catalogue less cluttered — vibe 000092's namespace
complaint, still untouched.  The probe suggests the organising principle,
because sorting its output needs the same categories a user needs:

| category | what it does to an expression | e.g. |
|---|---|---|
| **normalise** | reshapes, preserves content | `canonicalize`, `implicitize`, `simplify_scalars`, `collect_terms`, `fold_arithmetic` |
| **bridge** | invariant ⇄ components | `expand_in_basis`, `simplify_basis_dot`/`_cross`, `reassemble*`, `unroll_sums`, `eval_*_concrete` |
| **index algebra** | contracts / moves indices | `contract_delta`, `contract_eps_pair`, `contract_metric`, `insert_metric`, `contract_identity` |
| **operators** | ∇ and ∂ | `apply_operators`, `partial`, `expand_nabla`, `reassemble_nabla`, `fold_operator` |
| **engine** | goal-directed | `simplify`, `prove_equal`, `saturate`, `apply_identity` |

Note this is *not* the current module split (`td` / `tb` / `tender.steps`),
which is by implementation history.  A user asking "how do I get from components
back to direct notation?" wants the **bridge** row; they do not care that half of
it lives in `tb` because it takes a basis.

**The mechanism already exists.**  `tender.identities` gives every identity
`tags` and exposes `rules("cross", "leibniz")` and `group_names()`.  Steps can
carry the same tags, which gives `td.steps_in("bridge")`, sorted probe output,
and a documentation ordering — one idea, three payoffs, and consistent with a
pattern the project already chose.

## Other feedback forms — the brainstorm

1. **`applicable(expr, **context)`** — the above.  Ranked: content-changing
   first, `reordered only` collapsed to a count.
2. **`why_not(expr, step)`** — the inverse, and the higher-value one when the
   user *has* an expectation: "`contract_delta` did nothing because no δ carries
   a summed index".  Needs per-step precondition reporting, so it is real work;
   worth doing incrementally for the six most-used steps rather than all 41.
3. **`explain(before, after)`** — what a step changed, rendered.  The probe needs
   this internally anyway, and `tender.render.labeled` already exists to show it
   positionally.
4. **Categories in the probe's output** — see above.
5. **`reach(expr, goal, budget)`** — see below.

## Routes, in their proper role: found, then saved

The measured branching factor along challenge 000001's real pipeline (`a×(b×c)`
to concrete and back) is **2–12 advancing options per state at depth 8**, with a
deliberately crude classifier that over-counts.  That is small.  And expression
hash-consing gives a visited-set for free.

So the sequence a derivation needs is **searchable**, not merely guessable —
`applicable` is exactly the successor function a search needs, and a goal is a
predicate the library can already express ("no basis vector remains", "no free
index remains", "structurally equal to this target").

That reframes routes entirely, and settles the user's objection:

- not a hand-written catalogue that may or may not fit;
- but sequences **found** by search, then **saved and named** — the shipped ones
  and the user's own, exactly as proposed ("apply routes from a well-known list,
  as well as maintain user's own routes");
- with the search available when no saved route fits, which is precisely the
  case vibe 000105 had no answer for.

Vibe 000105's pipelines are then not a design, but *test data*: the nine
hand-written sequences in the corpus are what a search must be able to
rediscover.

## Order of work

1. `applicable` with content-change classification — small, self-contained,
   immediately useful, and it is the successor function everything else needs.
2. Step categories/tags, reusing the identities mechanism; probe output sorted
   by them; documentation follows.
3. `why_not` for the most-used steps.
4. `reach` — search, with the corpus's nine pipelines as the acceptance test.
5. Saved routes, once search can find them.

## Status

Design, revised after the user's objection to vibe 000105.  Item 1 is
prototyped (`applicable`, in scratch) and the numbers above are measured from
it; nothing is committed to the library yet.
