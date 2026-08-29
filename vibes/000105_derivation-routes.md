# 000105 A derivations API: name the destinations, not the moves

Before M5 doubles the amount of physics in the suite, the derivation surface
should stop being a catalogue of moves.  This note measures the complaint, says
what is actually wrong (three things, not one), and proposes a design.

## The complaint, measured

Across the 24 challenges and 12 examples, derivations call **41 distinct**
`td.*` / `tb.*` functions.  The long tail is real — 12 of them are used once —
but the head is more interesting: the corpus contains essentially **two
pipelines, hand-written nine times**.

*To symbolic components* (challenges 4, 5, 10, 16):

```
expand_in_basis → [expand_double_dot | expand_dyad_ops] → simplify_basis_dot
    → canonicalize → simplify_basis_dot → contract_delta → canonicalize → simplify
```

*To concrete numbers* (challenges 1, 14, 15, and two examples):

```
expand_in_basis → simplify_basis_cross ×2 → simplify_basis_dot → canonicalize
    → unroll_sums → eval_eps_concrete → eval_delta_concrete → fold_arithmetic
    → canonicalize
```

Nobody chose those orders from a doc.  They were arrived at, and the code says
so — challenge 000010 carries the comment `# dots exposed by canonicalize` on
its *second* `simplify_basis_dot`, which is a discovery, written down.

## Three problems wearing one coat

**1. There is no vocabulary for a destination.**  Every name in the catalogue is
a *move*: `contract_delta`, `unroll_sums`, `fold_arithmetic`.  But a derivation
is not thinking "I would like to contract a delta"; it is thinking "put this in
components", "get me numbers", "bring it back to direct notation".  The library
has exactly two functions named for a destination — `chart.evaluate` and
`chart.components` — and the second one's own doc-comment states the principle:

> …so the caller need not spell out the project-then-reduce pipeline by hand.

That is the right idea, applied twice and never generalised.

**2. Order and iteration are load-bearing, undocumented, and fail silently.**
Measured:

```python
x = expand_in_basis(a @ b, frame, Covariant)     # a_j e_j · b_i e_i
contract_delta(x)                                 # → unchanged.  No error.
contract_delta(canonicalize(simplify_basis_dot(x, frame)))   # → a_i b_i
```

Getting the order wrong produces a **no-op that looks exactly like "there was
nothing to do"** — which is vibe 000056's deciding usability failure, still
present at the pipeline level even though individual steps now honour their
no-op contract.

Iteration is the same story.  The corpus hand-rolls fixpoints with magic counts:
`for _ in range(2)` in challenge 000017 (commented "one pass per nesting level
of contraction"), `for i in range(4)` in challenge 000024, and — tellingly —
`for (int i = 0; i < 4; ++i)` in tender's *own* `tests/basis_test.cpp` helper.
Three places, three different guesses at how many times.

**3. When you are stuck, nothing says what is in the way.**  A pipeline that
stops short returns an expression, not a reason.  The user is left diffing
LaTeX to work out which move was missing — which is the actual shape of "it has
always been a pain to figure out which steps to use".

## The design

### A. Routes — named journeys, one per destination

```python
td.to_components(expr, frame)   # invariant → index form, nothing reducible left
td.to_concrete(expr, frame)     # → numbers: unrolled, ε/δ evaluated, arithmetic folded
td.to_invariant(expr, frame)    # components → direct notation
```

Each is the pipeline above, but: **order fixed once, centrally**, and **iterated
to a fixpoint** rather than a guessed count.  `nf_view`'s `fixpoint` combinator
already exists (M1) and is the right tool; no magic numbers anywhere.

These are not new capability — every one is a sequence the corpus already
performs.  They are the sequences given names, so they can be got right once and
tested once.  `chart.evaluate` and `chart.components` join them as the ∇ and
last-mile routes, and the set is then complete for the journeys the suite makes.

### B. Routes explain themselves

A route returns an `Expr` (so it composes like a step), and with `trace=True`
returns a `Derivation` instead — the audit trail that already exists, listing
which moves fired.  A route is then a shorthand, never a black box: when you
want to know what it did, ask.

### C. A route that stalls says what is in the way

This is the part that answers the actual complaint.  A route knows its
destination, so it can check it:

- `to_components` — no basis vector remains in a position where it could
  contract;
- `to_concrete` — no symbolic index remains;
- `to_invariant` — no coordinate component remains.

When the fixpoint settles short of that, the route reports **what is left and
where** ("2 terms still carry an uncontracted `e_i`: …"), rather than returning a
half-reduced expression silently.  That converts "nothing happened, why?" into a
pointer — and it is exactly the information the library already has and throws
away.

Whether that is a raised exception, a warning, or a status on the result is an
open question below.

### D. The catalogue is demoted, not deleted

The moves stay — the library, the C++ tests and power users all need them — but
they move behind `tender.steps`, which already exists and holds four names.
This also fixes a regression worth naming: vibe 000092 objected to ~30 exported
steps, M3 got `tender.derivation.__all__` down to 36, and it is **41 today**
(three of them added by this session's own work, with nothing demoted to
compensate).  A route layer gives the catalogue somewhere to be demoted *to*
without losing anything.

## What this does not fix

- It does not make the engine find routes on its own.  `prove_equal` and
  `simplify` already do goal-directed search where rules exist; routes are for
  the *coordinate-bridge* work, which is not rule-shaped.
- It does not remove the need to know that a cross needs `simplify_basis_cross`
  before it can reduce — it moves that knowledge from every caller into one
  place.
- It does not address vibe 000092's namespace complaint (`t` / `td` / `tb` /
  `tc`).  Routes could later be `Expr` methods; that is a separate decision.

## Open questions — for the user

1. **Names.**  `to_components` / `to_concrete` / `to_invariant`?  The `to_`
   prefix marks them as journeys and dodges two collisions I would otherwise
   hit: `chart.components` already means "the list of scalar components of a
   vector", and `chart.expand` already means "expand abstract *fields* onto the
   frame" — neither is what these do.
2. **Where do they live?**  Free functions in `tender.derivation` (consistent
   with the steps they replace), methods on the frame/chart (where the frame
   argument already lives, matching `chart.evaluate`), or `Expr` methods
   (`T.to_components(frame)`, best reading order, worst for discoverability)?
3. **A stalled route: raise, warn, or report?**  Raising is loudest and matches
   how `at` now refuses; a status field composes better with the verbs' existing
   `ProofStatus`/`SimplifyResult` shape.  My inclination is a status on the
   result plus `strict=True` to raise, but this is a taste call.
4. **Is `to_concrete` one route or two?**  Unrolling to numbers is really
   "symbolic → numeric" applied *after* "invariant → symbolic", and the corpus
   always composes them.  One route with a `concrete=True` flag, or two routes
   that compose?

## Status

Design proposal.  Nothing implemented; the four questions above want answers
first, since they decide naming and placement and rework would be wasteful.
