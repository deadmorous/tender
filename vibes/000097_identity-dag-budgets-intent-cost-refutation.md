# 000097 Identity DAG, budgets, intent-driven cost, refutation

Four design threads raised by the user after M2 landed (2026-08-24).  All are
M3-or-later work; this note settles what they mean and what evidence we have.

## 1. Identities and challenges are the same thing, arranged in a DAG

**User's framing.**  An identity is usually *derived*; that derivation is
exactly what a challenge is; and a derivation normally *uses* other,
already-derived identities.  Those uses are dependencies, and the dependency
relation makes a directed acyclic graph (DAG) of identities.

This dissolves the circularity worry recorded in vibe 000096 increment 4.
There, promoting a challenge by citing the library rule for that same identity
looked thin, and each promotion had to argue its own case in a docstring.  The
DAG makes the question structural instead of rhetorical:

- A challenge **is** the derivation of an identity — its proof obligation.
- The identity it proves is a **node**; the identities its derivation cites are
  its **in-edges**.
- Citing a rule is legitimate **iff** that rule is a node with its own
  derivation, and the citation does not close a cycle.
- The DAG's roots are the genuine axioms — the ε-δ contractions, the trace
  axioms, completeness of a basis — which are *definitions*, not theorems, and
  are verified by component reduction rather than derived.

What this buys, concretely:

- **Circularity becomes checkable**, not a matter of judgment: a cycle in the
  citation graph is a bug the harness can detect.  `cross-removal` citing
  itself is exactly a self-loop.
- **The scoreboard gains a real notion of depth**: a challenge proved directly
  from axioms is a different achievement from one standing on ten derived
  identities, and the DAG shows which.
- **Rule-group selection stops being guesswork**: the rules a proof may use are
  the transitive closure of its declared dependencies, so `td.rules("cross")`
  becomes "the dependencies this challenge declares" rather than a hand-picked
  bag.
- **A failing root invalidates its cone**: if an axiom's verification breaks,
  everything downstream is suspect, and the DAG says what.

Sketch of the shape (details when M3/M4 implements it):

```python
CHALLENGE = harness.declare(
    title="a×(b×I) = b⊗a − (a·b)I",
    tier="A",
    proves="cross-removal",              # the identity this challenge derives
    cites=["bac-cab", "completeness"],   # its in-edges
)
```

with the harness checking acyclicity, that every `cites` name exists, and that
the union of `cites` covers the rules the derivation actually fired (the
`fired` report already gives us that — an undeclared citation is detectable).

**Open question for the implementation:** identity *names* become a namespace
that must stay stable (renaming a rule breaks every `cites` edge).  Probably
worth a registry with explicit ids rather than free-text names.

## 2. Proving a false statement

**Measured.**  Every false statement comes back `Exhausted` — the rules run to
a fixed point without joining the two sides — and never a false `Proved`:

| statement | outcome |
|---|---|
| `a×(b×c) = c(a·b) − b(a·c)` (sign flipped) | Exhausted |
| `a×b = b×a` | Exhausted |
| `A·B = B·A` | Exhausted |
| `a = b` | Exhausted |
| `tr(I) = 4` (in 3-D) | Exhausted |
| `a×b = −(b×a)` (true, control) | **Proved**, 0 passes |

That is sound but unsatisfying: `Exhausted` conflates "this is false" with
"my rule set was too weak", and the user cannot tell which.  Saturation is a
semi-decision procedure — it can exhibit a proof, never a refutation — so the
distinction has to come from somewhere else.

**It can.**  Component expansion is a *decision* procedure for the chart-free
algebraic fragment, and it is already implemented (it is what every L1 test
does).  Measured on the same statements: expanding both sides to concrete WCS
components and comparing separates true from false exactly —

| statement | components equal |
|---|---|
| `a×(b×c) = b(a·c) − c(a·b)` | True |
| `a×(b×c) = c(a·b) − b(a·c)` | **False** |
| `a×b = b×a` | **False** |
| `a×b = −(b×a)` | True |

**Proposal (M3).**  A `refute` path, and a four-valued verb result:

- `Proved` — the sides joined in the e-graph.
- `Refuted` — component expansion produced different normal forms, with the
  differing component reported as the counterexample.
- `Exhausted` — rules ran out, components could not decide (a differential or
  chart-dependent statement).
- `Budget` — nothing concluded.

`prove_equal` could opportunistically attempt refutation when saturation
exhausts, which turns today's most confusing answer into a definite one for
the whole tier-A fragment.  Cost is bounded: component expansion in 3-D is the
same work the L1 tests already do per challenge.

## 3. Budgets in user units: time and memory

**Today.**  `SaturateBudget{max_passes = 30, max_nodes = 10'000}` — engine
units.  They are the right *mechanism* (they are what the loop can cheaply
check) but the wrong *interface*: a user has no way to know what 10k nodes
costs, and the number means different things for different rule sets.

**User's proposal**, which I agree with: express the budget as a **time cap**
and a **memory cap**, settable on the Context as a default and overridable per
call.

Design notes for the implementation:

- Keep passes/nodes as the internal mechanism; time and memory are the public
  dial.  Poll a steady clock and an estimated node-memory figure between
  passes (the same place the node check already sits), so the cost of checking
  stays negligible.
- **Determinism matters more than precision here.**  A time cap makes a run's
  outcome depend on the machine, which would make CI flaky and a challenge's
  result unreproducible.  So: time/memory caps are *safety valves* for
  interactive use, while the challenge suite pins deterministic
  passes/nodes budgets.  Both must be expressible; the report says which
  fired.
- Memory is estimable rather than measurable: node count × a per-node
  estimate, not RSS.  Say so in the API, or the number will be believed too
  precisely.
- Context-level default + per-call override is the right layering; the same
  layering should carry the *rule groups*, since "which rules are in scope" is
  the other thing a user sets once and occasionally overrides.

## 4. Cost must follow user intent

**The observation** (vibe 000096 increment 4): `engine_simplify` reaches the
answer from the problem alone only when the target is *cheaper* under the
ε-weighted cost.  It rewrites `a×I → I×a` and `I·a → a`, but keeps `a×(b×c)`
rather than expanding it — correctly, since the compact form has fewer nodes.

**User's point, and it is the right one:** cheapness is a fine default but it
is not what the user necessarily wants.  "Get rid of the two crosses" is a
legitimate goal, and there must be a way to say it.  A hardcoded cross penalty
would be wrong — it only makes sense *as an expression of intent*.

**Proposal (M3):** make the cost function a parameter of the verb, with named
intents covering the common cases and a raw weight map underneath:

```python
td.simplify(expr, rules, prefer="fewest_crosses")     # eliminate ×
td.simplify(expr, rules, prefer="fewest_eps")         # today's default
td.simplify(expr, rules, prefer="smallest")           # plain node count
td.simplify(expr, rules, cost={"Cross": 1000})        # raw, for anything else
```

Notes:

- The ε-weight already in `node_cost` is exactly this mechanism used once,
  hardcoded — the vibe-000046 decision to prefer δ-form over ε-form *is* an
  intent ("contract the ε's away"), so generalizing it is a refactor of
  something already proven, not a new idea.
- The e-graph makes this cheap: the *same* saturated graph can be extracted
  under different costs, so exploring "what does this look like with no
  crosses?" costs one extraction, not one saturation.
- This also answers a question M2 left open — whether `simplify` is useful for
  expansion-direction identities.  With `prefer="fewest_crosses"`, expanding
  bac-cab becomes the *preferred* extraction, so the same rule serves both
  directions depending on what the user asked for.
- Intents should be named for the *goal*, not the mechanism, so the vocabulary
  stays meaningful to a mechanician: "fewest crosses", "no ε", "componentwise",
  "operator form".

## Implementation status

- **(4) intent-driven cost — DONE** (a519a33).  `nf::CostModel` with a node
  baseline and per-kind extras (eps, cross, delta, identity, unary, div);
  named intents `fewest_eps` (the historical default), `smallest`,
  `fewest_crosses`, plus raw weights.  Python: `engine_simplify(...,
  prefer=..., cost={...})` and the documented `td.PREFER` map.  The cross
  weight counts *operators* (chain length − 1), not chain nodes.  Cost
  governs extraction only, never the search, so re-reading a saturated graph
  under another intent is one extraction.  Demonstrated: one expression, one
  rule set — `fewest_eps` keeps `a×(b×c)`, `fewest_crosses` returns
  `b(a·c) − c(a·b)`.  This also retires the M2 note that `simplify` was
  useless for expansion-direction identities.

- **(2) refutation — DONE**.  `ProofStatus` gains `Refuted`;
  `engine::decide_by_components` expands both sides on the World-Cartesian
  frame, evaluates ε and δ at concrete indices, folds, and compares — the
  same reduction the L1 challenges perform by hand, deliberately built from
  *independent* machinery so an e-graph bug cannot silently confirm itself.
  `prove_equal` runs it only when the rules exhaust (never on a budget trip,
  which concludes nothing).  Three outcomes, all tested:
  `a×b = b×a` → **refuted**; Lagrange with an empty rule set → `exhausted`
  with `components_agree`, i.e. *the claim looks true and your rules are
  incomplete*; anything holding a ∇ → undecided, since the procedure decides
  only the chart-free algebraic fragment and silence beats a wrong verdict.
  A residue check (leftover ε/δ/binder/∇) gates the negative, so "did not
  reduce" is never mistaken for "differs".

  Implementation note worth keeping: the first cut wrote the residue check as
  a substring search over rendered LaTeX.  That is the kind of shortcut that
  works until a tensor is named `delta_max`; it was replaced with a tree walk
  before landing.

- **(1) identity DAG — DONE**.  The reconciliation question — `challenges/`
  is development scaffolding while the DAG must be a library feature — is
  settled by making the dependency *one-directional*:

  * **The library owns the graph.**  `tender.identities` holds nodes
    (name, summary, `kind`, `cites`, `proof`, tags), the queries users need
    (`ancestors`, `descendants`, `depth`, `citable_for`, `rules_for`), and
    `register()` so a user's identity is a first-class node with dependencies
    of its own.  A node records its proof obligation as **inert data** — a
    challenge id, meaningless to the library, meaningful to the harness — so
    `tender` never imports `challenges`.
  * **The suite satisfies it.**  `harness.declare(proves=...)` names the
    identity (or identities) a challenge derives.
  * **One file reconciles them**: `challenges/test_dag.py`, development
    scaffolding, checks both directions — acyclicity, citations naming real
    nodes, no identity citing itself or a descendant, axioms owing no proof,
    every `proves` naming a real node, no two challenges claiming the same
    identity, node `proof` and challenge `proves` agreeing, and the open
    obligations being exactly the known set.

  The invariant that makes this work: **every derived identity owes a
  challenge, but not every challenge proves an identity** — 000007
  (cylindrical equilibrium), 000020 (operator tables) and 000006 (chart
  endpoints) verify endpoints against textbook results and have no node to
  point at.  So the arrow runs from identity to challenge, never back.

  Circularity is now structural rather than rhetorical.  `citable_for(name)`
  returns exactly the ancestors — never the identity itself, never anything
  resting on it — so passing it to `prove_equal` makes a derivation honest by
  construction instead of by review.  This retires the per-promotion
  circularity arguments M2 had to write in prose.

  Two findings from building it, both from the meta-test's first run:
  challenge 000003 derives *both* ε-δ contractions, so `proves` had to accept
  several — a single-valued field was a modelling error the check caught
  immediately; and `ddot-identity` has no derivation at all, which the DAG now
  states in the scoreboard ("**not yet derived**") instead of leaving implicit.

  The scoreboard gained an Identity DAG table: kind, depth above the axioms,
  what each rests on, and which challenge discharges it.

## Status

All four are M3 material and are recorded in vibe 000093's plan as such.
Order of value, highest first: (4) intent-driven cost — it changes what the
engine is *for*; (2) refutation — it removes the most confusing answer the
verbs can give; (1) the identity DAG — it makes the suite's structure honest
and checkable; (3) budgets in user units — real but narrower, and partly a
naming exercise over machinery that already works.
