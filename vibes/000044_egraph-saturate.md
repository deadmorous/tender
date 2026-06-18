# 000044 Saturation (Stage 1, part 2 — slice 3)

Third slice of the e-graph (vibe 000034, items #4 + #5): **equality
saturation**. `EGraph::saturate(rules, max_iterations) → passes` applies a set
of `Identity` rules everywhere, to a fixed point, merging each rewrite into the
graph. Afterwards `extract(find(root))` is the simplified form.

## Instantiate-into-egraph is just add(instantiate(...))

Vibe 000034 lists "instantiate the RHS into the e-graph" as a separate item.
With `ematch` returning `(class, MatchBinding)` and the existing
`instantiate(ctx, rhs, binding) → Expr*` (vibe 000039), inserting a rewrite's
RHS is `add(instantiate(...))` — `add` canonicalizes and hash-conses, which is
exactly what an inserted RHS needs. No e-node-level instantiation primitive is
required; the Expr-level one plus `add` is correct and DRY.

## The loop

Each pass is split into two phases (the standard egg read/write split):

1. **Read** — for every rule, `ematch` its (canonicalized) LHS and, for each
   match, `instantiate` its RHS. This only reads the graph (and path-compresses),
   so all matches are gathered against a *stable* graph.
2. **Write** — for each collected rewrite, `add` the RHS and, if its class
   differs from the matched class, `merge` them. Then one `rebuild` restores
   congruence.

A pass that merges nothing new is the fixed point. An `max_iterations` cap
(default 30) bounds rule sets that never settle (vibe 000034 §growth); tender's
size-reducing identities converge in a couple of passes — a single contraction
takes two (one to rewrite, one to confirm no change).

Because saturate reuses `ematch`, which searches *all* e-nodes of a class, a
rewrite fires even when the cheapest form of a class is something else — so a
contraction buried in `δ_{rs} + Σ_q δ^q_m δ^q_n` is rewritten and the enclosing
sum's extraction reflects it via congruence, with no manual step ordering. This
is the payoff over the linear `Derivation` pipeline (vibe 000034 §"replaces").

## Tests

`tests/egraph_test.cpp` Saturate.* (6): contracts a delta; contracts the
two-index eps-delta to `2 δ^k_l`; rewrites a contraction nested inside a sum
(congruence); reaches a fixed point and is idempotent (a second saturation runs
one no-op pass); leaves a non-matching graph unchanged; respects the iteration
cap. `egraph.cpp` 100% line coverage; 386 C++ / 95 Python pass.

## Next (vibe 000034)

#8 a curated **rule set** (delta-contraction, eps-delta, eventually distribution
and the parametric-slot + computed-RHS rules of vibe 000040), #9 Python
`td.saturate(expr, rules)`, and #10 benchmarks (node/pass counts on the full
eps-delta expansion). With saturation working, Stage 2 (vibe 000038) — exercising
the engine on coordinate-level identities — comes into reach.
