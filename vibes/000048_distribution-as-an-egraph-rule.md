# 000048 Distribution as a built-in e-graph rule

`expand_products` (derivation.cpp) fully distributes products over sums as a
linear pass. The e-graph needed distribution *available within saturation* — so
the engine can multiply out a sum-shaped factor on demand and re-contract the
result (the bac-cab shape `a_j (δδ − δδ) b^l c^m`), cost-driven rather than as a
mandatory pre-pass.

## The fork, and the decision

Writing distribution as a data `Identity` — `?a·(?b+?c) = ?a·?b + ?a·?c` —
needs **subtree pattern variables** (`?a,?b,?c` each binding a whole e-class).
The matcher has only *index* variables (`CountableIndex` in slots/binders, via
`MatchBinding`); there is no metavariable matching an arbitrary subtree. That is
the real content of "needs remainder/sequence-variable matching" (vibe 000040) —
though with **binary** products no true *sequence* variable is required: "the
rest of the factors" is just the other binary child, and repeated saturation
passes handle multi-factor products.

Two ways to land it were weighed:

1. **Built-in structural rule** — a dedicated saturation step, no matcher change.
2. **General subtree-variable matcher** — a pattern-metavariable representation,
   `MatchBinding` extended with subtree→e-class bindings, `match_node` /
   `instantiate` taught to bind and reconstruct them; distribution then a plain
   data `Identity`, reusable for future rules (theorems, basis identities).

**Chosen: (1), built-in first.** It is the small, alive, incremental step that
unblocks the engine now; it does not preclude (2), which we add when a *second*
rule needs subtree variables. Distribution is structural — of a kind with AC and
congruence — so making it an engine primitive rather than data is defensible, not
a hack.

## Implementation (egraph.cpp)

`collect_distribution_rewrites()` runs in the saturate **read phase** alongside
`ematch`, on the stable start-of-pass graph; its rewrites flow through the
existing shared write phase (add + merge) and the single rebuild. For every
e-node of a distributive kind — `TensorProduct, Dot, DDot, DDotAlt, Cross` —
whose child class holds a `Sum` node `S1+S2`, it emits `op(L,S1) + op(L,S2)`
(and the mirror for a sum on the left), built from the **cheapest** reconstructed
forms of the subtrees (`compute_best`, factored out of `extract`).

Two invariants shaped it:

- **Every e-node is canonical.** New e-nodes can't be synthesized raw — they
  would bypass `canonicalize` and pollute dedup/matching. So each distributed
  form is built as an Expr, `canonicalize`d, and `add_canon`ed, exactly like a
  rule RHS.
- **Canonical forms carry no `Difference`** (signs are coefficients), so only a
  `Sum` child is ever distributed; `X·(A−B)` is `X·(A + (−1)·B)` and the −1
  rides along (tested).

Termination: at the fixed point a product already equal to its distributed sum
re-derives the same canonical form, `add_canon` returns the same class, nothing
new merges. Where there is no sum factor, `collect_*` returns empty — a true
no-op, so existing saturations are unchanged (verified: full suite green, pass
counts unmoved).

## Tests (egraph_test.cpp, `Distribute`)

Right/left sum, over a difference, multi-factor (`X·Y·(A+B)`), Dot, and the
remaining ops (DDot/DDotAlt/Cross); `UnlocksAMatch` — with `A·B → C`, the target
`A·(B+D)` distributes so `A·B` becomes visible and rewrites, yielding `C + A·D`
(distribution composing with a data rule); `NoSumFactorIsNoOp`. Equivalences are
checked by `find()`-equality of the root and the expected distributed form
(extraction cost would keep the *smaller* undistributed product, so asserting on
`extract` would be wrong — the same lesson as ε-δ-1 in vibe 000046). 406 C++ / 98
Python pass; egraph.cpp 100%, overall 95%.

## Next

The general subtree-variable matcher (option 2) remains the eventual capability,
to be added when a rule beyond distribution needs it — likely the Stage-3 basis
work / `Theorem`s (vibe 000036). With distribution now in the engine, the
coordinate-level toolkit (Stage 2) is complete enough to attempt the invariant
examples once bases exist.
