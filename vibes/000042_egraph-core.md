# 000042 E-graph core (Stage 1, part 2 — slice 1)

First slice of the e-graph (vibe 000034): the **data-structure core** —
union-find, hash-consing, congruence rebuild, and cost-based extraction — built
and tested independently of the matcher and the saturation loop, as vibe 000034
§"Implementation plan" recommends. `tender/egraph.{hpp,cpp}`, `EGraph`.

## What canonicalize already gave us

Vibe 000034's plan lists "canonical child ordering" (item 1) and
"α-normalisation of bound indices" (item 7) as e-graph prerequisites. Both are
already done by `steps::canonicalize` (vibes 000037/000041): AC-sorted
children, signed-coefficient terms, α-normalised binders, and now materialised
implicit sums. So `EGraph::add` **canonicalizes on insertion** and hash-conses
the result. The payoff: AC- and α-equivalent inputs land in the same e-class
for free —

- `A + B` and `B + A` → one class;
- `Σ_i δ^i_i` and `Σ_j δ^j_j` → one class.

No commutativity/associativity rewrite rules, exactly the "canonical form"
mitigation of vibe 000034 §"Commutativity and associativity" (option 1).

## E-node representation

An `ENode` is an operator tag (`EKind`, kept independent of the `Expr::Node`
variant order) plus child e-class ids. Leaves (`TensorObject`, `ScalarLiteral`)
instead carry their whole `Expr*` and compare/​hash structurally — a leaf's
indices, levels, realms, spaces are part of its identity, not children.
`ExplicitSum`/`NoSum` additionally carry their (already α-canonical) binder id,
so two summations are the same e-node iff their bodies are the same e-class and
the canonical binder ids agree — α-equivalence handled by the prior
canonicalization, per vibe 000034 §"Interaction with index quantification".

E-node equality is `structural_eq` on leaves (the existing T0 leaf test);
hashing is a structural leaf hash that deliberately omits space pointers
(omitting a field only adds collisions, which equality then resolves). The
`Difference` node never appears — canonical forms carry signs as coefficients —
so its decode/encode arms are present only for variant totality and are
coverage-excluded.

## Algorithm

Textbook egg: `parent_` union-find with path halving; a hash-cons `memo_` from
canonical e-node to class id; per-class node lists and parent use-lists.
`merge` unions and queues the class; `rebuild` drains the worklist, re-canonicalises
each affected class's parent e-nodes, re-hash-conses them, and merges any that
became congruent — restoring `a == b ⇒ f(a) == f(b)`. `extract` computes a
least-node-count cost per class to a fixpoint, then rebuilds the cheapest
representative `Expr`.

`EGraph` is pImpl, so the header stays a thin handle and all of `ENode`, the
maps, and the algorithm live in the `.cpp`.

## Tests

`tests/egraph_test.cpp` (14): canonical round-trip; structural / AC / α
deduplication; subterm sharing; congruence propagation through `rebuild`;
cheapest-representative extraction; class-count drop on merge; round-trips
across every reachable node kind, bound nodes, void/concrete/label leaves;
movability; self-merge no-op. `egraph.cpp` at 100% line coverage; suite 371 C++
/ 95 Python green.

## What's next (remaining vibe 000034 items)

3. **E-class pattern matcher** — `match` over e-classes, not raw `Expr*` (the
   current `Identity` matcher works on `Expr*`; lift it to search e-class
   representatives / e-nodes).
4. **`instantiate` into the e-graph** — build RHS e-nodes from a binding.
5. **`saturate`** — the fixed-point loop applying a rule set.
8. **Rule set** — encode the index identities (delta-contraction, eps-delta, …)
   as `Identity` objects; this is where the parametric-slot + computed-RHS rule
   type (vibe 000040) gets designed in.
9. **Python** — expose `td.saturate(expr, rules)`.
10. **Benchmarks** — saturation on the full eps-δ expression; node/iteration
    counts.
