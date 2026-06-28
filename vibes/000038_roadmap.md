# 000038 Roadmap

A living roadmap — a spiral, not a waterfall. Each stage is exercised with
examples before the next is trusted; order may adjust as we learn.

## Stage 0 — Algebraic normal form (in progress)

`canonicalize(Expr)` per vibe 000037. The foundation everything else leans on:
it makes `structural_eq` an AC/α-aware equality so like-term collection, pattern
matching, and e-node deduplication work. Gives feedback immediately — usable as
`td.canonicalize` on hand-built expressions and inside existing linear
derivations, before any engine exists.

## Stage 1 — Simplification / matching engine

Two sub-steps (refinement of vibes 000033 + 000034):

1. **ANF-backed matcher + `apply_theorem`** (vibe 000033). A recursive matcher
   over canonical forms; named tensors in a theorem LHS act as pattern
   variables. Testable standalone, gives quick wins (apply named identities), and
   is reused by the e-graph.
2. **E-graph saturation** (vibe 000034). Union-find + e-node table keyed on
   canonical forms (the ANF supplies e-node dedup) + `saturate` + cost-based
   `extract`. Distribution and contraction enter here as *rewrite rules*, not as
   normalization — this is where the local ANF (which deliberately omits
   distributivity) is completed into full algebraic reasoning.

## Stage 2 — Exercise at the coordinate/symbol level

Feed hand-crafted examples that today's code can express (no bases yet) and
iterate on ANF + engine design:

- ε-δ identities (one- and two-index), generalized Kronecker.
- Kronecker contractions, traces, determinant/permutation-symbol identities.
- simple symmetric/antisymmetric index manipulations.

This is the first real feedback gate: revise ANF and the engine before adding the
invariant layer.

## Stage 3 — Vector bases and related operations

The invariant layer of vibe 000036, ported/rebuilt from attempt 1: a `Basis`
abstraction that emits indices (inheriting space + realm), cobasis, the metric
`g_{ij}` and `√g`, the polyad assembly `a = a^i e_i`, and the
coordinate↔invariant bridge (expand-in-basis / reassemble). Dyads and invariant
products become first-class. Distinguish the Levi-Civita **symbol** from the
**tensor** (vibe 000036 §4).

## Stage 4 — Well-known coordinate systems

Cartesian, cylindrical, spherical (and general curvilinear): metric, Christoffel
symbols, `√g`, built on Stage 3's basis machinery.

## Stage 5 — Differential operators

grad / div / rot, the covariant derivative, expressed in direct notation and
evaluated through the coordinate systems of Stage 4. The domain payoff for
computational mechanics.

## Why this order

Each stage is a strict dependency of the next: the engine needs the ANF; trusting
the engine needs Stage-2 exercise; the invariant layer is far easier to get right
on top of a solid coordinate engine; coordinate systems need bases; differential
operators need coordinate systems (metric, Christoffel). We deliberately get the
symbolic/coordinate core solid before introducing the invariant tensor layer.
