# 000046 Stage 2 kickoff: identity library, and a cost-function finding

Stage 2 of the roadmap (vibe 000038) — exercise the engine on coordinate-level
identities — begins here, chosen over building the basis layer first because the
invariant examples (bac-cab, a×I×b) *reduce* to this coordinate machinery, so
hardening it first de-risks them.

## The named identity library (vibe 000034 #8)

`tender/identities.{hpp,cpp}` — factory functions returning `Identity`, the rule
set the e-graph saturates with:

- `delta_contraction(ctx, space)` — `Σ_p δ^p_a δ^p_b = δ_{ab}`
- `delta_trace(ctx, space)` — `Σ_p δ^p_p = dim(space)` (per-space constant)
- `eps_delta_1(ctx)` — `Σ_i ε^{ijk} ε_{ilm} = δ^j_l δ^k_m − δ^j_m δ^k_l` (3D)
- `eps_delta_2(ctx)` — `Σ_i Σ_j ε^{ijk} ε_{ijl} = 2 δ^k_l` (3D)

Each is validated by saturating a fresh-indexed target and comparing the
extracted result against the existing hard-coded steps (`contract_eps_pair`,
direct δ results) as **oracles** — so the generic engine reproduces what the
bespoke steps do, exactly the vibe 000033 §2 plan (the hard-coded steps become
fast paths for the named identities).

## The finding: node-count extraction is the wrong objective

`eps_delta_1` exposed a real engine gap. The matcher, `ematch`, and `saturate`
all worked — a probe confirmed `passes=2`, the graph grew `4→12` e-nodes, and
the ε-form's class **was merged** with the δ-expansion's. But `td.saturate`
returned the *unchanged ε-form*, because:

> ε-δ-1's right-hand side (`δδ − δδ`, ~7 nodes) is **larger** than its left-hand
> side (the ε-sum, ~4 nodes). Node-count extraction correctly picks the cheaper
> ε-form.

(ε-δ-2 and delta-contraction happened to work only because *their* RHS is
smaller than their LHS.) So extraction needs a domain-aware objective, not raw
size.

**Fix:** a lexicographic cost — **minimize Levi-Civita symbols first, then node
count.** ε is precisely the object these identities exist to contract away; its
expansion into δ's is the goal even when it is larger. A per-e-node weight
(`kLeviCivitaWeight`, dominating any realistic node count) on ε leaves gives the
`(eps-count, node-count)` ordering. Weighting *summations* instead was rejected:
it would wrongly prefer unrolled sums over compact `Σ` forms. With the new cost,
ε-δ-1 extracts to the δ-expansion and every prior case is unchanged.

## Distribution: surfaced, not yet forced

Isolated identities don't need distribution — the RHS is simply added as a new
class. It becomes necessary the moment an identity's sum-shaped RHS sits inside a
larger product and must be multiplied out and re-contracted (e.g. bac-cab's
`a_j (δδ − δδ) b^l c^m`). `expand_products` exists as a linear step but **not as
an e-graph rule**; encoding it needs the remainder/sequence-variable matching of
vibe 000040. That is the next engine item, and it is better to hit it on simple
indexed inputs than mid-basis-build — which is why Stage 2 comes first.

## Tests

`tests/identities_test.cpp` (4): each library identity through `saturate`,
checked against an oracle. `identities.cpp` and `egraph.cpp` at 100% line
coverage; 390 C++ / 98 Python pass.

## Next

A few more identities as examples demand; expose the library to Python
(`tender.identities`); then the distribution-as-a-rule work (vibe 000040), after
which Stage 3 (vector bases, vibe 000036) can tackle bac-cab and a×I×b with a
trusted engine and a ready rule set — likely introducing the reserved `Theorem`
type (a derivation-with-history yielding an identity) for the basis-level proofs.
