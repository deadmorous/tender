# 000039 The identity matcher (Stage 1, part 1)

Implements the **ANF-backed matcher + application** of vibe 000038 Stage 1,
refining vibe 000033. This is the first piece of the simplification engine: a
recursive matcher over canonical forms plus a rewrite that applies a named
identity to the first matching subtree.

## Identity, not theorem

Vibe 000033 called the rewrite primitive a "theorem". That conflates two ideas.
In this system:

- A **theorem** is a *derivation that proves a result* and carries its history.
- An **identity** is the bare equality `LHS = RHS` that such a derivation yields.

The matcher applies *identities*. So the type is `Identity{name, lhs, rhs}`
(`tender/identity.hpp`). A future `Theorem` will **aggregate** a `Derivation`
and expose the `Identity` it proves — never inherit from `Identity` (CLAUDE.md
principle 8: aggregation over inheritance). The name is reserved; the type is
deferred until there is a consumer.

## What the pattern variables are

Vibe 000033 §4.1 described pattern variables as slot-less named tensors matching
whole subtrees. But its own worked example — `Σ_p δ^p_A δ^p_B = δ_{AB}` — needs
`A, B` to be free *indices* and `p` a bound index. Index identities want **index
matching**, not subtree matching. So:

- Every **free** `CountableIndex` of the LHS is a pattern variable: it binds to
  whatever index sits in the corresponding target slot, consistently across the
  whole match (`MatchBinding`: pattern id → target `IndexAssoc`).
- Indices **bound** by an `ExplicitSum`/`NoSum` are local (α) variables, bound to
  the target's binder when the two binders are matched node-for-node.
- Slot-less subtree variables are **not** supported (decided with the user:
  "slot-less pattern variables will not work").

Because binders are matched as pattern variables, the absolute α-normalized id a
binder carries after `canonicalize` is irrelevant — the pattern's binder binds to
the target's binder, and the bodies stay consistent. This same mechanism also
handles `delta-trace` (`δ^p_p = n`): reusing the bound id in both slots forces them
to agree via `try_bind` consistency, so — contrary to what this section originally
claimed — it is **not** an open limitation. **See vibe 000040** for the correction
and the real residual limit (a dimension-polymorphic / computed RHS, shared with the
general `delta-substitution`).

## Matching modulo AC

`apply_identity` canonicalizes both the target and `id.lhs` (ANF: sums flattened
and sorted, component products sorted, binders α-normalized). But canonical sort
order depends on real index ids, which differ from the pattern's, so pure
structural-over-canonical would fail to line up pattern variables. The matcher
therefore does **bounded backtracking** at the two commutative node kinds:

- `Sum` — flatten addends, find an order-independent assignment;
- component `TensorProduct` (gated on `is_component_valued`, vibe 000036's
  coordinate/invariant line) — same, modulo factor order.

Everything else is matched structurally (the ANF already normalizes rank-1
Dot/Cross order and sign, so those need no AC handling). The child lists are tiny
after canonicalization, so the worst-case factorial cost is negligible.

## Flow

`apply_identity(ctx, e, id)`:
1. `target = canonicalize(e)`, `lhs = canonicalize(id.lhs)`.
2. Walk `target` bottom-up (deepest first) via the shared `rewrite_tree`, with a
   first-match guard; at the first subtree where `match(lhs, ·)` succeeds,
   replace it with `instantiate(id.rhs, binding)`.
3. Re-canonicalize. The result is always canonical; with no match it equals
   `canonicalize(e)`.

`instantiate` substitutes each pattern index in `rhs` by its bound target index,
preserving `TensorTraits` on rebuilt objects.

## Supporting refactors

- `rewrite_tree` (bottom-up rebuild) lifted from `derivation.cpp` into a shared
  header `tender/rewrite.hpp` (DRY; used by the canonicalizer, the steps, and the
  matcher).
- `structural_eq` and `is_component_valued` promoted out of `derivation.cpp`'s
  anonymous namespace and declared in `derivation.hpp`. New `algebraic_eq(a,b) =
  structural_eq(canon a, canon b)` — the headline equality of the ANF (vibe
  000037), now public.

## What it closes

`delta-contraction` and the two-index `eps-delta-2` (`Σ_i Σ_j ε^{ijk} ε_{ijl} =
2 δ^k_l`) now go through the **generic** engine, reproducing what the hard-coded
`contract_delta` / `contract_eps_pair` steps do. Per vibe 000033 §2 those steps
become fast paths for identities the library will name.

Tests: `tests/identity_test.cpp` (8) and `python/tests/test_derivation.py`
(+5). Python surface: `td.Identity`, `td.apply_identity`, `td.structural_eq`,
`td.algebraic_eq`.

## Next

E-graph saturation (vibe 000034): union-find + e-node table keyed on canonical
forms, `saturate`, cost-based `extract`. Distribution and contraction enter
there as rewrite rules, completing the local ANF into full algebraic reasoning.
A named identity *library* (vibe 000033 §3) is built once the e-graph consumes
it; for now identities are constructed inline.
