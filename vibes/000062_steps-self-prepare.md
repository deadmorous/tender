# 000062 — Steps self-prepare (no manual canonicalize / expand_products)

Standing principle (user): **a derivation step must never require the caller to
run `canonicalize` or `expand_products` first.**  Sprinkling normalisation
between steps leaks internal-representation concerns into the user's derivation
script and is counter-intuitive.  Each step does whatever prep it needs as its
first action.

## The two prep needs

1. **Materialise** implicit Einstein sums → explicit `ExplicitSum` binders.
   Steps that pattern-match on binders need this: `reassemble`,
   `reassemble_completeness`, `contract_eps_pair`.
2. **Distribute** products over sums (`expand_products`) and float binders
   per-term.  `contract_delta` needs this so the δ-determinant an ε-pair
   contraction leaves behind — `Σ_m … (δ_ab δ_cd − δ_ad δ_cb)` — splits into
   separate terms, each of which contracts (a single distributed sum under a
   binder is otherwise left alone, by the guard from vibe 000060).

`canonicalize` (= `raise ∘ canonicalize_nf`) does (1) and the binder-floating
half of (2); `expand_products` does the product-distribution half.

## The pattern

```cpp
Expr const* prepped = e;
try { prepped = canonicalize(ctx, /*expand_products(ctx,*/ e /*)*/); }
catch (std::invalid_argument const&) { prepped = e; } // ill-formed sum → no prep
auto const* out = <core rewrite>(ctx, prepped);
return out == prepped ? e : implicitize(ctx, out); // no-op → original pointer
```

Two invariants this preserves:
- **`canonicalize` throws** on an ill-formed implicit sum (e.g. an Oblique
  same-level pair, `δ^i_j δ^i_j`).  For these steps that just means "nothing to
  prepare", so catch and fall back to the raw input — matching the long-standing
  `contract_delta` behaviour.  (`ContractEpsPair.WalksEveryNodeKindWithoutMatch`
  exercises this.)
- **No-op returns the original pointer.**  Many tests assert
  `EXPECT_EQ(step(e), e)`; since prep allocates a fresh canonical tree, compare
  the core's output against `prepped` and return the original `e` when nothing
  fired.  (`contract_delta` already returned `e` on no-fire, so this fell out.)

## Done in this pass

`contract_eps_pair`, `contract_delta` (now `canonicalize ∘ expand_products`),
`reassemble`, `reassemble_completeness`.  Both worked routes for `(a×I)×b` now
run with **no** `canonicalize`/`expand_products` between steps and converge to
`−(a·b) I + b ⊗ a`:

```
A: expand_in_basis → apply_identity(bac_cab) → simplify_basis_dot
     → contract_delta → reassemble
B: expand_in_basis → simplify_basis_cross → contract_eps_pair
     → contract_delta → reassemble
```

`ContractDelta.DistributedSumUnchanged` became
`ContractDelta.DistributesSumThenContracts` (the old no-op is now the correct
`a_j − c_j`).

## Tradeoff / still open

- Each step re-canonicalises its input, so a long chain re-normalises
  repeatedly.  Accepted for intuitiveness over micro-performance; revisit if a
  tight loop needs it (a cheaper `materialize`-only prep would suffice for the
  group-(1) steps, but `materialize` is currently file-local to derivation.cpp).
- Not yet audited for the principle: other binder/​sum-sensitive steps
  (`fold_sums`, `unroll_sums`, `contract_identity`, `distribute_contraction`,
  `expand_double_dot`, …).  Extend the same pattern as they come up.  See
  [[steps-self-prepare]].
