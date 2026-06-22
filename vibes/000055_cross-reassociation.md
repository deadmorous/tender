# 000055 Cross re-association around a rank-≥2 fence

A canonicalization normal form: `(x × M) × z → x × (M × z)` when the middle
operand `M` is rank ≥ 2.  Removes a binary-tree association that was blocking
the pattern matcher, without ever assuming the cross product is associative
where it is not.

## The problem

A user writes `a × I × b`.  The surface (and Python's `%`) parse it
**left-associated** as `(a × I) × b` = `Cross(Cross(a, I), b)`.  To make
progress one wants to apply the commute theorem `I × x = x × I` to the `I × b`
subterm — but `I × b` is **not a node** in that tree, so the matcher (which is
strictly binary-positional on `Cross`, identity.cpp) cannot see it.  The
redundant-looking parentheses are, internally, a hard commitment.

## Why it is sound (and where it is not)

`a × M × b` is **unambiguous when `M` is rank ≥ 2**.  `M = Σ p ⊗ q`; the `⊗`
fences the two crosses onto disjoint legs — the left cross lands on `p`, the
right on `q`:

```
(a×M)×b = Σ (a×p) ⊗ (q×b) = a×(M×b)
```

So the bracketing is immaterial.  This is **not** general associativity of the
cross: for a rank-1 middle, `(a × b) × c` is the **vector triple product**,
genuinely non-associative (that is bac-cab) — both crosses fight over the same
(unfenced) vector.  The discriminator is exactly the **middle operand's rank**.

## What canon does

In `canon`'s `Cross` arm (after the rank-1 anticommutation rule),
`reassociate_cross_fence(l, r)` fires when `l = Cross(x, M)` with `x` rank 1,
`M` rank ≥ 2, and `r = z` rank 1, rewriting to `x × (M × z)`.  We normalize to
the **right-associated** form because it exposes the `M × z` subterm (e.g.
`I × b`) for the matcher.  Idempotent: the right-associated form has no
`Cross(Cross(...), z)` shape, so it is a fixed point; the rule never runs
backward.

Outer operands are gated to rank 1 (the `vector × tensor × vector` shape) — the
clearly-correct, clearly-useful case.  Higher-rank outer operands (`M × N × b`)
are out of scope.

## Payoff

`apply_identity` canonicalizes its input first, so a left-associated `(a×I)×b`
is re-associated to `a×(I×b)` **before** matching, and the commute identity
`I×x = x×I` fires on the now-exposed `I×b`, giving `a×(b×I)` — regardless of how
the user bracketed.  Test: `BasisFeasibility.CrossReassociationExposesIdentity
ForMatch`.  Canon-level tests: `Canonicalize.CrossReassociatesAround{Identity
Fence,GenericRank2Fence}` and the non-associativity guard
`CrossDoesNotReassociateVectorTripleProduct`.

## Scope / future

- Only the 3-operand `vector × rank≥2 × vector` chain; deeper or wider chains
  (`M × N × b`, nested fences) are not normalized.
- This is the **associative-but-not-commutative** sibling of the matcher's
  existing flatten+AC handling of `Sum` and component `TensorProduct`.  The
  broader "binary tree blocks matching" theme (associative sub-chain matching
  for `⊗`/`+`) remains a separate, larger piece (kept out of the binary core).
