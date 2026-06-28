# 000066 — The zero literal: one untyped, absorbing, rank-0 scalar

## Question

While cleaning up `identities.ipynb` (the `alternative_Ixa()` cell had hand-built
`zero = fold_equal_addends(a + 0 - a)`, a rank-inconsistent sum), the question
arose: **can we create an expression with a single zero term, and what is the
right way — for rank 0 and for higher rank?**

## Findings (current behavior)

There is exactly **one** zero in the IR: a single `ScalarLiteral{0}` node.
`infer_rank` reports it as rank 0 (`src/derivation.cpp:930`). There is **no
rank-tagged zero tensor** anywhere in the type system.

- **Right way to write it:** `t.scalar(0, ctx=ctx)` (renders `0`), or just the
  integer `0` in arithmetic — auto-promoted via `py_to_scalar`.

- **Zero is absorbing and untyped.** Any cancellation, of any rank, collapses to
  this same rank-0 scalar; the rank is dropped:
  ```
  canonicalize(a - a)  # a rank 1  -> 0   (rank 0)
  canonicalize(B - B)  # B rank 2  -> 0   (rank 0)
  canonicalize(a * 0)  #           -> 0   (rank 0)
  ```
  All zeros compare equal: `algebraic_eq(a-a, B-B)`,
  `algebraic_eq(B-B, scalar(0))` are both True.

- **Rank 0:** `t.scalar(0, ctx=ctx)` is the canonical and only zero literal.

- **Rank > 0:** no first-class typed zero exists. Options today:
  1. **Derive it** — `canonicalize(X - X)` / `X*0` for any rank-n `X` yields the
     canonical zero (rank-0 scalar). This is what `fold_equal_addends` produces
     internally and is what proofs/cancellation should use.
  2. **Retain rank** — not expressible today; would need a new node/factory
     (a declared zero tensor of given rank). Design addition, not present.

## Caveat: sums are not rank-checked

`a + 0 - a` is silently accepted because `infer_rank` on `Sum`/`Difference`
trusts whichever side has a known rank (`src/derivation.cpp:940-949`); nothing
rejects a rank mismatch. So the rank-0 `0` acts as a universal additive
identity (`a + 0 → a`) — convenient, but a genuine mistake like `a + B` would
also pass `infer_rank` rather than erroring.

## Guidance

- To assert equality, use `algebraic_eq(x1, x2)` directly.
- To *show* a collapse to zero, `canonicalize(x1 - x2)` (or the now
  self-preparing `fold_equal_addends`, [[fold-equal-addends-self-prepare]],
  vibe 000065) — never fabricate `a + 0 - a`.

## Possible future work (not done)

- Bind `infer_rank` to Python so ranks can be inspected/asserted.
- Add a typed `zero(rank, ctx)` factory + rank-checking on `Sum`/`Difference`,
  if a rank-retaining zero is ever needed.
