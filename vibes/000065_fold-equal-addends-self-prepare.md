# 000065 — `fold_equal_addends` self-prepares (dummy-rename-aware folding)

## Context

While systematizing the basic identities into `tender-sandbox/identities.ipynb`,
an `alternative_Ixa()` cell hit a surprise.  Two derivations of the same tensor,

```
x1 = transform(I×a) = a_k e_j ε_{jki} e_i
x2 = transform(a×I) = a_k ε_{kij} e_j e_i
```

are algebraically equal — `td.algebraic_eq(x1, x2)` is `True`.  Yet

```python
dx = td.fold_equal_addends(x1 - x2)   # did NOTHING; expected 0
```

left the difference standing.

## Diagnosis

`x1` and `x2` denote the same tensor written two ways at once: their **summed
dummy indices are named differently**, and ε is permuted.  `algebraic_eq` works
because it **canonicalizes both sides first** (dummy α-normalization + ε/δ
ordering), collapsing them to one normal form.

`fold_equal_addends`, however, was a **purely structural** fold: it matched
addends exactly as written (`structural_eq` on the extracted cores), so two
terms equal only up to dummy renaming never landed in the same group.  Not a
bug in the fold — just too narrow a contract, and one that violated the
self-prepare principle ([[steps-self-prepare]], vibe 000060/000061): a step
should never require the caller to canonicalize first.

`canonicalize` already subsumes the entire fold *plus* the renaming:
`canonicalize(x1 - x2) == 0`, `canonicalize(A + A) == 2A`, etc.

## Resolution

Split the step in two:

- **`fold_equal_addends_structural`** — the old body, unchanged.  Bare
  structural fold; merges addends written identically only.  For callers that
  have already put the addends in a common frame.
- **`fold_equal_addends`** — now self-preparing: `canonicalize` on entry (so
  equal terms share a normal form), then the structural fold, then
  `implicitize` to restore the implicit-sum convention.  Mirrors
  `contract_eps_pair`'s `implicitize(canonicalize(...))` shape.  `canonicalize`
  throws on an ill-formed implicit sum; that is treated as "nothing to prepare"
  and falls back to the input.

`x1 - x2` now folds to `0` directly — no manual canonicalize.

### Surface

- C++: `steps::fold_equal_addends_structural` + self-preparing
  `steps::fold_equal_addends` (`src/derivation.cpp`, `derivation.hpp`).
- Python: `td.fold_equal_addends` (self-preparing) and new
  `td.fold_equal_addends_structural`; bindings `_fold_equal_addends`,
  `_fold_equal_addends_structural` in `python/_core.cpp`.

### Tests

- C++ `FoldEqualAddends.SelfPreparingCancelsAcrossDummyRenaming`:
  `δ^i_i − δ^j_j` (two traces, different dummy names) — structural fold leaves
  it, self-preparing fold → scalar 0.  The existing structural tests were
  repointed at `fold_equal_addends_structural`.
- Python `test_fold_equal_addends_self_prepares_across_dummy_renaming`: the
  literal `I×a` vs `a×I` case from the notebook.

## Note on the notebook's hand-built zero

`alternative_Ixa()` also fabricated `zero = fold_equal_addends(a + 0 - a)` to
compare against — a rank-inconsistent sum (`a` is rank-1, `0` a scalar).  Not
needed: `canonicalize`/`fold_equal_addends` produce the genuine zero of the
correct rank from `x1 - x2` itself.
