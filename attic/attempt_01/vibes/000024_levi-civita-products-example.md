# Phase 13.92 — Levi-Civita products example and ddot/Trace expansion

_Date: 2026-06-11_

## Context

With `eps_delta` proved as a `Theorem` (Phase 13.91), the next step was to
demonstrate how to use it — and how to work around its limits — through a
concrete example file covering three Levi-Civita product identities:

1. `ε_{ijk}ε_{isp} = δ_{js}δ_{kp} − δ_{jp}δ_{ks}` (abstract, one shared index)
2. `Σ_{ij} ε_{ijk}ε_{ijp} = 2δ_{kp}` (two shared indices)
3. `ε_{ijk}ε_{ijk} = 6` (all three shared — a scalar)

## New infrastructure

### `merge_index_step(old_id, new_id)`

Wraps `substitute_index` as a `DerivationStep` so that renaming one abstract
index to another (e.g. "set s = j") can appear as a named step in a `Derivation`.
Declared in `basis.hpp`, implemented in `basis.cpp`, bound in `_tender.cpp`,
exported from `python/tender/__init__.py`.

### `DoubleContract` bilinearity in `simplify_basis_dot_step`

The previous implementation could only reduce
`ddot(TP(TP(A_prefix, A_mid), A_last), TP(B_first, B_second))`
and silently passed through `ddot(Sum, ...)` or `ddot(Scale*T, ...)`.
Three new cases were added:

- **Sum distribution**: `ddot(a+b, r)` → `ddot(a,r) + ddot(b,r)` (and symmetrically for rhs Sum). Needed because `expand_levi_civita_first_step` produces a `Sum` of scaled basis-vector products.
- **Scale unwrap**: `ddot(α*A, B)` → `α * ddot(A, B)` (and symmetrically).
- **Case 2 — left-nested rhs**: `ddot(TP(TP(A_prefix, A_mid), A_last), TP(TP(B_prefix, B_mid), B_last))` → `TP(TP(A_prefix, B_prefix), Product(Contract(A_mid, B_mid), Contract(A_last, B_last)))`. This handles `ddot(eps_expanded, eps_expanded)` where both sides have the structure `TP(TP(TP(lcs, e^i), e^j), e^k)`.

### `Trace` reduction in `simplify_basis_dot_step`

New cases in `simplify_basis_dot_impl` for `Trace`:

- Distribute over `Sum`: `tr(a+b+...) → tr(a)+tr(b)+...`
- Distribute over `Scale`: `tr(α*A) → α*tr(A)`
- Reduce `tr(TP(rank1, rank1))` → `Contract(rank1, rank1)` (i.e. `tr(a⊗b) = a·b`)
- Pull rank-0 scalar from rhs: `tr(TP(A, s)) → s * tr(A)`
- Pull rank-0 scalar from lhs: `tr(TP(s, A)) → s * tr(A)`

### `replace_first_lct_impl` — Trace traversal

The DFS that replaces the first `LeviCivitaTensor` in a tree was not entering
`Trace` nodes. Added the missing case so that `trace(ddot(eps, eps))` can have
its `eps` occurrences expanded by `expand_levi_civita_first_step`.

## Abstract index limitation — δ_{jj} ≠ 3

The identities involving two or three shared Levi-Civita indices require
evaluating δ_{jj} = dim = 3 (the trace of the identity in 3D space).

tender's abstract index system folds `make_kronecker_delta(j, j)` → `RationalConst(1)` at construction time.  This is correct for a *fixed* index value δ_{j=k, j=k} = 1, but gives the wrong answer when j is a *summation dummy* (where δ_{jj} should equal dim).

Consequence: `merge_index_step(s_id, j_id)` applied to
`δ_{js}δ_{kp} − δ_{jp}δ_{ks}` yields `1·δ_{kp} − δ_{jp}δ_{kj}`, and
subsequent KD contraction gives `δ_{kp} − δ_{kp} = 0` instead of `2δ_{kp}`.

The correct answer requires knowing dim = 3.

## Approach taken in the example

**`examples/tensor_expansions.py`** uses two strategies:

- **Abstract** for derivation_1 (`ε_{ijk}ε_{isp}`, one shared index):
  `contract_eps_pair_step()` produces the Kronecker-delta form directly and
  correctly because no trace of δ is needed.

- **Concrete WCS** for derivations 2 and 3: an `eps_wcs_expansion(cs)` helper
  builds the explicit sum of 6 signed rank-3 basis-vector products:
  `Σ_{ijk} ε_{ijk} e^i⊗e^j⊗e^k`.

  - derivation_2 uses direct Python arithmetic over component indices to produce
    the 3×3 result matrix confirming `Σ_{ij} ε_{ijk}ε_{ijp} = 2δ_{kp}`.
  - derivation_3 computes `trace(ddot(eps_wcs, eps_wcs))` through
    `simplify_basis_dot_step` to get the scalar `6`.

## Test coverage

15 new C++ tests in `basis_test.cpp` cover all new code paths:
`MergeIndex` (3 tests), `DoubleContractDistributesOverSum`,
`DoubleContractRhsSumDistributes`, `DoubleContractScaleOnLhsUnwraps`,
`DoubleContractScaleOnRhsUnwraps`, `DoubleContractLeftNestedRhs`,
`TraceOfRank2TensorProduct`, `TraceDistributesOverSum`,
`TraceOfScaleTimesTP`, `TraceOfTpWithScalarOnRhs`,
`TraceOfTpWithScalarOnLhs`, `SbvBasisCobasisReversed`,
`ReplaceFirstLct.ReplacesLctInTrace`.
Line coverage: **90.2%** (above the 90% CI gate).
