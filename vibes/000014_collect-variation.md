# Phase 12.5 — Collect-on-variation step

## Problem

After applying IBP and `localize_step(V)` to the PVW expression, the resulting
state mixes pointwise (already-localized) volume terms with surface integrals
that still sit under `∫_∂V`:

```
∫_∂V (σ·n)·δu dS  −  ∇·σ·δu  −  f·δu  −  ∫_∂V t·δu dS  +  ρü·δu
```

Applying `localize_step(∂V)` to this state strips the two surface integrals but
leaves the three volume terms in place, so the final surface state wrongly
contains all five terms.

The correct derivation groups the expression **by domain before localizing**:

```
∫_∂V (σ·n − t)·δu dS  +  ∫_V (−∇·σ − f + ρü)·δu dV
```

Each grouped integral can then be localized independently, yielding exactly the
terms relevant to that domain.

## Design

### `collect_step(v)`

A new `DerivationStep` factory in `integral.hpp` / `integral.cpp`:

```cpp
auto collect_step(Expr* v) -> DerivationStep;
```

The step name is `"collect(" + v->python() + ")"`.

**What it does** — given the root expression, it:

1. Flattens the outermost `Sum` into a list of terms (non-`Sum` roots are
   treated as a one-element list).
2. For each term, tries to **extract a coefficient** with respect to `v`:
   - `Contract(A, v)` or `Contract(v, A)` → coefficient `A`, domain *pointwise*
   - `Scale(k, Contract(A, v))` → coefficient `Scale(k, A)`, domain *pointwise*
   - `Integral(D, inner)` — recurse on `inner` with the same extraction logic;
     if successful, the coefficient belongs to domain `D`
   - `Scale(k, Integral(D, inner))` — same, but multiply extracted coeff by `k`
   - Anything else → not collected; kept as a residual term
3. Groups collected coefficients by domain.
4. For each domain group, forms `Integral(D, Contract(Sum(coeffs), v))` (or
   `Contract(Sum(coeffs), v)` for the pointwise group).
5. Returns `Sum(grouped_terms + residual_terms)`.

### Rank constraint

`v` must have rank ≥ 1 (a scalar variation has no meaningful contraction).
The step throws `std::invalid_argument` if `v->rank() == 0`.

### Matching rule — `Contract(A, v)` vs `Contract(v, A)`

Both orderings are accepted.  The extracted coefficient preserves the
non-`v` side; the reconstructed term always uses `Contract(coeff, v)`.

### Interaction with `Scale` wrappers

`Scale(k, Integral(D, Contract(A, v)))` and
`Integral(D, Scale(k, Contract(A, v)))` are equivalent physically.  The
extractor handles both, folding `k` into the coefficient so the reconstruction
is always `Integral(D, Contract(coeff, v))`.

## Updated PVW derivation flow

```
pvw  ──IBP──▶  ∫_∂V (σ·n)·δu dS − ∫_V (∇·σ)·δu dV − ∫_V f·δu dV
                 − ∫_∂V t·δu dS + ∫_V ρü·δu dV

     ──collect(δu)──▶  ∫_∂V (σ·n − t)·δu dS + ∫_V (−∇·σ − f + ρü)·δu dV

     ──localize(V)──▶  ∫_∂V (σ·n − t)·δu dS + (−∇·σ − f + ρü)·δu

     ──localize(∂V)──▶ (σ·n − t)·δu + (−∇·σ − f + ρü)·δu
```

The last line still contains both terms (the pointwise volume equation rides
along), but each localization step is now applied to a correctly grouped state.
To isolate just the BC, the caller uses the state **after** `localize(∂V)` on
the collect result that still has both integrals.

In practice `pvw_continuum.py` will do:

```python
ibp_result  = Derivation([apply_integration_by_parts_step(V)]).apply(State(pvw))
collected   = Derivation([collect_step(delta_u)]).apply(ibp_result[-1])
vol_history = Derivation([localize_step(V)]).apply(collected[-1])
srf_history = Derivation([localize_step(dV)]).apply(collected[-1])
```

## Files

| File | Change |
|---|---|
| `src/include/tender/integral.hpp` | Declare `collect_step` |
| `src/integral.cpp` | Implement `collect_step` |
| `tests/integral_test.cpp` | Add `CollectStep.*` tests |
| `python/_tender.cpp` | Bind `collect_step` |
| `python/tender/__init__.py` | Export `collect_step` |
| `python/test_tender.py` | Add Python-level tests |
| `examples/pvw_continuum.py` | Fix derivation to use collect |

## Questions resolved

**Q: Should collect descend into nested non-`Integral` nodes (e.g. `Contract`
within `Contract`)?**  
A: No.  Descending into nested contractions would require symbolic matching of
sub-expressions that is out of scope here.  Collect only extracts at the
topmost contraction with `v` within each integral.

**Q: What if a term has `v` appearing more than once (e.g. `v·v`)?**  
A: Leave it as a residual — don't attempt to extract.

**Q: What if two terms belong to the same domain but different integrals
(same domain object)?**  
A: They are grouped into one integral over that domain, which is mathematically
correct (linearity of integration).
