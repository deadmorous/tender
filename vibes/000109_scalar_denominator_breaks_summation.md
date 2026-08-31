# 000109 A scalar denominator broke the summation scope

Reported from an interactive session: starting from the invariant `∇·T + f`
with the isotropic Hooke stress `T = λ tr(ε) I + 2μ ε`, `ε = sym(∇u)`,
`expand_nabla` returned an expression whose terms carried **single, dangling
indices** — some with an `i` and a `j`, some with only an `i`, and `f` with
none.  That is not a valid tensor expression at all, and nothing said so.

Two smaller symptoms came with it, and both turned out to be the same defect
seen from a different angle: the expansion carried **explicit Σ binders** where
implicit summation was expected, and a later `contract_delta` **left some δ's
uncontracted**.

## The defect

`is_term` (src/summation.cpp) decides what one Einstein-summation scope is: a
pure multilinear tree of tensors, products and contractions.  It listed
`ScalarDiv` among the scope *boundaries*, beside `Sum` and `Difference`.

That is wrong, and the reason it is wrong is one line of algebra:

```
X · (Y/c)  =  (X·Y)/c
```

An index repeated across the division is the same contraction it would be
without it.  Treating the division as a boundary meant `materialize` descended
into the two sides *independently*, so a repeated index straddling it was never
recognised as contracted — and canonicalize's α-renaming, which works per
scope, then renamed the two occurrences apart.

The minimal case:

```
e_i · ((∂_i u)/2)      →      Σ_j ½ e_j · (∂_i u)
```

`e_i` and `∂_i` are *the same index* — `expand_nabla` builds them that way, a
frame vector and a free-index ∂ sharing one id, which is the whole mechanism by
which `∇ = e_i ∂_i` sums.  After canonicalization they are `j` and `i`, each
used once.  The binder over `j` is the second symptom; the surviving δ's the
third, since a δ whose partner was renamed away has nothing left to contract
against.

## Why it surfaced now, and not in the examples

The trigger is a scalar denominator *inside* a ∇ operand.  `sym(A) = (A+Aᵀ)/2`
is exactly that, and it is how anyone writes a strain — so the answer to "how
did this survive twenty-four challenges" is that the maintained derivations
spell the isotropic stress as `λ(∇·u)I + μ(∇u + (∇u)ᵀ)`, with the 2 already
distributed and no division anywhere.  `examples/navier_lame.py` builds `T`
that way; challenge 8 does too.  Nobody had written `sym(∇u)` into a ∇ and
handed it to the library.

Worth recording as a testing lesson rather than a coding one: the corpus agreed
with itself on a spelling, and the agreement hid a defect in the spelling it did
not use.  A challenge suite is a sample, and a uniform sample is a narrow one.

## The fix

`ScalarDiv` is a term when its numerator is one **and its denominator carries no
index**:

```cpp
[](ScalarDiv const& d) {
    return is_term(d.left) && !carries_countable_index(d.right);
},
```

with `collect_term_uses` descending into the numerator only.

The denominator guard is deliberate.  An index under a division is *not* a
linear contraction — `a_i / b_i` is not a sum — so a denominator that carries
one keeps the old boundary behaviour, where the summation is deferred rather
than counted wrongly.  Deferring is the conservative failure; counting is not.

## What this does not fix

Two further defects were found in the same session while following the reported
pipeline to its end, both downstream of this one, both filed but **not fixed
here**:

1. **`reassemble` drops ∂-marks** — *now fixed by refusing; see below.*

2. **`contract_delta` cannot contract through a ∂-mark.**  In
   `δ_jk (∂_j ∂_k u_i) e_i` both of the δ's indices sit on derivative marks
   rather than tensor slots, so the step does not see them as carriers and the
   δ survives.  This is a gap rather than a corruption — the expression stays
   valid — but it blocks `δ_jk ∂_j ∂_k → ∂_j ∂_j`, which is how a Laplacian
   should appear.

## The second defect: `reassemble` now refuses a marked component

`δ_jk (∂_j ∂_k u_i) e_i` reassembled to `u`.  The completeness fold
`u_i e_i → u` is right; the two derivatives riding on `u_i` were discarded in
silence.

The cause is one line, in two places.  Both fold paths rebuild the invariant
with

```cpp
make_tensor_object(ctx, coord->name, {}, rank, coord->dim);
```

— name, rank and dimension, and nothing else.  The `deriv_marks` on the
component have nowhere to go, so they simply do not come along.

**Refusal, not carriage.**  Of the three options — carry the marks through,
refuse, or drop — dropping is the worst and was what happened.  Carrying them
through is the real feature and a larger question: `∂_j ∂_k u_i e_i` wants to
become `∂_j ∂_k u`, and once the δ contracts, `Δu` — which is
`reassemble_nabla`'s job, and it already exists.  So `reassemble` refuses and
says which step does understand ∂-marks.  That matches what the same fold
engine learned once before, when `reassemble_nabla` returned a bare `u` for
`(u·∇)u` and was made to refuse (vibe 000108).

**Per term, not per expression.**  The guard sits where each fold path accepts
a coordinate carrier, so a term whose components are unmarked still folds
beside one that is refused:

```
δ_jk (∂_j ∂_k u_i) e_i + a_i e_i   →   δ_jk (∂_j ∂_k u_i) e_i + a
```

**It cost nothing.**  953 C++ tests, 561 Python, 69 challenges and 12 examples
all pass unchanged with the refusal in place — no maintained derivation was
relying on the silent drop.  That is worth stating: a refusal added to a fold
engine is exactly the kind of change that can quietly break a working route,
and here it did not.

## Status

**Both fixed and verified.**  954 C++ tests, 563 Python, 69 challenges, 12
examples.

- the scope defect: four Python regressions
  (`TestScalarDivisionKeepsTheIndexLink`) and two C++ tests
  (`Canonicalize.AScalarDenominatorKeepsTheSummationScope`,
  `AnIndexedDenominatorIsStillABoundary`);
- the fold defect: two Python tests (`test_reassemble_refuses_a_component_
  carrying_derivatives`, `test_reassemble_still_folds_the_terms_that_have_no_
  marks`) and one C++
  (`BasisFilter.ReassembleRefusesAComponentCarryingDerivatives`).

Every one was checked against the unfixed build and fails there.

**Still open:** `contract_delta` cannot contract a δ whose indices sit on
derivative marks, so `δ_jk ∂_j ∂_k → ∂_j ∂_j` does not happen and the δ
survives.  A gap rather than a corruption — the expression stays valid — but it
is what stands between the reported pipeline and a Laplacian.
