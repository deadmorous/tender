# 000109 Seven defects behind one invalid derivation

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

## The third: `contract_delta` now contracts through a ∂ direction

`δ_jk (∂_j ∂_k u_i) e_i` — a Laplacian in components — came back exactly as
written.  `contract_delta` looked for its partner among tensor *slots* only,
and both of the δ's indices sat on derivative marks.

The inconsistency is the point: `collect_term_uses` has counted a free mark's
direction as an occurrence of its index since vibe 000078 — that is *why* `e_i`
and `∂_i` sum together.  The summation machinery and the contraction step
disagreed about what carries an index, and the contraction step was the one in
the wrong.

`find_partner` now sees a free mark, and `substitute_index` rewrites the mark's
`link` alongside the slots — restoring the canonical mark order afterwards,
since a changed link can reorder them and `∂_i∂_j` must keep hash-consing with
`∂_j∂_i`.  Only a *countable* target: fixing a free direction to a concrete one
is a different move, and `substitute` (which takes a `ConcreteIndex`)
deliberately still does not do it.

With this the reported pipeline reaches the right component form:

```
λ (∂_i ∂_j u_j) e_i + μ (∂_j ∂_j u_i) e_i + μ (∂_i ∂_j u_j) e_i + f
```

— which is `(λ+μ)∇(∇·u) + μΔu + f`, written out.

## The fourth: `reassemble_nabla` refuses a component form

That component form then went in to `reassemble_nabla` and came out as
`∇ u_i` — a bare unapplied ∇ times a component with a dangling index, one of
the two derivatives gone.

The classifier reads a term as a ∇-expansion: frame vectors are gradient legs,
`e_ℓ·e_m` pairs are Laplacians, one factor is the operand.  In
`(∂_j ∂_j u_i) e_i` the `e_i` is *not* a leg — it belongs to the field's own
index, `u_i` — and the `∂_j ∂_j` pair has no frame vector at all.  Read as a
leg, `e_i` became a ∇; read as nothing, the ∂'s vanished.

The invariant that separates the two forms is one line: **every free ∂_i in a
∇-expansion is paired with a frame vector e_i** — that pairing *is* the
expansion of `∇ = e_i ∂_i`.  A term with an orphaned direction is a component
form, and the fold declines it, saying which order does work:

> a ∂ direction here has no frame vector to pair with, so this is a component
> form rather than a ∇ expansion — the field's own indices carry the frame
> vectors.  Reassemble ∇ before expanding the operand in a basis, or keep the
> operand abstract

`reassemble_nabla` gained a `StepReport` to carry that, so it explains itself
like the rest of the catalogue rather than falling back to the synthesized
"its pattern is not present".

**The route that does work** is the one the maintained examples take — keep the
operand abstract, fold ∇ back first:

```python
td.derive(nabla @ T + f,
          [b.expand_nabla, td.contract_identity, td.canonicalize,
           b.reassemble_nabla])
#  f + μ Δu + ∇(λ ∇·u + μ ∇·u)
```

which is vibe 000080's "keep the operand abstract, expand the basis last",
now enforced by a refusal instead of left as lore.

## The fifth: coefficients did not pool across a ∇ fence

`2 λ ½ ∇∇·u` would not fold to `λ ∇∇·u`, while the identical term over a plain
vector folded to `λ Y`.  Canon's contract is *one rational coefficient per
term*, and it was keeping that promise only in the absence of an operator.

The cause is the fence of vibe 000096 increment 3: a ⊗-chain carrying an
operator is kept **whole** so it reaches `encapsulate` as one factor, and the
literals inside it were therefore invisible to the pooling.  In
`2 ⊗ (½ λ ∇ ∇·u)` the `2` is pooled and the `½` is not.

∇ acts rightward, so a literal to its **left** is outside its reach and belongs
in the coefficient like any other; those are now peeled off the chain before it
is fenced.  Leading only — a literal to the *right* of the operator is inside
its scope, and hoisting it would be a linearity argument this pass has no
business making.

The shape arises from like-term collection (`λ½X + λ½X → 2·(λ½X)`), which is
why it took a derivation with a `sym` in it to produce.  Note that canon was
*idempotent* on the bad form: a wrong fixed point, not an unfinished one, which
is why no amount of re-running found it.

## The sixth: `factor_common` factored across a ∇

`λ ∇(∇·u) + μ ∇(∇·u)` came back as `(∇λ + ∇μ) ∇·u`: `∇·u` lifted out from
*inside* the gradient, leaving the gradient of a constant.  Not equal to what
went in, by the library's own `algebraic_eq`.

An unapplied operator reaches everything to its right in its product — the same
reading `apply_operators` takes — so a factor standing there is the operator's
operand and cannot be lifted out of the term.  A candidate common factor is now
rejected when a **bare** ∇ or ∂ factor precedes it.

"Bare" is what keeps vibe 000080's own example working: the `∇·u` in
`λ(∇·u) + μ(∇·u)` is a completed contraction, not an operator standing over its
neighbours, so it still factors to `(λ+μ)(∇·u)`.  The two cases differ by
exactly one top-level ∇ factor, which is the thing to look at whenever a step
"knows" it may move a factor.

## Status

**All seven fixed and verified.**  958 C++ tests, 575 Python, 69 challenges, 12
examples.

- the scope defect: four Python regressions
  (`TestScalarDivisionKeepsTheIndexLink`) and two C++ tests
  (`Canonicalize.AScalarDenominatorKeepsTheSummationScope`,
  `AnIndexedDenominatorIsStillABoundary`);
- the fold defect: two Python tests (`test_reassemble_refuses_a_component_
  carrying_derivatives`, `test_reassemble_still_folds_the_terms_that_have_no_
  marks`) and one C++
  (`BasisFilter.ReassembleRefusesAComponentCarryingDerivatives`).

- the δ/∂ contraction and the ∇-expansion refusal: five Python tests
  (`TestContractingThroughADerivativeMark`,
  `TestReassembleNablaRefusesAComponentForm`).

- the coefficient pooling and the factoring: three Python tests
  (`TestCoefficientsPoolAcrossANablaFence`,
  `TestFactorCommonRespectsTheOperatorsReach`) and two C++
  (`Canonicalize.LiteralsLeftOfAnOperatorJoinTheCoefficient`,
  `FactorCommon.DoesNotFactorAcrossAnOperatorsReach`).

- the Laplacian pair: two Python tests and two C++
  (`Chart.ReassembleNablaReadsAContractedDirectionPairAsALaplacian`,
  `Chart.ReassembleNablaStillRefusesAComponentForm`).

Every one was checked against the unfixed build and fails there.

## What the six had in common

Four of the six are the same mistake in different clothes: **a step decided
what one term is, and got the boundary wrong.**  `is_term` put a scalar
division outside the term; the flattening put a literal inside a fence;
`factor_common` reached past an operator; `reassemble_nabla` read a component
index as a frame leg.  Each is a judgement about *scope* — what a term
contains, and what an operator reaches — made in one place and disagreeing with
the same judgement made elsewhere.

The other two are the fold engines dropping what they could not carry
(`reassemble`) or not seeing what was there (`contract_delta` and the ∂-marks).

Worth stating because it predicts where the next one is: any step that
partitions a term into "what I act on" and "the rest" is making this call, and
the calls are not written down in one place.

## The seventh: `∂_i ∂_i u` is a Laplacian

The refusal above left one shape unfolded, and it is the one every elasticity
derivation ends on.  Keeping the operand abstract and reducing the frame —
`expand_nabla`, `contract_identity`, `reduce_frame` — lands `∇·(∇⊗u)` on

```
∂_i ∂_i u
```

A field with two free ∂-marks on the same direction, the δ already contracted
away.  That is `Δu`, and it is the *same pair* the classifier has always read
as a Laplacian when it is still spelled `e_ℓ·e_m` — only one step further on.
So a repeated free direction, both marks on one field and nowhere else in the
term, now counts as a `∇·∇`, folded by the machinery already there.

**The mirror condition is what keeps this safe.**  Admitting an orphan pair
would have readmitted the component form of §4 — `(∂_j ∂_j u_i) e_i` has a
paired direction `j` too — so the guard gained its other half: a **frame
vector** whose index is not a ∂ direction is not a gradient leg either.  In the
component form `e_i` belongs to the field's own slot, and that is now what the
refusal says.

The two halves together state the invariant plainly: in a ∇-expansion, ∂'s and
frame vectors pair off, and the only thing that may go unpaired is a ∂ with
another ∂ on the same field.  Everything else is a component form, and the fold
declines it by name.

With this the whole derivation closes, from the invariant to the equation:

```python
b = ts.using(basis=wcs, chart=chart)
td.derive(nabla @ T + f,
          [b.expand_nabla, td.contract_identity, b.reduce_frame,
           b.reassemble_nabla])
#  f + μ Δu + λ ∇(∇·u) + μ ∇(∇·u)
```

`reduce_frame` rather than `expand_in_basis` is the whole difference: it takes
the frame dots the operand *needs* without componentising the field, so the
operand is still `u` when the ∇'s are folded back.

All four fixed; see below for the third and fourth.
