# 000100 Transformations blocked by context — a problem statement

A *problem catalogue*, in the spirit of vibe 000056, not a design.  The same
failure keeps recurring in a new costume: a transformation knows how to act on
a clean shape and refuses when that shape appears **embedded in context** —
extra scalar factors, a surrounding contraction, a sum, a binder, an operator
fence.  Each instance has been patched individually.  The patches work; the
pattern keeps coming back.

The hope this note is written toward: a design in which such transformations
are *naturally* applicable, rather than blocked by small obstacles each time.

## The pattern, across the project's history

| where | the clean shape | what blocked it |
|---|---|---|
| vibe 000056 | bac-cab on `a×(b×c)` | a rank-2 fence in `a×(b×I)`; needed `distribute_contraction` first — the "undiscoverable step sequence" |
| vibe 000063/64 | `a×B×c` cross removal | interleaved sum/sign peeling; `reassemble` needed a prep pass |
| vibe 000091 | `T··ε` dyad expansion | scaled and signed sums around the dyads |
| M1 (vibe 000095) | every additive step | *six* shapes (`Sum`/`Difference`/`Negate`/`ScalarDiv`/scaled-sum/binder) each step re-peeled by hand |
| M2 (vibe 000096) | `X··I = tr X` | canon sorts symmetric chains by name, so the rule matched only some targets |
| M4 (vibe 000099) | ε-reassembly | an expanded cross cannot fold back |
| **here** | `reassemble` | never descends into a contraction operand — **fixed**, vibe 000103 |

Seven instances, one shape.  Each was individually reasonable to fix, and the
fixes were real improvements — `distribute_bilinear` consolidated the additive
peeling, AC matching fixed the chain sorting.  But none of them removed the
*category*.

## The worked example: `a·b` in an oblique basis (challenge 000018)

Measured, not supposed:

| expression | `reassemble` folds it? |
|---|---|
| `a_i e^i` | **yes** → `a` |
| `(a_i e^i) · b` | no — *even with the other side already invariant* |
| `(a_i e^i) · (b_j e^j)` | no |
| the same, in an **orthonormal** basis | no |
| `a_i b_j g^ij` (the oblique dot) | no |

The first line shows the capability exists.  The second shows it evaporating
the moment the same subexpression acquires a neighbour.

Why the orthonormal round-trip nevertheless works (challenge 000017): it does
**not** use this capability.  `simplify_basis_dot` + `contract_delta` remove
the basis vectors *altogether*, leaving `a_i b_i` — a different shape, folded
by a different path.  In an oblique basis `g^ij` is not δ, nothing is
eliminated, and the fold never happens.

The user's reframing, which is the right one: `g^ij` **is** `e^i·e^j` by
definition, so the metric should need no special handling at all —

```
g^ij a_i b_j = (e^i·e^j) a_i b_j = (a_i e^i)·(b_j e^j) = a·b
```

Each step there is either a definition or an association.  Nothing deep is
being asked for.  That the library cannot follow it is the symptom.

## Diagnosis: shape-matching on a shape-normalizing IR

`Nf` deliberately destroys grouping.  A term is `coeff · [scalars] · [tensors]`
with the scalars *sorted*, so the adjacency between a coordinate `a_i` and its
basis vector `e^i` — the very thing `reassemble` pattern-matches on — is
normalized away.  The information survives (the summed index still ties them),
but the *syntax* does not.

So the recurring failure has a single cause: **operations implemented as
syntactic shape matchers are fighting an IR whose job is to normalize syntax
away.**  Every new context is a new shape, so the shape list grows without
end, and each growth arrives as a bug report.

Vibe 000062's "steps must self-prepare" was an early recognition of this.  It
helped, but it is a workaround: it makes each step *tolerate* pollution by
canonicalizing or distributing internally, rather than being *indifferent* to
it.  Tolerance must be re-implemented per step; indifference would be
structural.

## What already gets this right

`nf_match`'s identity matcher does **sub-product** and **sub-chain** matching:
the pattern's factors may sit "among extra factors of a larger term", and a
chain pattern may match a contiguous run inside a longer chain.  That is
exactly pollution-immunity, and it is already implemented — which is why the
M2 verbs handle cases the hand-written steps refuse.

The blocked operations are precisely the ones *not* expressed as rules.

## Design directions

Not a decision — the options, with what each costs.

**(A) Index-directed instead of shape-directed.**  Reassembly should look for
the *semantic relation* it exploits — a summed index tying a coordinate slot
to a basis vector — and fold that pair wherever the two sit in the term.
Position, neighbours and nesting become irrelevant because they are never
consulted.  This is the narrowest fix and probably the correct one for
reassembly specifically; it likely subsumes the ε case and the metric case at
once.

**(B) Express more transformations as rules over `Nf`.**  Anything expressible
as an identity inherits sub-product matching for free.  The limit: reassembly
is not obviously a rule — `Σ_i a_i e^i → a` must bind "the invariant whose
components these are", which is a meta-level operation, not a pattern
variable.  Worth establishing *which* of the blocked operations are
rule-expressible; several may be.

**(C) Selective application.**  `td.at(expr, path, step)` (vibe 000054) already
retargets any step at any position, which side-steps context by letting the
caller point.  It does not solve the problem — the user must find the
position, and after canonicalization the position may not exist — but it is
the existing escape hatch and should be documented as such.

**(D) Do nothing structural; keep patching.**  Recorded to be argued against
honestly: each patch has been cheap, and seven of them have shipped.  The cost
is not in any one patch but in the *discoverability* tax on the user, which is
what vibe 000056 identified and what the verbs were built to remove.

## Open questions

1. Is (A) general, or only right for reassembly?  Are there blocked
   operations with no "semantic relation" to key on?
2. Which of the seven historical instances would (A) and (B) have prevented?
   That is a concrete test of whether either is the real fix — worth doing
   before building anything.
3. Does the answer change what `Nf` should keep?  If several operations need
   the coordinate↔basis-vector association, perhaps it should be represented
   rather than reconstructed.
4. Does the e-graph subsume this?  Saturation explores forms rather than
   demanding one, so "polluted" and "clean" can coexist as one e-class — but
   only for operations expressible as rules, which returns to (B).

## Status

Problem statement only.  Nothing here is scheduled; it is written so the next
instance is recognised as the seventh of a kind rather than a fresh surprise.
