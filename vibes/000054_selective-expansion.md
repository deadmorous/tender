# 000054 Selective basis expansion

A *current-view* note (no design detail yet — that follows once the picture
settles).  Captures the need and the one hard constraint that shapes every
option.

## The need

`expand_in_basis` walks the whole tree and expands **every** expandable
invariant (every `I`, every generic named tensor).  In a derivation you often
want to expand **one** occurrence — e.g. in `a × I × b`, turn *only* the `I`
into `Σ_i e_i ⊗ e^i`, leaving `a` and `b` symbolic — so the subsequent
coordinate manipulation stays small and targeted.

Two naive ideas, both rejected:

- **A rewrite rule `I → e_i ⊗ e^i`** — crutchy, and (the deciding flaw) it
  cannot pick *one* `I` among several.
- **Expand-then-prune** — expand everything, then try to re-fold what you didn't
  want; fragile and backwards.

## The hard constraint: hash-consing

Expressions are hash-consed: every bare `I` is the **same pointer**.  So you
**cannot** name "this `I`, not that one" by value or identity — textually
identical subterms are one object.  Selection of a single occurrence is
therefore inherently **positional**, not by-value.

This is the fact that rules out the rule-application idea and points all viable
options toward addressing an occurrence *by its place in the tree*.  It also
links this work to other targeting needs (a targeted `apply_identity`, the
cross re-association exposure) — they all want the same positional-addressing
primitive.

## Direction (not yet committed)

- A cheap, partial win available immediately: a **kind/predicate filter** on
  `expand_in_basis` (expand only `Identity`, or only a named tensor) — separates
  "expand `I` but not other invariants" without disambiguating *multiple* `I`s.
- The robust answer is **positional addressing** (a path / occurrence selector,
  or a shared `rewrite_at` primitive).  Details — what a "path" is, how it
  survives hash-consing and canonicalization — are deferred to the next note.

## Status

View only.  Re-association (vibe 000055, the sibling usability fix surfaced by
the same `a × I × b` exercise) is implemented first; selective-expansion design
detail comes after.
