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
- The robust answer is **positional addressing**, detailed below.

## What a "path" is

Each node holds `Expr const*` children, identical subterms are **one shared
node** (hash-consing), so a node has no unique parent and cannot name its own
occurrence.  The only handle on *this* `I` is the **route from the root**.

A **path** is a sequence of child-selectors, one per level (a Dewey address):

| node kind | children → selector |
|---|---|
| binary (`Sum`, `TensorProduct`, `Dot`, `Cross`, …) | `left`=0, `right`=1 |
| unary (`Negate`, `Trace`, `Transpose`, …) | `operand`=0 |
| `ExplicitSum` / `NoSum` | `body`=0 (`bound`=1) |
| `TensorObject`, `ScalarLiteral` | leaf — no children |

In `Cross(Cross(a, I), b)` the path to `I` is `[0, 1]`.  The primitive is
`rewrite_at(ctx, e, path, f)`: walk the path, apply `f` at the target, rebuild
the spine above it (everything off-path stays shared — cheap).  The same
primitive also yields targeted `apply_identity` and "fire on *this* subterm" —
the shared positional-addressing enabler.

## The canon-stability hazard

A path addresses **one specific tree**.  Hash-consing is fine (a path is a
route, not a node identity; identical subtrees have different paths), but
**canonicalization invalidates a path** — reordering sums/products, floating
binders, α-renaming, cross re-association (vibe 000055) all reshape the tree.

Sequencing sidesteps it: **canonicalize first**, compute the path on the
canonical tree, then rewrite — no canon in between.  For `a × I × b` this is
natural — canon's re-association exposes `a × (I × b)`, and *then* the `I` sits
at a stable path.

## Addressing options

1. **Raw path `vector<int>`** — the primitive.  Cheap, explicit; caller computes
   it against the tree it will rewrite.  Pair with `find_occurrences(e,
   predicate) → vector<path>` ("find the `I`s, take the first").  Everything else
   layers on this.
2. **Occurrence index** — `target = Nth(kind=Identity, k)`: "the k-th expandable
   `I` in pre-order."  Best ergonomics for the common case; still tree-relative,
   but easy to *describe*.
3. **Predicate + scope** — the kind/predicate filter (the cheap ~10-line win)
   optionally narrowed to a sub-path ("the `I` under this cross").
4. **Marker node** — wrap the target in a transparent `Mark` that **rides
   through canon** (canon preserves and recurses; it deliberately breaks
   hash-cons sharing for that one occurrence), expand "the marked one," unmark.
   The only **canon-stable** option, but a new node type wired through every
   visitor (the cost `Trace`/`Transpose` paid).  Heaviest; deferred unless
   cross-canon stability is actually needed.

**Leaning:** `rewrite_at(path, f)` + `find_occurrences` (small, in
`rewrite.hpp`), with `Nth`/predicate selectors as sugar; keep the marker node as
a future option.  Open: are `Nth`/occurrence-index ergonomics enough, or is the
marker (canon-stable) route wanted; and should selective expansion *own* a
target-selector argument or just compose with a general `rewrite_at`.

## Visual feedback: labeled LaTeX (exploratory — needs detail)

Idea (user): render the expression to LaTeX **enriched with labels sitting above
nodes** — perhaps only leaves — each label carrying the node's **path** or its
**occurrence index**, so the user can *read off* the address to select.  Turns
"which `I`?" from guessing into pointing at a printed label.

Fits the existing renderer (`src/render.cpp`): a recursive visitor
`render(Expr const&)` over node kinds.  A labeled variant threads the current
path down the recursion and wraps chosen nodes' output in an annotation — e.g.
`\overset{\scriptstyle [0,1]}{I}` (label above), a small superscript tag, or a
colored box.  Leaves are the natural anchor (atoms, unambiguous placement);
interior labels need careful placement to stay legible.

Open detailization questions:
- **Label content** — full path `[0,1]` (precise, verbose) vs. a flat
  **occurrence index** per kind (`I`#1, `I`#2; compact, needs a legend mapping
  index → path) vs. both.
- **Which nodes** — leaves only (simplest), the expandable invariants (so an
  interior `I` in a dyad is labeled), or every node (busy).
- **Stability** — labels must be computed on the **same tree** the subsequent
  `rewrite_at` addresses (the canon-stability hazard): render *after* canon, hand
  back the path the label denotes.
- **Round-trip** — the label shown should be exactly the token passed back
  (`expand_at([0,1])` / `expand_nth("I", 1)`), so the visual and the API share
  one vocabulary.
- **Render hooks** — a new `render_latex_labeled(e, map, policy) → {latex,
  labels}` returning the annotated LaTeX *and* the label→path table; existing
  `render_latex` untouched.  Likely a `RenderHint`-style policy for which nodes
  get labels.

## Status

View + options captured; **not committed**.  Re-association (vibe 000055, the
sibling usability fix from the same `a × I × b` exercise) is implemented.
Awaiting further thoughts before choosing an addressing option and detailing the
labeled-render design.
