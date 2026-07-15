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

View + options captured.  Committed design below (2026-07-15).  Re-association
(vibe 000055, the sibling usability fix from the same `a × I × b` exercise) is
implemented.

---

# Committed design (2026-07-15) — IMPLEMENTED (increments 1–5 + labeled view)

**Status: DONE** (2026-07-16), including the labeled-LaTeX visual feedback.
Increments 1–5: `children` / `with_children` (rewrite.hpp, rewrite_tree
re-expressed on them, heap-free via `CappedVec`), `subexpr_at` / `rewrite_at` /
`replace_at` / `find_occurrences` / `addend_paths`, and the Python surface
`Expr.at` / `.replace_at` / `.rewrite_at` / `.find(kind=/name=)` / `.addends()`
+ the free combinator `td.at(expr, path, step)`. Validated by selectively
expanding one `I` in `a×I×b` (neighbours stay symbolic) and selectively
reassembling one coordinate term (vibe 000075 gap D mechanism).

**Labeled view (the visual feedback):** `Expr.paths(which)` enumerates node
paths by policy (`all`/`atoms`/`tensors`/`wellknown`); `tender.render.labeled(
expr, which)` returns a Jupyter/terminal display object — the whole expression
above a **path → part** legend. *Design choice:* the legend renders each part
**independently** rather than superimposing tags on the one pretty rendering.
The pretty renderer diverges from the structural (`children`) tree in several
arms — Δ-folding (`∇·(∇⊗X)`→`Δ X`), product flattening, subtraction rewrite,
`X⊗∇`→`(∇X)ᵀ` — so threading a `children`-path through it to place in-line tags
risks emitting a tag whose path silently points at the *wrong* node. Rendering
each part on its own is exact by construction under both hash-consing and every
fold (verified: a legend path reaches the `X` inside a folded `Δ X`). If literal
above-node tags are wanted later, the marker-node route (option 4) is the safe
way and plugs into the same paths.

845 C++ + 316 Python pass; rewrite.hpp 93% line coverage. Tests:
`tests/rewrite_test.cpp`, `python/tests/test_select.py`,
`python/tests/test_render.py`.



The scope grew: alongside *selective application* (apply a step to one
occurrence, not the whole tree) the user wants to **extract a specific part of
an expression into another expression** (e.g. one term).  Both are the **same
primitive** — address a node *by its path from the root* — so one addressing
core serves both.

## Decisions (locked with the user)

- **Addressing model: lightweight paths.**  `Path = vector<int>` (Dewey
  address) + `find`/`Nth`, governed by the sequencing rule *canonicalize →
  address → rewrite, with no canonicalize in between* (the canon-stability
  hazard above).  The **marker node** (option 4, canon-stable) is **not** built
  now; it can be added later as an alternative front-end feeding the very same
  `rewrite_at`, so nothing here is thrown away if it is wanted.
- **Visual feedback (labeled LaTeX): deferred to a later increment.**  Ship the
  headless core first, prove it on a real derivation, then add
  `render_latex_labeled` once the addressing vocabulary has settled.  The label a
  future renderer prints must be exactly the token passed back to `at` /
  `find` (one vocabulary), which the headless API fixes first.
- **Selective *step* application composes; it is not per-step plumbing.**  One
  generic combinator `at(expr, path, step)` applies **any** `(Expr) -> Expr`
  step to the sub-Expr at `path` and splices the result back.  So
  `at(path_to_I, λs. expand_in_basis(s, frame))` expands *only that* `I`, and the
  same combinator retargets `apply_identity`, `reassemble`, … — no step learns a
  "target" argument.  (The cheap kind/predicate *filter* on `expand_in_basis` is
  then unnecessary: the path lands on the exact node, so expanding "the whole
  subtree at that path" **is** expanding that one occurrence.)

## Architecture (bottom-up)

**1. Navigation primitives** (`rewrite.hpp`).  Factor the per-node-kind child
structure — currently hand-written once inside `rewrite_tree` — into two small
functions so no second giant visitor is written (DRY):

- `children(e) -> small_vector<Expr const*>` — the Expr children in selector
  order (binary: `left`=0, `right`=1; unary: `operand`=0; `ExplicitSum`:
  `body`=0, `bound`=1 *when present*; `NoSum`/`ScalarFn`/`Deriv`: their one child;
  `Pow`: `base`=0, `exponent`=1; leaves: empty).
- `with_children(ctx, e, new_children) -> Expr const*` — rebuild `e` with new
  children, **copying non-Expr fields** (the bound `index`, `ScalarFn::kind`,
  optional `ExplicitSum::bound` arity) from `e`.  Returns `e` unchanged (pointer
  reuse) when every child is identical.

`rewrite_tree` is then re-expressed as
`f(ctx, with_children(ctx, e, map(rewrite_tree, children(e))))` — same pointer-
reuse and bottom-up-then-`f` semantics as today.

**2. Path operations** (`rewrite.hpp`), all layered on the two primitives:

- `subexpr_at(e, path) -> Expr const*` — **extraction / navigation.**  The
  returned sub-Expr shares the same ctx/arena, so it is a first-class expression
  any step runs on.  Out-of-range selector ⇒ error (no silent nullptr).
- `rewrite_at(ctx, e, path, f) -> Expr const*` — apply `f` at the target,
  rebuild only the spine above it; everything off-path stays **shared** (sibling
  pointer identity preserved — cheap).  **Selective application.**
- `replace_at(ctx, e, path, sub)` = `rewrite_at` with a const `f` — splice a
  worked-on sub back in.  Gives the **round-trip**: `sub = at(e, p)` → work on
  `sub` → `e2 = replace_at(e, p, sub2)`.
- `find_occurrences(e, predicate) -> vector<Path>` in **pre-order**, with
  predicate sugar: by kind (well-known `Identity`, a generic `TensorObject`), by
  name.  `Nth(kind, k)` = `find_occurrences(...)[k]`.  A convenience selector
  enumerates the **top-level addends** of a canonical sum (walk the
  `Sum`/`Difference` spine, one path per addend) so "extract *a term*" is a
  natural call.

**3. Python surface.**  Methods on `Expr` (`PyExpr` already carries
`{ctx, expr}`, so `derive` re-wraps): `.at(path)` (extract), `.replace_at(path,
sub)`, `.rewrite_at(path, fn)`, `.find(kind=…/name=…)`, plus the free combinator
`tender.derivation.at(expr, path, step)`.  A `Path` is just a `list[int]`.

## Increment plan (each buildable, tested, ≥90% coverage, clang-formatted)

1. **Navigation primitives** — `children`/`with_children`; re-express
   `rewrite_tree` on them.  Tests: `with_children ∘ children` round-trips every
   node kind incl. `ExplicitSum` with/without bound; `rewrite_tree` behaviour
   unchanged; pointer reuse when nothing changes.
2. **Path ops** — `subexpr_at`, `rewrite_at`, `replace_at`.  Tests: navigate to
   a deep node; rewrite at a path; splice; **off-path sharing** (untouched
   sibling keeps its pointer); bad-path error.
3. **`find_occurrences` + predicates + Nth + addend enumeration.**  Tests: find
   both `I`s in `a×I×b×I` in order; Nth; enumerate addends of a canonical sum.
4. **Python surface + `at` combinator.**  Tests: selectively `expand_in_basis`
   *one* `I` in `a × I × b`, leaving `a`, `b` symbolic; extract a term from a
   canonical sum, `simplify` it alone, splice back (round-trip equal to the
   whole-tree result); `.find` reads off the same paths.
5. **Validation on a real need** — apply the primitive to the pending
   ∇-only-expansion + 2nd-derivative reassembly step (vibe 000075 gap D / the
   strain-compat notebook, vibe 000080): expand only the `∇`/one leg, reassemble
   that term.  Confirms the primitive unblocks the deferred work; not
   necessarily finishing that whole derivation here.

**Deferred (later increment):** `render_latex_labeled(e, map, policy) ->
{latex, label→path}` for visual selection — label content (raw path vs
occurrence-index-per-kind + legend) and which nodes get labels decided then; the
printed token must equal the `at`/`find` argument.

See [[route-b-curvilinear-derivations]] (vibe 000075 gap D — the pending need),
[[vibe80-notebook-gaps-sprint]] (selective expansion deferred there),
[[canonicalize-preserves-nabla-fence]] (canon reshaping — the stability hazard).
