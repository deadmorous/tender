# 000056 Expression representation — a rethink

A **problem catalogue**, not a design.  A series of small, unrelated-looking
failures keep surfacing from the same place: the way expressions are
represented, canonicalized, matched, and rendered.  They are individually
patchable, but they recur, and each patch is a guess.  Before building more
identity/basis machinery on top, we want to fix the foundation.  This note
records the symptoms and the suspected structural causes so we can design the
fix deliberately.

## The thesis (user)

> The system is useless if one has to guess things like "the missing step" —
> there is simply no intuitive knowledge of it.  No point moving further until
> we achieve something really neat about expressions, transformations, and
> matching, because it affects everything we build on top.

The deciding usability failure: proving `a × (b × I) = b⊗a − (a·b) I` required
inserting `distribute_contraction` (to expose the rank-1 triple product) and
`expand_products` (to split the difference) and a `canonicalize` (to make the
last term foldable) — **none of which the user could have known to call**.  A
correct, total derivation that a user cannot discover is not a usable system.

## Motivating session: `a × (b × I)`

```python
id_bac_cab = td.Identity("Ixa", a%(b%c), b*(a@c) - c*(a@b))
Ix = tb.expand_in_basis(I, frame, co)
x  = a % (b % Ix)
x  = td.apply_identity(id_bac_cab)(x)     # ← silently does nothing
```

`apply_identity` is a no-op here because after expansion the term is
`a × (b × (eᵢ⊗eⁱ))`: the inner cross is **not** a vector triple product (the
`⊗` is a rank-2 fence), so bac-cab correctly refuses.  The "missing steps" that
make it work — and which the user had no way to guess — are:

```python
x = td.distribute_contraction(td.canonicalize(a%(b%Ix)))  # → Σᵢ (a×(b×eᵢ))⊗eⁱ
x = td.apply_identity(id_bac_cab)(x)                       # → Σᵢ [b(a·eᵢ) − eᵢ(a·b)]⊗eⁱ
x = td.expand_products(x)                                  # split the −
x = tb.reassemble_completeness(x, frame)                   # → b⊗a − (a·b) I
```

and even then the final fold only completes if a `canonicalize` is threaded in
at the right point (see problem 4).

## The problems

### 1. Load-bearing steps that render as no-ops

`canonicalize` and `distribute_contraction` change the **grouping** of factors,
but the rendered output looks identical, so they appear to do nothing — yet the
derivation fails without them.  Worse, the requirement is non-deterministic from
the user's seat: *"it worked without `canonicalize`, too"* in one arrangement
and not in another, because the needed normalization was sometimes already
present.  The user cannot see what a step did, nor predict when it is required.

*Suspected cause:* the canonical form is not actually canonical (or not the form
the next step expects), and the renderer hides the structural difference that
the transform relies on.  Two expressions that must be treated differently print
the same.

### 2. Implicit summation silently becomes explicit

Before `apply_identity` the sum over `i` is implicit (Einstein); after, it is a
materialized `Σᵢ` binder, and stays that way.  The surface form is unstable
across a rewrite: the same mathematical object is shown two different ways
depending on its processing history.

*Suspected cause:* matching/rewriting materializes binders (`implicitize` is not
re-applied), so explicit/implicit is a leaky representational duality rather
than a single normalized form.

### 3. Unary minus rendered after a plus

Output contains `… + −eᵢ …` instead of `… − eᵢ …`.  `Negate` as a standalone
unary node and `Difference`/`Sum` do not normalize against each other, so
`A + (−B)` survives instead of becoming `A − B`.

*Suspected cause:* sign is represented two ways (a `Negate` node and a
`Difference` node) with no single normal form; the renderer faithfully prints
the un-normalized tree.

### 4. Cannot fold `eᵢ (a·b) eᵢ → (a·b) I`

The residual term is `Σᵢ −eᵢ (a·b) eᵢ`.  Completeness shape B needs *two bare
legs `eᵢ` and only scalar other factors*.  It fails here because:

- the term is wrapped in a **`Negate`**, which hides the leg structure from the
  pattern (the first factor reads as `−eᵢ`, not a bare `eᵢ`); and
- the scalar `(a·b)` sits **between** the two legs and is not floated out —
  the system does not treat `a·b` as a scalar free to commute past either `eᵢ`
  to bring the legs together.

`canonicalize` happens to fix this (it re-normalizes the sign/order), which is
exactly the invisible load-bearing step of problem 1.

*Suspected cause:* (a) sign duality again; (b) **scalars do not commute freely**
in the representation — a product is a rigid binary tree, so a scalar wedged
between two legs blocks a pattern that should see them as adjacent.  This is the
same binary-tree-blocks-matching theme as the cross bracketing in `a%I%b`
(vibe 000054) and cross re-association (vibe 000055).

### 5. Parentheses render incorrectly

`a % I % b` renders with parens that are redundant or wrong; `b × (eᵢ⊗eⁱ)`
renders **without** the parens that would disambiguate it from `(b×eᵢ)⊗eⁱ`
(this artifact is what made problem 1's regroupings invisible during diagnosis).
The printed grouping cannot be trusted, which compounds every problem above —
the user reads structure off the output and the output lies.

*Suspected cause:* renderer precedence/associativity handling is ad hoc and does
not reflect the actual node structure faithfully.

## The common roots

Most symptoms collapse onto a few representational decisions:

- **Binary-tree grouping of n-ary associative operators** (`+`, `⊗`, and the
  factor chain) — bracketing is arbitrary, blocks matching, and is what the
  renderer keeps hiding or mis-parenthesizing.  (Problems 1, 4, 5; theme of
  000054.)
- **Scalars are not free to commute** — a product's factor order is structural,
  so a scalar between two tensor legs is an obstacle rather than a movable
  weight.  (Problem 4.)
- **Sign has two representations** (`Negate` node vs `Difference` node) with no
  normal form.  (Problems 3, 4.)
- **Implicit/explicit summation is a leaky duality**, not one normalized form.
  (Problem 2.)
- **Rendering is not a faithful, total function of structure** — it elides and
  mis-places parens, so it cannot be used to understand what transforms did.
  (Problems 1, 5.)

Cutting across all of them: **transformations and their necessity are
invisible**.  A user cannot see what a step changed, nor know which step is
needed next.  Whatever representation we choose must make the canonical form
*the* form (so "missing steps" disappear), and make rendering a trustworthy
window onto it.

## Threads to pull (for discussion — not decided)

- A **flat associative chain** for `+`, `⊗`, and the factor product, with
  explicit scalar coefficients factored to the front, so commuting a scalar and
  re-bracketing a chain are free (kills problems 4, 5 and the 000054 theme in
  one move).
- A **single signed/normalized additive form** (signed terms, no free-floating
  `Negate`) so `A + −B` cannot occur (problems 3, 4).
- **One summation form** — pick implicit or explicit as canonical and always
  re-normalize to it after a rewrite (problem 2).
- A **faithful renderer** driven by real operator precedence/associativity, and
  a way to *show what a transform changed* (diff/highlight), so steps stop being
  invisible (problems 1, 5; links to the labeled-LaTeX idea in 000054).

## Status

Problem catalogue only — **no design yet, nothing committed beyond this note.**
The motivating derivation does complete today, but only via unguessable steps;
that is the bug.  Next: discuss the representation redesign before building any
further identity/basis machinery on top.
