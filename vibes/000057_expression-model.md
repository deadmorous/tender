# 000057 Expression model — sum of signed `*`-terms

The representation redesign motivated by the problem catalogue in
[000056](000056_expression-representation-rethink.md).  Goal: a representation in
which the canonical form is *the* form, canon is total and idempotent, and the
"missing steps" of 000056 cannot exist.  This note fixes the grammar, the
theorem that makes it sound, the canonical form, the canon algorithm, and the
one open question (already narrowed to summation).

## Grammar (surface)

```
Expr     := Term  |  Term Expr            -- unordered set of terms (+ commutes)
Term     := Sign Factors
Sign     := + | -
Factors  := Factor  |  Factor op Factors  -- list of (Factor, op), last op = noop
Factor   := Number | TensorObject | ( Expr )
op       := * | / | @ | // | : | %
```

A `Term` is a non-empty list of `(factor, op)` pairs.  Surface syntax is
**permissive** — the user writes any chain — and canon rewrites it into the
strict canonical form below.

Operators: `*` tensor product (⊗, juxtaposition), `/` scalar division, `@` dot
(·), `:` double dot, `//` double-dot alt, `%` cross (×).

## The theorem this rests on: contraction is local

In direct (coordinate-free) notation a contraction only ever joins **adjacent
legs** in the factor string.  There is no way to dot `a`'s leg with `d`'s leg
across `b ⊗ c` without first transposing — non-adjacent contraction simply has
no direct-notation spelling.  Two consequences:

1. **Bracketing is immaterial for the contraction family `{* @ : //}`** (and
   `/`).  Each contracts a fixed number of indices at the *interface* between
   the running-left object's tail and the next factor's head; re-bracketing
   never changes which indices meet.  So the value of a chain depends only on
   the `(factor, op)` sequence, not the grouping:
   ```
   (a⊗b)·c = a⊗(b·c) = (b·c) a
   (a⊗B)·C = a⊗(B·C),   A:(B⊗C) = (A:B) C,   …
   ```

2. **`%` (cross) is the sole exception, and only among rank-1 factors.**  `a×b`
   builds a rank-1 result out of *both* operands via ε — not an interface
   contraction — so `(a×b)×c ≠ a×(b×c)`.  But `a×M×b` with `M` rank ≥ 2 *is*
   associative: the ⊗ inside `M` fences the two crosses onto disjoint legs
   (proved in [000055](000055_cross-reassociation.md)).  The only genuinely
   non-associative case is a run of crosses among rank-1 factors.

Because every contraction localizes to a contiguous run of factors, each one can
be **encapsulated into a single composite `(Expr)` factor**.  What remains
joining factors at the top level is only juxtaposition.

## Canonical form: a graded `*`-monoid

> **`*` is the only operator in a canonical term.**  Every other operator is
> pushed inside an `(Expr)` factor; `/` folds into the coefficient.  A term is
> ```
> coeff  ·  [ scalar factors, sorted ]  ·  [ rank-≥1 factors, positional ]
> ```

Three regions, joined by `*`:

1. **Coefficient** — one rational `Number` (all numeric literals multiplied; `/`
   folds in as a reciprocal).  Always first; `/` never survives as a chain op.
2. **Scalar factors** — every rank-0 sub-result (`(a@b)`, `(A:B)`, …).  They
   commute, so **sorted** by `expr_cmp`; equal ones may collect into powers.
3. **Rank-≥1 factors** `T1 … TN` — kept in **positional order** (⊗ is
   non-commutative; nothing to sort).  The only freedom is cross
   anticommutation, whose sign is lifted to the term's `Sign`.

Region is decided by **result rank, not by operator**: `(a@b)` → rank 0 → region
2; `(A@b)` → rank 1 → region 3.  Same `@`, placed differently.  This makes
`infer_rank` load-bearing for canon — it must be total and trustworthy.

A **factor** is `Number | TensorObject | (Expr)`.  A composite `(Expr)` factor is
atomic to the `*`-chain but has a recursively canonical interior:

- a pure contraction chain (`A@B@c`) is stored **flat** — the interface theorem
  says bracketing is immaterial, so there is nothing to choose;
- commutative contractions (`a@b = b@a`, `A:B = B:A`) get canonical operand order
  inside;
- a cross is **always** a parenthesized factor (`(a%b)`, `(a%M%b)`) — under
  all-`*` it cannot be a top-level op, so this needs no separate decision — with
  its anticommutation sign lifted out.

Contraction is **encapsulated, not eliminated**: `A·B` (matrix product) is one
irreducible rank-2 object and stays the single factor `(A@B)`.  Canon never tries
to split it; it only forbids `@` from being a *joining* operator.

## What canon does (and does not) do

**Does:** flatten chains; lift each maximal contraction/cross sub-chain into an
`(Expr)` factor of its result rank; float scalars to region 2 and sort them;
fold `/` and numeric literals into the coefficient; normalize sign (cross
anticommutation, term `Sign`); collect like terms at the additive level
(identical canonical chains add coefficients, so `a + (−a) → 0`).

**Does not:** distribute products over sums, ever.  `(a+b)` is an opaque `(Expr)`
factor of known rank.  Distribution/expansion are **explicit, named,
user-visible transforms** (`distribute_contraction`, `expand_products`), chosen
by the user — never hidden prerequisites.

This is the property that kills 000056's "invisible load-bearing step": canon
never touches multiplicative/distributive structure, so it is **total and
idempotent** — running it twice never differs, and you never have to guess
whether to run it.  Matching may recurse into `(Expr)` factors; that is the
trade for keeping them opaque to canon.

### How the model dissolves the 000056 problems

- **Scalar wedged between legs** (`Σᵢ −eᵢ (a·b) eᵢ` won't fold): `(a·b)` floats to
  region 2, the two bare `eᵢ` become adjacent, shape-B completeness folds them to
  `I`.  Falls out of the canonical form — no missing step.
- **Unary minus after a plus** / **explicit-sum drift**: a single per-term `Sign`
  plus like-term collection makes `A + −B` and dangling `Negate` structurally
  impossible.
- **Bracketing blocks matching** / **bad parens**: there is no top-level
  bracketing to block anything — only a flat `*`-chain.

## Bound indices and summation

A `Term` **carries its set of bound (dummy) indices, inferred from its factors**
— the same realm-driven discipline as today, not a new mechanism:

- summation convention is **realm-driven**: a repeated index over the
  appropriate realm is summed implicitly;
- it is an **error** when the implicit-summation rules break — an index occurring
  more than twice, or with inconsistent variance/levels;
- the default is **overridden** by `sum` / `nosum`.

The override is stored **on the `Term`, exactly like `Sign`** — a per-index
summation mode `{ default | sum | nosum }`, a small field on the term, *not* an
`ExplicitSum` / `NoSum` wrapper node.  This is the same move the model makes for
sign (a `Sign` field, not a `Negate` node) and is what lets implicit and explicit
forms share one normal form (dissolving 000056's explicit-sum drift): the inferred
bound-index set plus its mode map lives on the `Term`, scoping the `Σ` relative to
the `*`-chain, and is what completeness folds and Einstein contraction read.

## Status

Model converged and recorded; **nothing implemented.**  The data model is now
settled: a `Term` has a `Sign`, a per-index summation-mode field, an inferred
bound-index set, and the three-region `*`-chain of factors.  Open work, in order:

1. spec the **canon algorithm** (chain flatten → contraction/cross encapsulation
   → rank-based region placement → scalar sort → sign + summation-mode
   normalization → like-term collection) against the existing `materialize` /
   `float_sums` / `canon` passes;
2. spec the **rendering algorithm** as a faithful, total function of this
   structure (precedence, no spurious parens) — links to the labeled-LaTeX idea
   in [000054](000054_selective-expansion.md);
3. an **incremental implementation plan** honouring the "software-as-a-plant"
   principle — the system buildable and green at every step, no dead branches.

Relation to siblings: [000056](000056_expression-representation-rethink.md) is the
problem catalogue this answers; [000054](000054_selective-expansion.md) first
raised the flat-chain idea and the addressing/labeling needs;
[000055](000055_cross-reassociation.md) proved the cross-fence associativity this
model relies on.
