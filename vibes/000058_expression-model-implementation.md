# 000058 Expression model — algorithms & implementation plan

Concrete realisation of the model in
[000057](000057_expression-model.md): the canon algorithm (lower `Expr → Nf`),
the rendering algorithm (`Nf → LaTeX`), and a **parallel-IR** implementation plan
that keeps the system buildable + green at every step (CLAUDE.md principle 1,
"software-as-a-plant").

Decision (settled): **parallel IR** — the new normal form `Nf` grows *beside* the
existing binary `Expr`, which stays the surface input language.  Nothing in
`Expr` is mutated in place; the old canon/render/match paths stay alive until the
new ones replace them, then are pruned.

## The data model (`Nf`)

```
Nf     := [ Term ]                       -- additive set, canonical term order
Term   := { Sign sign                    -- term-level, like today's intent (not a Negate node)
          , Rational coeff               -- region 1: one numeric magnitude
          , SumModes modes               -- per bound index: default | sum | nosum
          , [Index]  bound               -- inferred dummy indices, α-canonical ids
          , [Factor] scalars             -- region 2: rank-0 factors, sorted
          , [Factor] tensors }           -- region 3: rank-≥1 factors, positional
Factor := Atom TensorObject              -- a leaf (with its index labels)
        | Contraction [(Factor, COp)]    -- flat assoc chain over {@ : //}; rank known
        | Cross        [Factor]          -- a % chain (anticommutation sign already lifted)
        | Paren        Nf                -- an opaque parenthesised sum, rank known
COp    := @ | : | //
```

`sign`+`coeff` together are a signed rational; they are kept as a term field
(the "store sign like today" decision of 000057), and combine during like-term
collection.  `Number` factors never survive canon — they fold into `coeff`; `/`
folds in as a reciprocal.  A `Contraction` of result rank 0 is a *scalar* factor
(region 2); rank ≥ 1 puts it in region 3.  Region is chosen by `infer_rank`, not
by the operator.

## Canon algorithm: lower `Expr → Nf`

Total and idempotent.  Lowers in passes; **never distributes** products over
sums.  Reuses existing machinery where noted.

1. **Summation resolution.**  Reuse `materialize`: apply realm-driven implicit
   summation, raise the existing errors (an index occurring > 2 times,
   inconsistent variance/levels), honour `sum`/`nosum`.  Record the inferred
   bound-index set and the per-index mode map on each term.

2. **Additive flatten → signed terms.**  Expand the *outermost* additive layer
   only: `Sum → concat`, `Difference(a,b) → a-terms ++ sign-flip(b-terms)`,
   `Negate → sign-flip`.  A sum sitting *inside* a product is **not** touched —
   it becomes a `Paren` factor (canonicalised recursively).  Replaces the role of
   `canon_additive` for the top layer.

3. **Per term: multiplicative flatten + encapsulate.**
   - flatten associative `*` and the contraction chains;
   - each maximal contraction sub-chain (ops in `{@ : //}`) bounded by `*` →
     one `Contraction` factor (stored flat — the interface theorem of 000057
     makes its bracketing immaterial);
   - each cross group → `Cross`, lifting anticommutation sign to `sign` and
     re-associating around rank-≥2 fences (reuse `reassociate_cross_fence`,
     000055);
   - a sum-as-factor → `Paren`;
   - `Number` / `ScalarDiv` → fold into `coeff`.

4. **Region placement.**  `infer_rank` each factor: rank 0 → `scalars`, rank ≥ 1
   → `tensors`.  This is the step that floats a wedged scalar (`(a·b)`) out from
   between two legs — the 000056 fold failure dissolves here.

5. **Normalise within regions.**
   - `coeff`: reduce the rational.
   - `scalars`: canonicalise each interior (commutative contractions `a@b`,
     `A:B` get canonical operand order via `expr_cmp`), then **sort** the list by
     the total factor order; optionally collect equals into powers.
   - `tensors`: positional (⊗ non-commutative); only interior operand-order /
     cross-sign normalisation.
   - `bound`: α-rename dummy ids to canonical numbers (reuse the
     `bound_canon_id` / `canon_sum_stack` Fubini-ordering trick) so α-equivalent
     terms collide.

6. **Like-term collection.**  Term key = (`tensors`, `scalars`, `modes`,
   `bound`).  Merge equal keys by adding signed `coeff`s; drop zero; flip `sign`
   as needed.  This is where `a + (−a) → 0` and `2a + 3a → 5a` happen — 000056's
   sign-drift and `+ −` artefacts become structurally impossible.

7. **Order the term set** by a total term order → deterministic `Nf`.

**Idempotence check:** `lower(raise(nf)) == nf` (see raise, below).  This is the
property that retires "is canon needed again?" — the question of 000056.

## Raise `Nf → Expr`

So the public API can keep returning `Expr`.  Deterministic:
`coeff · scalars · tensors` as a fixed-convention product (pick left-assoc),
`Contraction`→`Dot`/`DDot` chain, `Cross`→`Cross`, `Paren`→`Sum`, `sign`→leading
`Negate`/fold into a `Difference` at the additive level, and `modes` →
`ExplicitSum`/`NoSum` only where an override differs from the realm default.
Round-trip law: `lower ∘ raise = id` on canonical `Nf`.

## Rendering algorithm: `Nf → LaTeX`

A **faithful, total** function of `Nf` — never emits a misleading or spurious
paren (000056 problem 5).  Parens come from a precedence table, *not* from the
(now-flat) canonical grouping.

- **Term:** render `sign` (`-` or nothing), then `coeff` (omit when 1), then
  `scalars` then `tensors`, juxtaposed with thin spacing.
- **Precedence (loosest→tightest):** `+` (additive) < juxtaposition `*` <
  contraction `{@ : //}` < cross `%` < atom.
- **Paren rule:** wrap a child iff its top operator's precedence is **strictly
  lower** than the parent context, or equal with non-associative placement.
  Consequences: a `Paren` (sum) factor is always parenthesised inside a product;
  a scalar `Contraction` like `(a·b)` abutting a tensor is parenthesised to block
  the `a·(b⊗c)` misreading; a lone cross term at top level drops its parens.
- **Cross:** a rank-1 `%` run prints with explicit parens (genuinely
  non-associative); a fenced `a%M%b` may print flat.
- **Labelled variant** (000054): thread the path / occurrence id down the
  recursion and wrap chosen factors in `\overset{\scriptstyle label}{·}`,
  returning `{latex, label→path}`.  Built on the same traversal; `render` stays
  untouched.

## Implementation plan (parallel IR)

Each commit builds and is green.  Feasibility examples (principle 5) are the
end-to-end gate; benchmarks (principle 4) guard canon.

**Stage 1 — isolated `Nf` type (purely additive, no consumers).**
- C1  `Nf`/`Term`/`Factor` structs, builders, hash-cons/equality, unit tests.
      **DONE** — `src/include/tender/nf.hpp`, `src/nf.cpp`, `tests/nf_test.cpp`
      (16 tests).  Notes: `bound`/`modes` paired into one `BoundIndex` list
      (kept in lockstep) rather than parallel arrays; no separate `Sign` field
      (the signed `Rational coeff` carries the sign — settled at review);
      equality is structural and positional (scalar sorting / α-renaming are
      canon's job, not equality's); "hash-cons/equality" realised as structural
      `equal` + consistent `hash` (no global cons table — the arena `Context`
      matches `Expr` and does not dedup), so callers can hash-cons later.
- C2  total orders (factor order, term order) on top of `expr_cmp`, tests.
      **DONE** — extracted the leaf comparators (`name_view_cmp`, `space_cmp`,
      `index_assoc_cmp`, `tensor_object_cmp`) out of `derivation.cpp`'s
      `expr_cmp` into `tender/tensor_order.{hpp,cpp}`; `expr_cmp`'s TensorObject
      arm and Nf's `compare`/`equal` now share that one atom key (DRY).  Added
      `nf::compare(Factor|Term|Nf)` three-way orders + 6 tests
      (`compare == 0 ⇔ equal`, antisymmetry).  Suite green (550).

**Stage 2 — lowering `Expr → Nf` (the canon algorithm), behind `canonicalize_nf`.**
Old `canonicalize` untouched; each pass its own commit + tests + differential
check.
- C3  additive flatten → signed terms (no distribution); like-term skeleton.
      **DONE** — lowering grows in a new `tender/nf_lower.{hpp,cpp}` module
      (passes added one per commit; the `canonicalize_nf` entry point is wired
      only at C10).  `additive_flatten(Expr) -> [SignedExpr{sign, body}]`
      expands the outermost `Sum`/`Difference`/`Negate` layer without
      distributing; a sum inside a product stays an opaque leaf `body` (still
      an `Expr` — multiplicative decomposition is C4/C5).  9 tests.  Like-term
      collection itself is deferred to C9 (operates on built `Term`s).  Suite
      green (559).
- C4  multiplicative flatten; `Number`/`/` → `coeff`.
      **DONE** — `multiplicative_flatten(SignedExpr) -> ProductParts{coeff,
      factors}` flattens the outermost `*` (`TensorProduct`) chain, folding
      `ScalarLiteral`, `Negate`, and numeric `ScalarDiv` (`x / literal`) into
      `coeff` (seeded by sign); a non-numeric divisor and any
      contraction/cross/sum node stay opaque `Expr` factors (encapsulation is
      C5/C6).  Factor order preserved.  8 tests.  Suite green (567).
- C5  contraction encapsulation + rank-based region placement.
- C6  cross encapsulation + anticommutation sign lift (reuse 000055).
- C7  scalar floating + scalar sort + interior commutative-operand ordering.
- C8  summation: bound-index inference, mode map, α-renaming (reuse
      `materialize` / `bound_canon_id`).
- C9  like-term collection (cancellation, coeff merge) + term-set ordering.
- C10 `canonicalize_nf` entry point; **differential harness** vs old
      `canonicalize` on a corpus (divergences are bugs or signed-off
      improvements); canon benchmark.

**Stage 3 — render + raise.**
- C11 `render_nf` (precedence table, paren rule); golden tests for every
      000056 rendering case.
- C12 `raise` `Nf → Expr`; round-trip tests (`lower ∘ raise = id`).

**Stage 4 — adopt under the public API (the flip; the one non-additive stage).**
- C13 `canonicalize := raise ∘ lower`; update affected canon-shape assertions in
      one focused commit; **feasibility examples must stay green** (acceptance
      gate); the `a×(b×I)` derivation becomes a no-"missing-steps" test.

**Stage 5 — matcher on `Nf`.**
- C14 migrate `identity.cpp` matching to the all-`*` flat form (the original
      motivation: robust `a%I%b`-style matching, AC over `scalars`, positional
      over `tensors`); motivating cases become tests.

**Stage 6 — prune.**
- C15 remove dead old canon / old render / `Negate`-`Difference` handling.

Risk is concentrated in **C13** (the flip) and **C14** (matcher); C1–C12 and C15
are additive or subtractive only.

## Micro-decisions deferred to implementation

- Exact `Factor` variant set (is `Paren` distinct from a rank-0 `Contraction`?).
- ~~Whether `sign` is a separate field or just the sign bit of `coeff`~~
  **Settled (C1 review): no separate `Sign` field — `Rational coeff` is signed
  and carries the term's sign.**  Cross anticommutation flips `coeff`.
- Scalar power-collection: yes/no.
- `raise` product associativity convention (left vs right).
- Whether identities are stored lowered (`Nf`) or lowered on demand at match
  time.

## Status

Algorithms + plan recorded.  **Stage 1 complete** (C1: isolated `Nf` type —
structs, builders, structural equality + hashing; C2: shared leaf comparators
in `tensor_order.{hpp,cpp}` + `nf::compare` total orders).  22 Nf unit tests;
full suite green at 550.  **Stage 2 started**: C3 (additive flatten → signed
terms) and C4 (multiplicative flatten → `ProductParts`) done in
`nf_lower.{hpp,cpp}`; suite green at 567.  Next action: Stage 2 / C5
(contraction encapsulation + rank-based region placement).
Builds on [000057](000057_expression-model.md) (the model),
[000056](000056_expression-representation-rethink.md) (the motivation),
[000055](000055_cross-reassociation.md) (cross-fence reuse),
[000054](000054_selective-expansion.md) (labelled-render reuse).
