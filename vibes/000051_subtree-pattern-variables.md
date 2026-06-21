# 000051 Subtree pattern variables for user identities

The next considerable piece of work: make `apply_identity` (and the e-matcher)
fire on user-written invariant identities like `(a⊗b):(c⊗d) = (a·c)(b·d)`,
`a×(b×c) = b(a·c) − c(a·b)`, `a×I×b = b⊗a − (a·b)I`.  Today these silently fail
to match, which is the single most frustrating usability gap.

## Why they fail — two independent blockers

Surfaced while a user tried `Identity("q", (a*b).ddot(c*d), (a@c)*(b@d))` and
applied it to `I:I` expanded in a basis.

### Blocker A — no subtree pattern variables (the big one)

The matcher's pattern variables are the LHS's **free indices** only.  A
slot-less named tensor (`a`, `b`, …) in an LHS is a **literal**, not a wildcard
— it must match its target subtree *structurally/exactly* (identity.hpp says so
explicitly; the design rejected subtree variables in vibe 000033 §4.1 because
the *index* identities did not need them).  So `(a⊗b):(c⊗d)` only matches a
target literally built from those same `a,b,c,d` objects; it cannot match
`(eᵢ⊗eᵢ):(eⱼ⊗eⱼ)` because `a` ≠ `eᵢ`.

This is the deferred capability flagged across vibes 000033/000040/000048/000049.
Invariant (direct-notation) identities are inherently about arbitrary
**subtrees** `a,b,c,d`, so there is no index-only encoding — subtree variables
are genuinely required.

### Blocker B — binder wrapping

Even with subtree variables, the LHS `DDot(TP(a,b), TP(c,d))` does not match the
target `DDot(ExplicitSum(i,…), ExplicitSum(j,…))`: the summation binders sit
*between* the `DDot` and the dyads, so the structures do not line up.

Because every operation is linear, `(Σᵢ Aᵢ) : B = Σᵢ (Aᵢ : B)`, so the fix is to
**pull `ExplicitSum` binders out of linear operand positions to the head of the
term** (the user's "accumulate the sums on the outside" idea) during
canonicalization: `(Σᵢ eᵢ⊗eᵢ):(Σⱼ eⱼ⊗eⱼ) → Σᵢ Σⱼ (eᵢ⊗eᵢ):(eⱼ⊗eⱼ)`.  Then the
`DDot` sits directly over two bare dyads and the pattern matches under the
binders.  Sound; it is a canonical-form improvement on its own.

Both blockers must be cleared for the example to match via `apply_identity`.

## Plan

1. **Subtree pattern variables.**  A slot-less rank-≥1 named `TensorObject` in an
   LHS becomes a wildcard.  `MatchBinding` gains a name→`Expr const*` map; on
   first encounter bind the name to the target subtree, on re-encounter require
   `structural_eq` with the bound value.  `instantiate` substitutes the bound
   subtrees into the RHS.  Wire into both the Expr-matcher (`match`/`instantiate`
   in identity.cpp, used by `apply_identity`) and the e-matcher (egraph.cpp,
   used by `saturate`).  Index variables keep working unchanged.
2. **Binder-to-top canonicalization.**  In `canonicalize`, pull a null-bound
   `ExplicitSum` out of a contraction / tensor-product operand to the term head
   (linearity), fresh-renaming to avoid capture, and order the resulting binder
   stack with the existing Fubini normalization (vibe 000049).
3. Proving example: the `(a⊗b):(c⊗d) = (a·c)(b·d)` identity firing through
   `apply_identity` on the basis-expanded `I:I`.  Then revisit the catalog
   (vibe 000050): the subtree matcher is the keystone that promotes bac-cab and
   `a×I×b` from 🟡 to ✅ (with the ε-pair / δ-substitution work).

## Notes

- `expand_double_dot` (and the other structural steps) already cover specific
  cases *without* this — the value here is **general user-defined identities**,
  not any one reduction.
- Watch the cost of subtree matching: a slot-less wildcard matches *any*
  subtree, so unconstrained patterns could be expensive in the e-graph; keep the
  Expr-matcher (single bottom-up pass) as the first, cheaper consumer and
  measure before turning subtree rules loose in `saturate`.

## Implemented

Both blockers cleared; the user's `(a⊗b):(c⊗d)=(a·c)(b·d)` identity now fires
through `apply_identity` on the basis-expanded `I:I`, and the full `I:I = 3`
reduction runs through that *user-defined* identity.

- **Part 1 — subtree variables** (commit *subtree pattern variables*). A
  slot-less, non-well-known named `TensorObject` in an LHS is a wildcard.
  `MatchBinding` gained a `subtrees` (name→Expr) map + `find_subtree`;
  `match_node`'s `TensorObject` arm binds the whole target subtree
  (rank-checked via `infer_rank` when both ranks are known), consistently via
  `structural_eq`; `instantiate` expands a bound subtree variable in the RHS.
  Well-known (I/δ/ε) and slotted tensors stay literal, so the index identities
  are untouched.  Wired into the **Expr-matcher** (`apply_identity`); the
  **e-matcher (`saturate`) is deliberately left index-only** — no current rule
  needs subtree vars there, and the cost/representation (binding to e-classes)
  is a separate piece for when a saturation rule first needs it.
- **Part 2 — binder-to-top** (commit *binder-to-top canonicalization*).
  `float_sums` (run between `materialize` and `canon` in `canonicalize`) pulls a
  null-bound `ExplicitSum` out of multilinear operand positions to the term
  head — both legs of `TensorProduct`/`Dot`/`DDot`/`DDotAlt`/`Cross`, the
  `ScalarDiv` numerator, and the operand of `Negate`/`Trace`/`VectorInvariant`/
  `Transpose` — fresh-renaming to avoid capture; `Sum`/`Difference` operands and
  a `ScalarDiv` denominator are *not* floatable.  The resulting binder stack is
  Fubini-ordered by the existing `canon_sum_stack` (vibe 000049).

Next within this theme: the e-matcher integration (subtree vars in `saturate`),
which—together with ε-pair contraction in a product—promotes the 🟡 catalog
theorems (bac-cab, `a×I×b`) toward ✅ (vibe 000050).
