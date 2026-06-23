#pragma once

// Lowering `Expr → Nf` (the canon algorithm of vibe 000058), grown one pass
// per commit.  Nothing here is wired into the public API yet; the
// `canonicalize_nf` entry point and the differential harness arrive at C10.
// Until then these are the internal, individually-tested passes.

#include <tender/expr.hpp>
#include <tender/nf.hpp>

#include <vector>

namespace tender::nf
{

// ---- pass 2: additive flatten (C3) -------------------------------------

// One term of the outermost additive layer: a sign (+1 / -1) and its
// non-additive `body`.  The body is still an `Expr`; its decomposition into
// `Nf` factors (coeff, scalars, tensors) happens in later passes.
struct SignedExpr final
{
    int sign;
    Expr const* body;
};

// Expand the *outermost* additive layer only — `Sum` keeps the sign,
// `Difference` flips it on the right, `Negate` flips it — and **never
// distributes**: a `Sum`/`Difference` sitting inside a product is left intact
// inside `body` as one opaque leaf.  Mirrors derivation.cpp's
// `collect_signed_addends` for the Nf lowering (vibe 000058 pass 2).
[[nodiscard]] auto additive_flatten(Expr const* e) -> std::vector<SignedExpr>;

// ---- pass 3a: multiplicative flatten (C4) ------------------------------

// A signed term split into its numeric magnitude and its non-numeric
// multiplicative factors (region 1 vs the still-undifferentiated regions 2/3).
struct ProductParts final
{
    Rational coeff;                   // sign × all folded literals / divisors
    std::vector<Expr const*> factors; // positional; still raw `Expr`s
};

// Flatten the outermost `*` (`TensorProduct`) chain of `term.body`, folding the
// numeric pieces into `coeff` (seeded by the term's sign):
//   - a `ScalarLiteral` factor multiplies `coeff`;
//   - a `Negate` flips `coeff` and is descended;
//   - a `ScalarDiv` by a `ScalarLiteral` divides `coeff` and its dividend is
//     descended (a non-numeric divisor stays an opaque factor).
// Only `TensorProduct` is flattened — a contraction / cross / sum node is a
// single factor (its encapsulation into an `Nf` `Factor` is C5/C6).  Factor
// order is preserved (⊗ is non-commutative).
[[nodiscard]] auto multiplicative_flatten(SignedExpr const& term)
    -> ProductParts;

// ---- pass 3b: factor encapsulation (C5, C6) ----------------------------

// A `Factor` together with a sign (+1 / -1) lifted out of it during
// encapsulation — cross anticommutation produces such a sign, which must flow
// up to the term's `coeff` (there is no per-factor sign in the model).
struct SignedFactor final
{
    int sign;
    Factor const* factor;
};

// Encapsulate one multiplicative factor (a non-numeric, non-`*` `Expr`) into
// an `Nf` `Factor`, lifting any anticommutation sign:
//   - a bare `TensorObject` becomes an `Atom` (sign +1);
//   - a maximal `{@ : //}` contraction tree becomes a flat `Contraction`, its
//     operands encapsulated recursively and its bracketing dropped (the
//     interface theorem of 000057 makes it immaterial); operand signs multiply
//     out, and a binary *commutative* contraction gets its two operands in
//     canonical order — `·` only between two rank-1 vectors, `:` / `··` only
//     between two rank-2 tensors (higher-rank double contractions like `C:ε`
//     are directional and keep their order);
//   - a `Cross` becomes a `Cross`: a rank-1 pair is ordered canonically with
//     its anticommutation sign lifted (`a×b = -(b×a)`), and a rank-≥2 fence is
//     re-associated `(x×M)×z → x×(M×z)` (000055);
//   - a unary invariant (`Trace` / `VectorInvariant` / `Transpose`) becomes a
//     `Unary` factor over its encapsulated operand (sign passes through).
// Deferred to later commits (encapsulate throws `std::invalid_argument` until
// then): sums → `Paren` and a `⊗` nested inside an operand — these await the
// recursive `lower` / fence distribution assembled around C10.
[[nodiscard]] auto encapsulate(Context&, Expr const* factor) -> SignedFactor;

// ---- pass 4: region placement (C5) -------------------------------------

// Build a `Term` from `ProductParts`: carry `coeff`, then encapsulate each
// factor and place it by `infer_rank` — rank 0 → `scalars`, rank ≥ 1 →
// `tensors`.  This is the step that floats a wedged scalar (`a·b`) out from
// between two legs (the 000056 fold failure).  The commutative `scalars` region
// is then **sorted** into canonical order (tensors stay positional — ⊗ is
// non-commutative).  Throws if a factor's rank is unknown (region placement
// needs a trustworthy `infer_rank`).  Bound-index inference and like-term
// collection are later passes.
[[nodiscard]] auto place_factors(Context&, ProductParts const&) -> Term;

// ---- per-term lowering (passes 3+4+5) ----------------------------------

// Lower one signed term to an `Nf` `Term`, composing the passes above with an
// aggressive ⊗-fence distribution and summation resolution:
//   1. strip the leading `ExplicitSum` / `NoSum` binder stack off the term;
//   2. `distribute_contraction` pushes every `·` / `×` through its adjacent ⊗
//      leg to a fixpoint (`A·(b⊗c) → (A·b)⊗c`), floating the ⊗ up so it joins
//      factors at the top — a ⊗ never stays buried in a contraction operand;
//   3. resolve summation (vibe 000058 / C8): the term's bound indices are its
//      implicit realm contractions (mode `Default`) plus the stripped explicit
//      binders (an explicit Σ is `Default` when it merely confirms the realm
//      default, else `Sum`; a `NoSum` that suppresses a real contraction is a
//      free override kept with its original id).  The summed dummies are
//      α-renamed to canonical (negative) ids via a Fubini-minimizing
//      permutation search (substitution at the Expr level, minimized under Nf
//      `compare`), populating `Term::bound`;
//   4. `multiplicative_flatten` then splits the exposed ⊗ chain;
//   5. `place_factors` encapsulates and region-places.
// Distribution is ⊗-only: a *genuine sum* operand (`A·(b+c)`) is left sunk —
// distributing over a sum stays an explicit, user-invoked transform (000057).
// Such a sum becomes a `Paren`, which still awaits the recursive `lower`, so a
// term carrying a sum factor throws for now.  Explicit binders must sit at the
// term head (as `float_sums` arranges, wired in around C10); a ranged
// `ExplicitSum` (symbolic bound) is likewise deferred and throws.
[[nodiscard]] auto lower_term(Context&, SignedExpr const& term) -> Term;

// ---- pass 6: like-term collection + term-set ordering (C9) --------------

// Collect a flat list of lowered terms into the canonical additive set: sort by
// the like-term key (`compare_term_key` — tensors, scalars, bound, ignoring
// `coeff`), merge equal keys by adding their signed `coeff`s, drop any term
// whose merged `coeff` is zero, and leave the result in canonical term order.
// This is where `a + (−a) → 0` and `2a + 3a → 5a` happen — 000056's sign-drift
// and stray `+ −` artefacts become structurally impossible.  An empty result is
// the zero expression.  Context-free: it only merges already-built `Term`s.
[[nodiscard]] auto collect_terms(std::vector<Term> terms) -> std::vector<Term>;

// ---- entry point: lower `Expr → Nf` (C10) ------------------------------

// Canonicalize a surface `Expr` into the normal form `Nf`: additive flatten →
// per-term `lower_term` → `collect_terms`.  This is the public lowering used by
// the (future) flip at C13; until then it grows beside the existing `Expr`
// `canonicalize`.  Total and deterministic; a genuine-sum factor is canonical-
// ized recursively into a `Paren` (never distributed — 000057).
//
// Not yet handled (throws, via `lower_term` / `encapsulate`): an explicit
// `ExplicitSum` / `NoSum` binder not at a term head (the `Expr` canon's
// `float_sums` prep is not run here yet), a ranged `ExplicitSum`, and a `⊗`
// nested inside a contraction operand that fence distribution has not exposed.
[[nodiscard]] auto canonicalize_nf(Context&, Expr const* e) -> Nf const*;

// ---- raise: Nf → Expr (C12) --------------------------------------------

// Rebuild a surface `Expr` from a normal form `Nf` — the inverse of
// `canonicalize_nf` up to canonicalization: `canonicalize_nf(raise(nf)) == nf`
// for any canonical `nf` (the round-trip / idempotence property that retires
// the "is canon needed again?" question of 000056).  An empty `Nf` raises to
// the literal `0`.  A `Default` bound index stays implicit (its repeated slot
// ids carry the summation); `Sum` / `NoSum` overrides become head binders.
[[nodiscard]] auto raise(Context&, Nf const& nf) -> Expr const*;

} // namespace tender::nf
