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
//     out;
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
// between two legs (the 000056 fold failure).  Throws if a factor's rank is
// unknown (region placement needs a trustworthy `infer_rank`).  Bound-index
// inference, in-region normalization, and like-term collection are later
// passes.
[[nodiscard]] auto place_factors(Context&, ProductParts const&) -> Term;

} // namespace tender::nf
