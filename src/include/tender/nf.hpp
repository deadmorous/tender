#pragma once

// Normal form (Nf): the expression model of vibe 000058.
//
// Nf grows *beside* the binary `Expr` of expr.hpp (the surface input language)
// as a parallel IR.  Where `Expr` is a binary tree of operator nodes, an `Nf`
// is an additive set of signed `*`-terms — the "all-`*` canonical form" of
// vibe 000057: every non-`*` operator is pushed inside a composite `Factor`,
// and a term is `coeff · [scalar factors] · [rank-≥1 factors]`, where the
// (signed) rational `coeff` carries the term's sign — there is no separate
// `Sign` field, since `Rational` is already signed.
//
// This header (Stage 1 / C1) introduces the *isolated* data structures, their
// builders, structural equality, and structural hashing.  It has no consumers
// yet: nothing lowers `Expr` into `Nf` or renders it.  Those arrive in later
// commits (see vibe 000058's plan).

#include <tender/context.hpp>
#include <tender/expr.hpp> // TensorObject
#include <tender/index.hpp>
#include <tender/rational.hpp>

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

namespace tender::nf
{

// Forward declarations — composite factors hold `Factor const*` children, and
// a `Paren` factor holds an `Nf const*` body.
struct Factor;
struct Nf;

// ---- term-level scalars ------------------------------------------------

// Per bound-index summation mode.  `Default` follows the realm-driven Einstein
// convention; `Sum` / `NoSum` are explicit overrides, stored as a small field
// on the term (vibe 000057) rather than as `ExplicitSum` / `NoSum` wrapper
// nodes.
enum class SumMode : uint8_t
{
    Default,
    Sum,
    NoSum,
};

// Contraction operator joining two factors of a flat contraction chain.
enum class COp : uint8_t
{
    Dot,     // @  ·
    DDot,    // :
    DDotAlt, // // ··
};

// ---- Factor ------------------------------------------------------------

// A leaf: a named tensor object (reuses `Expr`'s `TensorObject`).  Numeric
// literals never appear here — they fold into the term coefficient.
struct Atom final
{
    TensorObject obj;
};

// A maximal contraction sub-chain over `{@ : //}`, stored *flat*: by the
// interface theorem (vibe 000057) its bracketing is immaterial, so only the
// `(factor, op)` sequence is kept.  `ops.size() == factors.size() - 1`;
// `ops[k]` joins `factors[k]` to `factors[k+1]`.
struct Contraction final
{
    std::vector<Factor const*> factors;
    std::vector<COp> ops;
};

// A `%` (cross) chain among its factors, with the anticommutation sign already
// lifted into the owning term's `coeff`.  Genuinely non-associative among
// rank-1 factors; a rank-≥2 fence makes a run associative (vibe 000055).
struct Cross final
{
    std::vector<Factor const*> factors;
};

// An opaque parenthesised sum: a `Factor` whose interior is a recursively
// canonical `Nf`.  Canon never distributes through it (vibe 000057).
struct Paren final
{
    Nf const* body;
};

// A unary invariant operator applied to a rank-2 factor: `tr(·)` (rank 0),
// `vec(·)` (rank 1), or `(·)^T` (rank 2).  Region is still decided by result
// rank, so a `tr(A)` factor lands among the other scalars.
enum class UnaryOp : uint8_t
{
    Trace,           // tr
    VectorInvariant, // vec
    Transpose,       // ^T
};

struct Unary final
{
    UnaryOp op;
    Factor const* operand;
};

struct Factor final
{
    using Node = std::variant<Atom, Contraction, Cross, Paren, Unary>;

    Node node;

    Factor(Factor const&) = default;
    Factor(Factor&&) noexcept = default;
    auto operator=(Factor const&) -> Factor& = default;
    auto operator=(Factor&&) noexcept -> Factor& = default;

    template <typename T>
        requires(!std::same_as<std::remove_cvref_t<T>, Factor>)
    /*implicit*/ Factor(T&& v) noexcept(
        std::is_nothrow_constructible_v<Node, T&&>) :
      node(std::forward<T>(v))
    {
    }
};

// ---- Term --------------------------------------------------------------

// One inferred dummy index with its summation mode.  (The vibe sketches
// `bound` and `modes` as parallel lists; pairing them keeps them in lockstep.)
struct BoundIndex final
{
    CountableIndex index;
    SumMode mode = SumMode::Default;
};

// A signed `*`-term: `coeff · [scalars] · [tensors]`.
//
//   coeff   : the term's signed rational magnitude (region 1).  `Rational` is
//             signed, so this carries the term's sign — there is no separate
//             `Sign` field, and `A + (-B)` cannot survive as structure.
//   bound   : inferred dummy indices, α-canonical ids, with sum modes.  Their
//             ids are shared with the factor slots they range over.
//   scalars : rank-0 factors (region 2) — commutative, kept sorted.
//   tensors : rank-≥1 factors (region 3) — positional (⊗ is non-commutative).
struct Term final
{
    Rational coeff = Rational{1};
    std::vector<BoundIndex> bound = {};
    std::vector<Factor const*> scalars = {};
    std::vector<Factor const*> tensors = {};
};

// ---- Nf ----------------------------------------------------------------

// An additive set of terms in canonical term order.  An empty term set is the
// zero expression.
struct Nf final
{
    std::vector<Term> terms;
};

// ---- visit -------------------------------------------------------------

// Visit a single Factor; unwraps the `.node` variant so visitors see the
// concrete node types.  Callable via ADL within tender::nf.
template <typename Visitor>
decltype(auto) visit(Visitor&& v, Factor const& f)
{
    return std::visit(std::forward<Visitor>(v), f.node);
}

// ---- builders ----------------------------------------------------------

// All composite nodes are arena-allocated by the Context, matching `Expr`;
// builders return plain `const*` valid for the Context's lifetime.

[[nodiscard]] auto make_atom(Context&, TensorObject) -> Factor const*;

// Precondition: factors non-empty and ops.size() == factors.size() - 1.
[[nodiscard]] auto make_contraction(
    Context&,
    std::vector<Factor const*> factors,
    std::vector<COp> ops) -> Factor const*;

// Precondition: factors non-empty.
[[nodiscard]] auto make_cross(Context&, std::vector<Factor const*> factors)
    -> Factor const*;

[[nodiscard]] auto make_paren(Context&, Nf const* body) -> Factor const*;

[[nodiscard]] auto make_unary(Context&, UnaryOp, Factor const* operand)
    -> Factor const*;

[[nodiscard]] auto make_nf(Context&, std::vector<Term> terms) -> Nf const*;

// ---- structural equality -----------------------------------------------

// Deep structural equality.  Factors compare by node shape and contents;
// terms by coeff, bound (with modes), and both factor regions
// (positionally — scalar ordering is canon's job, not equality's); Nfs by
// their term sequences.  CountableIndex ids are compared exactly (no
// α-renaming — that is canon's job, performed before equality is meaningful).
[[nodiscard]] auto equal(Factor const&, Factor const&) -> bool;
[[nodiscard]] auto equal(Term const&, Term const&) -> bool;
[[nodiscard]] auto equal(Nf const&, Nf const&) -> bool;

[[nodiscard]] inline auto equal(Factor const* a, Factor const* b) -> bool
{
    return a == b || (a && b && equal(*a, *b));
}
[[nodiscard]] inline auto equal(Nf const* a, Nf const* b) -> bool
{
    return a == b || (a && b && equal(*a, *b));
}

// ---- total order -------------------------------------------------------

// Total three-way order (strcmp convention: negative / zero / positive),
// consistent with `equal`: `compare(x, y) == 0` iff `equal(x, y)`.
//
//   Factor : by variant tag (Atom < Contraction < Cross < Paren), then by
//            contents — atoms by `tensor_object_cmp` (the same key as the
//            `Expr` canonical order), composites by their factor/op sequences,
//            parens by body.
//   Term   : by tensors, then scalars, then bound (ids + modes), then coeff —
//            so terms of the same tensor shape sort adjacently.
//   Nf     : lexicographically by term sequence.
//
// Used to sort the commutative scalar region within a term and to order the
// term set; the lowering passes apply it (canon), this commit only defines it.
[[nodiscard]] auto compare(Factor const&, Factor const&) -> int;
[[nodiscard]] auto compare(Term const&, Term const&) -> int;
[[nodiscard]] auto compare(Nf const&, Nf const&) -> int;

// ---- structural hashing ------------------------------------------------

// Hashes consistent with `equal`: equal structures hash equal.  Provided so
// callers can hash-cons or key on `Nf`/`Term`/`Factor` later.
[[nodiscard]] auto hash(Factor const&) -> std::size_t;
[[nodiscard]] auto hash(Term const&) -> std::size_t;
[[nodiscard]] auto hash(Nf const&) -> std::size_t;

} // namespace tender::nf
