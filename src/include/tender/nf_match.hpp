#pragma once

// Matching on the normal form `Nf` (vibe 000058 / C14).
//
// The identity matcher and the e-graph used to match an identity LHS against a
// *whole* canonical `Expr` node.  After the flip (C13) summation binders float
// to the term head, so a product identity (e.g. `Σ_p δ^p_a δ^p_b = δ_ab`) could
// only fire when that product was the *entire* term — it could not reduce a
// sub-product inside a larger term.  Matching on the flat `Nf` term —
// `coeff · scalars[AC] · tensors[positional] · bound{}` — fixes this: the LHS
// factors match a *sub-multiset* of a term's scalars and a *sub-sequence* of
// its tensors (binding a subset of the bound indices), and the rest of the term
// is carried through (the "a%I%b" robustness).
//
// This header (C14a) introduces the binding type and the *whole-term* matcher;
// partial sub-product matching (C14b) and the Nf-native `apply_identity` (C14c)
// build on it.

#include <tender/index.hpp>
#include <tender/nf.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tender::nf
{

// The result of a successful Nf match: pattern index ids paired with the target
// indices they bound, and pattern subtree-variable names paired with the target
// `Factor`s they bound.  Mirrors the `Expr`-level `MatchBinding`, but subtree
// variables bind to `Nf` `Factor`s (not `Expr` subtrees).
struct NfBinding final
{
    std::vector<std::pair<int, IndexAssoc>> indices;
    std::vector<std::pair<std::string, Factor const*>> subtrees;

    // The target index bound to pattern index `id`, or nullopt if unbound.
    [[nodiscard]] auto find(int id) const -> std::optional<IndexAssoc>;
    // The target factor bound to pattern subtree variable `name`, or nullptr.
    [[nodiscard]] auto find_subtree(std::string_view name) const
        -> Factor const*;
};

// Match a single pattern `Factor` against a target `Factor`, extending `bnd`.
// A slot-less, non-well-known `Atom` is a subtree variable that binds the whole
// target factor (rank-checked when both ranks are known); every other factor is
// matched structurally with index/subtree binding.
[[nodiscard]] auto match_factor(
    Factor const* pat, Factor const* tgt, NfBinding& bnd) -> bool;

// Match a whole pattern term against a whole target term: equal coeff, the
// `scalars` regions equal as multisets (bounded AC backtracking), the `tensors`
// regions equal positionally, and the `bound` index lists aligned (same modes,
// pattern ids bound to the target ids).  Returns true on success, extending
// `bnd`.  Partial sub-product matching is C14b.
[[nodiscard]] auto match_term(Term const& pat, Term const& tgt, NfBinding& bnd)
    -> bool;

// A successful *partial* match of a pattern term against a target term: the
// pattern's factors matched a sub-multiset of the target's scalars and a
// contiguous sub-run of its tensors, and its bound indices bound a subset of
// the target's bound dummies.  `leftover` is what remains of the target term
// once the matched pattern instance is removed — the carried-through factors,
// the surviving bound dummies, and `coeff = tgt.coeff / pat.coeff`. `tensor_at`
// is the offset in the target's tensor sequence where the matched run sat, i.e.
// where an instantiated RHS must be spliced back in (⊗ is non-commutative).
struct PartialMatch final
{
    NfBinding binding;
    Term leftover;
    std::size_t tensor_at = 0;
};

// Try to match `pat` as a sub-product of `tgt` (see `PartialMatch`).  Sound
// with respect to summation: a pattern *bound* index may consume a target dummy
// only when that dummy occurs in no leftover factor (tearing a live contraction
// is rejected), and a target dummy a pattern *free* index binds is never
// consumed (the RHS reintroduces it, so it survives into `leftover.bound`).
// Returns the match on success, nullopt otherwise.
[[nodiscard]] auto match_term_partial(Term const& pat, Term const& tgt)
    -> std::optional<PartialMatch>;

// Instantiate a rule's RHS `Nf` under a match binding: replace each pattern
// free index by the target index it bound, and each subtree variable by the
// target `Factor` it bound.  The RHS's own bound dummies are freshened to new
// ids so that splicing the result beside a leftover term (whose surviving
// dummies share the canonical negative id space) cannot collide; a final
// re-canonicalization renames them all back.  Pattern indices/variables absent
// from the binding are left as they are.
[[nodiscard]] auto instantiate_nf(Context&, Nf const* rhs, NfBinding const& bnd)
    -> Nf const*;

// Sub-chain rewrite: when the rule's LHS is a single `Contraction`/`Cross`
// chain factor (no scalars/bound), try to match its factor sequence as a
// contiguous sub-run inside one of `tgt`'s tensor chain factors of the same
// kind, and — on the first match — replace that run in place with the
// instantiated RHS chain, returning the rewritten term.  This reaches a rewrite
// *inside* a flat chain (e.g. `I×x = x×I` on the `I×b` of `a×I×b`) that the
// term-level partial matcher cannot, because the flat `Cross`/`Contraction`
// drops the bracketing that used to expose the sub-run as its own node.  `rhs`
// must likewise be a single same-kind chain factor.  Returns nullopt when the
// rule is not a chain rule or no sub-run matches.
[[nodiscard]] auto rewrite_subchain(
    Context&,
    Term const& lhs_term,
    Nf const* rhs,
    Term const& tgt) -> std::optional<Term>;

// Fire a single-term identity (`lhs_term = rhs`) on one target term `tgt`.  If
// the LHS partially matches a sub-product of `tgt`, return the term(s) that
// replace it — the leftover with the instantiated RHS spliced back in at the
// matched tensor offset (one result term per RHS term).  Otherwise, if the LHS
// is a chain rule matching a contiguous sub-run inside one of `tgt`'s chain
// factors, return the single rewritten term.  Returns nullopt when the rule
// does not fire on `tgt`.  The returned terms are *not* re-canonicalized (the
// RHS dummies are freshened, like terms unmerged); the caller raises and
// re-canonicalizes.  This is the per-term firing shared by `apply_identity` and
// the `NfEGraph` saturation loop.
[[nodiscard]] auto fire_identity_on_term(
    Context&,
    Term const& lhs_term,
    Nf const* rhs,
    Term const& tgt) -> std::optional<std::vector<Term>>;

} // namespace tender::nf
