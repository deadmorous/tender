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

} // namespace tender::nf
