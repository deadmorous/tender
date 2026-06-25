#pragma once

#include <tender/expr.hpp>

#include <functional>
#include <string>

namespace tender
{

// An Identity is a directed rewrite rule LHS = RHS over Expr trees (vibe
// 000033).  It is deliberately *not* a Theorem: a theorem is a derivation that
// proves a result and carries its history; an identity is the bare equality
// such a theorem may yield.  The two are separate types — a future Theorem will
// aggregate a Derivation and expose the Identity it proves, never inherit from
// it.
//
// Pattern variables come in two kinds (vibes 000033, 000051):
//
//  - *Index variables* — every free CountableIndex of the LHS matches whatever
//    index sits in the corresponding target slot, consistently across the
//    match. Indices bound by an ExplicitSum/NoSum are local (alpha) variables
//    matched to the target's binder.
//  - *Subtree variables* — a slot-less, non-well-known named TensorObject (e.g.
//    `a`, `A`) matches any whole subtree of compatible rank, consistently. This
//    is what invariant (direct-notation) identities like (a⊗b):(c⊗d)=(a·c)(b·d)
//    need.  Well-known tensors (I, δ, ε) and slotted tensors stay literal.
struct Identity final
{
    std::string name;
    Expr const* lhs;
    Expr const* rhs;
};

// Apply `id.lhs = id.rhs` to `e` on the flat normal form `Nf` (vibe 000058 /
// C14): `e` is canonicalized and lowered to `Nf`, the identity's LHS becomes a
// single pattern term, and it fires at the first target term where the LHS
// matches as a *sub-product* (its factors sit among extra factors of a larger
// term) or as a *sub-chain* (a contiguous run inside a `Contraction`/`Cross`
// factor).  The instantiated RHS replaces the matched part, the rest of the
// term and the other additive terms are carried through, and the result is
// raised back and re-canonicalized.  The return value is always canonical: if
// nothing matched (including a multi-term LHS, which has no Nf sub-sum matcher
// yet) it equals canonicalize(e).  The matching engine is `nf_match`.
[[nodiscard]] auto apply_identity(Context&, Expr const* e, Identity const& id)
    -> Expr const*;

namespace steps
{

// Derivation-step factory: drv.step(steps::apply_identity(id)).  The derivation
// history thus records one entry per identity application.
[[nodiscard]] auto apply_identity(Identity id)
    -> std::function<Expr const*(Context&, Expr const*)>;

} // namespace steps

} // namespace tender
