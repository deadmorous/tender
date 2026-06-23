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

} // namespace tender::nf
