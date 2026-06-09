"""tender.lib.theorems — proved theorems, verified at import time.

All theorems are constructed (and their proofs run) when this module is first
imported.  A failed proof raises ValueError and makes the module unimportable,
surfacing regressions immediately.
"""

from tender import (
    Theorem,
    tensor, dot, cross, tp,
    State, Derivation,
    wcs,
    expand_in_basis_step,
    simplify_basis_dot_step,
    contract_kronecker_step,
    apply_identity_auto,
)
from tender.lib.identities.epsilon import anti_commutativity, bac_cab as _bac_cab_id

# ---------------------------------------------------------------------------
# Shared tensors used in theorems
# ---------------------------------------------------------------------------

_a = tensor("a", 1)
_b = tensor("b", 1)
_c = tensor("c", 1)

# ---------------------------------------------------------------------------
# dot_commutativity:  a · b = b · a
#
# Proof: abstract component expansion of both sides reduces to the same
# AbstractIndexedSum normal form.
# ---------------------------------------------------------------------------

dot_commutativity = Theorem.by_components(
    "dot_commutativity",
    dot(_a, _b),
    dot(_b, _a),
    lhs_steps=[
        expand_in_basis_step(_a, wcs, covariant=True, abstract=True),
        expand_in_basis_step(_b, wcs, covariant=False, abstract=True),
        simplify_basis_dot_step(wcs),
        contract_kronecker_step(),
    ],
    rhs_steps=[
        expand_in_basis_step(_b, wcs, covariant=False, abstract=True),
        expand_in_basis_step(_a, wcs, covariant=True, abstract=True),
        simplify_basis_dot_step(wcs),
        contract_kronecker_step(),
    ],
)

# ---------------------------------------------------------------------------
# cross_anticommutativity:  a × b = -(b × a)
#
# Proof: apply anti_commutativity identity to lhs; result matches rhs.
# ---------------------------------------------------------------------------

_cross_lhs = cross(_a, _b)
_cross_rhs = -cross(_b, _a)

cross_anticommutativity = Theorem.by_derivation(
    "cross_anticommutativity",
    _cross_lhs,
    _cross_rhs,
    steps=[apply_identity_auto(anti_commutativity, _cross_lhs)],
)

# ---------------------------------------------------------------------------
# bac_cab:  a × (b × c) = b(a · c) − c(a · b)
#
# Proof: apply bac_cab identity to lhs; result matches rhs.
# ---------------------------------------------------------------------------

_bac_lhs = cross(_a, cross(_b, _c))
_bac_rhs = tp(_b, dot(_a, _c)) - tp(_c, dot(_a, _b))

bac_cab = Theorem.by_derivation(
    "bac_cab",
    _bac_lhs,
    _bac_rhs,
    steps=[apply_identity_auto(_bac_cab_id, _bac_lhs)],
)

__all__ = ["dot_commutativity", "cross_anticommutativity", "bac_cab"]
