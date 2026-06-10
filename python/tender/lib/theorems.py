"""tender.lib.theorems — proved theorems, verified at import time.

All theorems are constructed (and their proofs run) when this module is first
imported.  A failed proof raises ValueError and makes the module unimportable,
surfacing regressions immediately.
"""

from tender import (
    Theorem,
    tensor, dot, cross, tp, ddot, eps,
    State, Derivation,
    wcs,
    expand_in_basis_step,
    simplify_basis_dot_step,
    contract_kronecker_step,
    expand_levi_civita_first_step,
    contract_eps_pair_step,
    apply_identity_auto,
    _find_and_rewrite_all,
    _capture_step,
)
from tender.lib.identities.definitions import (
    cross_def,
    eps_anticomm,
    cross_def_rev,
    eps_delta,
)

# ---------------------------------------------------------------------------
# Shared tensors used in theorems
# ---------------------------------------------------------------------------

_a = tensor("a", 1)
_b = tensor("b", 1)
_c = tensor("c", 1)

# ---------------------------------------------------------------------------
# eps_delta:  ε:(a⊗(ε:(b⊗c))) = b(a·c) − c(a·b)
#
# Proof (non-circular — first-principles component expansion):
#   Step 1-2: Expand both ε tensors in basis (first the outer, then the inner)
#   Step 3-5: Expand a, b, c in basis (abstract component form)
#   Step 6:   simplify_basis_dot — collapses double-contractions, SBV·SBV → KD
#   Step 7-10: contract 4 Kronecker deltas (from SBV dot products)
#   Step 11:  contract ε-pair sharing one dummy index → δδ − δδ
#   Step 12-15: contract 4 remaining Kronecker deltas
# ---------------------------------------------------------------------------

eps_delta_theorem = Theorem.by_components(
    "eps_delta",
    ddot(eps, tp(_a, ddot(eps, tp(_b, _c)))),
    tp(_b, dot(_a, _c)) - tp(_c, dot(_a, _b)),
    lhs_steps=[
        expand_levi_civita_first_step(wcs),
        expand_levi_civita_first_step(wcs),
        expand_in_basis_step(_a, wcs, covariant=False, abstract=True),
        expand_in_basis_step(_b, wcs, covariant=True, abstract=True),
        expand_in_basis_step(_c, wcs, covariant=True, abstract=True),
        simplify_basis_dot_step(wcs),
        contract_kronecker_step(),
        contract_kronecker_step(),
        contract_kronecker_step(),
        contract_kronecker_step(),
        contract_eps_pair_step(),
        contract_kronecker_step(),
        contract_kronecker_step(),
        contract_kronecker_step(),
        contract_kronecker_step(),
    ],
    rhs_steps=[
        expand_in_basis_step(_b, wcs, covariant=True, abstract=True),
        expand_in_basis_step(_a, wcs, covariant=False, abstract=True),
        expand_in_basis_step(_c, wcs, covariant=True, abstract=True),
        simplify_basis_dot_step(wcs),
        contract_kronecker_step(),
    ],
)

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
# Proof (non-circular — does not use the cross-anticommutativity identity):
#   Step 1: a×b  = ε:(a⊗b)          [cross_def]
#   Step 2: ε:(a⊗b) = -ε:(b⊗a)     [eps_anticomm — antisymmetry of ε]
#   Step 3: -ε:(b⊗a) = -(b×a)       [cross_def_rev inside Scale(-1,…)]
# ---------------------------------------------------------------------------

_cross_lhs = cross(_a, _b)
_cross_rhs = -cross(_b, _a)

_ca_step1 = apply_identity_auto(cross_def, _cross_lhs)
_ca_after1 = Derivation([_ca_step1]).apply(State(_cross_lhs))[-1].expr

_ca_step2 = apply_identity_auto(eps_anticomm, _ca_after1)
_ca_after2 = Derivation([_ca_step1, _ca_step2]).apply(State(_cross_lhs))[-1].expr

_ca_matches3 = list(_find_and_rewrite_all(cross_def_rev, _ca_after2))
_ca_step3 = _capture_step(_ca_matches3[0][1], _ca_matches3[0][0])

cross_anticommutativity = Theorem.by_derivation(
    "cross_anticommutativity",
    _cross_lhs,
    _cross_rhs,
    steps=[_ca_step1, _ca_step2, _ca_step3],
)

# ---------------------------------------------------------------------------
# bac_cab:  a × (b × c) = b(a · c) − c(a · b)
#
# Proof (non-circular — does not use the BAC-CAB identity):
#   Step 1: a×(b×c) = ε:(a⊗(b×c))          [cross_def at root]
#   Step 2: ε:(a⊗(b×c)) = ε:(a⊗(ε:(b⊗c))) [cross_def in subtree]
#   Step 3: ε:(a⊗(ε:(b⊗c))) = b(a·c)-c(a·b) [eps_delta — ε-δ identity]
# ---------------------------------------------------------------------------

_bac_lhs = cross(_a, cross(_b, _c))
_bac_rhs = tp(_b, dot(_a, _c)) - tp(_c, dot(_a, _b))

_bc_step1 = apply_identity_auto(cross_def, _bac_lhs)
_bc_after1 = Derivation([_bc_step1]).apply(State(_bac_lhs))[-1].expr

_bc_matches2 = list(_find_and_rewrite_all(cross_def, _bc_after1))
_bc_step2 = _capture_step(_bc_matches2[0][1], _bc_matches2[0][0])
_bc_after2 = _bc_matches2[0][0]

_bc_step3 = apply_identity_auto(eps_delta, _bc_after2)

bac_cab = Theorem.by_derivation(
    "bac_cab",
    _bac_lhs,
    _bac_rhs,
    steps=[_bc_step1, _bc_step2, _bc_step3],
)

__all__ = ["dot_commutativity", "cross_anticommutativity", "bac_cab", "eps_delta_theorem"]
