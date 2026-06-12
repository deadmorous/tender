"""Definitional (axiomatic) identity layer for vector/tensor algebra.

These identities are not derived from simpler rules — they define the
primitive operations.  All other identities in the library are consequences
of these axioms.

Hierarchy (see vibe 000017 for design rationale):
    dot product        — primitive; bilinearity + basis evaluation
    trace              — defined by tr(a ⊗ b) = a · b
    identity tensor    — defined by I · a = a
    cross product      — defined by a × b = ε : (a ⊗ b)
    Levi-Civita ε      — defined by anti-symmetry + normalisation
    scalar commutativity — s t = t s  (rank-0 axiom)

Levi-Civita tensor identities (used in theorem proofs):
    eps_anticomm    — ε:(a⊗b) = -ε:(b⊗a)   (antisymmetry of ε)
    cross_def_rev   — ε:(a⊗b) = a×b          (reverse of cross_def; not in ALL)
    eps_delta       — ε:(a⊗(ε:(b⊗c))) = b(a·c) − c(a·b)  (ε-δ identity for BAC-CAB)
"""

from tender import (
    Identity,
    make_pattern_var,
    tensor,
    dot,
    tp,
    cross,
    trace,
    ddot,
    I as identity_tensor,
    eps,
)

# ---------------------------------------------------------------------------
# Pattern variables
# ---------------------------------------------------------------------------

_a = make_pattern_var("a")
_a.constrain_rank(1)
_b = make_pattern_var("b")
_b.constrain_rank(1)
_c = make_pattern_var("c")
_c.constrain_rank(1)
_s = make_pattern_var("s")
_s.constrain_rank(0)
_t_var = make_pattern_var("t")
_t_var.constrain_rank(0)
_u = make_pattern_var("u")
_u.constrain_rank(1)

# ---------------------------------------------------------------------------
# dot_bilinear
#   (a + b) · c = a·c + b·c
# Note: the Sum of a and b is built via pattern matching in search_apply;
# for direct use, state it as an Identity to be applied.
# ---------------------------------------------------------------------------

# Encoded as the corresponding right-hand side pattern identity.
# The LHS is dot(a + b, c) but tender's Identity machinery matches at
# structural level.  We represent dot_bilinear as a bilinearity axiom
# applied to vector sums.
#
# For Phase 13.2 the expand_step() already distributes contractions over
# sums, so dot_bilinear is subsumed by expand_step.  The identity is listed
# here for documentation completeness.

# ---------------------------------------------------------------------------
# tp_contract_right
#   (u ⊗ v) · w  =  (v · w) u
#
# Rank: (1+1) · 1 = 2·1 → 1 on lhs; (0) * 1 = 1 on rhs.
# ---------------------------------------------------------------------------

tp_contract_right = Identity(
    "tp-contract-right",
    lhs=dot(tp(_a, _b), _c),
    rhs=tp(_a, dot(_b, _c)),
)

# ---------------------------------------------------------------------------
# tp_contract_left
#   w · (u ⊗ v)  =  (w · u) v
# ---------------------------------------------------------------------------

tp_contract_left = Identity(
    "tp-contract-left",
    lhs=dot(_c, tp(_a, _b)),
    rhs=tp(dot(_c, _a), _b),
)

# ---------------------------------------------------------------------------
# trace_dyad
#   tr(a ⊗ b) = a · b
# This defines the trace for a dyad; linearity extends to all rank-2 tensors.
# ---------------------------------------------------------------------------

trace_dyad = Identity(
    "trace-dyad",
    lhs=trace(tp(_a, _b)),
    rhs=dot(_a, _b),
)

# ---------------------------------------------------------------------------
# identity_vec (already in identity_tensor module, re-exported here)
#   I · a = a
# ---------------------------------------------------------------------------

identity_vec = Identity(
    "identity-dot-vec",
    lhs=dot(identity_tensor, _a),
    rhs=_a,
)

# ---------------------------------------------------------------------------
# cross_def
#   a × b = ε : (a ⊗ b)
# Double contraction of rank-3 ε with rank-2 (a ⊗ b) → rank-1.
# ---------------------------------------------------------------------------

cross_def = Identity(
    "cross-def",
    lhs=cross(_a, _b),
    rhs=ddot(eps, tp(_a, _b)),
)

# ---------------------------------------------------------------------------
# scalar_comm
#   s t = t s   for rank-0 tensors s, t
# Commutativity of scalar multiplication is an axiom (not derivable without
# index enumeration in the direct-notation framework).
# ---------------------------------------------------------------------------

scalar_comm = Identity(
    "scalar-comm",
    lhs=_s * _t_var,
    rhs=_t_var * _s,
)

# ---------------------------------------------------------------------------
# eps_anticomm
#   ε : (a ⊗ b) = -ε : (b ⊗ a)
# The Levi-Civita tensor is antisymmetric in its first two tensor slots.
# This is an axiomatic property of ε (not derived from cross-product rules).
# Used in the proof of cross_anticommutativity in tender.lib.theorems.
# ---------------------------------------------------------------------------

eps_anticomm = Identity(
    "eps-anticomm",
    lhs=ddot(eps, tp(_a, _b)),
    rhs=-ddot(eps, tp(_b, _a)),
)

# ---------------------------------------------------------------------------
# cross_def_rev
#   ε : (a ⊗ b) = a × b
# Reverse direction of cross_def — used to fold ε:(…) back to a cross product.
# NOT added to ALL: together with cross_def it forms a reversible pair that
# would create cycles in search_apply.
# ---------------------------------------------------------------------------

cross_def_rev = Identity(
    "cross-def-rev",
    lhs=ddot(eps, tp(_a, _b)),
    rhs=cross(_a, _b),
)

# ---------------------------------------------------------------------------
# eps_delta
#   ε : (a ⊗ (ε : (b ⊗ c))) = b(a·c) − c(a·b)
# The ε-δ identity specialised to the BAC-CAB configuration.
# Provable from the total antisymmetry of ε and 3-D Kronecker contraction
# (ε_{ijk} ε_{jlm} = δ_{kl} δ_{im} − δ_{km} δ_{il}).
# Used in the proof of bac_cab in tender.lib.theorems.
# ---------------------------------------------------------------------------

eps_delta = Identity(
    "eps-delta",
    lhs=ddot(eps, tp(_a, ddot(eps, tp(_b, _c)))),
    rhs=tp(_b, dot(_a, _c)) - tp(_c, dot(_a, _b)),
)

# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

ALL = [
    tp_contract_right,
    tp_contract_left,
    trace_dyad,
    identity_vec,
    cross_def,
    scalar_comm,
    eps_anticomm,
    eps_delta,
]
