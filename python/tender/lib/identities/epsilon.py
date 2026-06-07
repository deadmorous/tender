"""Levi-Civita / cross-product identities."""

from tender import Identity, make_pattern_var, cross, dot, tp

# Pattern variables — all rank-1 vectors.
_a = make_pattern_var("a"); _a.constrain_rank(1)
_b = make_pattern_var("b"); _b.constrain_rank(1)
_c = make_pattern_var("c"); _c.constrain_rank(1)

# BAC-CAB rule:  a×(b×c) = b(a·c) − c(a·b)
bac_cab = Identity(
    "BAC-CAB",
    lhs=cross(_a, cross(_b, _c)),
    rhs=tp(_b, dot(_a, _c)) - tp(_c, dot(_a, _b)),
)

# Anti-commutativity:  a×b = -(b×a)
anti_commutativity = Identity(
    "cross-anticomm",
    lhs=cross(_a, _b),
    rhs=-cross(_b, _a),
)

# Note: the "double-cross" identity (a×b)×c = b(a·c) − a(b·c) is derivable
# from bac_cab + anti_commutativity via search_apply and is not listed here.

ALL = [bac_cab, anti_commutativity]
