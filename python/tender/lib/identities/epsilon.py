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

# Companion (CAB-BAC):  (a×b)×c = b(a·c) − a(b·c)
double_cross = Identity(
    "double-cross",
    lhs=cross(cross(_a, _b), _c),
    rhs=tp(_b, dot(_a, _c)) - tp(_a, dot(_b, _c)),
)

ALL = [bac_cab, double_cross]
