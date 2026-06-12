"""Phase 13.5 vector identity library.

Identities are either:
  - Proved    (validated by a formal derivation in tender.lib.theorems)
  - Derived   (produced by search_apply on the existing identity set)

Tier 1 — fundamental vector relations:
    dot_comm          a·b = b·a      [proved in theorems.dot_commutativity]
    cross_self_zero   a×a = 0
    cross_anticomm    a×b = -(b×a)   [proved in theorems.cross_anticommutativity]

Tier 2 — compound identities:
    double_cross      (a×b)×c = b(a·c) - a(b·c)
    jacobi            a×(b×c) + b×(c×a) + c×(a×b) = 0

See vibes/000017_phase13.5-identity-derivation.md and
vibes/000012_implementation-plan.md for design rationale.
"""

from tender import (
    Identity,
    make_pattern_var,
    cross,
    dot,
    tp,
    rational,
)

# ---------------------------------------------------------------------------
# Pattern variables — all rank-1 vectors
# ---------------------------------------------------------------------------

_a = make_pattern_var("a")
_a.constrain_rank(1)
_b = make_pattern_var("b")
_b.constrain_rank(1)
_c = make_pattern_var("c")
_c.constrain_rank(1)

# ---------------------------------------------------------------------------
# Tier 1 — asserted
# ---------------------------------------------------------------------------

# a·b = b·a
# Proof: expand both sides in the WCS and observe the component sums
# a^i b_i and b^i a_i are identical.
# See examples/dot_commutativity.py for the full derivation.
dot_comm = Identity(
    "dot-comm",
    lhs=dot(_a, _b),
    rhs=dot(_b, _a),
)

# a×a = 0
# Proof: a×a = ε:(a⊗a); since ε_ijk is antisymmetric in i,j while (a⊗a)_ij
# is symmetric, every term in the double contraction cancels.
cross_self_zero = Identity(
    "cross-self-zero",
    lhs=cross(_a, _a),
    rhs=rational(0),
)

# a×b = -(b×a)   [proved in tender.lib.theorems.cross_anticommutativity]
cross_anticomm = Identity(
    "cross-anticomm",
    lhs=cross(_a, _b),
    rhs=-cross(_b, _a),
)

# ---------------------------------------------------------------------------
# Tier 2 — derived
# ---------------------------------------------------------------------------

# (a×b)×c = b(a·c) - a(b·c)
# Proof:
#   (a×b)×c = -(c×(a×b))            [cross_anticomm: x→(a×b), y→c]
#   -(c×(a×b)) = -(a*(c·b) - b*(c·a))  [BAC-CAB: x→c, y→a, z→b]
#             = b*(c·a) - a*(c·b)
#             = b*(a·c) - a*(b·c)    [dot_comm on each scalar factor]
double_cross = Identity(
    "double-cross",
    lhs=cross(cross(_a, _b), _c),
    rhs=tp(_b, dot(_a, _c)) - tp(_a, dot(_b, _c)),
)

# a×(b×c) + b×(c×a) + c×(a×b) = 0   (Jacobi identity)
# Proof: apply BAC-CAB to each term and observe that all six tensor products
# cancel pairwise:
#   a×(b×c) = b(a·c) - c(a·b)
#   b×(c×a) = c(b·a) - a(b·c)
#   c×(a×b) = a(c·b) - b(c·a)
# Sum = 0 (using dot_comm to match scalar factors).
_zero = rational(0)
jacobi = Identity(
    "jacobi",
    lhs=cross(_a, cross(_b, _c)) + cross(_b, cross(_c, _a)) + cross(_c, cross(_a, _b)),
    rhs=_zero,
)

# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

ALL = [dot_comm, cross_self_zero, cross_anticomm, double_cross, jacobi]
