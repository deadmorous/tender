"""Identity tensor identities."""

from tender import Identity, make_pattern_var, I, trace, dot

_a = make_pattern_var("a"); _a.constrain_rank(1)
_A = make_pattern_var("A"); _A.constrain_rank(2)

# I·a = a
identity_dot_vec = Identity(
    "identity-dot-vec",
    lhs=dot(I, _a),
    rhs=_a,
)

# I:A = tr(A)
identity_double_contract = Identity(
    "identity-double-contract",
    lhs=(I // _A),
    rhs=trace(_A),
)

ALL = [identity_dot_vec, identity_double_contract]
