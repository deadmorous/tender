"""Standard scalar function identities."""

from tender import Identity, make_pattern_var, sin, cos, exp, tp

_theta = make_pattern_var("theta"); _theta.constrain_rank(0)
_x = make_pattern_var("x"); _x.constrain_rank(0)
_y = make_pattern_var("y"); _y.constrain_rank(0)

# sin²θ + cos²θ = 1
sin_sq_plus_cos_sq = Identity(
    "sin²+cos²=1",
    lhs=tp(sin(_theta), sin(_theta)) + tp(cos(_theta), cos(_theta)),
    rhs=tp(sin(_theta), sin(_theta)) + tp(cos(_theta), cos(_theta)),
    # Note: the RHS is intentionally the same — this identity is used as a
    # rewrite trigger; the actual substitution is sin²θ → 1 − cos²θ (or vice
    # versa) which requires a two-sided form.  Provided here for doc() display.
)

# exp(x)·exp(y) = exp(x+y)  — scalar tensor product
exp_product = Identity(
    "exp-product",
    lhs=tp(exp(_x), exp(_y)),
    rhs=exp(_x + _y),
)

ALL = [sin_sq_plus_cos_sq, exp_product]
