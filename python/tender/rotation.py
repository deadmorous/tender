"""tender.rotation — the ways to write a rotation, each verified when built.

Vibe 000110 I5.  An abstract `ws.rotation("P")` says *that* P is orthogonal;
these say *which* orthogonal tensor it is, without giving up the ability to
work with it as one symbol:

    n = ws.vector("n", unit=True)
    Q = ws.reflection("Q", n)              # I − 2 n⊗n,  improper
    P = ws.turn("P", n, theta)             # the turn tensor about n by θ

Each returns a **named symbol** carrying the orthogonality property, with its
formula registered as a defining identity — `ws.definition(P)` — so the algebra
runs on `P` and unfolds to the formula only where the formula is wanted.  That
is how one works by hand, and it is also what makes the property representable:
a property lives on a leaf, and every one of these forms is a *sum*.

**The stamp is earned, not asserted.**  Each constructor reduces `X·Xᵀ` and
`Xᵀ·X` and refuses to name anything that does not reach `I`, showing the
residual.  `ws.orthogonal_from` is the same path for a form the library has
never seen — which is the point, since the list of forms is not closed.

What the verification cannot check is the **sign**: `X·Xᵀ = I` holds for a
reflection as much as for a rotation, and tender has no determinant.  So
`proper=` is your assertion, recorded as one (vibe 000110 I5).

The reduction is *directed*, not a saturation.  Measured: `prove_equal` on the
turn tensor exhausts memory, while the same facts applied as directed rewrites
close in four rounds — vibe 000102's "a transformation, not a pattern",
arriving once more.
"""

from . import derivation as _td
from . import identities as _ti

__all__ = ["reduce_orthogonality", "verify_orthogonal"]

#: Rule groups the verification runs with, plus whatever the context declares.
#:
#: The whole `cross` group is deliberately *not* here: `cross-identity`
#: (`a × I = I × a`) rewrites the skew tensor into a shape the `rotation` rules
#: do not match, and the turn tensor then stalls on `(I × n)·n` — a rewrite
#: that is true, useful elsewhere, and here only takes the expression away from
#: the normal form the other rules expect.  `cross-self` is picked out by name
#: because the reduction genuinely needs it.
_GROUPS = ("rotation", "dyadic", "transpose")

_ROUNDS = 12


def _rules(ctx):
    return (
        _td.rules(*_GROUPS, ctx=ctx)
        + [_td.rule("cross-self", ctx)]
        + _ti.constraint_rules(ctx)
    )


def reduce_orthogonality(ctx, expr, rounds=_ROUNDS):
    """Reduce ``expr·exprᵀ`` as far as the rules and scalar simplifier go.

    Returns the reduced expression — ``I`` when the form is orthogonal.  The
    loop alternates rewriting with re-expansion and scalar simplification
    because each exposes work for the other: `cos²θ + sin²θ → 1` is a *step*,
    not a rule, so no amount of saturation would reach it.
    """
    e = _td.canonicalize(_td.expand_products(expr))
    rules = _rules(ctx)
    for _ in range(rounds):
        previous = e
        for rule in rules:
            e = _td.apply_identity(e, rule)
        e = _td.simplify_scalars(_td.canonicalize(_td.expand_products(e)))
        if _td.structural_eq(e, previous):
            break
    return e


def verify_orthogonal(ctx, expr, identity, rounds=_ROUNDS):
    """``None`` if ``expr`` is orthogonal, else the residual that stopped it.

    Both products are checked: `X·Xᵀ = I` alone leaves a left inverse
    unexamined, and for an expression (as opposed to a declared symbol) the two
    are separate facts until one of them is proved.
    """
    for label, product in (
        ("X·Xᵀ", expr @ expr.transpose()),
        ("Xᵀ·X", expr.transpose() @ expr),
    ):
        reduced = reduce_orthogonality(ctx, product, rounds)
        if not _td.algebraic_eq(reduced, identity):
            return f"{label} reduced to {reduced}, not I"
    return None
