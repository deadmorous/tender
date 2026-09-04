"""Zhilin's ω for a finite rotation vector θ n:

    ω = θ̇ n + sin θ · ṅ + (1 − cos θ) · n × ṅ

The hardest thing in vibe 000110, and the best single indicator that the
rotation machinery is real: everything else in the arc is definitional, and
this is the first result that could come out *wrong* rather than merely absent.
It exercises the turn tensor, the unit constraint on `n` together with its
consequence `n·ṅ = 0`, the axial-vector bridge, and the time chain at once.

**It is not reached, and the two routes stop at two different places.**  Both
are recorded here, because each names a capability rather than a mystery, and
between them they say what is actually missing.

*The invariant route* (L2 below) reduces the spin `Ṗ·Pᵀ` of the turn tensor to
within a factor of the answer: the residual is

    sin θ (cos θ − 1) · [ (ṅ × I) + (n × (n × ṅ)) × I ]

and the bracket vanishes because `n × (n × ṅ) = −ṅ`, which the library *does*
prove.  What it cannot do is fold the bracket into that shape: the reduction
leaves `ṅ⊗n − n⊗ṅ`, and turning that back into `(n × ṅ) × I` needs a rule whose
**left-hand side is two terms**, which the matcher cannot compile (the same
limitation `bac-cab-rev` is warned about under).

*The concrete route* (L1) takes a genuinely moving axis — `n = cos φ i +
sin φ j` with φ(t) — so the component procedure can decide everything, and it
**verifies**.  Getting there took one fix, and it was smaller than it looked:
the whole difference collapses to `cos φ (cos²φ + sin²φ − 1)(1 − cos θ)`, and
the Pythagorean fold could not see it because `cos³φ` is a single `Pow` node of
exponent 3, while the fold required exactly 2.  Peeling *two* from any exponent
≥ 2 and leaving the rest in the remainder closes it.

So the formula is verified, on a moving axis, and what is still red is the
*invariant* derivation — a matcher limitation, not a fact in doubt.
"""

import tender as t
import tender.basis as tb
import tender.derivation as td
import tender.identities as ti

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="ω = θ̇ n + sin θ ṅ + (1 − cos θ) n × ṅ for a finite rotation",
    tier="E",
    source="Zhilin, rotation tensors; vibe 000110 I6",
)


def _turn(ws, tm, axis, angle, identity):
    """The turn tensor written out, not stamped: this challenge differentiates
    the formula rather than working with the symbol."""
    nn = axis * axis
    return (
        nn
        + (identity - nn) * t.cos(angle)
        + (axis % identity) * t.sin(angle)
    )


@harness.level("L1")
def test_the_pieces_the_derivation_rests_on_do_hold():
    """`n·ṅ = 0` and `n × (n × ṅ) = −ṅ` — both proved, not assumed.

    Kept green so the red below is known to be about the *last* step and not
    about the setup.
    """
    ws = t.Workspace()
    tm = ws.time("t")
    n = tm.unit_field("n")
    ndot = td.apply_operators(tm.ddt() * n)
    rules = (
        td.rules("rotation", "transpose", "dyadic", "cross", ctx=ws.ctx)
        + ti.constraint_rules(ws.ctx)
        + tm.constraint_rules()
    )
    show("n·ṅ", td.apply_operators(tm.ddt() * (n @ n)))
    assert td.prove_equal(n @ ndot, t.scalar(0, ctx=ws.ctx), rules).proved
    assert td.prove_equal(n % (n % ndot), -ndot, rules).proved


@harness.level(
    "L2",
    expected=False,
    reason="folding ṅ⊗n − n⊗ṅ back into (n × ṅ) × I needs a rule with a "
    "two-term left-hand side, which the matcher cannot compile",
)
def test_the_invariant_derivation():
    """Differentiate the turn tensor, reduce its spin, read off ω."""
    ws = t.Workspace()
    tm = ws.time("t")
    I = ws.identity()
    n = tm.unit_field("n")
    theta = tm.field(r"\theta", 0, deps=[tm.t])
    P = _turn(ws, tm, n, theta, I)

    spin = td.apply_operators(tm.ddt() * P) @ P.transpose()
    reduced = tm.reduce(spin, rounds=25)
    show("spin, reduced", f"{len(reduced.addends())} terms")

    ndot = td.apply_operators(tm.ddt() * n)
    thetadot = td.apply_operators(tm.ddt() * theta)
    omega = (
        thetadot * n
        + t.sin(theta) * ndot
        + (t.scalar(1, ctx=ws.ctx) - t.cos(theta)) * (n % ndot)
    )
    harness.assert_algebraic_eq(reduced, tm.reduce(omega % I), "Zhilin's ω")


@harness.level("L1")
def test_verified_on_a_concrete_moving_axis():
    """n = cos φ i + sin φ j with φ(t): a moving axis, decidable in components."""
    ws = t.Workspace()
    tm = ws.time("t")
    E = ws.wcs()
    i, j, _ = (E.direction(m) for m in range(3))
    I = ws.identity()
    theta = tm.field(r"\theta", 0, deps=[tm.t])
    phi = tm.field(r"\varphi", 0, deps=[tm.t])
    n = t.cos(phi) * i + t.sin(phi) * j
    P = _turn(ws, tm, n, theta, tb.expand_identity(I, E))

    def reduce(e, rounds=10):
        for _ in range(rounds):
            previous = e
            e = td.expand_dyad_ops(td.canonicalize(td.expand_products(e)))
            e = tb.simplify_basis_cross(td.canonicalize(e), E)
            e = tb.simplify_basis_dot(td.canonicalize(e), E)
            e = td.unroll_sums(td.canonicalize(e))
            e = td.fold_arithmetic(
                td.eval_eps_concrete(td.eval_delta_concrete(td.canonicalize(e)))
            )
            e = td.simplify_scalars(td.canonicalize(td.fold_equal_addends(e)))
            e = td.simplify_scalars(td.canonicalize(td.collect_terms(e)))
            if td.structural_eq(e, previous):
                return e
        return e

    spin = reduce(td.apply_operators(tm.ddt() * P) @ P.transpose())
    omega = reduce(t.scalar(t.Rational(-1, 2), ctx=ws.ctx) * spin.vec())
    ndot = td.apply_operators(tm.ddt() * n)
    thetadot = td.apply_operators(tm.ddt() * theta)
    claim = reduce(
        thetadot * n
        + t.sin(theta) * ndot
        + (t.scalar(1, ctx=ws.ctx) - t.cos(theta)) * (n % ndot)
    )
    show("residual", reduce(omega - claim))
    harness.assert_algebraic_eq(omega, claim, "Zhilin's ω, on a moving axis")
