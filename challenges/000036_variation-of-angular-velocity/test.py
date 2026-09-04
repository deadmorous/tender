"""δω = (δo)˙ − ω × δo — the relation that makes rotations variational.

The sharpest test of whether vibe 000110's central claim was built right: δ and
d/dt are one construction handed different derivations, so the variation of an
angular velocity and the rate of a virtual rotation must be related by nothing
more than their commuting.  Everything here follows from that, from Poisson's
`Ṗ = ω × P` and `δP = δo × P`, and from `P·Pᵀ = I`.

Two levels of the same statement, and the difference between them is the
increment's honest boundary:

  * With **one** generalized coordinate the identity holds and is *verified* —
    including its cross term, which the library proves is zero rather than
    being told.  ω and δo are then parallel (both along the same axis), so
    `ω × δo` vanishes identically.
  * With **two**, the cross term is real, and the reduction stalls on the
    *integrability* content: what remains is the statement that
    `∂_r a_q − ∂_q a_r = −a_q × a_r` for the per-coordinate axial vectors, a
    consequence of `∂_q∂_r P = ∂_r∂_q P` (which canon knows — marks are sorted)
    that the rule set cannot yet reach.  See the red below for what it needs.

One thing this increment did change: the differentiated constraints are minted
**per independent variable**, not per operator.  `d/dt P` for a rotation of two
coordinates is `q̇ ∂_q P + ṙ ∂_r P`, so a rule about the whole spin is a rule
about a *sum* — and a multi-term left-hand side is exactly what the matcher
cannot compile.  Each partial spin is skew in its own right, and a sum of skew
terms is skew, so the finer statement is both truer and usable.
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="δω = (δo)˙ − ω × δo",
    tier="E",
    source="Zhilin / Eliseev, virtual rotations; vibe 000110 I8",
)


def _one_coordinate():
    ws = t.Workspace()
    tm = ws.time("t")
    q, qd = tm.coordinate("q", orders=1)
    return ws, tm, tm.rotation("P", deps=[q])


@harness.level("L1")
def test_the_two_derivations_commute_on_the_rotation():
    """δṖ = (δP)˙ — the invariant `ws.time` exists to own, on a rotation."""
    ws, tm, P = _one_coordinate()
    ap = td.apply_operators
    left = ap(tm.variation() * ap(tm.ddt() * P))
    right = ap(tm.ddt() * ap(tm.variation() * P))
    show("δṖ", left)
    harness.assert_algebraic_eq(left, right, "δ and d/dt commute on P")


@harness.level("L2")
def test_the_relation_holds_with_one_generalized_coordinate():
    """δω = (δo)˙ − ω × δo, with the cross term *proved* zero.

    Not assumed away: `ω × δo` is formed and reduced, and it comes out zero
    because both are along the same axis — `a × a = 0` doing the work.
    """
    ws, tm, P = _one_coordinate()
    ap = td.apply_operators
    omega = tm.angular_velocity(P)
    virtual = tm.angular_velocity(P, tm.variation())

    cross = tm.reduce(omega % virtual, rounds=8)
    show("ω × δo", cross)
    harness.assert_algebraic_eq(
        cross, t.scalar(0, ctx=ws.ctx), "ω and δo are parallel here"
    )

    left = tm.reduce(ap(tm.variation() * omega), rounds=12)
    right = tm.reduce(ap(tm.ddt() * virtual) - omega % virtual, rounds=12)
    show("δω", f"{len(left.addends())} terms")
    harness.assert_algebraic_eq(left, right, "δω = (δo)˙ − ω × δo")


@harness.level(
    "L2",
    expected=False,
    reason="the residual is the integrability condition ∂_r a_q − ∂_q a_r = "
    "−a_q × a_r; reaching it needs vec((a×I)·(b×I)) = −(a×b), which is true "
    "and absent, and the axial form of each partial spin",
)
def test_the_relation_holds_with_two_generalized_coordinates():
    """The general case, where `ω × δo` is not zero.

    What stalls is not the identity but a theorem beneath it: the two sides
    differ by terms in `(∂_q P·Pᵀ)_× × (∂_r P·Pᵀ)_×`, and closing them is the
    statement that the per-coordinate axial vectors satisfy
    `∂_r a_q − ∂_q a_r = −a_q × a_r` — which follows from the equality of mixed
    partials (`∂_q∂_r P = ∂_r∂_q P`, which canon already gives) once the axial
    vectors can be related to the spins they came from.
    """
    ws = t.Workspace()
    tm = ws.time("t")
    q, qd = tm.coordinate("q", orders=1)
    r, rd = tm.coordinate("r", orders=1)
    P = tm.rotation("P", deps=[q, r])
    ap = td.apply_operators
    omega = tm.angular_velocity(P)
    virtual = tm.angular_velocity(P, tm.variation())

    left = tm.reduce(ap(tm.variation() * omega), rounds=14)
    right = tm.reduce(ap(tm.ddt() * virtual) - omega % virtual, rounds=14)
    show("residual", str(tm.reduce(left - right, rounds=14))[:120])
    harness.assert_algebraic_eq(left, right, "δω = (δo)˙ − ω × δo")


@harness.level("L1")
def test_mixed_partials_of_a_rotation_agree():
    """∂_q∂_r P = ∂_r∂_q P — the fact the general case rests on.

    Kept green so the red above is known to be about the last step: canon sorts
    the applied-derivative marks, so this holds by normal form.
    """
    ws = t.Workspace()
    tm = ws.time("t")
    q, qd = tm.coordinate("q", orders=1)
    r, rd = tm.coordinate("r", orders=1)
    P = tm.rotation("P", deps=[q, r])
    harness.assert_algebraic_eq(
        td.partial(td.partial(P, q), r),
        td.partial(td.partial(P, r), q),
        "mixed partials agree",
    )
