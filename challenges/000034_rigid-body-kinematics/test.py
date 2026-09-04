"""A rigid body: composed rotations, velocity, and acceleration.

    ω(P₁·P₂) = ω₁ + P₁·ω₂
    v = ṙ_C + ω × (r − r_C)
    a = a_C + ω̇ × (r − r_C) + ω × (ω × (r − r_C))

All three invariantly, from `r = r_C + P·ρ` with `ρ` fixed in the body and two
applications of d/dt.  Nothing here is a formula quoted from a book: the Euler
and centripetal terms *come out*.

The composition law is the bridge from the rotation increments to kinematics,
because a real problem states an orientation as a product of rotations about
named axes.  Its derivation is where the proper/improper sign finally does
work: conjugating an axial vector, `P·(ω₂ × I)·Pᵀ = (P·ω₂) × I`, holds for a
rotation and flips for a reflection, so the transport rule a symbol mints
depends on what it was declared to be.

The acceleration is derived by differentiating the *velocity*, not the position
twice — which is how one does it by hand, and here it is also what keeps every
rewrite's right-hand side a single term.  A rule whose right side is a sum
cannot yet be spliced into a chain (vibe 000110 I4b), so `P̈ = ε×P + ω×(ω×P)`
would not fire inside `P̈·ρ`; going through the velocity never forms that shape.
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="rigid body: ω₁ + P₁·ω₂, and the velocity and acceleration of a point",
    tier="E",
    source="Zhilin, rigid-body kinematics; vibe 000110 I7",
    proves=["cross-dot-assoc-tensor", "cross-skew", "cross-of-cross-skew"],
)


def _body():
    ws = t.Workspace()
    tm = ws.time("t")
    P = tm.rotation("P")
    omega = tm.angular_velocity(P, name=r"\omega")
    return ws, tm, P, omega, tm.poisson(P)


@harness.level("L2")
def test_the_angular_velocities_of_composed_rotations_add():
    """ω(P·Q) = ω_P + P·ω_Q — the transported sum, derived.

    `Ṗ·Pᵀ` for the product expands to `Ṗ·Pᵀ + P·(Q̇·Qᵀ)·Pᵀ`; the first term is
    `ω_P × I` and the second is the transport of `ω_Q × I` through P.  The sign
    of that transport is the one thing the declaration has to supply.
    """
    ws = t.Workspace()
    tm = ws.time("t")
    I = ws.identity()
    P, Q = tm.rotation("P"), tm.rotation("Q")
    wP = tm.angular_velocity(P, name=r"\omega")
    wQ = tm.angular_velocity(Q, name=r"\varpi")
    extra = [tm.poisson(P), tm.poisson(Q)]

    product = P @ Q
    spin = td.apply_operators(tm.ddt() * product) @ product.transpose()
    show("spin of P·Q, raw", td.canonicalize(td.expand_products(spin)))
    reduced = tm.reduce(spin, extra)
    show("reduced", reduced)
    # Distributed on both sides: the cross over a sum is a rewrite, not a
    # normal form, so `(ω_P + P·ω_Q) × I` has to be spread before comparing.
    harness.assert_algebraic_eq(
        reduced,
        td.canonicalize(td.expand_products((wP + P @ wQ) % I)),
        "ω(P·Q) = ω_P + P·ω_Q",
    )


@harness.level("L2")
def test_the_velocity_of_a_point_of_the_body():
    """v = ṙ_C + ω × (r − r_C), with r − r_C = P·ρ."""
    ws, tm, P, omega, poisson = _body()
    rho = ws.tensor(r"\rho", rank=1)  # fixed in the body
    rC = tm.field("c", 1, deps=[tm.t])
    ap = td.apply_operators

    r = rC + P @ rho
    v = ap(tm.ddt() * r)
    show("v, raw", v)
    v = tm.reduce(v, [poisson])
    show("v, reduced", v)
    harness.assert_algebraic_eq(
        v, ap(tm.ddt() * rC) + omega % (P @ rho), "v = ṙ_C + ω × (r − r_C)"
    )


@harness.level("L2")
def test_the_acceleration_carries_its_euler_and_centripetal_terms():
    """a = a_C + ω̇ × (r − r_C) + ω × (ω × (r − r_C)).

    Two applications of d/dt and the Poisson rule twice; the two named terms
    come out rather than being put in.
    """
    ws, tm, P, omega, poisson = _body()
    rho = ws.tensor(r"\rho", rank=1)
    rC = tm.field("c", 1, deps=[tm.t])
    ap = td.apply_operators

    v = tm.reduce(ap(tm.ddt() * (rC + P @ rho)), [poisson])
    a = tm.reduce(ap(tm.ddt() * v), [poisson])
    show("a", a)

    arm = P @ rho
    epsilon = ap(tm.ddt() * omega)
    expected = (
        ap(tm.ddt() * ap(tm.ddt() * rC))
        + epsilon % arm
        + omega % (omega % arm)
    )
    harness.assert_algebraic_eq(a, expected, "Euler and centripetal terms")


@harness.level("L1")
def test_the_transport_sign_follows_the_declaration():
    """A rotation carries a cross product along; a reflection reverses it.

    The negative half is the point: the licence is attached to what the symbol
    was declared to be, not to the shape of the expression.
    """
    ws = t.Workspace()
    I = ws.identity()
    u = ws.tensor("u", rank=1)
    P = ws.rotation("P")
    Q = ws.orthogonal("Q", proper=False)

    show("proper", td.prove_equal((P @ (u % I)) @ P.transpose(), (P @ u) % I, []).status)
    assert td.prove_equal((P @ (u % I)) @ P.transpose(), (P @ u) % I, []).proved
    assert td.prove_equal(
        (Q @ (u % I)) @ Q.transpose(), -((Q @ u) % I), []
    ).proved
    # …and the rotation's sign is not available to the reflection.
    assert not td.prove_equal(
        (Q @ (u % I)) @ Q.transpose(), (Q @ u) % I, []
    ).proved


@harness.level("L1")
def test_the_two_rules_this_increment_added():
    """Both checked with no rules supplied, so the components answer.

    `cross-skew` earns its place by an accident worth recording: with
    `cross-dot-assoc-tensor` in the set, `(a × I)·(b × I)` can reduce to
    `a × (b × I)` *before* `skew-product` reaches it, and without a way out of
    that shape the turn tensor's verification stopped there.  Two routes into
    one place, so both need an exit.
    """
    ws = t.Workspace()
    I = ws.identity()
    a, b = ws.tensor("a", rank=1), ws.tensor("b", rank=1)
    B, C = ws.tensor("B", rank=2), ws.tensor("C", rank=2)
    for label, lhs, rhs in [
        ("(a × B)·C = a × (B·C)", (a % B) @ C, a % (B @ C)),
        ("a × (b × I) = b⊗a − (a·b) I", a % (b % I), b * a - (a @ b) * I),
        ("(a × b) × I = b⊗a − a⊗b", (a % b) % I, b * a - a * b),
    ]:
        result = td.prove_equal(lhs, rhs, [])
        show(label, f"{result.status}, agree={result.components_agree}")
        assert result.components_agree
