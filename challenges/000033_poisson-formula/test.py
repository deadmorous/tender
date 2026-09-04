"""The rate of a rotation is its angular velocity crossed into it.

    Ṗ = ω × P        with     ω = −½ (Ṗ·Pᵀ)_×

and Poisson's formula for a moving frame vector, `ė_k = ω × e_k`, is its
corollary.  Everything here is *derived*: nothing about ω is asserted beyond
the definition, and the definition is a formula, not a hypothesis.

The derivation is three links, and each is the library's own work:

    ω × I      = Ṗ·Pᵀ      the spin is skew (challenge 000032), and a skew
                           tensor is its axial vector crossed into I
    (ω × I)·P  = ω × P     the rank-2 form of the axial vector's action
    (Ṗ·Pᵀ)·P   = Ṗ         orthogonality

Then for a frame vector carried by the rotation, `e_k = P·E_k` with `E_k`
fixed: `ė_k = Ṗ·E_k = (ω × P)·E_k = ω × (P·E_k) = ω × e_k`, the middle step
being the associativity `(a × B)·c = a × (B·c)`.

Four rules arrived with this.  Two are one theorem read in both directions —
`½(A − Aᵀ) = −½(A_×) × I`, which *extracts* an axial vector, and its converse
which *consumes* one — and both are unconditional, for every rank-2 A, which
is what lets the skew case follow without a hypothesis to encode.  The other
two are rank-2 forms of facts the library had only at rank 1, and the second of
those is worth a word: `(a × B)·c = a × (B·c)` looks like plain associativity,
and canon flattens chains of *one* operator so `(A·B)·c` and `A·(B·c)` are
already one form.  It does not flatten across two, and rightly — with a rank-1
middle operand the two groupings are not both well-formed, since `b·c` is a
scalar and `a ×` a scalar is nothing.  Associativity here is rank-conditional,
which a flattening keyed on the operator cannot decide.
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="Ṗ = ω × P, and Poisson's ė = ω × e as its corollary",
    tier="E",
    source="Zhilin, angle-free rotation kinematics; vibe 000110 I6b",
    proves=[
        "skew-decomposition",
        "axial-to-skew",
        "skew-dot-tensor",
        "cross-dot-assoc",
    ],
)


def _setup():
    ws = t.Workspace()
    tm = ws.time("t")
    P = tm.rotation("P")
    return ws, tm, P, ws.identity()


def _rules(ws, tm):
    return (
        td.rules("rotation", "transpose", "dyadic", ctx=ws.ctx)
        + tm.constraint_rules()
    )


@harness.level("L1")
def test_the_four_rules_hold_in_components():
    """Checked with no rules supplied, so the component procedure answers."""
    ws, tm, P, I = _setup()
    A = ws.tensor("A", rank=2)
    B = ws.tensor("B", rank=2)
    a, c = ws.tensor("a", rank=1), ws.tensor("c", rank=1)
    half = t.scalar(t.Rational(1, 2), ctx=ws.ctx)
    mhalf = t.scalar(t.Rational(-1, 2), ctx=ws.ctx)
    for label, lhs, rhs in [
        ("½(A − Aᵀ) = −½(A_×) × I", (A - A.transpose()) * half, (mhalf * A.vec()) % I),
        ("(a × I)·B = a × B", (a % I) @ B, a % B),
        ("(a × B)·c = a × (B·c)", (a % B) @ c, a % (B @ c)),
    ]:
        result = td.prove_equal(lhs, rhs, [])
        show(label, f"{result.status}, components_agree={result.components_agree}")
        assert result.components_agree, f"{label} does not hold in components"


@harness.level("L2")
def test_the_spin_equals_its_axial_vector_crossed_into_the_identity():
    """ω × I = Ṗ·Pᵀ — the link that turns a skew tensor back into a spin.

    Directed rather than saturated, and the reason is worth recording: after
    `axial-to-skew` the spin's transpose sits inside a *parenthesised* sum,
    where no rule reaches it (vibe 000100), so the sum is distributed before
    the skewness fires.
    """
    ws, tm, P, I = _setup()
    rules = {r.name: r for r in _rules(ws, tm)}
    omega, spin = tm.angular_velocity(P), tm.spin(P)
    # The skewness is minted per *independent variable* (vibe 000110 I8), so
    # it is named for the coordinate rather than for the derivation.

    e = td.canonicalize(omega % I)
    show("ω × I", e)
    e = td.canonicalize(
        td.expand_products(td.apply_identity(e, rules["axial-to-skew"]))
    )
    show("after axial-to-skew", e)
    skewness = next(r for r in _rules(ws, tm) if "-spin-" in r.name)
    e = td.canonicalize(td.apply_identity(e, skewness))
    show("after the skewness", e)
    harness.assert_algebraic_eq(e, spin, "ω × I is the spin")


@harness.level("L2")
def test_the_rate_of_a_rotation_is_its_angular_velocity_crossed_into_it():
    """Ṗ = ω × P, derived by `tm.poisson`, which refuses if a link fails."""
    ws, tm, P, I = _setup()
    rule = tm.poisson(P)
    show("derived", f"{rule.lhs} = {rule.rhs}")

    omega = tm.angular_velocity(P)
    rules = _rules(ws, tm)
    # The two links the helper proves by saturation, stated here as well, so
    # the challenge does not merely trust the helper.
    assert td.prove_equal((omega % I) @ P, omega % P, rules).proved
    assert td.prove_equal(tm.spin(P) @ P, td.apply_operators(tm.ddt() * P), rules).proved
    harness.assert_algebraic_eq(
        rule.lhs, td.apply_operators(tm.ddt() * P), "the rule is about Ṗ"
    )


@harness.level("L2")
def test_poissons_formula_for_a_carried_frame_vector():
    """ė_k = ω × e_k for e_k = P·E_k with E_k fixed — the corollary."""
    ws, tm, P, I = _setup()
    E = ws.wcs()
    omega = tm.angular_velocity(P)
    rules = _rules(ws, tm) + [tm.poisson(P)]

    for k in range(3):
        e_k = P @ E.direction(k)
        rate = td.apply_operators(tm.ddt() * e_k)
        result = td.prove_equal(rate, omega % e_k, rules)
        show(f"d/dt (P·E_{k})", f"{result.status}, fired={sorted(result.fired)}")
        assert result.proved


@harness.level("L2")
def test_the_variation_gives_the_same_derivation():
    """δP = δo × P — the same three links, with δ in place of d/dt.

    The claim the whole arc is organised around: the virtual rotation is not
    separate machinery.  `tm.poisson` takes the derivation as an argument, and
    nothing else changes.
    """
    ws = t.Workspace()
    tm = ws.time("t")
    q, qd = tm.coordinate("q", orders=1)
    P = tm.rotation("P", deps=[q])
    rule = tm.poisson(P, tm.variation())
    show("δP", rule.lhs)
    show("δo × P", rule.rhs)
    harness.assert_algebraic_eq(
        rule.lhs,
        td.apply_operators(tm.variation() * P),
        "the rule is about δP",
    )
