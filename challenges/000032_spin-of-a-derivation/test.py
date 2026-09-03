"""Every derivation has a spin, and it is skew because the rotation is one.

For any derivation `D` and any orthogonal `P`, differentiating `P·Pᵀ = I` gives

    D(P)·Pᵀ + P·D(P)ᵀ = 0        ⟹        D(P)·Pᵀ  is skew

so `d/dt` and `δ` are not two constructions but one, handed different
derivations.  `d/dt` gives the angular velocity, `δ` gives the virtual
rotation, and the two spins come out as the same tensor with `q̇` replaced by
`δq` — which is the claim, made visible.

A skew tensor is written `w × I` here, never as a standalone Ω, so the spin's
content is its **axial vector**: `ω = −½ (Ṗ·Pᵀ)_×`, the −½ being the library's
own convention rather than a textbook's (`(a × I)_× = −2a`, measured).

Two things had to be true for any of this to work, and both are recorded here
because both were surprises:

  * **A rotation that turns is a field *and* a constrained symbol.**  Without
    the field dependence `d/dt P` is zero; without the constraint riding on the
    same object, `∂_t P` and `P` — which share a name — are two pattern
    *variables* in one, so every rule relating them binds the same variable
    twice and never fires.
  * **δ reaches a rotation only through the generalized coordinates.**  A
    rotation depending on `t` alone has `δP = 0`, and rightly: the variation
    varies the configuration, not the clock.
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="the spin D(P)·Pᵀ is skew, for d/dt and for δ alike",
    tier="E",
    source="Zhilin, angle-free rotation kinematics; vibe 000110 I6",
)


def _setup():
    ws = t.Workspace()
    tm = ws.time("t")
    q, qd = tm.coordinate("q", orders=1)
    P = tm.rotation("P", deps=[q])
    return ws, tm, q, qd, P


@harness.level("L2")
def test_skewness_is_derived_not_declared():
    """d/dt(P·Pᵀ) = Ṗ·Pᵀ + P·Ṗᵀ, and P·Pᵀ is I, whose derivative is zero.

    So the sum of the spin and its transpose vanishes — no rule about skewness
    is needed to *get* the fact; the rule minted below is the citable record of
    what this derivation proves.
    """
    ws, tm, q, qd, P = _setup()
    I = ws.identity()
    ap = td.apply_operators

    differentiated = ap(tm.ddt() * (P @ P.transpose()))
    show("d/dt(P·Pᵀ)", differentiated)
    show("d/dt(I)", ap(tm.ddt() * I))

    # The left side is the spin plus its transpose …
    spin = tm.spin(P)
    result = td.prove_equal(
        differentiated,
        spin + spin.transpose(),
        td.rules("transpose", ctx=ws.ctx),
    )
    show("d/dt(P·Pᵀ) = spin + spinᵀ", f"{result.status}, fired={result.fired}")
    assert result.proved, "d/dt(P·Pᵀ) should be the spin plus its transpose"
    # … and the right side is zero, because P·Pᵀ is the identity.
    harness.assert_algebraic_eq(
        ap(tm.ddt() * I), t.scalar(0, ctx=ws.ctx), "d/dt I = 0"
    )


@harness.level("L2")
def test_one_construction_two_derivations():
    """The spins of d/dt and δ are the same tensor, `q̇` against `δq`.

    This is the increment's whole claim in one assertion: the virtual rotation
    is not separate machinery, it is the angular velocity's construction handed
    a different derivation.
    """
    ws, tm, q, qd, P = _setup()
    rate = tm.spin(P)
    virtual = tm.spin(P, tm.variation())
    show("Ṗ·Pᵀ", rate)
    show("δP·Pᵀ", virtual)

    substituted = td.canonicalize(
        virtual.replace_at(virtual.find(name=r"\delta{q}")[0], qd)
    )
    harness.assert_algebraic_eq(
        substituted, td.canonicalize(rate), "δq ↦ q̇ carries one spin onto the other"
    )


@harness.level("L2")
def test_both_spins_are_skew():
    ws, tm, q, qd, P = _setup()
    rules = tm.constraint_rules()
    show("minted", ", ".join(r.name for r in rules))
    for label, operator in (("d/dt", tm.ddt()), ("δ", tm.variation())):
        spin = tm.spin(P, operator)
        result = td.prove_equal(spin.transpose(), -spin, rules)
        show(f"({label} spin)ᵀ = −({label} spin)", result.status)
        assert result.proved


@harness.level("L1")
def test_the_axial_vector_carries_the_library_s_own_convention():
    """ω = −½ (Ṗ·Pᵀ)_×, and the −½ is measured, not quoted.

    `(a × I)_× = −2a` in tender's ε and vec, so inverting `w × I` back to `w`
    is a factor of −½.  Checked here on a concrete vector so the convention is
    decidable rather than a matter of taste.
    """
    ws, tm, q, qd, P = _setup()
    E = ws.wcs()
    a = 1 * E.direction(0) + 2 * E.direction(1) + 5 * E.direction(2)
    I = ws.identity()
    skew = td.canonicalize(td.expand_products(a % I))
    recovered = td.simplify(
        t.scalar(t.Rational(-1, 2), ctx=ws.ctx) * skew.vec()
    )
    show("−½ (a × I)_×", recovered)
    harness.assert_components_equal(
        ws.cartesian_chart()[0], recovered, a, "the axial vector of a × I is a"
    )


@harness.level("L1")
def test_a_unit_vector_that_moves_has_n_dot_ndot_zero():
    """The same mechanism as the skewness, which is why they are one increment.

    `d/dt(n·n) = 2 n·ṅ` and `n·n` is 1, so `n·ṅ = 0` — the derivative of a
    constraint is a constraint.
    """
    ws, tm, q, qd, P = _setup()
    n = tm.unit_field("n", deps=[q])
    differentiated = td.apply_operators(tm.ddt() * (n @ n))
    show("d/dt(n·n)", differentiated)
    ndot = td.apply_operators(tm.ddt() * n)
    result = td.prove_equal(
        n @ ndot, t.scalar(0, ctx=ws.ctx), tm.constraint_rules()
    )
    assert result.proved


@harness.level("L1")
def test_a_rotation_of_time_alone_cannot_be_varied():
    """δP = 0 when P depends on the clock and not the configuration.

    Not a limitation — the statement that a variation varies the configuration.
    """
    ws = t.Workspace()
    tm = ws.time("t")
    tm.coordinate("q", orders=1)
    P = tm.rotation("P")  # deps default to time alone
    varied = td.apply_operators(tm.variation() * P)
    show("δP for P(t)", varied)
    harness.assert_algebraic_eq(varied, t.scalar(0, ctx=ws.ctx), "δP(t) = 0")
