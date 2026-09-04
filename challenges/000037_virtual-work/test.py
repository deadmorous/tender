"""Virtual work, and the generalized force it defines.

For a rigid body whose configuration is a generalized coordinate `q`:

    δr = δr_C + δo × (r − r_C)              the virtual displacement
    δA = F·δr = δq · Q_q ,  Q_q = F·∂_q r_C + (ρ × F)·q̂

and equilibrium is `Q_q = 0`, reached by *equating coefficients* — the `δq` are
arbitrary and independent, so `δA = 0` says each coefficient vanishes.  No
integral and no lemma over a domain: for finitely many degrees of freedom that
is the whole of the fundamental lemma, which is why this arc never needed the
definite integral (vibe 000111 owns that separately).

The virtual displacement is the velocity with δ in place of d/dt — literally
the same call with a different derivation — which is the arc's claim showing up
as an economy rather than a slogan.

Two things had to be got right first, and both are the same lesson:

  * **Poisson per coordinate.**  `∂_c P = ĉ × P`, not `D(P) = ω × P`.  The
    operator form is a *sum* for several coordinates and a *product* (`δq ∂_q P`)
    even for one, and neither can be matched inside a contraction chain.
  * **The axis needs a name.**  `ĉ = −½(∂_c P·Pᵀ)_×` mentions `∂_c P`, so a rule
    written with the formula rewrites its own right-hand side — measured, seven
    times before the reduction was stopped.  `q̂` reads as what it is: the axis
    about which `q` turns the body.
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="virtual work: δA = δq (F·∂_q r_C + (ρ × F)·q̂), and equilibrium",
    tier="E",
    source="Gantmacher, virtual work; vibe 000110 I8",
    proves=["triple-rotate"],
)


def _body():
    ws = t.Workspace()
    tm = ws.time("t")
    q, qd = tm.coordinate("q", orders=1)
    P = tm.rotation("P", deps=[q])
    rho = ws.tensor(r"\rho", rank=1)          # fixed in the body
    rC = tm.field("c", 1, deps=[q])           # the reference point
    return ws, tm, q, P, rho, rC, tm.poisson_rules(P)


@harness.level("L2")
def test_the_virtual_displacement_is_the_velocity_with_delta():
    """δr = δr_C + δo × (r − r_C), and v = ṙ_C + ω × (r − r_C).

    The same reduction, the same rules, one call apart.
    """
    ws, tm, q, P, rho, rC, rules = _body()
    ap = td.apply_operators
    r = rC + P @ rho

    virtual = tm.reduce(ap(tm.variation() * r), rules, rounds=10)
    rate = tm.reduce(ap(tm.ddt() * r), rules, rounds=10)
    show("δr", virtual)
    show("v", rate)

    axis = ws.field(r"\hat{q}", 1, deps=[q])
    arm = P @ rho
    qdot = td.apply_operators(tm.ddt() * q)
    dq = tm.variation_of(q)
    harness.assert_algebraic_eq(
        virtual, dq * td.partial(rC, q) + dq * (axis % arm), "δr"
    )
    harness.assert_algebraic_eq(
        rate, qdot * td.partial(rC, q) + qdot * (axis % arm), "v"
    )


@harness.level("L2")
def test_the_generalized_force_is_a_force_and_a_moment():
    """Q_q = F·∂_q r_C + (ρ × F)·q̂ — the moment appears, it is not put in."""
    ws, tm, q, P, rho, rC, rules = _body()
    ap = td.apply_operators
    F = ws.tensor("F", rank=1)

    virtual = tm.reduce(ap(tm.variation() * (rC + P @ rho)), rules, rounds=10)
    work = tm.reduce(F @ virtual, rules + [td.rule("triple-rotate", ws.ctx)], rounds=10)
    show("δA", work)

    coefficients = tm.coefficients(work)
    assert list(coefficients) == [r"\delta{q}"]
    Q = coefficients[r"\delta{q}"]
    show("Q_q", Q)

    axis = ws.field(r"\hat{q}", 1, deps=[q])
    arm = P @ rho
    harness.assert_algebraic_eq(
        Q, F @ td.partial(rC, q) + (axis % arm) @ F, "force term and moment term"
    )

    # The moment term *is* `(ρ × F)·q̂` — the rotation of a scalar triple
    # product — and the library proves that identity on atoms:
    a, b, c = (ws.tensor(n, rank=1) for n in "abc")
    assert td.prove_equal(
        (a % b) @ c, (b % c) @ a, td.rules("cross", "dyadic", ctx=ws.ctx)
    ).proved
    # …but not yet with a *compound* operand in place of `b`: `(q̂ × (P·ρ))·F`
    # and `((P·ρ) × F)·q̂` come back `exhausted`.  A pattern variable binds a
    # whole factor, and here the factor is a contraction — the same reach
    # problem vibe 000100 collects, met once more (vibe 000110 I8).
    assert not td.prove_equal(
        (axis % arm) @ F, (arm % F) @ axis, td.rules("cross", "dyadic", ctx=ws.ctx)
    ).proved


@harness.level("L2")
def test_equilibrium_is_reached_by_equating_coefficients():
    """δA = 0 for arbitrary δq means Q_q = 0 — no integral anywhere.

    The whole of the fundamental lemma for finitely many degrees of freedom.
    """
    ws, tm, q, P, rho, rC, rules = _body()
    ap = td.apply_operators
    F = ws.tensor("F", rank=1)
    work = tm.reduce(
        F @ tm.reduce(ap(tm.variation() * (rC + P @ rho)), rules, rounds=10),
        rules + [td.rule("triple-rotate", ws.ctx)],
        rounds=10,
    )
    equations = tm.coefficients(work)
    show("one equation per degree of freedom", ", ".join(equations))
    assert len(equations) == 1


@harness.level("L1")
def test_a_work_that_is_not_linear_in_the_variations_is_refused():
    """The lemma does not apply to `δq²`, and saying so beats a silent answer."""
    import pytest

    ws, tm, q, P, rho, rC, rules = _body()
    dq = tm.variation_of(q)
    with pytest.raises(ValueError, match="linear"):
        tm.coefficients(dq * dq)
    with pytest.raises(ValueError, match="variations"):
        tm.coefficients(ws.tensor("s", rank=0))
