"""Time, the generalized coordinates that move with it, and the variation δ.

The entry point of the applied-mechanics arc (vibe 000093 M5A item 1, brief in
vibe 000110).  Two claims, and neither needs a new algebraic mechanism:

  * **d/dt and δ are ordinary derivations** of the form `Σ_k c_k ∂_k` — d/dt
    with coefficients (1, q̇, q̈, …), δ with coefficients (δq, δq̇, …).  So
    Leibniz, the chain rule through declared dependence, and n-ary products all
    come from `apply_operators`, which has never heard of either operator
    (challenge 000024 certifies that mechanism in general; this is a new
    instance of it).
  * **δ and d/dt commute** — `δ(dL/dt) = d/dt(δL)` — which holds *only* because
    the variations are themselves members of the time chain, `d/dt δq = δq̇`.
    That is the invariant `Workspace.time` exists to own: assembled by hand the
    two operators come out non-commuting, and the vibe records the measurement.

The partial/total distinction that trips every Lagrangian-mechanics text needs
no representation here: it *is* the declared dependence.  `∂L/∂q` holds q̇ and
t fixed because q̇ is a separate declared dependency of L, not a function of q;
`dL/dt` chains because the operator says which coordinates move with t.
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="d/dt and δ are derivations, and they commute",
    tier="E",
    source="Gantmacher, Lectures in Analytical Mechanics §1; vibe 000110",
)


def _setup(orders=2):
    ws = t.Workspace()
    tm = ws.time("t")
    chain = tm.coordinate("q", orders=orders)
    return ws, tm, chain


@harness.level("L1")
def test_the_operators_act_as_declared_on_the_generators():
    """d/dt q = q̇, d/dt q̇ = q̈, δq̇ = δq̇ — and nothing silently zero.

    The chain is closed one order beyond what `coordinate` returns, so the last
    *returned* member still has a true successor rather than differentiating to
    nothing.
    """
    ws, tm, (q, qd, qdd) = _setup()
    ddt, delta = tm.ddt(), tm.variation()
    ap = td.apply_operators

    show("d/dt q", ap(ddt * q))
    show("d/dt q̇", ap(ddt * qd))
    show("d/dt q̈", ap(ddt * qdd))
    harness.assert_algebraic_eq(ap(ddt * q), qd, "d/dt q = q̇")
    harness.assert_algebraic_eq(ap(ddt * qd), qdd, "d/dt q̇ = q̈")
    harness.assert_algebraic_eq(
        ap(delta * qd), tm.variation_of(qd), "δq̇ is the variation of q̇"
    )
    # d/dt of the variation is the variation of the rate — the invariant the
    # commuting rests on.
    harness.assert_algebraic_eq(
        ap(ddt * tm.variation_of(q)),
        tm.variation_of(qd),
        "d/dt δq = δq̇",
    )


@harness.level("L1")
def test_both_are_derivations_over_a_product():
    """δ(fg) = (δf) g + f (δg), and the same for d/dt — no rule registered."""
    ws, tm, (q, qd, qdd) = _setup()
    f = tm.field("f", 0, deps=[q, tm.t])
    g = tm.field("g", 0, deps=[q, tm.t])
    ddt, delta = tm.ddt(), tm.variation()
    ap = td.apply_operators

    show("δ(fg)", ap(delta * (f * g)))
    harness.assert_algebraic_eq(
        ap(delta * (f * g)),
        ap(delta * f) * g + f * ap(delta * g),
        "δ is a derivation",
    )
    harness.assert_algebraic_eq(
        ap(ddt * (f * g)),
        ap(ddt * f) * g + f * ap(ddt * g),
        "d/dt is a derivation",
    )


@harness.level("L2")
def test_delta_and_ddt_commute_on_a_lagrangian():
    """δ(dL/dt) = d/dt(δL) for L(q, q̇, t), on the public surface.

    Every object comes from the factory; the derivation is four calls.  The
    chain rule through the declared dependence is what makes both sides
    non-trivial — L is a field of (q, q̇, t) and of nothing else, so the ∂_q̈
    and ∂_δq terms that would otherwise appear are absent by declaration.
    """
    ws, tm, (q, qd, qdd) = _setup()
    L = tm.field("L", 0, deps=[q, qd, tm.t])
    ddt, delta = tm.ddt(), tm.variation()
    ap = td.apply_operators

    show("dL/dt", ap(ddt * L))
    show("δL", ap(delta * L))
    lhs = ap(delta * ap(ddt * L))
    rhs = ap(ddt * ap(delta * L))
    show("δ(dL/dt)", lhs)
    show("d/dt(δL)", rhs)
    harness.assert_algebraic_eq(lhs, rhs, "δ and d/dt commute")
