"""A uniformly rotating frame: d/dt e_r = ω k × e_r = ω e_φ.

With i, j, k the fixed World Cartesian directions, t the time, ω a constant,
and the rotation angle φ = ω t,

    e_r = cos φ i + sin φ j          e_φ = −sin φ i + cos φ j

the two rates are

    d/dt e_r = ω k × e_r =  ω e_φ        d/dt e_φ = ω k × e_φ = −ω e_r

The middle member is Poisson's formula, `d/dt e = Ω × e` with the angular
velocity vector Ω = ω k: it says the same thing as the component answer, but
without naming a component — the form the angle-free rotation tensors of
vibe 000093's M5A item 2 will generalise.

Three things make this the natural first challenge of the time arc (Stepan's,
proposed while vibe 000110 was still being written):

  * it needs the **chain rule in time** — d/dt cos(ω t) = −ω sin(ω t) — through
    an angle that is an *expression* in t, not a coordinate of its own;
  * the fixed frame must be **constant in time** while the rotating one is not,
    which is exactly the `nonspatial` reading of time (vibe 000110 I3): i, j, k
    are constants because they are not fields, not because anything was
    declared about t;
  * it is the smallest statement in which a *vector* has a time derivative at
    all, so it is the first place the algebra and the mechanics meet.

Friction recorded rather than hidden: the cross of two concrete frame vectors
does not fold — `k × i` reduces to `−ε_{i13} e_i`, a bound sum, and getting `j`
out of it takes four further public steps (`unroll_sums`, `eval_eps_concrete`,
`fold_arithmetic`, `to_concrete`).  The route is honest and every step is
documented, but a `k × i → j` fold belongs in `simplify_basis_cross`.
"""

import tender as t
import tender.basis as tb
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="rotating frame: d/dt e_r = ω k × e_r = ω e_φ",
    tier="E",
    source="Poisson's formula for a uniformly rotating frame; vibe 000110",
)


def _setup():
    ws = t.Workspace()
    wcs = ws.wcs()
    i, j, k = (wcs.direction(n) for n in range(3))
    tm = ws.time("t")
    omega = ws.tensor("\\omega", rank=0)
    phi = omega * tm.t
    e_r = t.cos(phi) * i + t.sin(phi) * j
    e_phi = -t.sin(phi) * i + t.cos(phi) * j
    return ws, wcs, (i, j, k), tm, omega, (e_r, e_phi)


def _flat(e):
    """Distribute and canonicalize — the common form both sides are read in."""
    return td.canonicalize(td.expand_products(e))


def _reduce_cross(e, frame):
    """Evaluate a cross product of concrete frame vectors.

    `simplify_basis_cross` turns `k × i` into the ε form `−ε_{i13} e_i`; the
    remaining four steps unroll the bound index, evaluate the concrete ε
    symbols, fold the arithmetic, and name the surviving direction.
    """
    e = tb.simplify_basis_cross(td.expand_products(e), frame)
    e = td.unroll_sums(td.canonicalize(e))
    e = td.fold_arithmetic(td.eval_eps_concrete(e))
    return td.canonicalize(tb.to_concrete(td.canonicalize(e), frame))


@harness.level("L1")
def test_the_rates_are_the_other_frame_vector():
    """d/dt e_r = ω e_φ and d/dt e_φ = −ω e_r, in components.

    The only derivation step is the time derivative itself: `tm.ddt()` applied
    by Leibniz, differentiating cos(ω t) by the chain rule through φ = ω t.
    """
    ws, wcs, (i, j, k), tm, omega, (e_r, e_phi) = _setup()
    ddt = tm.ddt()

    dr = td.apply_operators(ddt * e_r)
    dphi = td.apply_operators(ddt * e_phi)
    show("d/dt e_r", dr)
    show("d/dt e_φ", dphi)
    harness.assert_algebraic_eq(_flat(dr), _flat(omega * e_phi), "d/dt e_r")
    harness.assert_algebraic_eq(
        _flat(dphi), _flat(-(omega * e_r)), "d/dt e_φ"
    )


@harness.level("L1")
def test_the_fixed_frame_does_not_move():
    """i, j, k are constant in time — the premise the whole thing rests on."""
    ws, wcs, (i, j, k), tm, omega, _ = _setup()
    zero = t.scalar(0, ctx=ws.ctx)
    for name, vector in (("i", i), ("j", j), ("k", k)):
        harness.assert_algebraic_eq(
            td.apply_operators(tm.ddt() * vector), zero, f"d/dt {name} = 0"
        )


@harness.level("L2")
def test_poissons_formula():
    """d/dt e_r = ω k × e_r, and d/dt e_φ = ω k × e_φ.

    The angular velocity vector is Ω = ω k, and the claim is that the rate of
    each rotating direction is Ω crossed into it — no component of the answer
    named anywhere in the statement.
    """
    ws, wcs, (i, j, k), tm, omega, (e_r, e_phi) = _setup()
    ddt = tm.ddt()
    Omega = omega * k

    for label, vector in (("e_r", e_r), ("e_φ", e_phi)):
        rate = _flat(td.apply_operators(ddt * vector))
        poisson = _reduce_cross(Omega % vector, wcs)
        show(f"d/dt {label}", rate)
        show(f"Ω × {label}", poisson)
        harness.assert_algebraic_eq(
            rate, poisson, f"Poisson's formula for {label}"
        )
