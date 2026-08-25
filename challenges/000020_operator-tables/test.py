"""Textbook operator tables: grad, div, Δ of general fields in cyl and sph.

For arbitrary f(r,θ,z) / u(r,θ,z) (and spherical counterparts), the chart
operators must reproduce the classic formulas, e.g.

    cyl:  ∇·u = ∂_r u_r + u_r/r + (1/r)∂_θ u_θ + ∂_z u_z
    sph:  Δf  = (1/r²)∂_r(r² ∂_r f) + (1/(r² sinθ))∂_θ(sinθ ∂_θ f)
              + (1/(r² sin²θ))∂²_φ f

Each row is a one-call evaluation on the public chart surface compared with
the hand-written textbook expression.
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="grad/div/Δ tables in cylindrical and spherical charts",
    tier="D",
    source="Borisenko–Tarapov, appendix of curvilinear formulas",
)


def _cyl():
    ws = t.Workspace()
    chart, (r, th, z) = ws.cylindrical_chart()
    return ws, chart, (r, th, z)


def _sph():
    ws = t.Workspace()
    chart, (r, th, ph) = ws.spherical_chart()
    return ws, chart, (r, th, ph)


@harness.level("L2")
def test_cylindrical_table():
    ws, cyl, (r, th, z) = _cyl()
    d = td.partial

    f = ws.field("f", 0)
    fr, fth, fz = (cyl.components(cyl.grad(f))[k] for k in range(3))
    show("(∇f)_r, (∇f)_θ, (∇f)_z", f"{fr.latex()}; {fth.latex()}; {fz.latex()}")
    harness.assert_algebraic_eq(fr, d(f, r), "(∇f)_r")
    harness.assert_algebraic_eq(fth, d(f, th) / r, "(∇f)_θ")
    harness.assert_algebraic_eq(fz, d(f, z), "(∇f)_z")

    u = ws.field("u", 1)
    uc = cyl.components(u)
    divu = cyl.div(u)
    show("∇·u", divu)
    harness.assert_algebraic_eq(
        divu,
        d(uc[0], r) + uc[0] / r + d(uc[1], th) / r + d(uc[2], z),
        "∇·u, cylindrical",
    )

    lap = cyl.laplacian(f)
    show("Δf", lap)
    harness.assert_algebraic_eq(
        lap,
        d(r * d(f, r), r) / r + d(d(f, th), th) / (r * r) + d(d(f, z), z),
        "Δf, cylindrical",
    )


@harness.level("L2")
def test_spherical_table():
    ws, sph, (r, th, ph) = _sph()
    d = td.partial
    sin = t.sin

    f = ws.field("f", 0)
    fr, fth, fph = (sph.components(sph.grad(f))[k] for k in range(3))
    harness.assert_algebraic_eq(fr, d(f, r), "(∇f)_r")
    harness.assert_algebraic_eq(fth, d(f, th) / r, "(∇f)_θ")
    harness.assert_algebraic_eq(fph, d(f, ph) / (r * sin(th)), "(∇f)_φ")

    u = ws.field("u", 1)
    uc = sph.components(u)
    divu = sph.div(u)
    show("∇·u", divu)
    harness.assert_algebraic_eq(
        divu,
        d(r * r * uc[0], r) / (r * r)
        + d(sin(th) * uc[1], th) / (r * sin(th))
        + d(uc[2], ph) / (r * sin(th)),
        "∇·u, spherical",
    )

    lap = sph.laplacian(f)
    show("Δf", lap)
    harness.assert_algebraic_eq(
        lap,
        d(r * r * d(f, r), r) / (r * r)
        + d(sin(th) * d(f, th), th) / (r * r * sin(th))
        + d(d(f, ph), ph) / (r * r * sin(th) * sin(th)),
        "Δf, spherical",
    )
