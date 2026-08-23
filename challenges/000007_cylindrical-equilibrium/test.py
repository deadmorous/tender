"""Equilibrium of a continuous medium in cylindrical coordinates.

For a symmetric stress field T(r, θ, z), the three components of ∇·T in the
physical frame e_r, e_θ, e_z must match the classic textbook equations:

    (∇·T)_r = ∂_r T_rr + (1/r)∂_θ T_rθ + ∂_z T_rz + (T_rr − T_θθ)/r
    (∇·T)_θ = ∂_r T_rθ + (1/r)∂_θ T_θθ + ∂_z T_θz + 2 T_rθ/r
    (∇·T)_z = ∂_r T_rz + (1/r)∂_θ T_θz + ∂_z T_zz + T_rz/r

Performed on the public surface: the abstract field goes straight into
`cyl.div`, which differentiates components AND moving basis vectors via the
frame connection; `cyl.components` surfaces the scalar equations.
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="∇·T in cylindrical coordinates = textbook equilibrium equations",
    tier="E",
    source="Eliseev, Mechanics of Elastic Bodies; examples/cyl_equilibrium.py "
    "(vibe 000073)",
)


@harness.level("L2")
def test_divergence_matches_textbook_equations():
    ws = t.Workspace()
    r, th, z = ws.coords("r", r"\theta", "z", nonneg=("r",))
    cyl = ws.chart(ws.wcs(), [r, th, z], [r * t.cos(th), r * t.sin(th), z])

    T = ws.field("T", 2, symmetric=True)  # T_ij = T_ji, depends on r, θ, z
    div_r, div_th, div_z = cyl.components(cyl.div(T))
    show("(∇·T)_r", div_r)
    show("(∇·T)_θ", div_th)
    show("(∇·T)_z", div_z)

    Tc = cyl.components(T)  # physical components T_ij = e_i·T·e_j
    d = td.partial

    harness.assert_algebraic_eq(
        div_r,
        d(Tc[0][0], r) + d(Tc[0][1], th) / r + d(Tc[0][2], z)
        + (Tc[0][0] - Tc[1][1]) / r,
        "(∇·T)_r",
    )
    harness.assert_algebraic_eq(
        div_th,
        d(Tc[0][1], r) + d(Tc[1][1], th) / r + d(Tc[1][2], z) + 2 * Tc[0][1] / r,
        "(∇·T)_θ",
    )
    harness.assert_algebraic_eq(
        div_z,
        d(Tc[0][2], r) + d(Tc[1][2], th) / r + d(Tc[2][2], z) + Tc[0][2] / r,
        "(∇·T)_z",
    )
