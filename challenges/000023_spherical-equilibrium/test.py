"""Equilibrium in spherical coordinates: ∇·T for a symmetric stress field.

The spherical companion of challenge 000007, and a harder test of the same
machinery: where the cylindrical frame turns only about one axis, the
spherical frame turns about two, so the connection contributes the cot θ terms
that make these equations awkward to derive by hand and easy to mistype.

For a symmetric T(r, θ, φ) the physical components of ∇·T are

  (∇·T)_r = ∂_r T_rr + (1/r)∂_θ T_rθ + (1/(r sinθ))∂_φ T_rφ
            + (2T_rr − T_θθ − T_φφ + T_rθ cot θ)/r

  (∇·T)_θ = ∂_r T_rθ + (1/r)∂_θ T_θθ + (1/(r sinθ))∂_φ T_θφ
            + (3T_rθ + (T_θθ − T_φφ) cot θ)/r

  (∇·T)_φ = ∂_r T_rφ + (1/r)∂_θ T_θφ + (1/(r sinθ))∂_φ T_φφ
            + (3T_rφ + 2T_θφ cot θ)/r

Performed on the public surface: the abstract field goes straight into
`sph.div`, which differentiates components *and* the moving frame via the
connection; `sph.components` reads off the equations.  Nothing is tabulated —
the cot θ terms are derived from the embedding.
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="∇·T in spherical coordinates = textbook equilibrium equations",
    tier="E",
    source="Lurie, Theory of Elasticity, appendix of coordinate formulas",
)


@harness.level("L2")
def test_divergence_matches_textbook_equations():
    ws = t.Workspace()
    sph, (r, th, ph) = ws.spherical_chart()

    T = ws.field("T", 2, symmetric=True)  # T_ij = T_ji, of r, θ, φ
    div_r, div_th, div_ph = sph.components(sph.div(T))
    show("(∇·T)_r", div_r)
    show("(∇·T)_θ", div_th)
    show("(∇·T)_φ", div_ph)

    Tc = sph.components(T)  # physical components T_ij = e_i·T·e_j
    d = td.partial
    sin, cos = t.sin, t.cos
    cot = cos(th) / sin(th)

    harness.assert_algebraic_eq(
        div_r,
        d(Tc[0][0], r) + d(Tc[0][1], th) / r + d(Tc[0][2], ph) / (r * sin(th))
        + (2 * Tc[0][0] - Tc[1][1] - Tc[2][2] + Tc[0][1] * cot) / r,
        "(∇·T)_r",
    )
    harness.assert_algebraic_eq(
        div_th,
        d(Tc[0][1], r) + d(Tc[1][1], th) / r + d(Tc[1][2], ph) / (r * sin(th))
        + (3 * Tc[0][1] + (Tc[1][1] - Tc[2][2]) * cot) / r,
        "(∇·T)_θ",
    )
    harness.assert_algebraic_eq(
        div_ph,
        d(Tc[0][2], r) + d(Tc[1][2], th) / r + d(Tc[2][2], ph) / (r * sin(th))
        + (3 * Tc[0][2] + 2 * Tc[1][2] * cot) / r,
        "(∇·T)_φ",
    )
