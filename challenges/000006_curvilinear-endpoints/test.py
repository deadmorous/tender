"""Curvilinear geometry and operator endpoints, cylindrical and spherical.

Everything is *derived* from the chart embedding (metric → scale factors →
physical frame → connection → operators), and each endpoint is a one-call
evaluation on the public chart surface:

    cyl:  g_θθ = r²,  h_θ = r,  ∇R = I,  ∇·(r e_r) = 2,  Δ(r²) = 4,
          ∇×(r e_θ) = 2 e_z
    sph:  h_φ = r sinθ,  Δ(r²) = 6
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="chart-derived operators: cyl and sph endpoints",
    tier="D",
    source="Borisenko–Tarapov, ch. on curvilinear coordinates; "
    "examples/curvilinear_operators.py",
)


def _cyl(ws):
    r, th, z = ws.coords("r", r"\theta", "z", nonneg=("r",))
    return ws.chart(ws.wcs(), [r, th, z], [r * t.cos(th), r * t.sin(th), z]), r, th, z


@harness.level("L2")
def test_cylindrical_metric_and_scale_factors():
    """g_θθ = r² (needs cos²+sin²→1) and h_θ = √(r²) = r (needs r ≥ 0)."""
    ws = t.Workspace()
    cyl, r, th, z = _cyl(ws)
    show("g_θθ", cyl.metric_component(1, 1))
    show("h_θ", cyl.scale_factor(1))
    harness.assert_algebraic_eq(cyl.metric_component(1, 1), r**2, "g_θθ = r²")
    harness.assert_algebraic_eq(cyl.scale_factor(1), r, "h_θ = r")


@harness.level("L2")
def test_cylindrical_operator_endpoints():
    """∇R = I, ∇·(r e_r) = 2, Δ(r²) = 4, ∇×(r e_θ) = 2 e_z — one call each."""
    ws = t.Workspace()
    cyl, r, th, z = _cyl(ws)
    ctx = ws.ctx
    e = [cyl.physical_frame().direction(k) for k in range(3)]

    grad_R = cyl.grad(cyl.position())
    div_radial = cyl.div(r * e[0])
    lap_r2 = cyl.laplacian(r**2)
    rot_swirl = cyl.rot(r * e[1])

    show("∇R", grad_R)
    show("∇·(r e_r)", div_radial)
    show("Δ(r²)", lap_r2)
    show("∇×(r e_θ)", rot_swirl)

    assert td.structural_eq(grad_R, t.identity(ctx)), "∇R should be I"
    harness.assert_algebraic_eq(div_radial, t.scalar(2, ctx=ctx), "∇·(r e_r)")
    harness.assert_algebraic_eq(lap_r2, t.scalar(4, ctx=ctx), "Δ(r²) in cyl")

    simp = lambda x: td.simplify_scalars(td.expand_products(x))
    harness.assert_algebraic_eq(
        simp(rot_swirl), simp(t.scalar(2, ctx=ctx) * e[2]), "∇×(r e_θ) = 2 e_z"
    )


@harness.level("L2")
def test_spherical_endpoints():
    """h_φ = r sinθ and Δ(r²) = 6, same machinery, spherical chart."""
    ws = t.Workspace()
    r, th, ph = ws.coords("r", r"\theta", r"\phi", nonneg=("r",))
    sph = ws.chart(
        ws.wcs(),
        [r, th, ph],
        [r * t.sin(th) * t.cos(ph), r * t.sin(th) * t.sin(ph), r * t.cos(th)],
    )
    show("h_φ", sph.scale_factor(2))
    show("Δ(r²)", sph.laplacian(r**2))
    harness.assert_algebraic_eq(sph.scale_factor(2), r * t.sin(th), "h_φ = r sinθ")
    harness.assert_algebraic_eq(
        sph.laplacian(r**2), t.scalar(6, ctx=ws.ctx), "Δ(r²) in sph"
    )
