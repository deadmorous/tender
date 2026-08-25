"""Strain compatibility: inc ε = ∇×(∇×ε)ᵀ equals its closed cross-free form.

The incompatibility tensor of a symmetric strain field ε satisfies

    inc ε = −∇∇θ + Δθ·I − (∇∇··ε)·I − Δε + 2(∇∇·ε)ˢ,      θ = tr ε,

and under inc ε = 0 collapses to the classical Saint-Venant equations.

L1 verifies the closed identity component-by-component in a Cartesian chart.
L2 runs the maintained as-performed derivation (examples/strain_compatibility.py):
∇ expanded with ε abstract, the a×B×c cross-removal identity derived
in-codebase, the reduction reassembled into invariant operators, and the
classical form reached under the trace condition — the example asserts each
stage internally.
"""

import pathlib
import runpy

import tender as t
import tender.derivation as td

from challenges import harness

CHALLENGE = harness.declare(
    title="inc ε = ∇×(∇×ε)ᵀ closed identity + Saint-Venant form",
    tier="E",
    source="Eliseev, Mechanics of Elastic Bodies §compatibility; "
    "examples/strain_compatibility.py (vibes 000075/000078/000080)",
)


@harness.level("L1")
def test_closed_identity_in_cartesian_components():
    ws = t.Workspace()
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    cart, (x, y, z) = ws.cartesian_chart()

    inc = cart.components(cart.rot(cart.rot(eps).transpose()))

    theta = t.tr(eps)
    gg = cart.components(cart.grad(cart.grad(theta)))  # ∇∇θ
    de = cart.components(cart.div(cart.grad(eps)))  # Δε
    gd = cart.components(cart.grad(cart.div(eps)))  # ∇∇·ε
    lap = cart.laplacian(theta)  # Δθ
    dd = cart.div(cart.div(eps))  # ∇∇··ε

    for i in range(3):
        for j in range(3):
            rhs = -gg[i][j] - de[i][j] + gd[i][j] + gd[j][i]
            if i == j:
                rhs = rhs + lap - dd
            harness.assert_chart_zero(
                cart,
                cart.expand(inc[i][j]) - cart.expand(rhs),
                f"inc ε closed identity, component ({i},{j})",
            )


@harness.level("L2")
def test_performed_derivation_via_maintained_example():
    """The full as-performed derivation, asserted stage-by-stage inside the
    maintained example (reduction, reassembly, closed form, classical form,
    plus the cylindrical-chart verification)."""
    example = (
        pathlib.Path(__file__).resolve().parents[2]
        / "examples"
        / "strain_compatibility.py"
    )
    runpy.run_path(str(example), run_name="__main__")
