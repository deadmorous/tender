"""Navier–Lamé: ∇·T for Hooke stress reduces to μ Δu + (λ+μ) ∇(∇·u).

With T = λ(∇·u)I + μ(∇u + (∇u)ᵀ), the equilibrium divergence ∇·T is the
elastic operator acting on u.

L2 performs the reduction with **u abstract** (no components ever appear):
expand ∇ to its free-index frame form, apply the ∂'s by Leibniz, fold e_i·I,
reassemble the ∂-roles back into invariant operators, and factor — the
operator identity  ∇·T = μ ∇·∇u + ∇((λ+μ) ∇·u).

L1 verifies the endpoint component-by-component in a Cartesian chart.
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="∇·T → μΔu + (λ+μ)∇(∇·u), u abstract",
    tier="E",
    source="Lurie, Theory of Elasticity; examples/navier_lame.py (vibe 000080)",
)


def _setup():
    ws = t.Workspace()
    u = ws.field("u", 1)
    nabla = t.nabla(ctx=ws.ctx)
    I = t.identity(ws.ctx)
    lam = t.tensor(r"\lambda", 0, ctx=ws.ctx)
    mu = t.tensor(r"\mu", 0, ctx=ws.ctx)
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    return ws, u, nabla, I, lam, mu, cart


@harness.level("L2")
def test_performed_with_u_abstract():
    ws, u, nabla, I, lam, mu, cart = _setup()
    T = lam * (nabla @ u) * I + mu * (nabla * u + (nabla * u).transpose())
    divT = nabla @ T
    show("T (Hooke stress)", T)
    show("∇·T (coordinate-free)", divT)

    interior = td.contract_identity(td.canonicalize(cart.expand_nabla(divT)))
    assert "u_{" not in interior.latex(), "u must stay abstract"
    show("∇ expanded, u abstract", interior)

    reass = cart.reassemble_nabla(td.canonicalize(interior))
    show("reassembled into ∇ operators", td.collect_terms(reass))

    operator_form = (
        lam * (nabla * (nabla @ u))  # λ∇(∇·u)
        + mu * (nabla * (nabla @ u))  # μ∇(∇·u)   (from ∇·((∇u)ᵀ))
        + mu * (nabla @ (nabla * u))  # μ∇·∇u     (from ∇·(∇u))
    )
    harness.assert_algebraic_eq(reass, operator_form, "reassembly")

    nl = td.factor_common(td.collect_terms(reass))
    show("Navier–Lamé endpoint", nl)
    assert r"\lambda" in nl.latex() and r"\mu" in nl.latex()


@harness.level("L1")
def test_endpoint_verified_in_components():
    """∇·T = μ ∇·∇u + (λ+μ) ∇(∇·u), all 3 Cartesian components equal.

    (The maintained example verifies the same identity in a cylindrical chart
    as well; see examples/navier_lame.py §5.)"""
    ws, u, nabla, I, lam, mu, cart = _setup()

    def hooke_div(chart):
        gradu = chart.grad(u)
        T = lam * chart.div(u) * I + mu * (gradu + gradu.transpose())
        return chart.div(T)

    lhs = hooke_div(cart)
    rhs = mu * cart.div(cart.grad(u)) + (lam + mu) * cart.grad(cart.div(u))
    harness.assert_components_equal(cart, lhs, rhs, "Navier–Lamé, Cartesian")
