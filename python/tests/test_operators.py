"""First-class ∇ / ∂_q operators (vibe 000070 P8)."""

import tender as t
import tender.derivation as td
from tender.operators import nabla, d, laplacian, evaluate


def _chart(ws):
    x, y, z = ws.coords("x", "y", "z")
    return ws.chart(ws.wcs(), [x, y, z], [x, y, z]), (x, y, z)


def test_nabla_builds_symbolically():
    # Decision 1/2: ∇ is a chart-free symbol; building stays symbolic.
    ws = t.Workspace()
    cart, _ = _chart(ws)
    f = cart.field("f", 0)
    v = cart.field("v", 1)
    assert (nabla * f).latex() == "\\nabla f"
    assert (nabla @ v).latex() == "\\nabla \\cdot \\mathbf{v}"
    assert (nabla % v).latex() == "\\nabla \\times \\mathbf{v}"
    assert (nabla @ (nabla * f)).latex() == "\\nabla \\cdot \\nabla f"


def test_compound_operands_are_parenthesised():
    # A compound operand (sum, or another cross) is wrapped in parens so the
    # rendering is unambiguous (vibe 000071).
    ws = t.Workspace()
    cart, _ = _chart(ws)
    R = cart.radius_vector()
    I = ws.identity()
    assert "\\left(" in (nabla % R).latex()  # ∇×(x i + y j + z k)
    assert "\\left(" in (nabla % (R % I)).latex()  # ∇×((…) × I)
    # A bare field is not parenthesised.
    f = cart.field("f", 0)
    assert "\\left(" not in (nabla * f).latex()
    assert "\\left(" not in (nabla @ (nabla * f)).latex()


def test_nabla_evaluates_to_m6_operators():
    # The operators are thin wrappers over the chart's M6 operators.
    ws = t.Workspace()
    cart, _ = _chart(ws)
    R = cart.radius_vector()
    v = cart.field("v", 1)
    assert td.structural_eq((nabla * R).evaluate(cart), cart.grad(R))
    assert td.structural_eq((nabla @ R).evaluate(cart), cart.div(R))
    assert td.structural_eq((nabla % v).evaluate(cart), cart.rot(v))
    # Free-function form too.
    assert td.structural_eq(evaluate(nabla * R, cart), cart.grad(R))


def test_laplacian_atom_agrees_with_div_grad():
    # Decision 3: Δ is a citable atom that evaluates through div(grad), so it and
    # nabla @ (nabla * f) agree by construction.
    ws = t.Workspace()
    cart, _ = _chart(ws)
    f = cart.field("f", 0)
    assert laplacian(f).latex() == "\\Delta f"
    assert td.structural_eq(
        laplacian(f).evaluate(cart), (nabla @ (nabla * f)).evaluate(cart)
    )
    assert td.structural_eq(laplacian(f).evaluate(cart), cart.laplacian(f))


def test_partial_operator():
    ws = t.Workspace()
    cart, (x, y, z) = _chart(ws)
    f = cart.field("f", 0)
    assert (d(x) * f).latex() == "\\partial_{x} f"
    assert td.structural_eq((d(x) * f).evaluate(cart), td.partial(f, x))
    assert td.structural_eq(d(x)(f).evaluate(cart), td.partial(f, x))


def test_directional_derivative_custom_operator():
    # The flagship payoff: a custom operator v·∇ built from ∇.  (v·∇)R = v.
    ws = t.Workspace()
    cart, _ = _chart(ws)
    R = cart.radius_vector()
    WCS = ws.wcs()
    v = (
        t.scalar(2, ctx=ws.ctx) * WCS.basis(0)
        + t.scalar(3, ctx=ws.ctx) * WCS.basis(1)
        + WCS.basis(2)
    )
    op = nabla.along(v)
    assert op.latex().endswith("\\cdot \\nabla)")
    assert td.algebraic_eq((op * R).evaluate(cart), v)


def test_first_class_deriv_and_apply_operators():
    # vibe 000077 steps A/B: td.deriv is the unapplied ∂ operator; apply_operators
    # carries out application (Leibniz = commutation).
    ws = t.Workspace()
    x = ws.coords("x")[0]
    f = t.field("f", 0, ctx=ws.ctx)

    dx = td.deriv(x)
    assert dx.latex() == "\\partial_{x}"
    # unapplied: operator then operand (a product), then apply
    assert (dx * f).latex() == "\\partial_{x} \\, f"
    assert td.structural_eq(td.apply_operators(dx * f), td.partial(f, x))
    # ∂_x x = 1
    assert td.structural_eq(
        td.apply_operators(dx * x), t.scalar(1, ctx=ws.ctx))
    # the (∂_x x)·f example: unapplied ∂_x acts greedily → f + x ∂_x f
    greedy = td.apply_operators(dx * x * f)
    expect = f + x * td.partial(f, x)
    assert td.algebraic_eq(greedy, expect)


def test_first_class_nabla_reproduces_grad():
    # vibe 000077 step C: chart.nabla() is the first-class ∇; applying it with ⊗
    # equals the gradient.  nabla.at(chart) exposes the same from the ∇ symbol.
    ws = t.Workspace()
    cart, _ = _chart(ws)
    f = cart.field("f", 0)

    nab = cart.nabla()
    assert "\\partial_{" in nab.latex()          # inspectable, carries ∂ operators
    assert nab.rank == 1                          # a vector operator
    assert td.algebraic_eq(td.apply_operators(nab * f), cart.grad(f))
    # the ∇ symbol's .at bridge gives the same operator.
    assert td.structural_eq(nabla.at(cart), nab)
