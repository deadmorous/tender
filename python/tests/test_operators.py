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


def test_chart_free_nabla_node():
    # vibe 000078 increment 1: t.nabla() is the chart-free ∇ operator *Expr* — a
    # rank-1 invariant vector operator, distinct from the deferred operators.py
    # symbol.  grad/div/rot are the ordinary product nodes with ∇ on the left.
    ctx = t.Context()
    nab = t.nabla(ctx=ctx)
    assert nab.latex() == "\\nabla"
    assert nab.rank == 1
    eps = t.field(r"\varepsilon", 2, symmetric=True, ctx=ctx)
    assert (nab * eps).latex().startswith("\\nabla ")     # grad ∇⊗ε
    assert (nab @ eps).latex().startswith("\\nabla \\cdot")  # div ∇·ε
    assert (nab % eps).latex().startswith("\\nabla \\times")  # rot ∇×ε
    # inc ε = ∇×(∇×ε)ᵀ builds chart-free with ε abstract (no components).
    inc = nab % (nab % eps).transpose()
    assert inc.rank == 2
    assert "varepsilon_{" not in inc.latex()


def test_expand_nabla_free_index_interior():
    # vibe 000078 increment 2: expand_nabla lowers the chart-free inc ε =
    # ∇×(∇×ε)ᵀ to the free-index interior e_i × (e_j × ∂_i∂_j ε)ᵀ — ε stays
    # abstract (no components), only second derivatives appear.
    ctx = t.Context()
    import tender.basis as tb
    import tender.chart as tc
    ref = tb.wcs(ctx)
    x = t.coordinate("x", chart_id=9, slot=0, ctx=ctx)
    y = t.coordinate("y", chart_id=9, slot=1, ctx=ctx)
    z = t.coordinate("z", chart_id=9, slot=2, ctx=ctx)
    cart = tc.CoordinateChart(ref, [x, y, z], [x, y, z])
    eps = t.field(r"\varepsilon", 2, symmetric=True, ctx=ctx)
    nab = t.nabla(ctx=ctx)

    inc = nab % (nab % eps).transpose()
    ex = cart.expand_nabla(inc)
    assert ex.rank == 2
    assert "varepsilon_{" not in ex.latex()   # ε never componentised
    assert "partial" in ex.latex()            # ∂'s present


def test_expand_nabla_components_match_brute_force():
    # The free-index inc ε, componentised, equals brute-force ∇×(∇×ε)ᵀ term by
    # term — the classic strain-compatibility component (∂²_zz ε_yy − 2 ∂²_yz
    # ε_yz + ∂²_yy ε_zz) falls out of the (0,0) entry.
    ctx = t.Context()
    import tender.basis as tb
    import tender.chart as tc
    ref = tb.wcs(ctx)
    x = t.coordinate("x", chart_id=9, slot=0, ctx=ctx)
    y = t.coordinate("y", chart_id=9, slot=1, ctx=ctx)
    z = t.coordinate("z", chart_id=9, slot=2, ctx=ctx)
    cart = tc.CoordinateChart(ref, [x, y, z], [x, y, z])
    eps = t.field(r"\varepsilon", 2, symmetric=True, ctx=ctx)
    nab = t.nabla(ctx=ctx)

    free_form = cart.componentize_nabla(cart.expand_nabla(nab % (nab % eps).transpose()))
    brute = cart.rot(cart.rot(eps).transpose())
    a = cart.components(free_form)
    b = cart.components(brute)
    assert all(
        td.algebraic_eq(cart.expand(a[i][j]), b[i][j])
        for i in range(3)
        for j in range(3)
    )


def test_expand_nabla_nested_operator_compositions():
    # vibe 000078: a composed operator whose operand *itself* contains an
    # unapplied ∇ — grad(div ε), div(div ε), Δε — must apply rightmost-first
    # so the inner ∇ resolves before the outer ∂ differentiates it (regression:
    # this used to throw "differentiating a ∂ operator").  The expanded free
    # form matches the chart-operator composition, component by component.
    ws = t.Workspace()
    cart, _ = _chart(ws)
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    nab = t.nabla(ctx=ws.ctx)

    graddiv_free = cart.componentize_nabla(cart.expand_nabla(nab * (nab @ eps)))
    a = cart.components(graddiv_free)
    b = cart.components(cart.grad(cart.div(eps)))
    assert all(
        td.algebraic_eq(cart.expand(a[i][j]), b[i][j])
        for i in range(3)
        for j in range(3)
    )
