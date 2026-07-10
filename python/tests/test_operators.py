"""First-class ∇ / ∂_q operators (vibe 000070 P8)."""

import tender as t
import tender.derivation as td
import tender.basis as tb
from tender.operators import nabla, d, laplacian, evaluate


def _chart(ws):
    x, y, z = ws.coords("x", "y", "z")
    return ws.chart(ws.wcs(), [x, y, z], [x, y, z]), (x, y, z)


def _derive(initial, steps):
    der = td.Derivation(initial)
    for s in steps:
        der.step(s)
    return der.current


def _cross_removal_identities(ctx):
    # Derive the a×B×c cross-removal identity in-codebase (vibe 000078 Q3),
    # then the strain interior identity a×(c×E)ᵀ = <cross-free δ/dyad RHS> for a
    # symmetric E — the transpose-cross helper a×(c×E)ᵀ = −a×E×c composed with
    # a×B×c.  Both are proven by construction, not hand-asserted.
    basis = tb.wcs(ctx)
    co = tb.Variance.Covariant
    a = t.tensor("a", 1, ctx=ctx)
    c = t.tensor("c", 1, ctx=ctx)
    B = t.tensor("B", 2, ctx=ctx)
    E = t.field("E", 2, ctx=ctx, symmetric=True)
    I = t.identity(ctx)
    axIxb = _derive(
        a % I % c,
        (
            lambda x: tb.expand_in_basis(x, basis, co),
            lambda x: tb.simplify_basis_cross(x, basis),
            td.contract_eps_pair,
            td.contract_delta,
            lambda x: tb.reassemble(x, basis),
        ),
    )
    id_alt = td.Identity(
        "axIxb_alt", td.fold_equal_addends(axIxb + a @ c * I), a % I % c + a @ c * I
    )
    axBxc = _derive(
        a % B % c,
        (
            lambda x: tb.expand_in_basis(x, basis, co),
            td.apply_identity(id_alt),
            lambda x: tb.expand_in_basis(x, basis, co),
            lambda x: tb.simplify_basis_cross(x, basis),
            lambda x: tb.simplify_basis_dot(x, basis),
            td.contract_delta,
            td.contract_eps_pair,
            td.contract_delta,
            td.contract_eps_pair,
            td.contract_delta,
            lambda x: tb.reassemble(x, basis),
        ),
    )
    id_axBxc = td.Identity("axBxc", a % B % c, axBxc)
    id_inc = td.Identity(
        "inc",
        a % (c % E).transpose(),
        td.canonicalize(-td.apply_identity(id_axBxc)(a % E % c)),
    )
    return id_axBxc, id_inc


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


def test_expand_nabla_double_divergence_and_transpose():
    # ∇·(∇·ε): the two ∂-summation indices must stay distinct (vibe 000078 bug
    # 3a — a premature intermediate canon once aliased them, collapsing it).
    # (∇⊗(∇·ε))ᵀ: a transposed grad-div's ∂_i e_j = 0 term must fold, keeping
    # rank 2 (bug 3c).  Both are reassembly-target building blocks.
    ws = t.Workspace()
    cart, _ = _chart(ws)
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    nab = t.nabla(ctx=ws.ctx)

    def matches(free_expr, chart_expr):
        f = cart.componentize_nabla(cart.expand_nabla(free_expr))
        a = cart.components(f)
        b = cart.components(chart_expr)
        return all(
            td.algebraic_eq(cart.expand(a[i][j]), b[i][j])
            for i in range(3)
            for j in range(3)
        )

    assert cart.expand_nabla(nab @ (nab @ eps)).rank == 0
    assert matches((nab * (nab @ eps)).transpose(), cart.grad(cart.div(eps)).transpose())


def test_expand_nabla_scalar_div_grad_is_laplacian():
    # vibe 000079: ∇·(∇f) for a SCALAR field is the scalar Δf (rank 0), not a
    # rank-2 dyad.  The inner grad ∇f = (∂_i f) e_i is a scalar-scaled frame
    # vector; the outer ∇· must contract e_ℓ with that e_i.  Differentiating the
    # constant e_i leaves a Leibniz connection term `0 ⊗ ∂_i f` of rank 0 beside
    # the real rank-1 term; left in the Sum it made infer_rank misread the
    # operand as a scalar, so make_dot degraded the `·` to `⊗`.  Fixed by folding
    # forced zeros in the deferred derivative.
    ws = t.Workspace()
    cart, _ = _chart(ws)
    f = ws.field("f", 0)
    nab = t.nabla(ctx=ws.ctx)

    expanded = cart.expand_nabla(nab @ (nab * f))
    assert expanded.rank == 0  # Δf is a scalar, not a dyad
    # Componentized and expanded, it is the chart Laplacian Δf.
    comp = cart.componentize_nabla(expanded)
    assert td.algebraic_eq(cart.expand(comp), cart.laplacian(f))
    # And it reassembles back to the operator form ∇·∇f.
    assert td.algebraic_eq(cart.reassemble_nabla(expanded), nab @ (nab * f))


def _closed_identity_holds(chart, eps):
    # inc ε == −∇∇θ + Δθ·I − (∇∇··ε)·I − Δε + 2(∇∇·ε)ˢ , componentwise.  Both
    # sides are coordinate-free tensors, so this must hold in every frame.
    theta = t.tr(eps)
    inc = chart.components(chart.rot(chart.rot(eps).transpose()))
    gg = chart.components(chart.grad(chart.grad(theta)))
    de = chart.components(chart.div(chart.grad(eps)))
    gd = chart.components(chart.grad(chart.div(eps)))
    lap = chart.laplacian(theta)
    dd = chart.div(chart.div(eps))

    def is_zero(e):
        return td.simplify_scalars(td.canonicalize(chart.expand(e))).latex() == "0"

    for i in range(3):
        for j in range(3):
            r = (gg[i][j] * (-1)) + (de[i][j] * (-1)) + gd[i][j] + gd[j][i]
            if i == j:
                r = r + lap - dd
            if not is_zero(chart.expand(inc[i][j]) - chart.expand(r)):
                return False
    return True


def test_strain_compatibility_closed_identity_cartesian():
    # vibe 000078 increment 5: the strain-compatibility closed identity, proven
    # in a Cartesian frame.
    ws = t.Workspace()
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    assert _closed_identity_holds(cart, eps)


def test_strain_compatibility_closed_identity_cylindrical():
    # vibe 000078 increment 5: the same identity in a *curvilinear* (cylindrical)
    # frame — the connection terms fall out of the operators on their own.
    ws = t.Workspace()
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    r, th, z = ws.coords("r", r"\theta", "z", nonneg=("r",))
    cyl = ws.chart(ws.wcs(), [r, th, z], [r * t.cos(th), r * t.sin(th), z])
    assert _closed_identity_holds(cyl, eps)


def test_axbxc_identity_derives():
    # vibe 000078 increment 3: a×B×c cross-removal derives in-codebase to the
    # closed 6-term invariant form (the Q3 recipe), and applying it back to a
    # frame-vector target is cross-free.
    ws = t.Workspace()
    id_axBxc, _ = _cross_removal_identities(ws.ctx)
    B = t.tensor("B", 2, ctx=ws.ctx)
    basis = tb.wcs(ws.ctx)
    i, j = ws.ctx.alloc_index(), ws.ctx.alloc_index()
    ei, ej = basis.covariant_vector(i), basis.covariant_vector(j)
    out = td.apply_identity(id_axBxc)(ei % B % ej)
    assert "times" not in out.latex()  # no cross products remain


def test_strain_phase1_reduction():
    # vibe 000078 increment 3 (Phase-1): the free-index interior
    # inc ε = e_i×(e_j×∂_i∂_j ε)ᵀ reduces — ε abstract — through the derived
    # a×(c×ε)ᵀ identity to a cross-free δ/dyad sum, equal component-by-component
    # to the brute-force interior (the increment-2 oracle).
    ws = t.Workspace()
    cart, _ = _chart(ws)
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    nab = t.nabla(ctx=ws.ctx)
    _, id_inc = _cross_removal_identities(ws.ctx)

    interior = cart.expand_nabla(nab % (nab % eps).transpose())
    phase1 = td.canonicalize(td.apply_identity(id_inc)(interior))
    assert "times" not in phase1.latex()  # Phase-1 is cross-free

    a = cart.components(cart.componentize_nabla(phase1))
    b = cart.components(cart.componentize_nabla(interior))
    assert all(
        td.algebraic_eq(cart.expand(a[i][j]), cart.expand(b[i][j]))
        for i in range(3)
        for j in range(3)
    )


def test_strain_phase2_reassembly():
    # vibe 000078 increment 4 (Phase-2, the heart): reassemble_nabla reads each
    # frame-vector ↔ ∂-mark pair's role in the Phase-1 sum and folds it back into
    # chart-free ∇ operators, yielding the closed compatibility identity
    #   inc ε = −∇∇θ + Δθ·I − (∇∇··ε)I − Δε + ∇∇·ε + (∇∇·ε)ᵀ   (θ = tr ε).
    ws = t.Workspace()
    cart, _ = _chart(ws)
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    nab = t.nabla(ctx=ws.ctx)
    I = t.identity(ws.ctx)
    _, id_inc = _cross_removal_identities(ws.ctx)

    interior = cart.expand_nabla(nab % (nab % eps).transpose())
    phase1 = td.canonicalize(td.apply_identity(id_inc)(interior))
    reass = cart.reassemble_nabla(phase1)

    th = t.tr(eps)
    closed = (
        -(nab @ (nab @ eps)) * I           # −(∇∇··ε) I
        + (nab @ (nab * th)) * I           # +Δθ I
        - (nab @ (nab * eps))              # −Δε
        - (nab * (nab * th))               # −∇∇θ  (scalar Hessian: symmetric, no ᵀ)
        + (nab * (nab @ eps))              # +∇∇·ε
        + (nab * (nab @ eps)).transpose()  # +(∇∇·ε)ᵀ
    )
    assert td.algebraic_eq(reass, closed)


def test_reassemble_scalar_hessian_drops_transpose():
    # vibe 000080 Increment 5: the scalar Hessian ∇∇θ (θ = tr ε) reassembles
    # symmetric — no redundant transpose — while the genuine rank-2 gradient
    # (∇(∇·ε))ᵀ keeps its transpose.
    ws = t.Workspace()
    cart, _ = _chart(ws)
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    nab = t.nabla(ctx=ws.ctx)
    _, id_inc = _cross_removal_identities(ws.ctx)

    interior = cart.expand_nabla(nab % (nab % eps).transpose())
    phase1 = td.canonicalize(td.apply_identity(id_inc)(interior))
    reass = cart.reassemble_nabla(phase1)
    tex = reass.latex()

    # −∇∇θ appears un-transposed …
    assert r"\nabla \, \nabla \, \operatorname{tr}(\boldsymbol{\varepsilon})" in tex
    # … but the true rank-2 gradient (∇(∇·ε))ᵀ still carries a transpose.
    assert r"(\nabla \, (\nabla \cdot \boldsymbol{\varepsilon}))^{\mathsf{T}}" in tex


def test_laplacian_render_recognition():
    # vibe 000080 Increment 3: ∇·(∇⊗X) renders as ΔX, not the misleading ∇².
    ws = t.Workspace()
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    nab = t.nabla(ctx=ws.ctx)
    assert (nab @ (nab * eps)).latex() == r"\Delta \boldsymbol{\varepsilon}"
    th = t.tr(eps)
    assert (nab @ (nab * th)).latex() == r"\Delta \operatorname{tr}(\boldsymbol{\varepsilon})"


def test_div_hooke_stress_reduces_toward_navier_lame():
    # vibe 000080 Increment 8: ∇·T for the isotropic Hooke stress
    #   T = λ(∇·u)I + μ(∇u + (∇u)ᵀ),  u abstract
    # reduces (expand ∇ → apply ∂ by Leibniz → contract e·I → reassemble) to the
    # Navier–Lamé operator form λ∇(∇·u) + μ∇(∇·u) + μ∇·∇u.  Exercises the two
    # infra fixes: apply_operators resolving the inner ∇·u nested in (∇·u)I, and
    # reassemble_nabla carrying the scalar Lamé coefficients through (they were
    # dropped before).  Like-term collection into (λ+μ)∇(∇·u) — the ScalarDiv
    # (sym) route — is the remaining Increment 7(b1)/8 step.
    ws = t.Workspace()
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    nab = t.nabla(ctx=ws.ctx)
    I = t.identity(ws.ctx)
    lam = t.tensor(r"\lambda", 0, ctx=ws.ctx)
    mu = t.tensor(r"\mu", 0, ctx=ws.ctx)
    u = ws.field("u", 1)

    T = lam * (nab @ u) * I + mu * (nab * u + (nab * u).transpose())
    interior = td.contract_identity(td.canonicalize(cart.expand_nabla(nab @ T)))
    reass = cart.reassemble_nabla(td.canonicalize(interior))

    expected = (
        lam * (nab * (nab @ u))   # λ∇(∇·u)
        + mu * (nab * (nab @ u))  # μ∇(∇·u)  (from ∇·((∇u)ᵀ))
        + mu * (nab @ (nab * u))  # μ∇·∇u    (from ∇·(∇u))
    )
    assert td.algebraic_eq(reass, expected)
    # the Lamé constants survive reassembly (they were being dropped).
    assert r"\lambda" in reass.latex() and r"\mu" in reass.latex()


def test_div_of_scalar_times_identity_grad_div():
    # The ∇·((∇·u) I) = ∇(∇·u) piece in isolation: the inner divergence ∇·u sits
    # inside a ⊗-factor, so apply_operators must resolve it before the outer ∂,
    # then e_i·I → e_i and the term reassembles to a grad-div (vibe 000080 Inc 8).
    ws = t.Workspace()
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    nab = t.nabla(ctx=ws.ctx)
    I = t.identity(ws.ctx)
    u = ws.field("u", 1)
    e = nab @ ((nab @ u) * I)
    interior = td.contract_identity(td.canonicalize(cart.expand_nabla(e)))
    reass = cart.reassemble_nabla(td.canonicalize(interior))
    assert td.algebraic_eq(reass, nab * (nab @ u))


def test_canonicalize_keeps_nabla_on_the_left():
    # vibe 000080 Increment 6 (Issue 1): a value-preserving canonical reorder
    # used to leave ∇ on the *right* (∇·(∇·ε) → (∇·ε)·∇, reads as ∇ acting on
    # nothing).  Render-time operator-left normalisation puts it back.
    ws = t.Workspace()
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    nab = t.nabla(ctx=ws.ctx)
    # double divergence: ∇ stays left, no trailing "·∇".
    dd = td.canonicalize(nab @ (nab @ eps)).latex()
    assert dd == r"\nabla \cdot (\nabla \cdot \boldsymbol{\varepsilon})"
    # transpose of a grad-div: ∇ left, as (∇(∇·ε))ᵀ (not (∇·ε)∇).
    gt = td.canonicalize((nab * (nab @ eps)).transpose()).latex()
    assert gt == r"(\nabla (\nabla \cdot \boldsymbol{\varepsilon}))^{\mathsf{T}}"
    # value is unchanged by the reorder.
    assert td.algebraic_eq(td.canonicalize(nab @ (nab @ eps)), nab @ (nab @ eps))


def test_navier_lame_endpoint():
    # vibe 000080 Increment 8 endpoint: ∇·T for the isotropic Hooke stress
    # reduces all the way to the Navier–Lamé operator form μ∇·∇u + ∇((λ+μ)∇·u)
    # (= μΔu + (λ+μ)∇(∇·u)) via expand ∇ → apply ∂ → e·I fold → reassemble →
    # collect_terms → factor_common.
    ws = t.Workspace()
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    nab = t.nabla(ctx=ws.ctx)
    I = t.identity(ws.ctx)
    lam = t.tensor(r"\lambda", 0, ctx=ws.ctx)
    mu = t.tensor(r"\mu", 0, ctx=ws.ctx)
    u = ws.field("u", 1)

    T = lam * (nab @ u) * I + mu * (nab * u + (nab * u).transpose())
    interior = td.contract_identity(td.canonicalize(cart.expand_nabla(nab @ T)))
    reass = cart.reassemble_nabla(td.canonicalize(interior))
    nl = td.factor_common(td.collect_terms(reass))

    assert (
        nl.latex()
        == r"\mu \, \nabla \cdot \nabla \, \mathbf{u} + \nabla (\lambda + \mu) \, \nabla \cdot \mathbf{u}"
    )
    # correctness: factor_common only regroups coefficients, so nl and
    # collect_terms(reass) distribute to the same fully-expanded form.
    ct = td.collect_terms(reass)
    assert td.structural_eq(td.expand_products(nl), td.expand_products(ct))


def _navier_lame_holds(chart, ctx):
    # ∇·T == μ∇·∇u + (λ+μ)∇(∇·u) componentwise, for the isotropic Hooke stress
    # T = λ(∇·u)I + μ(∇u + (∇u)ᵀ).  Both sides are coordinate-free vectors, so
    # this must hold in every frame — the bare-∇-independent endpoint witness.
    lam = t.tensor(r"\lambda", 0, ctx=ctx)
    mu = t.tensor(r"\mu", 0, ctx=ctx)
    I = t.identity(ctx)
    u = t.field("u", 1, ctx=ctx)
    gradu = chart.grad(u)
    stress = lam * chart.div(u) * I + mu * (gradu + gradu.transpose())
    lhs = chart.components(chart.div(stress))
    rhs = chart.components(
        mu * chart.div(chart.grad(u)) + (lam + mu) * chart.grad(chart.div(u))
    )

    def is_zero(e):
        return td.simplify_scalars(td.canonicalize(chart.expand(e))).latex() == "0"

    return all(
        is_zero(chart.expand(lhs[i]) - chart.expand(rhs[i])) for i in range(3)
    )


def test_navier_lame_endpoint_cartesian():
    # vibe 000080 Increment 8: the Navier–Lamé endpoint, proven componentwise in
    # a Cartesian frame (the example witness, guarded in the suite).
    ws = t.Workspace()
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    assert _navier_lame_holds(cart, ws.ctx)


def test_navier_lame_endpoint_cylindrical():
    # vibe 000080 Increment 8: the same endpoint in a *curvilinear* (cylindrical)
    # frame — the connection terms fall out of the operators on their own.
    ws = t.Workspace()
    r, th, z = ws.coords("r", r"\theta", "z", nonneg=("r",))
    cyl = ws.chart(ws.wcs(), [r, th, z], [r * t.cos(th), r * t.sin(th), z])
    assert _navier_lame_holds(cyl, ws.ctx)
