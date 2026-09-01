"""The core ∇ operator: building it, expanding it into a frame, lowering it
onto a chart, and reassembling the result back into ∇ form.

Named for `tender.operators` once, which was a deferred-evaluation DSL over
these same capabilities; that module is now in the attic and everything here
uses the core route — `t.nabla()`, a real `Expr`.
"""

import re

import tender as t
import tender.derivation as td
import tender.basis as tb
import tender.chart as tc


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
            id_alt,
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
        td.canonicalize(-td.apply_identity(a % E % c, id_axBxc)),
    )
    return id_axBxc, id_inc


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
    out = td.apply_identity(ei % B % ej, id_axBxc)
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
    phase1 = td.canonicalize(td.apply_identity(interior, id_inc))
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
    # dimension-awareness is identity-neutral (vibe 000081), so a plain I here
    # compares equal to reassemble_nabla's dimensioned one.
    I = t.identity(ws.ctx)
    _, id_inc = _cross_removal_identities(ws.ctx)

    interior = cart.expand_nabla(nab % (nab % eps).transpose())
    phase1 = td.canonicalize(td.apply_identity(interior, id_inc))
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


def test_reassemble_nabla_dimensions_identity_and_trace_folds():
    # vibe 000081 B1: the derivation keeps the dimension-agnostic identity (the
    # basis machinery needs the slotless form), but reassemble_nabla stamps the
    # chart's dimension onto the I in its invariant output — so tr(inc ε) folds
    # tr(c·I)→c·n and the whole thing reduces to Δ tr(ε) − ∇·(∇·ε).  The
    # dimensioned I still renders as a clean `I` (no slot decoration).
    ws = t.Workspace()
    cart, _ = _chart(ws)
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    nab = t.nabla(ctx=ws.ctx)
    _, id_inc = _cross_removal_identities(ws.ctx)  # bare I inside

    interior = cart.expand_nabla(nab % (nab % eps).transpose())
    phase1 = td.canonicalize(td.apply_identity(interior, id_inc))
    reass = cart.reassemble_nabla(phase1)
    assert reass.latex().count(r"\mathbf{I}") == 2  # clean I, no ^{•·}_{·•}

    tr_inc = td.fold_equal_addends_structural(td.expand_dyad_ops(t.tr(reass)))
    theta = t.tr(eps)
    target = nab @ (nab * theta) - nab @ (nab @ eps)  # Δ tr ε − ∇·(∇·ε)
    assert td.fold_equal_addends_structural(tr_inc - target).latex() == "0"


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
    phase1 = td.canonicalize(td.apply_identity(interior, id_inc))
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


def test_invariant_laplacian_constructor():
    # vibe 000083 Part B: t.laplacian(X) is the official invariant Laplacian
    # ΔX = ∇·(∇⊗X) — no new node, just a thin constructor over nabla@(nabla*X).
    ws = t.Workspace()
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    nab = t.nabla(ctx=ws.ctx)

    # scalar operand → Δ tr(ε), rank-2 operand → Δε
    assert t.laplacian(t.tr(eps)).latex() == r"\Delta \operatorname{tr}(\boldsymbol{\varepsilon})"
    assert t.laplacian(eps).latex() == r"\Delta \boldsymbol{\varepsilon}"

    # structurally identical to the explicit ∇·(∇⊗X) it abbreviates
    assert td.structural_eq(t.laplacian(t.tr(eps)), nab @ (nab * t.tr(eps)))
    assert td.structural_eq(t.laplacian(eps), nab @ (nab * eps))


def test_canonicalize_preserves_nabla_laplacian_nesting():
    # vibe 000085: canonicalize must NOT float the abstract-∇ operator fence
    # ∇·(∇⊗X) into (∇·∇)⊗X.  That float is a moving-frame correctness trap and,
    # crucially, irreversible — a canonicalize done automatically at the start of
    # some later step would silently break the ability to expand ΔX in a chart.
    # Canonicalize is now non-destructive on the ∇-nesting (and idempotent).
    ws = t.Workspace()
    X = ws.field("X", 1)
    nab = t.nabla(ctx=ws.ctx)
    lap = nab @ (nab * X)  # ∇·(∇⊗X) = ΔX
    floated = (nab @ nab) * X  # (∇·∇)⊗X — the old corrupted form

    c = td.canonicalize(lap)
    assert td.structural_eq(c, lap), "canonicalize floated the ∇ fence"
    assert not td.structural_eq(c, floated)
    assert td.structural_eq(c, td.canonicalize(c)), "not idempotent"
    assert c.latex() == r"\Delta \mathbf{X}"

    # a scalar coefficient rides through and still renders the clean Laplacian,
    # not a parenthesised (ΔX): μΔX / 2μΔX (the navier_lame endpoint shape).
    mu = t.tensor(r"\mu", 0, ctx=ws.ctx)
    assert td.canonicalize(mu * lap).latex() == r"\mu \, \Delta \mathbf{X}"
    two = t.scalar(2, ctx=ws.ctx)
    assert td.canonicalize(two * mu * lap).latex() == r"2 \, \mu \, \Delta \mathbf{X}"

    # A *product* operand (vibe 000086): Δ(u e) must stay nested through
    # canonicalize even though the scalar u puts the ∇ off the immediate fence
    # leg — the barrier keys on "contains an abstract ∇ anywhere", so the float
    # is refused and it still renders Δ(u e), μΔ(u e) (not "∇·∇ u e").
    u = ws.field("u", 0)
    e = ws.field("e", 1)
    lap_ue = nab @ (nab * (u * e))
    c_ue = td.canonicalize(lap_ue)
    assert not td.structural_eq(c_ue, (nab @ nab) * (u * e))  # not floated
    assert td.structural_eq(c_ue, td.canonicalize(c_ue))  # idempotent
    assert c_ue.latex() == r"\Delta (u \, \mathbf{e})"
    assert td.canonicalize(mu * lap_ue).latex() == r"\mu \, \Delta (u \, \mathbf{e})"


def test_reassemble_second_order_leibniz_bilinear():
    # vibe 000087: Δ(u e) expands (Leibniz) and reassembles to the second-order
    # Leibniz rule (Δu)e + 2(∇u)·(∇⊗e) + u(Δe).  The two middle copies are a
    # *bilinear* cross term — two ∂-marked fields joined by an inter-gradient dot
    # — that reassemble_nabla used to mis-fold to Δe (dropping ∇u).
    ws = t.Workspace()
    u = ws.field("u", 0)
    e = ws.field("e", 1)
    nab = t.nabla(ctx=ws.ctx)
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])

    lap = nab @ (nab * (u * e))
    interior = td.contract_identity(td.canonicalize(cart.expand_nabla(lap)))
    reass = td.collect_terms(cart.reassemble_nabla(td.canonicalize(interior)))

    rhs = (
        t.laplacian(u) * e
        + t.scalar(2, ctx=ws.ctx) * ((nab * u) @ (nab * e))
        + u * t.laplacian(e)
    )
    assert td.algebraic_eq(reass, rhs), reass.latex()


def test_reassemble_second_order_leibniz_vector_dyad():
    # vibe 000087: Δ(a⊗b) for two VECTOR fields exercises the transpose branch of
    # the bilinear fold — (∇a)ᵀ·(∇⊗b) — and the coefficient-order fix: the
    # single-operand term must be (Δa)⊗b, NOT b⊗Δa (an undifferentiated rank-1
    # factor keeps its ⊗ position relative to the operand).
    ws = t.Workspace()
    a = ws.field("a", 1)
    b = ws.field("b", 1)
    nab = t.nabla(ctx=ws.ctx)
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])

    lap = nab @ (nab * (a * b))
    interior = td.contract_identity(td.canonicalize(cart.expand_nabla(lap)))
    reass = td.collect_terms(cart.reassemble_nabla(td.canonicalize(interior)))

    rhs = (
        t.laplacian(a) * b
        + t.scalar(2, ctx=ws.ctx) * ((nab * a).transpose() @ (nab * b))
        + a * t.laplacian(b)
    )
    assert td.algebraic_eq(reass, rhs), reass.latex()
    # the (Δa)⊗b term keeps its leg order (not b⊗Δa) — a structural check
    assert r"(\Delta \mathbf{a}) \, \mathbf{b}" in reass.latex()
    assert r"\mathbf{b} \, \Delta \mathbf{a}" not in reass.latex()


def test_reassemble_second_order_leibniz_dot_product():
    # vibe 000088: Δ(u·v) for two vector fields — a *contracted* (dot) operand.
    # reassemble_nabla used to return 4·Δ(u·v) (it mis-scoped the δ-pair Laplacian
    # to the whole u·v and could not emit the double contraction).  The structural
    # path now folds it to (Δu)·v + 2∇u:∇v + u·Δv, scoping the Laplacian to the
    # mark-carrying sub-field and emitting the DDot cross term.
    ws = t.Workspace()
    u = ws.field("u", 1)
    v = ws.field("v", 1)
    nab = t.nabla(ctx=ws.ctx)
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])

    lap = t.laplacian(u @ v)  # Δ(u·v)
    interior = td.contract_identity(td.canonicalize(cart.expand_nabla(lap)))
    reass = td.collect_terms(cart.reassemble_nabla(td.canonicalize(interior)))

    rhs = (
        t.laplacian(u) @ v
        + t.scalar(2, ctx=ws.ctx) * ((nab * u).ddot(nab * v))
        + u @ t.laplacian(v)
    )
    assert td.algebraic_eq(reass, rhs), reass.latex()
    # not the old 4·Δ(u·v) bug, and it does contain the double contraction
    assert "4" not in reass.latex()
    assert ":" in reass.latex()


def _flat(c):
    """Flatten a components() result (vector list or rank-2 matrix) to a list."""
    out = []
    if isinstance(c, list):
        for x in c:
            out.extend(_flat(x))
    else:
        out.append(c)
    return out


def test_chart_evaluate_lowers_invariant_nabla_cylindrical():
    # vibe 000084: chart.evaluate lowers an invariant core-∇ expression to the
    # curvilinear-correct chart operators — no hand-rewrite via cyl.grad/div/rot.
    # Verified component-by-component on a CYLINDRICAL chart against the direct
    # operators, and the full Navier–Lamé ∇·T against its operator-built endpoint.
    ws = t.Workspace()
    u = ws.field("u", 1)
    lam = t.tensor(r"\lambda", 0, ctx=ws.ctx)
    mu = t.tensor(r"\mu", 0, ctx=ws.ctx)
    I = t.identity(ws.ctx)
    nab = t.nabla(ctx=ws.ctx)
    r, th, zc = ws.coords("r", r"\theta", "z", nonneg=("r",))
    cyl = ws.chart(ws.wcs(), [r, th, zc], [r * t.cos(th), r * t.sin(th), zc])

    def is_zero(e):
        return td.simplify_scalars(td.canonicalize(cyl.expand(e))).latex() == "0"

    def same(a, b):
        ca, cb = _flat(cyl.components(a)), _flat(cyl.components(b))
        return len(ca) == len(cb) and all(
            is_zero(cyl.expand(ca[i]) - cyl.expand(cb[i])) for i in range(len(ca))
        )

    assert same(cyl.evaluate(nab @ u), cyl.div(u))
    assert same(cyl.evaluate(nab * u), cyl.grad(u))
    assert same(cyl.evaluate(nab % u), cyl.rot(u))
    assert same(cyl.evaluate(nab @ (nab * u)), cyl.laplacian(u))  # ∇·(∇⊗u) = Δu

    # The whole Navier–Lamé ∇·T, written coordinate-free, evaluated in cyl.
    T = lam * (nab @ u) * I + mu * (nab * u + (nab * u).transpose())
    nl = mu * cyl.div(cyl.grad(u)) + (lam + mu) * cyl.grad(cyl.div(u))
    assert same(cyl.evaluate(nab @ T), nl)

    # a canonicalized invariant evaluates the same as the raw one (vibe 000085).
    raw = mu * (nab @ (nab * u))
    assert same(cyl.evaluate(td.canonicalize(raw)), cyl.evaluate(raw))


def test_chart_evaluate_reduced_navier_lame_endpoint():
    # vibe 000084: chart.evaluate must also handle the *reduced* invariant a
    # derivation leaves — the factor_common'd Navier–Lamé endpoint μΔu +
    # (λ+μ)∇(∇·u).  That form uses the floated Laplacian (∇·∇)⊗u and the
    # operator-left-normalised gradient (∇·u)⊗∇ (∇ on the right), and factors the
    # constant (λ+μ) inside the gradient — all of which evaluate now lowers.
    ws = t.Workspace()
    u = ws.field("u", 1)
    lam = t.tensor(r"\lambda", 0, ctx=ws.ctx)
    mu = t.tensor(r"\mu", 0, ctx=ws.ctx)
    I = t.identity(ws.ctx)
    nab = t.nabla(ctx=ws.ctx)
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    r, th, zc = ws.coords("r", r"\theta", "z", nonneg=("r",))
    cyl = ws.chart(ws.wcs(), [r, th, zc], [r * t.cos(th), r * t.sin(th), zc])

    # reduce ∇·T once (Cartesian) to the factored endpoint nl (coordinate-free)
    divT = nab @ (lam * (nab @ u) * I + mu * (nab * u + (nab * u).transpose()))
    reass = cart.reassemble_nabla(
        td.canonicalize(td.contract_identity(td.canonicalize(cart.expand_nabla(divT))))
    )
    nl = td.factor_common(td.collect_terms(reass))

    def is_zero(chart, e):
        return td.simplify_scalars(td.canonicalize(chart.expand(e))).latex() == "0"

    def same(chart, a, b):
        ca, cb = _flat(chart.components(a)), _flat(chart.components(b))
        return len(ca) == len(cb) and all(
            is_zero(chart, chart.expand(ca[i]) - chart.expand(cb[i]))
            for i in range(len(ca))
        )

    for chart in (cart, cyl):
        nl_vec = mu * chart.div(chart.grad(u)) + (lam + mu) * chart.grad(chart.div(u))
        # the reduced nl and the original ∇·T both evaluate to the same endpoint
        assert same(chart, chart.evaluate(nl), nl_vec)
        assert same(chart, chart.evaluate(nl), chart.evaluate(divT))


def test_chart_evaluate_floated_and_back_forms():
    # evaluate recognises the floated Laplacian (∇·∇)⊗X and the reordered
    # gradient X⊗∇ (operator on the right), not just the nested/left forms.
    ws = t.Workspace()
    u = ws.field("u", 1)
    nab = t.nabla(ctx=ws.ctx)
    r, th, zc = ws.coords("r", r"\theta", "z", nonneg=("r",))
    cyl = ws.chart(ws.wcs(), [r, th, zc], [r * t.cos(th), r * t.sin(th), zc])

    def is_zero(e):
        return td.simplify_scalars(td.canonicalize(cyl.expand(e))).latex() == "0"

    def same(a, b):
        ca, cb = _flat(cyl.components(a)), _flat(cyl.components(b))
        return len(ca) == len(cb) and all(
            is_zero(cyl.expand(ca[i]) - cyl.expand(cb[i])) for i in range(len(ca))
        )

    assert same(cyl.evaluate((nab @ nab) * u), cyl.laplacian(u))  # floated (∇·∇)⊗u
    assert same(cyl.evaluate((nab @ u) * nab), cyl.grad(cyl.div(u)))  # back (∇·u)⊗∇


def test_chart_evaluate_cross_chart_position_gradient():
    # vibe 000090: charts over the same (memoised) WCS reference share a manifold,
    # so ∇R = I is chart-independent.  A Cartesian position, evaluated in a
    # cylindrical chart, reprojects its WCS coords (x = r cosθ, …) via the target
    # embedding and folds to I — forward direction (WCS quantity → curvilinear).
    ws = t.Workspace()
    nab = t.nabla(ctx=ws.ctx)
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    r, th, zc = ws.coords("r", r"\theta", "z", nonneg=("r",))
    cyl = ws.chart(ws.wcs(), [r, th, zc], [r * t.cos(th), r * t.sin(th), zc])

    a = nab * cart.position()  # ∇⊗R, R Cartesian
    assert cart.evaluate(a).latex() == r"\mathbf{I}"
    assert cyl.evaluate(a).latex() == r"\mathbf{I}"  # reprojected + folded

    b = nab * cyl.position()  # ∇⊗R, R cylindrical
    assert cyl.evaluate(b).latex() == r"\mathbf{I}"  # native


def test_chart_evaluate_cross_chart_reverse_direction():
    # The reverse direction (a curvilinear-expressed quantity evaluated in a
    # different chart), vibe 000090 approach B.  The inverse embedding is never
    # written down — r = √(x²+y²) and θ = atan2(y, x) would need an arctangent
    # tender does not have — because only the *derivatives* of the inverse are
    # ever needed, and for an orthogonal chart those are ∂q^a/∂x^b = (e_a·i_b)/h_a,
    # which the chart already knows.  ∇R = I is chart-independent, so both
    # directions must agree.
    ws = t.Workspace()
    nab = t.nabla(ctx=ws.ctx)
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    r, th, zc = ws.coords("r", r"\theta", "z", nonneg=("r",))
    cyl = ws.chart(ws.wcs(), [r, th, zc], [r * t.cos(th), r * t.sin(th), zc])

    assert td.algebraic_eq(
        cart.evaluate(nab * cyl.position()), t.identity(ws.ctx)
    )


def test_chart_without_a_square_frame_has_no_cross_chart_jacobian():
    # A planar chart over a 3-D reference has no physical frame, so no
    # Jacobian — and evaluating a quantity of one in a sibling must say so
    # rather than silently returning 0.
    ws = t.Workspace()
    nab = t.nabla(ctx=ws.ctx)
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    pol, (pr, pth) = ws.polar_chart()
    import pytest

    with pytest.raises(ValueError, match="Jacobian"):
        cart.evaluate(nab * (pr * t.cos(pth)))


def test_workspace_wcs_is_memoised():
    # The world frame is unique per workspace (vibe 000090): every ws.wcs() call
    # returns the same basis, so charts built over it share one reference.
    ws = t.Workspace()
    assert ws.wcs().basis_id == ws.wcs().basis_id


def test_chart_evaluate_bare_nabla_raises():
    # A bare ∇, or a ∇·∇ Laplacian operator with no operand, is not evaluable —
    # with a clear message (not the obscure earlier "bare ∇" wording).
    ws = t.Workspace()
    nab = t.nabla(ctx=ws.ctx)
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    import pytest

    with pytest.raises(ValueError, match="not applied to any field"):
        cart.evaluate(nab)
    with pytest.raises(ValueError, match="Laplacian operator"):
        cart.evaluate(nab @ nab)  # bare ∇·∇, no operand


def test_apply_operators_no_op_without_deriv():
    # vibe 000083 Part A: with no concrete Deriv to apply, apply_operators is a
    # genuine no-op — it must NOT canonicalize (which would float the scalar off
    # the bare ∇, detaching it from the would-be Laplacian, the I2/B3 wall).
    ws = t.Workspace()
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    nab = t.nabla(ctx=ws.ctx)

    hessian_trace = t.tr(nab * (nab * t.tr(eps)))  # tr(∇⊗∇⊗θ), θ = tr ε
    # apply_operators leaves it untouched ...
    assert td.structural_eq(td.apply_operators(hessian_trace), hessian_trace)
    # ... so expand_dyad_ops still recovers the Laplacian Δθ, matching the
    # pipeline that never went through apply_operators.
    via_apply = td.expand_dyad_ops(td.apply_operators(hessian_trace))
    direct = td.expand_dyad_ops(hessian_trace)
    assert td.structural_eq(via_apply, direct)
    assert td.structural_eq(via_apply, t.laplacian(t.tr(eps)))


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
        == r"\mu \, \Delta \mathbf{u} + (\lambda + \mu) \, \nabla (\nabla \cdot \mathbf{u})"
    )
    # correctness: factor_common's constant-hoist ∇((λ+μ)∇·u)→(λ+μ)∇(∇·u) is an
    # operator-linearity rewrite that expand_products can't model, so the
    # endpoint identity ∇·T = μ∇·∇u + (λ+μ)∇(∇·u) is proven componentwise
    # (independent of the bare-∇ display) in the dedicated
    # test_navier_lame_endpoint_{cartesian,cylindrical}; here the pipeline is
    # confirmed to reach that clean factored display.
    assert _navier_lame_holds(cart, ws.ctx)


def _explicit_hooke(chart, u, lam, mu, I, ctx):
    # T = λ(∇·u)I + μ(∇u + (∇u)ᵀ).
    gradu = chart.grad(u)
    return lam * chart.div(u) * I + mu * (gradu + gradu.transpose())


def _standard_hooke(chart, u, lam, mu, I, ctx):
    # T = λ tr(ε)I + 2με with ε = sym(∇u) — the textbook elasticity form, whose
    # scalar-halved sym part exercises the constant-denominator diff rule.
    return lam * chart.div(u) * I + t.scalar(2, ctx=ctx) * mu * td.sym(chart.grad(u))


def _navier_lame_holds(chart, ctx, stress=_explicit_hooke):
    # ∇·T == μ∇·∇u + (λ+μ)∇(∇·u) componentwise, for an isotropic Hooke stress.
    # Both sides are coordinate-free vectors, so this must hold in every frame —
    # the bare-∇-independent endpoint witness.
    lam = t.tensor(r"\lambda", 0, ctx=ctx)
    mu = t.tensor(r"\mu", 0, ctx=ctx)
    I = t.identity(ctx)
    u = t.field("u", 1, ctx=ctx)
    lhs = chart.components(chart.div(stress(chart, u, lam, mu, I, ctx)))
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


def test_navier_lame_endpoint_standard_sym_form():
    # vibe 000080, sym-form (b): the textbook stress T = λ tr(ε)I + 2με with
    # ε = sym(∇u) = (∇u+(∇u)ᵀ)/2 reduces to the SAME clean endpoint.  The scalar
    # /2 rides out via the constant-denominator diff rule (the full quotient rule
    # used to orphan the ∂-mark indices and drop the second derivatives).
    ws = t.Workspace()
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    nab = t.nabla(ctx=ws.ctx)
    I = t.identity(ws.ctx)
    lam = t.tensor(r"\lambda", 0, ctx=ws.ctx)
    mu = t.tensor(r"\mu", 0, ctx=ws.ctx)
    u = ws.field("u", 1)

    T = lam * (nab @ u) * I + t.scalar(2, ctx=ws.ctx) * mu * td.sym(nab * u)
    interior = td.contract_identity(td.canonicalize(cart.expand_nabla(nab @ T)))
    reass = cart.reassemble_nabla(td.canonicalize(interior))
    nl = td.factor_common(td.collect_terms(reass))
    assert (
        nl.latex()
        == r"\mu \, \Delta \mathbf{u} + (\lambda + \mu) \, \nabla (\nabla \cdot \mathbf{u})"
    )
    # and it holds componentwise in a Cartesian and a cylindrical frame.
    assert _navier_lame_holds(cart, ws.ctx, stress=_standard_hooke)
    ws2 = t.Workspace()
    r, th, zc = ws2.coords("r", r"\theta", "z", nonneg=("r",))
    cyl = ws2.chart(ws2.wcs(), [r, th, zc], [r * t.cos(th), r * t.sin(th), zc])
    assert _navier_lame_holds(cyl, ws2.ctx, stress=_standard_hooke)


# ---------------------------------------------------------------------------
# Ported from the retired tender.operators DSL, onto the core route.  Each
# covers behaviour of the ∇ *node*, which the DSL only wrapped.
# ---------------------------------------------------------------------------


def test_compound_operands_are_parenthesised():
    # A compound operand (a sum, or another cross) is wrapped so the rendering
    # is unambiguous (vibe 000071); a bare field is not.
    ws = t.Workspace()
    cart, _ = _chart(ws)
    nab = t.nabla(ctx=ws.ctx)
    R = cart.position()
    I = ws.identity()

    assert "(" in (nab % R).latex()  # ∇×(x i + y j + z k)
    assert "((" in (nab % (R % I)).latex()  # ∇×((…) × I)

    f = cart.field("f", 0)
    assert "(" not in (nab * f).latex()
    # ∇·∇f renders as Δf — never ∇², which is not a ring element.
    assert (nab @ (nab * f)).latex() == "\\Delta f"


def test_directional_derivative_of_the_position_vector():
    # The flagship payoff, on the core route: (v·∇)R = v, for a constant v.
    ws = t.Workspace()
    cart, _ = _chart(ws)
    nab = t.nabla(ctx=ws.ctx)
    R = cart.position()
    wcs = ws.wcs()
    v = (
        t.scalar(2, ctx=ws.ctx) * wcs.basis(0)
        + t.scalar(3, ctx=ws.ctx) * wcs.basis(1)
        + wcs.basis(2)
    )

    out = td.simplify(td.contract_identity(cart.evaluate(v @ (nab * R))))
    assert td.algebraic_eq(out, v)


def test_chart_nabla_is_the_gradient_when_applied():
    # vibe 000077 step C: chart.nabla() is the first-class ∇ — an inspectable
    # rank-1 operator carrying ∂'s — and applying it with ⊗ is the gradient.
    ws = t.Workspace()
    cart, _ = _chart(ws)
    f = cart.field("f", 0)

    nab = cart.nabla()
    assert "\\partial_{" in nab.latex()
    assert nab.rank == 1
    assert td.algebraic_eq(td.apply_operators(nab * f), cart.grad(f))

# ---------------------------------------------------------------------------
# vibe 000109: a ½ inside a ∇ operand must not orphan the direction indices
# ---------------------------------------------------------------------------


def _index_counts(term):
    """How often each index name is written in *term*."""
    names = re.findall(r"_\{(\w)\}", term.latex())
    return {n: names.count(n) for n in set(names)}


class TestScalarDivisionKeepsTheIndexLink:
    """`expand_nabla` links each frame vector e_i to its own ∂_i.

    A scalar denominator used to break the Einstein scope, so canonicalize
    α-renamed the pair apart and the expansion came back with single, dangling
    indices — invalid, and silently so.  `sym(∇u) = (∇u + (∇u)ᵀ)/2` is what a
    reader meets this with first, which is to say every stress.
    """

    def _setup(self):
        ws = t.Workspace()
        u = t.field("u", 1, ctx=ws.ctx)
        nabla = t.nabla(ctx=ws.ctx)
        chart, _ = _chart(ws)
        return ws, chart, u, nabla

    def test_a_halved_operand_matches_the_unhalved_one(self):
        ws, chart, u, nabla = self._setup()
        assert td.algebraic_eq(
            chart.expand_nabla(nabla @ (u / 2)),
            chart.expand_nabla(nabla @ u) / 2,
        )

    def test_it_leaves_no_dangling_index(self):
        # Every index in a term is written exactly twice — the property that
        # makes the expression mean anything at all.
        ws, chart, u, nabla = self._setup()
        out = chart.expand_nabla(nabla @ td.sym(nabla * u))
        for path in out.addends():
            term = out.at(path)
            counts = _index_counts(term)
            assert all(c == 2 for c in counts.values()), (
                f"{term.latex()} uses an index once: {counts}"
            )

    def test_no_summation_is_left_explicit(self):
        # The Σ's were the visible symptom: an index the scope no longer
        # recognised as contracted got a binder of its own.
        ws, chart, u, nabla = self._setup()
        out = chart.expand_nabla(nabla @ td.sym(nabla * u))
        assert "\\sum" not in out.latex()

    def test_the_isotropic_stress_divergence_expands(self):
        # The reported case: ∇·T + f for the Hooke stress built from sym(∇u).
        ws, chart, u, nabla = self._setup()
        f = t.field("f", 1, ctx=ws.ctx)
        I = t.identity(ws.ctx)
        lam = t.tensor(r"\lambda", 0, ctx=ws.ctx)
        mu = t.tensor(r"\mu", 0, ctx=ws.ctx)
        eps = td.sym(nabla * u)
        T = lam * eps.tr() * I + 2 * mu * eps
        out = chart.expand_nabla(nabla @ T + f)
        assert "\\sum" not in out.latex()
        for path in out.addends():
            term = out.at(path)
            counts = _index_counts(term)
            assert all(c == 2 for c in counts.values()), term.latex()

class TestContractingThroughADerivativeMark:
    """A δ must contract against a ∂-direction, not only against a slot.

    `δ_jk ∂_j ∂_k u_i e_i` is a Laplacian in components, and it stayed exactly
    as written because `contract_delta` read only tensor *slots* — while the
    summation machinery had always counted a free ∂-mark's direction as an
    occurrence of its index (vibe 000109).
    """

    def _components(self):
        ws = t.Workspace()
        frame = ws.wcs()
        u = t.field("u", 1, ctx=ws.ctx)
        nabla = t.nabla(ctx=ws.ctx)
        chart, _ = _chart(ws)
        return ws, frame, chart, tb.simplify_basis_dot(
            tb.expand_in_basis(
                chart.expand_nabla(nabla @ (nabla * u)), frame
            ),
            frame,
        )

    def test_the_delta_contracts_the_two_directions(self):
        ws, frame, chart, comps = self._components()
        assert "delta" in comps.latex()
        out = td.contract_delta(comps)
        assert "delta" not in out.latex()
        # ∂_j ∂_j u_i e_i — one direction, twice: the Laplacian.
        assert out.latex().count("partial_{j}") == 2

    def test_it_still_refuses_when_there_is_no_partner(self):
        # A δ whose index appears nowhere else is not a contraction.
        ws = t.Workspace()
        i = t.alloc_index(ws.ctx)
        j = t.alloc_index(ws.ctx)
        d = t.delta(
            t.Realm.Orthonormal, t.space_3d, t.Level.Lower, t.Level.Lower, i, j
        )
        assert td.structural_eq(td.contract_delta(d), d)


class TestReassembleNablaRefusesAComponentForm:
    """A ∂ with no frame vector is not a ∇ expansion.

    `(∂_j ∂_j u_i) e_i` came back as `∇ u_i` — a bare unapplied ∇ times a
    component with a dangling index, one derivative lost.  The classifier read
    the `e_i` as a gradient leg, though it belongs to the field's own index
    (vibe 000109).
    """

    def _setup(self):
        ws = t.Workspace()
        frame = ws.wcs()
        u = t.field("u", 1, ctx=ws.ctx)
        nabla = t.nabla(ctx=ws.ctx)
        chart, _ = _chart(ws)
        return ws, frame, chart, u, nabla

    def test_the_free_index_form_still_folds(self):
        ws, frame, chart, u, nabla = self._setup()
        abstract = chart.expand_nabla(nabla @ (nabla * u))
        assert "Delta" in chart.reassemble_nabla(abstract).latex()

    def test_the_component_form_is_refused_with_a_reason(self):
        import tender.steps as ts

        ws, frame, chart, u, nabla = self._setup()
        comps = td.contract_delta(
            tb.simplify_basis_dot(
                tb.expand_in_basis(
                    chart.expand_nabla(nabla @ (nabla * u)), frame
                ),
                frame,
            )
        )
        result = ts.info("reassemble_nabla").run(comps, chart=chart)
        assert not result.fired
        assert "no ∂ to pair with" in result.reason
        assert td.structural_eq(result.expr, comps)

    def test_a_contracted_direction_pair_is_a_laplacian(self):
        """The one orphan that *is* readable: `∂_i ∂_i u` is `Δu`.

        Once `reduce_frame` contracts the δ, the two directions pair with each
        other and no frame vector is left — the same pair the classifier reads
        as a Laplacian when it is still spelled `e_ℓ·e_m`, one step further on
        (vibe 000109).
        """
        ws, frame, chart, u, nabla = self._setup()
        reduced = td.canonicalize(
            tb.reduce_frame(
                chart.expand_nabla(nabla @ (nabla * u)), frame
            )
        )
        assert "partial" in reduced.latex() and "Delta" not in reduced.latex()
        assert "\\Delta \\mathbf{u}" in chart.reassemble_nabla(reduced).latex()

    def test_the_route_that_keeps_the_operand_abstract_reaches_navier_lame(self):
        # The endpoint, end to end: ∇·T + f for the isotropic Hooke stress.
        import tender.steps as ts

        ws, frame, chart, u, nabla = self._setup()
        f = t.field("f", 1, ctx=ws.ctx)
        I = t.identity(ws.ctx)
        lam = t.tensor(r"\lambda", 0, ctx=ws.ctx)
        mu = t.tensor(r"\mu", 0, ctx=ws.ctx)
        eps = td.sym(nabla * u)
        T = lam * eps.tr() * I + 2 * mu * eps
        b = ts.using(basis=frame, chart=chart)

        got = td.canonicalize(
            td.derive(
                nabla @ T + f,
                [
                    b.expand_nabla,
                    td.contract_identity,
                    b.reduce_frame,
                    b.reassemble_nabla,
                ],
            ).current
        )
        want = (
            mu * (nabla @ (nabla * u))
            + lam * (nabla * (nabla @ u))
            + mu * (nabla * (nabla @ u))
            + f
        )
        assert td.algebraic_eq(got, want)

    def test_an_expression_with_no_derivative_says_so(self):
        import tender.steps as ts

        ws, frame, chart, u, nabla = self._setup()
        result = ts.info("reassemble_nabla").run(u, chart=chart)
        assert not result.fired
        assert "no ∂ here" in result.reason

class TestCoefficientsPoolAcrossANablaFence:
    """Canon promises one rational coefficient per term — ∇ or no ∇.

    A ⊗-chain carrying an operator is kept whole so the fence survives, and the
    literals *inside* it were invisible to the pooling: `2 ⊗ (½ λ ∇∇·u)` stayed
    `2 ½ λ ∇∇·u`, where the same term over a plain vector folded to `λ Y`
    (vibe 000109).  A literal to the left of ∇ is outside its reach.
    """

    def _pieces(self):
        ws = t.Workspace()
        lam = t.tensor(r"\lambda", 0, ctx=ws.ctx)
        u = t.field("u", 1, ctx=ws.ctx)
        nabla = t.nabla(ctx=ws.ctx)
        return ws, lam, t.scalar(t.Rational(1, 2)), nabla * (nabla @ u)

    def test_a_literal_outside_the_operator_joins_the_coefficient(self):
        ws, lam, half, grad_div = self._pieces()
        folded = td.canonicalize(lam * half * grad_div)
        assert td.structural_eq(
            td.canonicalize(2 * folded), td.canonicalize(lam * grad_div)
        )

    def test_the_fence_still_holds(self):
        # Pooling the coefficient must not flatten the operator's scope: the
        # gradient stays a gradient of ∇·u.
        ws, lam, half, grad_div = self._pieces()
        out = td.canonicalize(2 * td.canonicalize(lam * half * grad_div))
        assert td.algebraic_eq(out, lam * grad_div)
        assert "\\nabla \\, \\nabla \\cdot" in out.latex()

    def test_a_plain_term_folds_the_same_way(self):
        # The parity that was broken: with and without an operator.
        ws, lam, half, _ = self._pieces()
        y = t.tensor("Y", 1, ctx=ws.ctx)
        out = td.canonicalize(2 * td.canonicalize(lam * half * y))
        assert td.structural_eq(out, td.canonicalize(lam * y))


class TestFactorCommonRespectsTheOperatorsReach:
    """A factor inside a gradient cannot be lifted out of the term.

    `λ ∇(∇·u) + μ ∇(∇·u)` came back as `(∇λ + ∇μ) ∇·u` — `∇·u` pulled out from
    *inside* the ∇, leaving the gradient of a constant (vibe 000109).
    """

    def _pieces(self):
        ws = t.Workspace()
        return (
            ws,
            t.tensor(r"\lambda", 0, ctx=ws.ctx),
            t.tensor(r"\mu", 0, ctx=ws.ctx),
            t.nabla(ctx=ws.ctx),
            t.field("u", 1, ctx=ws.ctx),
        )

    def test_it_leaves_a_term_whose_common_factor_is_inside_a_gradient(self):
        ws, lam, mu, nabla, u = self._pieces()
        grad_div = nabla * (nabla @ u)
        e = lam * grad_div + mu * grad_div
        out = td.factor_common(e)
        assert td.structural_eq(out, e)
        assert td.algebraic_eq(out, e)

    def test_the_case_it_was_written_for_still_folds(self):
        # vibe 000080's own example: `∇·u` is a completed contraction, not an
        # operator standing over its neighbours, so it factors as before.
        ws, lam, mu, nabla, u = self._pieces()
        div = nabla @ u
        out = td.factor_common(lam * div + mu * div)
        assert td.algebraic_eq(out, lam * div + mu * div)
        assert "(\\lambda + \\mu)" in out.latex()

class TestToContraction:
    """`a_i = a·e_i` — the inverse half of a component expansion (vibe 000109).

    On its own it looks like a step backwards.  Paired with the completeness
    fold `(X·e_i) e_i → X` it takes a componentized expression back to
    invariants *carrying its ∂ marks*, which the direct fold cannot: that one
    rebuilds the invariant from a name and rank, so the marks have nowhere to
    go, and it refuses rather than drop them.
    """

    def _setup(self):
        ws = t.Workspace()
        frame = ws.wcs()
        u = t.field("u", 1, ctx=ws.ctx)
        nabla = t.nabla(ctx=ws.ctx)
        chart, _ = _chart(ws)
        return ws, frame, chart, u, nabla

    def _components(self):
        ws, frame, chart, u, nabla = self._setup()
        comps = td.contract_delta(
            tb.simplify_basis_dot(
                tb.expand_in_basis(
                    chart.expand_nabla(nabla @ (nabla * u)), frame
                ),
                frame,
            )
        )
        return ws, frame, chart, comps

    def test_it_writes_a_component_as_its_contraction(self):
        ws, frame, chart, comps = self._components()
        out = tc.to_contraction(comps, chart)
        # u_i became u·e_i, and the ∂ marks moved onto the invariant.
        assert "u_{" not in out.latex()
        assert "\\mathbf{u}) \\cdot \\mathbf{e}" in out.latex()
        assert out.latex().count("partial") == 2

    def test_the_componentized_form_now_folds_all_the_way_back(self):
        # The route that did not exist: from components to Δu, three steps.
        ws, frame, chart, comps = self._components()
        got = td.derive(
            comps,
            [
                lambda e: tc.to_contraction(e, chart),
                lambda e: tb.reassemble(e, frame),
                lambda e: chart.reassemble_nabla(e),
            ],
        ).current
        u = t.field("u", 1, ctx=ws.ctx)
        nabla = t.nabla(ctx=ws.ctx)
        assert td.algebraic_eq(got, nabla @ (nabla * u))

    def test_a_curvilinear_frame_is_refused_rather_than_losing_the_connection(self):
        # ∂(a·e_i) = (∂a)·e_i only where ∂e_i = 0.
        import tender.steps as ts

        ws = t.Workspace()
        cyl, (r, th, z) = ws.cylindrical_chart()
        a = t.field("a", 1, ctx=ws.ctx)
        marked = td.partial(tb.expand_in_basis(a, cyl.physical_frame()), r)
        result = ts.info("to_contraction").run(marked, chart=cyl)
        assert not result.fired
        assert "not constant" in result.reason
        assert td.structural_eq(result.expr, marked)

    def test_an_unmarked_component_needs_no_constant_frame(self):
        # `a_i = a·e_i` is unconditional; only moving ∂'s across e_i is not.
        ws = t.Workspace()
        cyl, _ = ws.cylindrical_chart()
        a = t.field("a", 1, ctx=ws.ctx)
        out = tc.to_contraction(
            tb.expand_in_basis(a, cyl.physical_frame()), cyl
        )
        assert "\\mathbf{a} \\cdot \\mathbf{e}" in out.latex()

    def test_it_says_so_when_there_is_no_component(self):
        import tender.steps as ts

        ws, frame, chart, u, nabla = self._setup()
        result = ts.info("to_contraction").run(u, chart=chart)
        assert not result.fired
        assert "no component" in result.reason


class TestTheComponentRouteClosesToNavierLame:
    """The reported pipeline, with the one step it was missing."""

    def test_it_reaches_the_equation(self):
        import tender.steps as ts

        ws = t.Workspace()
        frame = ws.wcs()
        u = t.field("u", 1, ctx=ws.ctx)
        f = t.field("f", 1, ctx=ws.ctx)
        nabla = t.nabla(ctx=ws.ctx)
        I = t.identity(ws.ctx)
        lam = t.tensor(r"\lambda", 0, ctx=ws.ctx)
        mu = t.tensor(r"\mu", 0, ctx=ws.ctx)
        eps = td.sym(nabla * u)
        T = lam * eps.tr() * I + 2 * mu * eps
        chart, _ = _chart(ws)
        b = ts.using(basis=frame, chart=chart)

        got = td.derive(
            nabla @ T + f,
            [
                b.expand_nabla,
                b.expand_in_basis,
                b.simplify_basis_dot,
                b.contract_delta,
                b.to_contraction,
                b.reassemble,
                b.reassemble_nabla,
                td.canonicalize,
            ],
        ).current
        want = (
            mu * (nabla @ (nabla * u))
            + lam * (nabla * (nabla @ u))
            + mu * (nabla * (nabla @ u))
            + f
        )
        assert td.algebraic_eq(got, want)


class TestAlphaRenamingKeepsTheMarksInStep:
    """Renaming a binder must rename the ∂ links it binds.

    `substitute_index_id` renamed only the slots, so a binder over a direction
    that lives *only* on marks desynced: the binder moved to a fresh id, the
    marks kept the old one, and canon materialized a second binder beside an
    empty one — `Σ_? Σ_i ∂_i∂_i u` (vibe 000109).
    """

    def test_no_empty_binder_survives(self):
        ws = t.Workspace()
        frame = ws.wcs()
        u = t.field("u", 1, ctx=ws.ctx)
        nabla = t.nabla(ctx=ws.ctx)
        chart, _ = _chart(ws)
        lap = td.canonicalize(
            tb.reduce_frame(chart.expand_nabla(nabla @ (nabla * u)), frame)
        )
        i = t.alloc_index(ws.ctx)
        shape = (lap @ frame.covariant_vector(i)) * frame.covariant_vector(i)

        assert "?" not in td.canonicalize(shape).latex()
        folded = tb.reassemble(shape, frame)
        assert "?" not in folded.latex()
        assert td.algebraic_eq(folded, lap)

