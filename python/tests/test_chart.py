"""Coordinate charts (vibe 000069 M4): derive the orthogonal-curvilinear
geometry (radius vector, tangent basis, metric, scale factors, physical frame)
from a coordinate mapping."""

import pytest

import tender as t
import tender.basis as tb
import tender.chart as tc
import tender.derivation as td


def cartesian_2d(ctx):
    i = t.tensor("i", rank=1, ctx=ctx)
    j = t.tensor("j", rank=1, ctx=ctx)
    return tb.make_orthonormal_basis(
        [i, j],
        t.space_2d,
        value_names=["x", "y"],
        vector_symbols=["i", "j"],
    )


def make_polar(ctx):
    ref = cartesian_2d(ctx)
    r = t.coordinate("r", chart_id=1, slot=0, nonneg=True, ctx=ctx)
    th = t.coordinate(r"\theta", chart_id=1, slot=1, ctx=ctx)
    chart = tc.CoordinateChart(
        ref, [r, th], [r * t.cos(th), r * t.sin(th)]
    )
    return ref, r, th, chart


def test_polar_tangent_vectors():
    ctx = t.Context()
    ref, r, th, chart = make_polar(ctx)
    i, j = ref.basis(0), ref.basis(1)

    gr = chart.tangent_vector(0)
    assert td.algebraic_eq(gr, t.cos(th) * i + t.sin(th) * j)

    gt = chart.tangent_vector(1)
    assert td.algebraic_eq(gt, -(r * t.sin(th)) * i + (r * t.cos(th)) * j)


def test_polar_metric():
    ctx = t.Context()
    _, r, _, chart = make_polar(ctx)
    assert td.algebraic_eq(chart.metric_component(0, 0), t.scalar(1, ctx=ctx))
    assert td.algebraic_eq(chart.metric_component(1, 1), r**2)
    assert td.algebraic_eq(chart.metric_component(0, 1), t.scalar(0, ctx=ctx))


def test_polar_scale_factors():
    ctx = t.Context()
    _, r, _, chart = make_polar(ctx)
    assert td.algebraic_eq(chart.scale_factor(0), t.scalar(1, ctx=ctx))
    assert td.algebraic_eq(chart.scale_factor(1), r)


def test_polar_physical_basis():
    ctx = t.Context()
    ref, r, th, chart = make_polar(ctx)
    fb = chart.physical_basis()
    i, j = ref.basis(0), ref.basis(1)

    assert fb.dim == 2
    assert fb.is_orthonormal
    assert td.algebraic_eq(fb.basis(0), t.cos(th) * i + t.sin(th) * j)
    assert td.algebraic_eq(fb.basis(1), -(t.sin(th)) * i + t.cos(th) * j)


def test_polar_matches_hand_written():
    ctx = t.Context()
    _, _, _, chart = make_polar(ctx)
    derived = chart.physical_basis()
    hand = tb.polar_2d(ctx)
    assert derived.dim == hand.dim
    assert derived.is_orthonormal == hand.is_orthonormal


def test_spherical_azimuth_row():
    ctx = t.Context()
    ref = tb.wcs(ctx)
    r = t.coordinate("r", chart_id=3, slot=0, nonneg=True, ctx=ctx)
    th = t.coordinate(r"\theta", chart_id=3, slot=1, ctx=ctx)
    ph = t.coordinate(r"\phi", chart_id=3, slot=2, ctx=ctx)
    chart = tc.CoordinateChart(
        ref,
        [r, th, ph],
        [
            r * t.sin(th) * t.cos(ph),
            r * t.sin(th) * t.sin(ph),
            r * t.cos(th),
        ],
    )
    # g_phiphi = r^2 sin^2(theta); h_phi = r sin(theta).
    assert td.algebraic_eq(
        chart.metric_component(2, 2), r**2 * t.sin(th) ** 2
    )
    assert td.algebraic_eq(chart.scale_factor(2), r * t.sin(th))

    # e_phi = -sin(phi) i + cos(phi) j.
    fb = chart.physical_basis()
    i, j = ref.basis(0), ref.basis(1)
    assert td.algebraic_eq(fb.basis(2), -(t.sin(ph)) * i + t.cos(ph) * j)


def test_polar_basis_derivative():
    ctx = t.Context()
    _, _, _, chart = make_polar(ctx)
    fb = chart.physical_basis()
    # radial derivatives vanish
    assert td.algebraic_eq(chart.basis_derivative(0, 0), t.scalar(0, ctx=ctx))
    assert td.algebraic_eq(chart.basis_derivative(1, 0), t.scalar(0, ctx=ctx))
    # d_phi e_r = e_phi, d_phi e_phi = -e_r
    assert td.algebraic_eq(chart.basis_derivative(0, 1), fb.basis(1))
    assert td.algebraic_eq(chart.basis_derivative(1, 1), -fb.basis(0))


def _coeffs_eq(got, exp):
    return len(got) == len(exp) and all(
        td.algebraic_eq(g, e) for g, e in zip(got, exp)
    )


def test_polar_connection_coefficients():
    ctx = t.Context()
    _, _, _, chart = make_polar(ctx)
    z = t.scalar(0, ctx=ctx)
    one = t.scalar(1, ctx=ctx)
    assert _coeffs_eq(chart.connection_coefficients(0, 0), [z, z])
    assert _coeffs_eq(chart.connection_coefficients(1, 0), [z, z])
    assert _coeffs_eq(chart.connection_coefficients(0, 1), [z, one])
    assert _coeffs_eq(
        chart.connection_coefficients(1, 1), [t.scalar(-1, ctx=ctx), z]
    )


def test_spherical_connection_coefficients():
    ctx = t.Context()
    ref = tb.wcs(ctx)
    r = t.coordinate("r", chart_id=3, slot=0, nonneg=True, ctx=ctx)
    th = t.coordinate(r"\theta", chart_id=3, slot=1, ctx=ctx)
    ph = t.coordinate(r"\phi", chart_id=3, slot=2, ctx=ctx)
    chart = tc.CoordinateChart(
        ref,
        [r, th, ph],
        [
            r * t.sin(th) * t.cos(ph),
            r * t.sin(th) * t.sin(ph),
            r * t.cos(th),
        ],
    )
    z = t.scalar(0, ctx=ctx)
    one = t.scalar(1, ctx=ctx)
    # d_theta e_r = e_theta; d_phi e_r = sin(theta) e_phi.
    assert _coeffs_eq(chart.connection_coefficients(0, 1), [z, one, z])
    assert _coeffs_eq(chart.connection_coefficients(0, 2), [z, z, t.sin(th)])
    # d_phi e_phi = -sin(theta) e_r - cos(theta) e_theta.
    assert _coeffs_eq(
        chart.connection_coefficients(2, 2), [-t.sin(th), -t.cos(th), z]
    )


def make_cylindrical(ctx):
    ref = tb.wcs(ctx)
    r = t.coordinate("r", chart_id=2, slot=0, nonneg=True, ctx=ctx)
    th = t.coordinate(r"\theta", chart_id=2, slot=1, ctx=ctx)
    z = t.coordinate("z", chart_id=2, slot=2, ctx=ctx)
    chart = tc.CoordinateChart(
        ref, [r, th, z], [r * t.cos(th), r * t.sin(th), z]
    )
    return r, th, z, chart


def _inv_eq(a, b):
    """Equality of invariant tensors: distribute (x) over + and simplify first."""
    return td.algebraic_eq(
        td.simplify_scalars(td.expand_products(a)),
        td.simplify_scalars(td.expand_products(b)),
    )


def test_cartesian_gradient_is_identity():
    # Intrinsic operators (vibe 000071): grad R = I, div R = 3, rot R = 0, with R
    # on the chart's own frame R = x e_x + y e_y + z e_z.
    ctx = t.Context()
    ref = tb.wcs(ctx)
    x = t.coordinate("x", chart_id=7, slot=0, ctx=ctx)
    y = t.coordinate("y", chart_id=7, slot=1, ctx=ctx)
    z = t.coordinate("z", chart_id=7, slot=2, ctx=ctx)
    chart = tc.CoordinateChart(ref, [x, y, z], [x, y, z])
    fb = chart.physical_frame()
    R = sum(
        (c * fb.direction(k) for k, c in enumerate([x, y, z][1:], start=1)),
        [x, y, z][0] * fb.direction(0),
    )
    assert td.structural_eq(chart.grad(R), t.identity(ctx))
    assert td.algebraic_eq(chart.div(R), t.scalar(3, ctx=ctx))
    assert td.algebraic_eq(chart.rot(R), t.scalar(0, ctx=ctx))


def test_cylindrical_gradient():
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    fb = chart.physical_frame()
    one = t.scalar(1, ctx=ctx)
    # grad(theta) = (1/r) e_theta, intrinsically on the frame's e_theta.
    assert td.structural_eq(
        td.canonicalize(chart.grad(th)),
        td.canonicalize((one / r) * fb.direction(1)),
    )
    # grad(r^2) = 2r e_r
    assert td.structural_eq(
        td.canonicalize(chart.grad(r**2)),
        td.canonicalize((t.scalar(2, ctx=ctx) * r) * fb.direction(0)),
    )


def test_cylindrical_divergence_and_laplacian():
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    v = r * chart.physical_frame().direction(0)  # r e_r
    assert td.algebraic_eq(chart.div(v), t.scalar(2, ctx=ctx))
    assert td.algebraic_eq(chart.laplacian(r**2), t.scalar(4, ctx=ctx))


def test_field_derivative_survives_index_materialization():
    # vibe 000073: unroll_sums / index substitution must preserve a field's ∂
    # directions (field_derivs).  Materializing the dummy indices of an expanded
    # ∂S used to drop the ∂, yielding constant components.  Use a Cartesian frame
    # (∂e_i = 0), where expanding a field derivative is legal (no connection).
    ctx = t.Context()
    ref = tb.wcs(ctx)
    x = t.coordinate("x", chart_id=5, slot=0, ctx=ctx)
    y = t.coordinate("y", chart_id=5, slot=1, ctx=ctx)
    z = t.coordinate("z", chart_id=5, slot=2, ctx=ctx)
    chart = tc.CoordinateChart(ref, [x, y, z], [x, y, z])
    frame = chart.physical_frame()
    S = t.field("S", 2, deps=[x, y], ctx=ctx)
    expanded = tb.expand_in_basis(td.partial(S, x), frame, tb.Variance.Covariant)
    unrolled = td.canonicalize(td.unroll_sums(expanded))
    assert r"\partial_{x}" in unrolled.latex()  # the derivative survives


def test_physical_basis_and_frame_share_identity():
    # vibe 000073 Gap 3: physical_basis() and physical_frame() return the same
    # basis vectors, so a field on one and an operator result on the other still
    # reduce together.  Previously they minted different identities.
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    frame = chart.physical_frame()
    basis = chart.physical_basis()
    for k in range(3):
        assert td.structural_eq(frame.direction(k), basis.direction(k))
    # A plain field expanded on physical_basis() reduces its dots against the
    # frame the operators use — the shared identity is what makes this work.
    T = t.field("T", 2, deps=[r, th], ctx=ctx)
    X = tb.expand_in_basis(T @ frame.direction(0), basis, tb.Variance.Covariant)
    X = tb.simplify_basis_dot(td.unroll_sums(X), basis)
    assert r"\cdot" not in X.latex()  # all e_i·e_j contracted away


def test_expand_in_basis_refuses_field_derivative_on_moving_frame():
    # vibe 000073: a bare basis has no connection Ω, so expanding ∂T on a moving
    # frame would silently drop the connection terms.  The guard refuses loudly.
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    frame = chart.physical_frame()
    T = t.field("T", 2, deps=[r, th], ctx=ctx)
    with pytest.raises(ValueError, match="moving frame"):
        tb.expand_in_basis(td.partial(T, r), frame, tb.Variance.Covariant)


def test_div_of_e_theta_dyad_does_not_crash():
    # vibe 000073 Gap 2: div of a dyad whose contracted leg is e_θ used to
    # throw "encapsulate: unsupported factor node" — the connection term
    # ∂_θ e_θ = −e_r left a raw ⊗ inside a dot operand.  Now it reduces.
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    e = [chart.physical_frame().direction(k) for k in range(3)]
    a = t.field("a", 0, deps=[r, th], ctx=ctx)
    # ∇·(a e_θ e_θ) = −(a/r) e_r + (∂_θ a / r) e_θ
    div = td.simplify_scalars(td.expand_products(chart.div(a * (e[1] * e[1]))))
    expected = -(a / r) * e[0] + (td.partial(a, th) / r) * e[1]
    assert td.simplify_scalars(td.expand_products(div - expected)).latex() == "0"


def test_expand_in_basis_components_are_fields():
    # vibe 000073: a component of a field must itself be a field, so div can
    # differentiate it.  Before the fix ∂_r T_rr came out 0 (constant component).
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    frame = chart.physical_frame()
    e = [frame.direction(k) for k in range(3)]
    T = t.field("T", 2, ctx=ctx)
    Tc = td.canonicalize(td.unroll_sums(
        tb.expand_in_basis(T, frame, tb.Variance.Covariant)))
    # Recover T_rr and check it differentiates (is a field, not a constant).
    Trr = td.simplify_scalars(tb.simplify_basis_dot(
        td.expand_products(e[0] @ Tc @ e[0]), frame))
    assert not td.algebraic_eq(td.partial(Trr, r), t.scalar(0, ctx=ctx))


def test_cylindrical_divergence_matches_textbook():
    # vibe 000073: the full ∇·T reproduces the standard cylindrical equilibrium
    # equations, incl. the (T_rθ + T_θr)/r shear term in the θ-component.
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    frame = chart.physical_frame()
    e = [frame.direction(k) for k in range(3)]
    T = t.field("T", 2, ctx=ctx)
    Tc = td.canonicalize(td.unroll_sums(
        tb.expand_in_basis(T, frame, tb.Variance.Covariant)))

    def red(x):
        x = tb.simplify_basis_dot(td.expand_products(x), frame)
        return td.simplify_scalars(td.fold_arithmetic(
            td.eval_delta_concrete(td.canonicalize(x))))

    def comp(v, i):
        return red(v @ e[i])

    def Tij(i, j):
        return red(e[i] @ Tc @ e[j])

    def d(x, c):
        return td.simplify_scalars(td.partial(x, c))

    div_T = chart.div(Tc)
    exp_r = d(Tij(0,0),r) + d(Tij(1,0),th)/r + d(Tij(2,0),z) + (Tij(0,0)-Tij(1,1))/r
    exp_th = d(Tij(0,1),r) + d(Tij(1,1),th)/r + d(Tij(2,1),z) + (Tij(0,1)+Tij(1,0))/r
    exp_z = d(Tij(0,2),r) + d(Tij(1,2),th)/r + d(Tij(2,2),z) + Tij(0,2)/r
    assert td.simplify_scalars(comp(div_T, 0) - exp_r).latex() == "0"
    assert td.simplify_scalars(comp(div_T, 1) - exp_th).latex() == "0"
    assert td.simplify_scalars(comp(div_T, 2) - exp_z).latex() == "0"


def test_route_a_div_of_abstract_field_and_components():
    # vibe 000073 Route A: cyl.div(T) on an ABSTRACT field expands-first
    # internally, and cyl.components surfaces the scalar equations — matching the
    # explicit Route B expansion, connection terms and all.
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    T = t.field("T", 2, deps=[r, th], ctx=ctx)

    # Route A: abstract field straight into the operator, then surface.
    eqs = chart.components(chart.div(T))
    assert len(eqs) == 3

    # Route B reference: expand T explicitly, then div, then project.
    frame = chart.physical_frame()
    e = [frame.direction(k) for k in range(3)]
    Tc = td.canonicalize(td.unroll_sums(
        tb.expand_in_basis(T, frame, tb.Variance.Covariant)))
    div_ref = chart.div(Tc)

    def red(x):
        x = tb.simplify_basis_dot(td.expand_products(x), frame)
        return td.simplify_scalars(td.fold_arithmetic(
            td.eval_delta_concrete(td.canonicalize(x))))

    for i in range(3):
        assert td.simplify_scalars(eqs[i] - red(div_ref @ e[i])).latex() == "0"


def test_expand_of_invariant_matches_div_components():
    # vibe 000073: chart.expand(div(T)) surfaces the same components as
    # chart.components(div(T)) — the general 'expand an invariant' path.
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    T = t.field("T", 2, deps=[r, th], ctx=ctx)
    e = [chart.physical_frame().direction(k) for k in range(3)]
    full = chart.expand(chart.div(T))
    eqs = chart.components(chart.div(T))
    for i in range(3):
        assert td.simplify_scalars(chart.components(full)[i] - eqs[i]).latex() == "0"


def test_mixed_coordinate_subscript_spacing():
    # vibe 000073 Gap 4: a LaTeX-command index (\theta) followed by a Latin one
    # (r) must render as "\theta r", not the invalid command "\thetar".
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    frame = chart.physical_frame()
    T = t.field("T", 2, ctx=ctx)
    Tc = td.canonicalize(td.unroll_sums(
        tb.expand_in_basis(T, frame, tb.Variance.Covariant)))
    s = Tc.latex()
    assert r"\thetar" not in s
    assert r"T_{\theta r}" in s   # spaced
    assert "T_{rr}" in s          # plain Latin stays unspaced


def test_spherical_laplacian():
    ctx = t.Context()
    ref = tb.wcs(ctx)
    r = t.coordinate("r", chart_id=3, slot=0, nonneg=True, ctx=ctx)
    th = t.coordinate(r"\theta", chart_id=3, slot=1, ctx=ctx)
    ph = t.coordinate(r"\phi", chart_id=3, slot=2, ctx=ctx)
    chart = tc.CoordinateChart(
        ref,
        [r, th, ph],
        [r * t.sin(th) * t.cos(ph), r * t.sin(th) * t.sin(ph), r * t.cos(th)],
    )
    assert td.algebraic_eq(chart.laplacian(r**2), t.scalar(6, ctx=ctx))


def test_cylindrical_rot():
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    fb = chart.physical_frame()
    v = r * fb.direction(1)  # r e_theta
    # rot(r e_theta) = 2 e_z (uniform vorticity), intrinsically on the frame.
    assert td.structural_eq(
        td.canonicalize(chart.rot(v)),
        td.canonicalize(t.scalar(2, ctx=ctx) * fb.direction(2)),
    )


def test_collect_terms_on_second_gradient():
    # ∇∇(f(r) sin θ) comes out as six raw terms; collect_terms groups them by
    # dyad to one per e_i⊗e_j (vibe 000071).
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    fb = chart.physical_frame()
    from tender.operators import nabla

    f = t.field("f", 0, deps=[r], ctx=ctx)
    x = (nabla * (nabla * (f * t.sin(th)))).evaluate(chart)
    collected = td.collect_terms(x)
    tex = collected.latex()
    # Four distinct dyads, each appearing once.
    for a, b in [("r", "r"), ("r", "\\theta"), ("\\theta", "r"), ("\\theta", "\\theta")]:
        dyad = "\\mathbf{e}_{" + a + "} \\, \\mathbf{e}_{" + b + "}"
        assert tex.count(dyad) == 1, (a, b, tex)


def test_rot_of_radius_cross_identity():
    # A field built with the identity tensor and a cross reduces on the frame
    # (vibe 000071): ∇×(R×I) = −2I, with R on the chart's frame.
    ctx = t.Context()
    ref = tb.wcs(ctx)
    x = t.coordinate("x", chart_id=1, slot=0, ctx=ctx)
    y = t.coordinate("y", chart_id=1, slot=1, ctx=ctx)
    z = t.coordinate("z", chart_id=1, slot=2, ctx=ctx)
    chart = tc.CoordinateChart(ref, [x, y, z], [x, y, z])
    fb = chart.physical_frame()
    I = t.identity(ctx)
    R = x * fb.direction(0) + y * fb.direction(1) + z * fb.direction(2)
    want = td.canonicalize(t.scalar(-2, ctx=ctx) * I)
    assert td.structural_eq(td.canonicalize(chart.rot(R % I)), want)


def test_basis_to_basis_expansion():
    # vibe 000071 P4: a frame result can be brought to WCS on demand, and a WCS
    # vector expressed in a curvilinear frame; the round-trip recovers the frame.
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    fb = chart.physical_frame()
    ref = tb.wcs(ctx)
    e_r = fb.direction(0)

    # to_reference: e_r = cos θ i + sin θ j.
    er_wcs = t.cos(th) * ref.basis(0) + t.sin(th) * ref.basis(1)
    assert td.algebraic_eq(chart.to_reference(e_r), er_wcs)

    # express: WCS i in the cylindrical frame is cos θ e_r − sin θ e_θ.
    want = t.cos(th) * fb.direction(0) - t.sin(th) * fb.direction(1)
    assert td.structural_eq(
        td.canonicalize(chart.express(ref.basis(0))), td.canonicalize(want)
    )
    # Round-trip WCS → frame recovers e_r.
    assert td.algebraic_eq(chart.express(chart.to_reference(e_r)), e_r)


def test_intrinsic_position():
    # vibe 000072 Obs 6: position() is the radius vector in the chart's own frame
    # (r e_r + z e_z), so grad(position) folds to I with no mixed frame.
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    fb = chart.physical_frame()
    want = r * fb.direction(0) + z * fb.direction(2)
    assert td.structural_eq(
        td.canonicalize(chart.position()), td.canonicalize(want)
    )
    assert td.structural_eq(chart.grad(chart.position()), t.identity(ctx))


def test_cartesian_frame_reuses_reference():
    # vibe 000072 Obs 1: the identity chart's physical frame IS the reference
    # basis, so it prints i, j, k (shared basis id), and express keeps those names.
    ctx = t.Context()
    ref = tb.wcs(ctx)
    x = t.coordinate("x", chart_id=7, slot=0, ctx=ctx)
    y = t.coordinate("y", chart_id=7, slot=1, ctx=ctx)
    z = t.coordinate("z", chart_id=7, slot=2, ctx=ctx)
    chart = tc.CoordinateChart(ref, [x, y, z], [x, y, z])
    fb = chart.physical_frame()
    assert fb.basis_id == ref.basis_id
    R = chart.radius_vector()
    # express prints under the reference names i, j, k — not e_x, e_y, e_z.
    ex_latex = chart.express(R).latex()
    assert "\\mathbf{i}" in ex_latex and "e_{x}" not in ex_latex
    assert td.algebraic_eq(chart.to_reference(chart.express(R)), R)
    # A curvilinear frame is a distinct basis.
    _, _, _, cyl = make_cylindrical(ctx)
    assert cyl.physical_frame().basis_id != ref.basis_id


def test_express_folds_and_projects_all_legs():
    # vibe 000072 Obs 8: a cross-frame ∇R re-expressed in the Cartesian frame
    # folds to I (express and to_reference both collapse the resolution), and a
    # rank-2 dyad gets *both* legs projected onto the target frame.
    ctx = t.Context()
    ref = tb.wcs(ctx)
    x = t.coordinate("x", chart_id=7, slot=0, ctx=ctx)
    y = t.coordinate("y", chart_id=7, slot=1, ctx=ctx)
    z = t.coordinate("z", chart_id=7, slot=2, ctx=ctx)
    cart = tc.CoordinateChart(ref, [x, y, z], [x, y, z])
    r, th, zc, cyl = make_cylindrical(ctx)

    grad_R = cyl.grad(cyl.radius_vector())
    assert td.structural_eq(cart.express(grad_R), t.identity(ctx))
    assert td.structural_eq(cyl.to_reference(grad_R), t.identity(ctx))

    # i⊗j in the cylindrical frame: both legs on e_r/e_θ (no leftover i, j).
    fb = cyl.physical_frame()
    er, et = fb.direction(0), fb.direction(1)
    cs, sn = t.cos(th), t.sin(th)
    want = (
        sn * cs * (er * er)
        + cs * cs * (er * et)
        - sn * sn * (et * er)
        - sn * cs * (et * et)
    )
    ij = ref.basis(0) * ref.basis(1)
    assert td.structural_eq(
        td.simplify_scalars(td.canonicalize(cyl.express(ij))),
        td.simplify_scalars(td.canonicalize(want)),
    )


def test_physical_frame_rebuilds_on_chart_id_reuse():
    # vibe 000072 Obs 2: reusing a chart_id for a different chart rebuilds the
    # frame (fingerprint-validated cache) instead of returning the stale one.
    ctx = t.Context()
    ref = tb.wcs(ctx)
    r = t.coordinate("r", chart_id=5, slot=0, nonneg=True, ctx=ctx)
    th = t.coordinate(r"\theta", chart_id=5, slot=1, ctx=ctx)
    ph = t.coordinate(r"\varphi", chart_id=5, slot=2, ctx=ctx)
    sph = tc.CoordinateChart(
        ref,
        [r, th, ph],
        [
            r * t.sin(th) * t.cos(ph),
            r * t.sin(th) * t.sin(ph),
            r * t.cos(th),
        ],
    )
    sid = sph.physical_frame().basis_id

    z = t.coordinate("z", chart_id=5, slot=2, ctx=ctx)
    cyl = tc.CoordinateChart(
        ref, [r, th, z], [r * t.cos(th), r * t.sin(th), z]
    )
    assert cyl.physical_frame().basis_id != sid  # not the stale spherical frame


def test_chart_construction_validates_slots():
    # vibe 000072 Obs 3: coords must be slot 0..n-1 of one chart_id, and the
    # embedding must have one component per reference direction.
    import pytest

    ctx = t.Context()
    ref = tb.wcs(ctx)
    r = t.coordinate("r", chart_id=2, slot=0, ctx=ctx)
    th = t.coordinate(r"\theta", chart_id=2, slot=1, ctx=ctx)
    z = t.coordinate("z", chart_id=2, slot=2, ctx=ctx)

    with pytest.raises(ValueError, match="slots 0"):
        tc.CoordinateChart(ref, [th, r, z], [r, th, z])  # slots 1,0,2
    with pytest.raises(ValueError, match="one component per reference"):
        tc.CoordinateChart(ref, [r, th], [r, th])  # embedding size 2 ≠ 3

    other = t.coordinate("s", chart_id=3, slot=2, ctx=ctx)  # foreign chart_id
    with pytest.raises(ValueError, match="one chart_id"):
        tc.CoordinateChart(ref, [r, th, other], [r, th, other])


def test_tensor_field_operators():
    # A tensor field is no longer seen as constant (vibe 000070 P7): div T is a
    # symbolic vector, grad f a symbolic vector of partials, and a field declared
    # on one coordinate is constant in the others.
    ctx = t.Context()
    ref = tb.wcs(ctx)
    x = t.coordinate("x", chart_id=1, slot=0, ctx=ctx)
    y = t.coordinate("y", chart_id=1, slot=1, ctx=ctx)
    z = t.coordinate("z", chart_id=1, slot=2, ctx=ctx)
    chart = tc.CoordinateChart(ref, [x, y, z], [x, y, z])

    T = chart.field("T", 2)
    div_T = chart.div(T)
    assert not td.algebraic_eq(div_T, t.scalar(0, ctx=ctx))  # no longer zero

    # grad of a scalar field f(x) only has the x-component; ∂_y f = 0.
    fx = t.field("f", 0, deps=[x], ctx=ctx)
    assert td.algebraic_eq(td.partial(fx, y), t.scalar(0, ctx=ctx))
    assert not td.algebraic_eq(td.partial(fx, x), t.scalar(0, ctx=ctx))

    # Mixed partials are symmetric.
    g = chart.field("g", 0)
    assert td.structural_eq(
        td.partial(td.partial(g, x), y), td.partial(td.partial(g, y), x)
    )


def test_rot_of_constant_is_zero():
    # rot of a cross of two constant (non-field, non-frame) vectors is 0.
    ctx = t.Context()
    ref = tb.wcs(ctx)
    x = t.coordinate("x", chart_id=1, slot=0, ctx=ctx)
    y = t.coordinate("y", chart_id=1, slot=1, ctx=ctx)
    z = t.coordinate("z", chart_id=1, slot=2, ctx=ctx)
    chart = tc.CoordinateChart(ref, [x, y, z], [x, y, z])
    a = t.tensor("a", rank=1, ctx=ctx)
    b = t.tensor("b", rank=1, ctx=ctx)
    assert td.algebraic_eq(chart.rot(a % b), t.scalar(0, ctx=ctx))
