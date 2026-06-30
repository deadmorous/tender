"""Coordinate charts (vibe 000069 M4): derive the orthogonal-curvilinear
geometry (radius vector, tangent basis, metric, scale factors, physical frame)
from a coordinate mapping."""

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
    # The operators return invariant tensors: grad R = I, div R = 3, rot R = 0.
    ctx = t.Context()
    ref = tb.wcs(ctx)
    x = t.coordinate("x", chart_id=7, slot=0, ctx=ctx)
    y = t.coordinate("y", chart_id=7, slot=1, ctx=ctx)
    z = t.coordinate("z", chart_id=7, slot=2, ctx=ctx)
    chart = tc.CoordinateChart(ref, [x, y, z], [x, y, z])
    R = chart.radius_vector()
    # grad R folds the resolution of identity Σ_k e_k⊗e_k back to I itself
    # (vibe 000070 P4), so the result is structurally the identity tensor.
    assert td.structural_eq(chart.gradient(R), t.identity(ctx))
    # The raw (unfolded) form is the concrete dyad sum.
    identity = sum(
        (ref.basis(k) * ref.basis(k) for k in range(1, 3)), ref.basis(0) * ref.basis(0)
    )
    assert _inv_eq(chart.gradient(R, fold_identity=False), identity)
    assert td.algebraic_eq(chart.divergence(R), t.scalar(3, ctx=ctx))
    assert td.algebraic_eq(chart.rot(R), t.scalar(0, ctx=ctx))


def test_cylindrical_gradient():
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    fb = chart.physical_basis()
    one = t.scalar(1, ctx=ctx)
    # grad(theta) = (1/r) e_theta
    assert _inv_eq(chart.gradient(th), (one / r) * fb.basis(1))
    # grad(r^2) = 2r e_r
    assert _inv_eq(chart.gradient(r**2), (t.scalar(2, ctx=ctx) * r) * fb.basis(0))


def test_cylindrical_divergence_and_laplacian():
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    v = r * chart.physical_basis().basis(0)  # r e_r
    assert td.algebraic_eq(chart.divergence(v), t.scalar(2, ctx=ctx))
    assert td.algebraic_eq(chart.laplacian(r**2), t.scalar(4, ctx=ctx))


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
    v = r * chart.physical_basis().basis(1)  # r e_theta
    # rot(r e_theta) = 2 e_z = 2k (uniform vorticity).
    assert _inv_eq(chart.rot(v), t.scalar(2, ctx=ctx) * tb.wcs(ctx).basis(2))


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
    div_T = chart.divergence(T)
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


def test_rot_of_radius_cross_identity():
    # A rank-2 field built with the identity tensor and a cross — R×I, the skew
    # tensor with (R×I)·a = R×a — reduces instead of crashing (vibe 000070 P6).
    # ∇×(R×I) = I − 3I = −2I.
    ctx = t.Context()
    identity = t.identity(ctx)
    ref = tb.wcs(ctx)
    x = t.coordinate("x", chart_id=1, slot=0, ctx=ctx)
    y = t.coordinate("y", chart_id=1, slot=1, ctx=ctx)
    z = t.coordinate("z", chart_id=1, slot=2, ctx=ctx)
    chart = tc.CoordinateChart(ref, [x, y, z], [x, y, z])
    R = chart.radius_vector()
    want = t.scalar(-2, ctx=ctx) * identity
    assert td.structural_eq(chart.rot(R % identity), td.canonicalize(want))
    # A cross of two constant vectors differentiates to 0 (gracefully, no crash).
    a = t.tensor("a", rank=1, ctx=ctx)
    b = t.tensor("b", rank=1, ctx=ctx)
    assert td.algebraic_eq(chart.rot(a % b), t.scalar(0, ctx=ctx))
