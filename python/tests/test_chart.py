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


def test_cylindrical_gradient():
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    one, zero = t.scalar(1, ctx=ctx), t.scalar(0, ctx=ctx)
    # grad(theta) = (1/r) e_theta
    assert _coeffs_eq(chart.gradient(th), [zero, one / r, zero])
    # grad(r^2) = 2r e_r
    assert _coeffs_eq(chart.gradient(r**2), [t.scalar(2, ctx=ctx) * r, zero, zero])


def test_cylindrical_divergence_and_laplacian():
    ctx = t.Context()
    r, th, z, chart = make_cylindrical(ctx)
    zero = t.scalar(0, ctx=ctx)
    assert td.algebraic_eq(
        chart.divergence([r, zero, zero]), t.scalar(2, ctx=ctx)
    )
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
    zero = t.scalar(0, ctx=ctx)
    # rot(r e_theta) = 2 e_z (uniform vorticity).
    assert _coeffs_eq(
        chart.rot([zero, r, zero]), [zero, zero, t.scalar(2, ctx=ctx)]
    )
