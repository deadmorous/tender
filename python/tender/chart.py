"""tender.chart — coordinate charts and derived curvilinear geometry.

A :class:`CoordinateChart` is a coordinate mapping from chart coordinates
``q^i`` to an orthonormal reference (Cartesian) frame.  From the mapping the
whole orthogonal-curvilinear geometry is *derived* — the radius vector, the
holonomic tangent basis ``g_i = ∂R/∂q^i``, the metric ``g_ij``, the scale
factors ``h_i``, and the physical orthonormal frame ``e_i = g_i / h_i`` —
rather than hand-supplied (vibe 000069 M4)::

    import tender
    import tender.basis as tb
    import tender.chart as tc

    ctx = tender.Context()
    cart = tb.wcs(ctx)
    r = tender.coordinate("r", chart_id=1, slot=0, nonneg=True, ctx=ctx)
    th = tender.coordinate("\\theta", chart_id=1, slot=1, ctx=ctx)
    z = tender.coordinate("z", chart_id=1, slot=2, ctx=ctx)
    chart = tc.CoordinateChart(
        cart, [r, th, z], [r * tender.cos(th), r * tender.sin(th), z])

    chart.metric_component(1, 1).latex()   # r^{2}
    chart.scale_factor(1).latex()          # r
    e = chart.physical_basis()             # e_r, e_theta, e_z
"""

from tender._core import chart as _c

CoordinateChart = _c.CoordinateChart

__all__ = [
    "CoordinateChart",
]
