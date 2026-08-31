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


# ---------------------------------------------------------------------------
# The chart's rewriting steps, as functions.
#
# Every one of these is a method on `CoordinateChart` and stays one; these are
# thin aliases in the shape the rest of the library uses — the expression
# first, the thing it is done with second, exactly as `tender.basis` takes
# `(expr, basis)`.  Written this way they can be catalogued in
# :mod:`tender.steps`, which methods on a parameter object could not be, and a
# third of the moves in real derivations happen on a chart (vibe 000108 §14).
# ---------------------------------------------------------------------------


def expand_nabla(expr, chart):
    """Expand every chart-free ∇ into the free-index frame form ``e_i ∂_i``.

    The first move of a ∇ derivation: it turns an invariant operator into
    something the index machinery can work on.  Constant unit-scale (Cartesian)
    frames only — see :func:`evaluate` for the curvilinear route.
    """
    return chart.expand_nabla(expr)


def componentize_nabla(expr, chart):
    """Lower an :func:`expand_nabla` result to concrete components."""
    return chart.componentize_nabla(expr)


def reassemble_nabla(expr, chart):
    """Fold a reduced free-index expression back into chart-free ∇ operators."""
    return chart.reassemble_nabla(expr)


def evaluate(expr, chart):
    """Lower an invariant core-∇ expression onto this chart's operators.

    ``∇·X → div``, ``∇⊗X → grad``, ``∇×X → rot``, ``∇·(∇⊗X) → Δ`` — the
    curvilinear-correct route, moving-frame terms and all, without rewriting
    by hand through grad/div/rot (vibe 000084).
    """
    return chart.evaluate(expr)


def expand(expr, chart):
    """Expand every abstract tensor field into components on the physical frame.

    A field derivative is expanded by Leibniz *with the connection*, so the
    moving-frame terms are kept.
    """
    return chart.expand(expr)


def express(expr, chart):
    """Re-express in this chart's physical frame — the general change of basis."""
    return chart.express(expr)


def to_reference(expr, chart):
    """Re-express in the reference (WCS) frame: ``e_r → cos θ i + sin θ j``."""
    return chart.to_reference(expr)


__all__ = [
    "CoordinateChart",
    "componentize_nabla",
    "evaluate",
    "expand",
    "expand_nabla",
    "express",
    "reassemble_nabla",
    "to_reference",
]
