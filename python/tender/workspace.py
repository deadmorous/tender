"""tender.workspace — a context-bound facade (vibe 000070 P1/P2).

Driving the differential layer meant threading an explicit ``ctx`` through every
constructor and restating each coordinate's ``chart_id`` and ``slot`` by hand,
kept in sync with its position in the chart.  :class:`Workspace` removes both:
it owns a :class:`~tender.Context`, forwards the common factories with ``ctx``
bound, and mints chart coordinate atoms with their ``chart_id``/``slot`` filled
in automatically::

    import tender as t

    ws = t.Workspace()
    I = ws.identity()
    WCS = ws.wcs()
    x, y, z = ws.coords("x", "y", "z")            # auto chart_id + slots
    cart = ws.chart(WCS, [x, y, z], [x, y, z])
    R = cart.radius_vector()

The explicit ``ctx=`` API still works unchanged; the facade is purely additive.
"""

from . import _core
from . import basis as _basis
from . import chart as _chart

__all__ = ["Workspace"]


class Workspace:
    """A context plus ctx-bound factories and coordinate minting."""

    def __init__(self):
        self.ctx = _core.Context()
        self._next_chart_id = 1

    # ---- expression factories (ctx implicit) ----------------------------

    def tensor(self, name, rank=None):
        """A named tensor object in this workspace's context."""
        return _core.tensor(name, rank, ctx=self.ctx)

    def field(self, name, rank, deps=None):
        """A tensor field (vibe 000070 P7); deps=None means all coordinates."""
        return _core.field(name, rank, deps=deps, ctx=self.ctx)

    def scalar(self, value):
        """A scalar literal (int or Rational)."""
        return _core.scalar(value, ctx=self.ctx)

    def identity(self):
        """The identity tensor I."""
        return _core.identity(ctx=self.ctx)

    def coordinate(self, name, chart_id=0, slot=0, nonneg=False):
        """A single chart coordinate variable (use :meth:`coords` for a set)."""
        return _core.coordinate(
            name, chart_id=chart_id, slot=slot, nonneg=nonneg, ctx=self.ctx
        )

    # ---- bases ----------------------------------------------------------

    def wcs(self):
        """The world (orthonormal Cartesian) coordinate system i, j, k."""
        return _basis.wcs(self.ctx)

    def cylindrical(self):
        return _basis.cylindrical(self.ctx)

    def spherical(self):
        return _basis.spherical(self.ctx)

    def polar_2d(self):
        return _basis.polar_2d(self.ctx)

    # ---- coordinate minting (P1) ----------------------------------------

    def coords(self, *names, chart_id=None, nonneg=()):
        """Mint a set of coordinate atoms, slots filled in by position.

        All coordinates share one ``chart_id`` (a fresh one per call unless
        given), and take slots 0, 1, 2, … in the order named — so the chart's
        slots need not be restated.  ``nonneg`` is the collection of names known
        to be ≥ 0 (e.g. a radius), which licenses √(r²) → r.  Returns a list, so
        ``x, y, z = ws.coords("x", "y", "z")`` unpacks.
        """
        if chart_id is None:
            chart_id = self._next_chart_id
            self._next_chart_id += 1
        nn = set(nonneg)
        return [
            _core.coordinate(
                name, chart_id=chart_id, slot=i, nonneg=name in nn, ctx=self.ctx
            )
            for i, name in enumerate(names)
        ]

    # ---- chart (P1) -----------------------------------------------------

    def chart(self, reference, coords, embedding):
        """A coordinate chart from a reference basis, coords, and embedding."""
        return _chart.CoordinateChart(reference, coords, embedding)
