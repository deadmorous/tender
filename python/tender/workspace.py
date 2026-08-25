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
        self._wcs = None

    # ---- expression factories (ctx implicit) ----------------------------

    def tensor(self, name, rank=None):
        """A named tensor object in this workspace's context."""
        return _core.tensor(name, rank, ctx=self.ctx)

    def field(self, name, rank, deps=None, symmetric=False):
        """A tensor field (vibe 000070 P7); deps=None means all coordinates.

        ``symmetric=True`` marks a rank-2 field symmetric (T_ij = T_ji).
        """
        return _core.field(
            name, rank, deps=deps, symmetric=symmetric, ctx=self.ctx
        )

    def scalar(self, value):
        """A scalar literal (int or Rational)."""
        return _core.scalar(value, ctx=self.ctx)

    def identity(self, space=None):
        """The identity tensor I.

        It carries its dimension (vibe 000082), so ``tr(I)`` folds to ``n``;
        ``space`` defaults to 3-D — there is no dimension-agnostic identity.
        Pass ``space=t.space_2d`` etc. for other dimensions.
        """
        return _core.identity(ctx=self.ctx, space=space)

    def coordinate(self, name, chart_id=0, slot=0, nonneg=False):
        """A single chart coordinate variable (use :meth:`coords` for a set)."""
        return _core.coordinate(
            name, chart_id=chart_id, slot=slot, nonneg=nonneg, ctx=self.ctx
        )

    # ---- bases ----------------------------------------------------------

    def wcs(self):
        """The world (orthonormal Cartesian) coordinate system i, j, k.

        Memoised: the world frame is unique per workspace, so every call returns
        the *same* basis.  Charts built over it therefore share one reference —
        the precondition for relating their coordinates across charts (vibe
        000090), e.g. evaluating a Cartesian quantity in a cylindrical chart.
        """
        if self._wcs is None:
            self._wcs = _basis.wcs(self.ctx)
        return self._wcs

    def nabla(self):
        """The coordinate-free ∇ operator, bound to this workspace's context.

        A real :class:`~tender.Expr`, so it composes with everything: build
        the physics without choosing coordinates —

            nabla = ws.nabla()
            T = lam * (nabla @ u) * I + mu * (nabla * u + (nabla * u).transpose())

        — then hand it to a chart with :meth:`chart.evaluate` when you want
        components.  ``nabla * T`` is the gradient, ``nabla @ T`` the
        divergence, ``nabla % T`` the rotor.

        Use this when the *statement* should be coordinate-free.  When you
        simply want a derivative in a chart you already have, the chart's own
        ``grad`` / ``div`` / ``rot`` / ``laplacian`` say so more directly.
        """
        return _core.nabla(ctx=self.ctx)

    # ---- named charts (vibe 000098) -------------------------------------
    #
    # The standard coordinate systems, by name.  Each mints its own
    # coordinates — with `nonneg` set where it matters, which is not a
    # detail: a radius declared non-negative is what licenses √(r²) → r, and
    # forgetting it makes scale factors fail to simplify in ways that are
    # tedious to trace back.  Each returns `(chart, coords)`, so:
    #
    #     cyl, (r, th, z) = ws.cylindrical_chart()
    #
    # For a coordinate system tender does not name, build the chart directly
    # from its embedding — see `examples/custom_chart.py`.

    def cartesian_chart(self, names=("x", "y", "z")):
        """The Cartesian chart — the identity embedding on the world frame."""
        coords = self.coords(*names)
        return self.chart(self.wcs(), coords, list(coords)), coords

    def cylindrical_chart(self, names=("r", r"\theta", "z")):
        """Cylindrical (r, θ, z):  x = r cosθ,  y = r sinθ,  z = z."""
        r, th, z = self.coords(*names, nonneg=(names[0],))
        embedding = [r * _core.cos(th), r * _core.sin(th), z]
        return self.chart(self.wcs(), [r, th, z], embedding), (r, th, z)

    def spherical_chart(self, names=("r", r"\theta", r"\phi")):
        """Spherical (r, θ, φ):  x = r sinθ cosφ,  y = r sinθ sinφ,  z = r cosθ.

        θ is the polar angle from the z axis (the physics convention, and
        Lurie's).
        """
        r, th, ph = self.coords(*names, nonneg=(names[0],))
        embedding = [
            r * _core.sin(th) * _core.cos(ph),
            r * _core.sin(th) * _core.sin(ph),
            r * _core.cos(th),
        ]
        return self.chart(self.wcs(), [r, th, ph], embedding), (r, th, ph)

    def polar_chart(self, names=("r", r"\theta")):
        """Plane polar (r, θ), embedded in the z = 0 plane of the world frame."""
        r, th = self.coords(*names, nonneg=(names[0],))
        embedding = [r * _core.cos(th), r * _core.sin(th), self.scalar(0)]
        return self.chart(self.wcs(), [r, th], embedding), (r, th)

    # ---- the well-known *bases* (superseded by the charts above) ---------
    #
    # A basis alone carries no coordinate map, so it cannot differentiate:
    # prefer the named charts, which derive the frame, metric, scale factors
    # and connection from the embedding.  Kept for the basis-level tests.

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
