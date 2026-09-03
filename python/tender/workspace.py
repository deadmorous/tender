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
from . import derivation as _td
from . import basis as _basis
from . import chart as _chart
from . import mechanics as _mechanics
from . import rotation as _rotation

__all__ = ["Workspace"]


class Workspace:
    """A context plus ctx-bound factories and coordinate minting."""

    def __init__(self):
        self.ctx = _core.Context()
        self._next_chart_id = 1
        self._wcs = None
        # name -> the expression a constructed rotation stands for (vibe
        # 000110 I5); read by `definition`, never by the algebra.
        self._definitions = {}

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

    # ---- constrained symbols (vibe 000110 I4) ---------------------------

    def vector(self, name, unit=False):
        """A rank-1 tensor; ``unit=True`` declares ``|n| = 1``.

        The constraint is a property of the symbol, not of this expression:
        it mints the rewrite rule ``n·n → 1`` and it is in force in every claim
        about ``n``, whoever built the expression.
        """
        if not unit:
            return _core.tensor(name, 1, ctx=self.ctx)
        return _core.constrained_tensor(name, 1, "unit", ctx=self.ctx)

    def rotation(self, name, axis=None, angle=None):
        """A proper orthogonal tensor: ``P·Pᵀ = Pᵀ·P = I``, det P = +1.

        With just a name, an *abstract* rotation — some rotation, nothing more.
        With an axis and an angle, **the** rotation about that axis by that
        angle, built from the three-term formula and verified on construction::

            P = n⊗n + (I − n⊗n) cos θ + (n × I) sin θ

        `axis` must be a declared unit vector (``ws.vector(..., unit=True)``);
        the formula is registered as the symbol's definition, so
        ``ws.definition(P)`` unfolds it where the formula is wanted.

        Rotations compose — a ``·``-chain of them is a rotation — which needs
        no declaring: it follows from the minted rules and the transpose group.
        For a tensor containing a reflection use :meth:`reflection`, or
        :meth:`orthogonal` with ``proper=False``.
        """
        if (axis is None) != (angle is None):
            raise ValueError(
                "give both an axis and an angle, or neither: "
                "ws.rotation('P') is an abstract rotation, and "
                "ws.rotation('P', n, theta) is the rotation about n by theta"
            )
        if axis is None:
            return self.orthogonal(name, proper=True)
        from . import _core as _c

        I = self.identity()
        nn = axis * axis
        form = nn + (I - nn) * _c.cos(angle) + (axis % I) * _c.sin(angle)
        return self.orthogonal_from(name, form, proper=True)

    def orthogonal(self, name, proper=True):
        """An orthogonal tensor, proper (a rotation) or improper.

        ``proper`` is your assertion and is recorded as one: ``P·Pᵀ = I`` holds
        for both kinds and tender has no determinant, so nothing here can check
        it (vibe 000110 I5).  It matters because the transport rules — cross
        products survive a rotation and are reversed by a reflection — depend
        on it.
        """
        return _core.constrained_tensor(
            name, 2, "orthogonal", proper=proper, ctx=self.ctx
        )

    def orthogonal_from(
        self, name, expr, proper=True, rounds=None, frames=()
    ):
        """Name *expr* as an orthogonal tensor — after verifying that it is one.

        The general path, and the reason the list of shipped forms below need
        not be closed (vibe 000110 I5): hand in a form the library has never
        seen, and it is checked against the constraints already declared before
        it is stamped.

            n = ws.vector("n", unit=True)
            Q = ws.orthogonal_from("Q", I - 2*(n*n), proper=False)

        Returns a *symbol* carrying the property, with `expr` registered as its
        definition (see :meth:`definition`).  Raises with the residual if the
        form does not reduce — a refusal, not a warning, because a stamp that
        might be wrong is worse than no stamp.

        ``proper`` cannot be verified: `X·Xᵀ = I` holds for a reflection too and
        tender has no determinant.  It is your assertion, recorded as one.
        """
        kw = {} if rounds is None else {"rounds": rounds}
        residual = _rotation.verify_orthogonal(
            self.ctx, expr, self.identity(), frames=frames, **kw
        )
        if residual is not None:
            raise ValueError(
                f"{name!r} is not orthogonal, so it was not declared: "
                f"{residual}.  If the form is right and the library merely "
                f"cannot see it, the missing step is a rule, not a stamp."
            )
        symbol = self.orthogonal(name, proper=proper)
        self._definitions[name] = (symbol, expr)
        return symbol

    def reflection(self, name, n):
        """The reflection `I − 2 n⊗n` in the plane ⟂ `n`; improper.

        `n` must be a declared unit vector (`ws.vector(..., unit=True)`) —
        without `n·n = 1` the form is not orthogonal and the verification says
        so rather than stamping it anyway.
        """
        I = self.identity()
        return self.orthogonal_from(name, I - 2 * (n * n), proper=False)

    def frame_rotation(self, name, frame, reference):
        """The rotation carrying `reference` onto `frame`: ``P = Σ e_i ⊗ E_i``.

        Both frames must be orthonormal.  The result is proper when they share
        an orientation and improper otherwise — the one sign the library can
        settle for itself, because a `Basis` records its handedness as the sign
        of its cell volume.

        Verified like the others, but the reduction needs the frames as well as
        the rules: `e_i·e_j = δ_ij` is knowledge held by the basis, not an
        identity about symbols.
        """
        if not (frame.is_orthonormal and reference.is_orthonormal):
            raise ValueError(
                f"{name!r}: both frames must be orthonormal — "
                "P = Σ e_i ⊗ E_i is a rotation only between orthonormal frames"
            )
        if frame.dim != reference.dim:
            raise ValueError(
                f"{name!r}: the frames have different dimensions "
                f"({frame.dim} and {reference.dim})"
            )
        form = None
        for k in range(frame.dim):
            dyad = frame.direction(k) * reference.direction(k)
            form = dyad if form is None else form + dyad
        proper = _td.algebraic_eq(frame.volume, reference.volume)
        return self.orthogonal_from(
            name, form, proper=proper, frames=(frame, reference)
        )

    def definition(self, symbol):
        """The defining identity of a constructed rotation, for unfolding.

            P = ws.turn("P", n, theta)
            td.apply_identity(expr, ws.definition(P))    # P → its formula

        Raises for a symbol that has no definition — an abstract
        `ws.rotation("P")` is a rotation and nothing more.
        """
        # Matched structurally rather than by the rendered name: a rank-2
        # symbol renders bolded, and string surgery on LaTeX is not identity.
        for name, (declared, expr) in self._definitions.items():
            if _td.structural_eq(declared, symbol):
                return _td.Identity(f"{name}-definition", declared, expr)
        raise ValueError(
            f"{symbol} has no definition: it was declared orthogonal, not "
            "constructed.  Only the forms built by reflection / turn / "
            "orthogonal_from carry one."
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

    def coordinate(
        self, name, chart_id=0, slot=0, nonneg=False, nonspatial=False
    ):
        """A single chart coordinate variable (use :meth:`coords` for a set).

        ``nonspatial=True`` marks an independent variable that is not a
        coordinate of space (time — see :meth:`time`): ∂ then passes through
        anything describing the frame, so ∂ₜ(∇⊗u) = ∇⊗(∂ₜu).
        """
        return _core.coordinate(
            name,
            chart_id=chart_id,
            slot=slot,
            nonneg=nonneg,
            nonspatial=nonspatial,
            ctx=self.ctx,
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

    def coords(self, *names, chart_id=None, nonneg=(), nonspatial=False):
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
                name,
                chart_id=chart_id,
                slot=i,
                nonneg=name in nn,
                nonspatial=nonspatial,
                ctx=self.ctx,
            )
            for i, name in enumerate(names)
        ]

    # ---- time and the configuration chain (vibe 000110) -----------------

    def time(self, name="t"):
        """Time, the generalized coordinates moving with it, and δ.

        Returns a :class:`~tender.mechanics.Time` owning its own coordinate
        group::

            tm = ws.time("t")
            q, qd, qdd = tm.coordinate("q", orders=2)
            L = tm.field("L", 0, deps=[q, qd, tm.t])
            ddt, delta = tm.ddt(), tm.variation()

        Both operators are ordinary ``Σ c_k ∂_k`` derivations, so
        :func:`tender.derivation.apply_operators` applies them; what the
        factory owns is that they *commute* (vibe 000110).
        """
        return _mechanics.Time(self, name)

    # ---- chart (P1) -----------------------------------------------------

    def chart(self, reference, coords, embedding):
        """A coordinate chart from a reference basis, coords, and embedding."""
        return _chart.CoordinateChart(reference, coords, embedding)
