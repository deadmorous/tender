"""tender.operators — first-class ∇ and ∂_q (vibe 000070 P8).

``∇`` (nabla) and ``∂_q`` are made composable expression objects, so the
differential operators are built rather than called, and users can write their
own (the directional derivative ``v·∇``, the material derivative ``∂_t + v·∇``,
…).  Following the settled decisions:

1. **Chart-free ∇** — :data:`nabla` is a pure symbol; the chart is supplied at
   evaluation, so a ∇ expression can be built and inspected without choosing a
   coordinate system.
2. **Explicit evaluation** — building stays symbolic (``(nabla @ (nabla * f))``
   renders as ``∇·∇f``); nothing computes until you call ``.evaluate(chart)``
   (or :func:`evaluate`).
3. **Laplacian derivable from ∇·∇** — :func:`laplacian` is a citable atom that
   renders as ``Δ`` but evaluates through the single ``div(grad)`` rule, so it
   and ``nabla @ (nabla * f)`` agree by construction.

This is the explicit-evaluation surface; it lowers to the chart's M6 operators
(:meth:`grad` / :meth:`div` / :meth:`rot` / :meth:`laplacian`) and
the M2 partial differentiator, so those become thin wrappers::

    import tender as t
    from tender.operators import nabla, d, laplacian

    ws = t.Workspace()
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    f = cart.field("f", 0)

    expr = nabla @ (nabla * f)      # symbolic ∇·∇f
    print(expr.latex())            # "\\nabla \\cdot \\nabla f"
    lap = expr.evaluate(cart)      # == cart.laplacian(f)
"""

from . import derivation as _td

__all__ = ["Nabla", "nabla", "d", "laplacian", "evaluate"]


def _needs_parens(s):
    """True if `s` has a top-level (paren/brace-depth 0) additive or contraction
    operator, so it must be parenthesised when it becomes an operator's operand
    (e.g. ∇×(x i + y j + z k), ∇×(R×I))."""
    depth = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c in "{(":
            depth += 1
        elif c in "})":
            depth -= 1
        elif depth == 0:
            if s.startswith(" + ", i) or s.startswith(" - ", i):
                return True
            if s.startswith("\\times", i) or s.startswith("\\cdot", i):
                return True
        i += 1
    return False


def _operand_latex(x):
    s = x.latex()
    return "\\left(" + s + "\\right)" if _needs_parens(s) else s


def _eval_operand(x, chart):
    if isinstance(x, DifferentialExpr):
        return x.evaluate(chart)
    return x  # a plain invariant Expr (the field/tensor the operator acts on)


class DifferentialExpr:
    """A deferred application of ∇ / ∂_q, evaluated against a chart.

    Composable: an operand may itself be a :class:`DifferentialExpr`, so
    ``nabla @ (nabla * f)`` nests ``div`` over ``grad``.
    """

    def __init__(self, kind, operand, coord=None):
        # kind ∈ {"grad", "div", "rot", "laplacian", "partial"}
        self.kind = kind
        self.operand = operand
        self.coord = coord  # set for "partial"

    # -- evaluation (explicit, decision 2) --

    def evaluate(self, chart):
        """Lower to an invariant Expr in `chart`'s reference frame."""
        inner = _eval_operand(self.operand, chart)
        if self.kind == "grad":
            return chart.grad(inner)
        if self.kind == "div":
            return chart.div(inner)
        if self.kind == "rot":
            return chart.rot(inner)
        if self.kind == "laplacian":
            # Δ ≡ ∇·∇ — the single defining rule (decision 3); chart.laplacian
            # is itself div(grad), so the atom and nabla @ (nabla * f) agree.
            return chart.laplacian(inner)
        if self.kind == "partial":
            return _td.partial(inner, self.coord)
        if self.kind == "directional":
            # (v·∇)T = v · ∇T.  Keep grad's resolution of identity unfolded so
            # (v·∇)R = v · Σ_k e_k⊗e_k = v reduces through the frame dot.
            return chart.dot(self.coord, chart.grad(inner, fold_identity=False))
        raise ValueError(f"unknown differential operator {self.kind!r}")

    # -- symbolic rendering (stays inspectable before evaluation) --

    def latex(self):
        op = _operand_latex(self.operand)
        if self.kind == "grad":
            return "\\nabla " + op
        if self.kind == "div":
            return "\\nabla \\cdot " + op
        if self.kind == "rot":
            return "\\nabla \\times " + op
        if self.kind == "laplacian":
            return "\\Delta " + op
        if self.kind == "partial":
            return "\\partial_{" + _coord_name(self.coord) + "} " + op
        if self.kind == "directional":
            return "(" + self.coord.latex() + " \\cdot \\nabla) " + op
        raise ValueError(f"unknown differential operator {self.kind!r}")

    def __repr__(self):
        return f"<DifferentialExpr {self.latex()!r}>"


def _coord_name(coord):
    # The coordinate's display letter, for ∂_x rendering.
    s = coord.latex()
    return s[len("\\mathbf{") : -1] if s.startswith("\\mathbf{") else s


class Nabla:
    """The chart-free invariant ∇ operator (decision 1).

    Reusing the expression operators: ``nabla * T`` is the gradient (⊗),
    ``nabla @ T`` the divergence (·), ``nabla % T`` the rotor (×).
    """

    def __mul__(self, operand):
        return DifferentialExpr("grad", operand)

    def __matmul__(self, operand):
        return DifferentialExpr("div", operand)

    def __mod__(self, operand):
        return DifferentialExpr("rot", operand)

    def along(self, v):
        """The directional (scalar) operator v·∇ — a custom operator built from
        ∇ (vibe 000070 P8).  Apply with ``nabla.along(v) * T``."""
        return _DirectionalOp(v)

    def latex(self):
        return "\\nabla"

    def __repr__(self):
        return "<Nabla ∇>"


class _DiffOp:
    """The scalar differential operator ∂_q for a coordinate q (chart implicit
    in q's (chart_id, slot)).  Apply with ``d(q) * T`` or ``d(q)(T)``."""

    def __init__(self, coord):
        self.coord = coord

    def __mul__(self, operand):
        return DifferentialExpr("partial", operand, coord=self.coord)

    def __call__(self, operand):
        return DifferentialExpr("partial", operand, coord=self.coord)

    def latex(self):
        return "\\partial_{" + _coord_name(self.coord) + "}"

    def __repr__(self):
        return f"<DiffOp {self.latex()!r}>"


class _DirectionalOp:
    """The directional derivative operator v·∇ (vibe 000070 P8), a custom scalar
    operator built from ∇.  Apply with ``nabla.along(v) * T`` or ``op(T)``."""

    def __init__(self, vec):
        self.vec = vec

    def __mul__(self, operand):
        return DifferentialExpr("directional", operand, coord=self.vec)

    def __call__(self, operand):
        return DifferentialExpr("directional", operand, coord=self.vec)

    def latex(self):
        return "(" + self.vec.latex() + " \\cdot \\nabla)"

    def __repr__(self):
        return f"<DirectionalOp {self.latex()!r}>"


def d(coord):
    """The differential operator ∂_q for the coordinate variable `coord`."""
    return _DiffOp(coord)


def laplacian(operand):
    """The Laplacian Δ as a citable atom (decision 3), Δ ≡ ∇·∇."""
    return DifferentialExpr("laplacian", operand)


def evaluate(expr, chart):
    """Evaluate a :class:`DifferentialExpr` against `chart` (free-function form
    of :meth:`DifferentialExpr.evaluate`)."""
    return expr.evaluate(chart)


# The chart-free ∇ symbol (decision 1): a shared, stateless singleton.
nabla = Nabla()
