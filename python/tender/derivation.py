"""tender.derivation — derivation steps and history tracking.

A derivation records an expression at each rewriting step::

    import tender
    import tender.derivation as td

    ctx = tender.Context()
    i = ctx.alloc_index()
    expr = tender.explicit_sum(i,
               tender.delta(tender.Realm.Oblique, tender.space_3d,
                            tender.Level.Upper, tender.Level.Lower, i, i))

    drv = td.Derivation(expr)
    drv.step(td.unroll_sums).step(td.eval_delta_concrete).step(td.fold_arithmetic)
    print(drv.current.latex())   # "3"

Steps are plain callables ``(Expr) -> Expr``, so users can define custom steps
and pass them to :meth:`Derivation.step` alongside the built-in ones.
"""

from tender._core import derivation as _d

__all__ = [
    "Derivation",
    "unroll_sums",
    "eval_delta_concrete",
    "fold_arithmetic",
    "expand_products",
    "expand_eps",
    "fold_sums",
    "contract_delta",
]


class Derivation:
    """Sequence of rewriting steps applied to an expression.

    ``history[0]`` is the initial expression; ``history[k]`` is the result
    after applying the k-th step.
    """

    def __init__(self, initial):
        self._history = [initial]

    def step(self, step_fn):
        """Apply *step_fn* to the current expression; return *self* for chaining."""
        self._history.append(step_fn(self._history[-1]))
        return self

    @property
    def history(self):
        """All expressions from initial through each applied step."""
        return list(self._history)

    @property
    def current(self):
        """The most recently produced expression."""
        return self._history[-1]

    @property
    def initial(self):
        """The expression this derivation started from."""
        return self._history[0]


def unroll_sums(expr):
    """Expand each ``ExplicitSum`` with a concrete index space into a ``Sum`` tree."""
    return _d._unroll_sums(expr)


def eval_delta_concrete(expr):
    """Replace ``δ(a, b)`` with concrete indices by ``1`` (a == b) or ``0`` (a != b)."""
    return _d._eval_delta_concrete(expr)


def fold_arithmetic(expr):
    """Constant-fold arithmetic: reduce ``Sum``/``Difference``/``TensorProduct``/``ScalarDiv``/``Negate`` of scalar literals."""
    return _d._fold_arithmetic(expr)


def expand_products(expr):
    """Distribute TensorProduct over Sum/Difference (expand brackets)."""
    return _d._expand_products(expr)


def expand_eps(expr):
    """Expand every rank-3 Levi-Civita symbol to its 6-term Kronecker-delta cofactor expansion."""
    return _d._expand_eps(expr)


def fold_sums(expr):
    """Detect concrete N-addend Sum cycles and fold them into ``ExplicitSum`` over a fresh index."""
    return _d._fold_sums(expr)


def contract_delta(expr):
    """Contract ``ExplicitSum{m, δ^m_a · δ^m_b}`` into ``δ_{ab}``."""
    return _d._contract_delta(expr)
