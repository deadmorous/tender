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
    "contract_eps_pair",
    "fold_equal_addends",
]


class Derivation:
    """Sequence of rewriting steps applied to an expression.

    ``history[0]`` is the initial expression; ``history[k]`` is the result
    after applying the k-th step.

    Pass an ``index_map`` (a :class:`tender.IndexNameMap`) at construction to
    keep index names consistent across all rendering calls on the history.
    """

    def __init__(self, initial, index_map=None):
        self._history = [initial]
        self.index_map = index_map

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

    def latex(self, k, index_map=None):
        """Render history step k to LaTeX, using the derivation's index map."""
        imap = index_map or self.index_map
        return self._history[k].latex(imap)


def unroll_sums(expr, *indices):
    """Expand ``ExplicitSum`` nodes into concrete ``Sum`` trees.

    If *indices* are provided, only unroll sums whose summation index appears
    in that list; raises ``ValueError`` if none of the given indices are found
    as an ``ExplicitSum`` in *expr*.  With no *indices*, all sums with a
    concrete index space are expanded (original behaviour).
    """
    if not indices:
        return _d._unroll_sums(expr)
    idx_list = list(indices)
    if not _d._has_explicit_sum_for(expr, idx_list):
        ids = ", ".join(str(i.id) for i in idx_list)
        raise ValueError(
            f"No ExplicitSum found for any of the given indices (ids: {ids})"
        )
    return _d._unroll_sums_for(expr, idx_list)


def eval_delta_concrete(expr):
    """Replace ``δ(a, b)`` with concrete indices by ``1`` (a == b) or ``0`` (a != b)."""
    return _d._eval_delta_concrete(expr)


def fold_arithmetic(expr):
    """Constant-fold arithmetic: reduce ``Sum``/``Difference``/``TensorProduct``/``ScalarDiv``/``Negate`` of scalar literals.

    Also normalises ``X + (-Y)`` → ``X - Y`` and ``X - (-Y)`` → ``X + Y``.
    """
    return _d._fold_arithmetic(expr)


def expand_products(expr):
    """Distribute product nodes (TensorProduct, Dot, DDot, DDotAlt, Cross) over Sum/Difference."""
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


def contract_eps_pair(expr):
    """Contract a pair of Levi-Civita symbols sharing summed indices.

    Maps ``Σ_{i…} ( ε^{… i…} ⊗ ε_{… i…} )`` directly to the generalized
    Kronecker delta, with no concrete-WCS unrolling::

        Σ_i  ε^{ijk} ε_{iml}  → δ^j_m δ^k_l − δ^j_l δ^k_m
        Σ_ij ε^{ijk} ε_{ijl}  → 2 δ^k_l

    Only 3D, and a body that is exactly the product of two rank-3 ε symbols,
    are supported; anything else is returned unchanged.
    """
    return _d._contract_eps_pair(expr)


def fold_equal_addends(expr):
    """Group identical addends: ``X + X → 2X``, ``n·X + X → (n+1)·X``, etc."""
    return _d._fold_equal_addends(expr)
