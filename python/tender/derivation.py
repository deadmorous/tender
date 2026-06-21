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

from tender import _core
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
    "contract_identity",
    "distribute_contraction",
    "expand_double_dot",
    "contract_eps_pair",
    "fold_equal_addends",
    "canonicalize",
    "Identity",
    "apply_identity",
    "saturate",
    "structural_eq",
    "algebraic_eq",
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


def contract_identity(expr):
    """Contract the identity tensor in a dot product: ``I·x → x``, ``x·I → x``."""
    return _d._contract_identity(expr)


def expand_double_dot(expr):
    """Expand a double contraction of dyads by definition.

    ``(a⊗b) : (c⊗d) → (a·c)(b·d)`` and ``(a⊗b) ·· (c⊗d) → (a·d)(b·c)``;
    scalar factors are pulled through and the contraction distributes over sums
    and summation binders, so it also fires on indexed / implicitly-summed dyads
    (e.g. ``(Σ_i e_i⊗e_i):(Σ_j e_j⊗e_j)``).  A double dot whose sides are not
    both dyads is left unchanged.
    """
    return _d._expand_double_dot(expr)


def distribute_contraction(expr):
    """Distribute a contraction (``·`` or ``×``) over the adjacent leg of a tensor product.

    ``op(L, A⊗B) → op(L,A) ⊗ B`` and ``op(A⊗B, R) → A ⊗ op(B,R)``, so e.g.
    ``a × (u ⊗ v) → (a × u) ⊗ v``.  One pass (right operand first); apply again
    for deeper nesting.
    """
    return _d._distribute_contraction(expr)


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


def canonicalize(expr):
    """Rewrite into algebraic normal form (vibe 000037).

    Sorts commutative operands, carries signs in a single rational coefficient
    per term, combines like terms, and α-normalises dummies.  Materialises
    implicit Einstein contractions into ``explicit_sum`` first (vibe 000028), so
    a repeated index means the same with or without an explicit sum; an
    ill-formed term (e.g. an Oblique same-level pair with no override) raises
    ``ValueError``.  Does NOT distribute products over sums.  Two expressions
    equal under the normal-form theory T0 produce structurally identical results.
    """
    return _d._canonicalize(expr)


class Identity:
    """A directed rewrite rule ``lhs = rhs`` over expressions (vibe 000033).

    The *free indices* of ``lhs`` are pattern variables: each matches whatever
    index sits in the corresponding target slot, consistently across a match.
    Indices bound by an ``explicit_sum``/``no_sum`` inside ``lhs`` are local
    (alpha) variables, matched to the target's binder.

    An Identity is *not* a theorem: a theorem is a derivation that proves a
    result and carries its history; an identity is the bare equality such a
    theorem yields.
    """

    def __init__(self, name, lhs, rhs):
        self.name = name
        self.lhs = lhs
        self.rhs = rhs

    def __call__(self, expr):
        """Use the identity as a derivation step: ``drv.step(identity)``."""
        return _d._apply_identity(expr, self.lhs, self.rhs, self.name)

    def __repr__(self):
        return f"Identity({self.name!r})"


def apply_identity(identity):
    """Return a derivation step that applies *identity* to its argument.

    The step rewrites the first (deepest-first) subtree matching ``identity.lhs``
    into the instantiated ``identity.rhs``; the result is canonical.  If nothing
    matches, the result equals :func:`canonicalize` of the input.
    """
    return lambda expr: identity(expr)


def saturate(expr, rules, max_iterations=30):
    """Equality-saturate *expr* under *rules*, returning the simplest result.

    *rules* is an iterable of :class:`Identity`.  Each rule's ``lhs = rhs`` is
    applied everywhere it matches, to a fixed point (or until *max_iterations*
    passes), inside an e-graph; the cheapest extracted expression is returned.
    Unlike a linear :class:`Derivation`, no manual step ordering is needed — a
    rewrite nested inside a larger expression is found and applied automatically.

    All of *expr* and the rules' expressions must share one :class:`tender.Context`.
    """
    rules = list(rules)
    lhss = [r.lhs for r in rules]
    rhss = [r.rhs for r in rules]
    return _d._saturate(expr, lhss, rhss, max_iterations)


def structural_eq(a, b):
    """Deep structural equality of two expression trees."""
    return _core._structural_eq(a, b)


def algebraic_eq(a, b):
    """Algebraic equality in theory T0: ``structural_eq`` of the canonical forms."""
    return _core._algebraic_eq(a, b)
