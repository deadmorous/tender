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

Most work should not need them.  The goal-directed verbs — :func:`prove_equal`
and :func:`engine_simplify`, driven by :func:`rules` — say *what* you want
rather than which rewriting to apply in which order, which is the difference
between a derivation you can write and one you have to discover.  The purely
internal steps live in :mod:`tender.steps`; they remain importable,
but reaching for one usually means the verb surface is missing something.
"""

import warnings

from tender import _core
from tender._core import derivation as _d

# `__all__` is the *advertised* surface, not the reachable one: every name below
# still imports and works.  The moves left out are catalogued in
# :mod:`tender.steps` under their category — they are what the vocabulary is
# built from rather than vocabulary themselves (vibe 000106).
__all__ = [
    "Derivation",
    "NoOpStep",
    "contract_delta",
    "contract_identity",
    "contract_eps_pair",
    "collect_terms",
    "partial",
    "deriv",
    "apply_operators",
    "contract_metric",
    "insert_metric",
    "fold_operator",
    "simplify_scalars",
    "simplify",
    "Identity",
    "apply_identity",
    "rule",
    "prove_equal",
    "rules",
    "rule_groups",
    "citable_for",
    "engine_simplify",
    "PREFER",
    "ProofResult",
    "BudgetExceeded",
    "Budget",
    "set_default_budget",
    "default_budget",
    "structural_eq",
    "algebraic_eq",
    "at",
    "explore",
]


def explore(expr, needs=None, scope=None, gui=None, max_height="520px", **context):
    """Open an interactive derivation session on *expr* (vibe 000108).

    A thin forwarder to :func:`tender.explore.explore`, kept here because this
    is where a derivation starts.  Say what the steps get by kind, and it is
    passed to every one that wants it::

        s = td.explore(a @ b, {"basis": frame})    # or basis=frame

    Leave it out and *your* namespace is searched for an object of each kind,
    so a session needs no setup beyond the cell you already wrote::

        s = td.explore(a @ b)     # finds the Basis, passes it to every step

    In a notebook this also opens the widget from :mod:`tender.gui`; elsewhere
    it is the session alone, which works the same from a terminal — a widget
    needs a browser to draw in, and there is no terminal fallback.
    """
    from .explore import _caller_scope
    from .explore import explore as _explore

    return _explore(
        expr,
        needs=needs,
        scope=scope if scope is not None else _caller_scope(2),
        gui=gui,
        max_height=max_height,
        **context,
    )


class NoOpStep(UserWarning):
    """A derivation step changed nothing (vibe 000095 increment 3).

    Steps obey the no-op contract: a step that has nothing to do returns its
    input unchanged.  :meth:`Derivation.step` surfaces that as this warning,
    because a silent no-op is how derivations go wrong invisibly (vibe
    000056 §1) — either the step was unnecessary (drop it) or it was expected
    to fire and did not (the derivation has stalled).  Pass ``optional=True``
    for a step that is legitimately conditional.
    """


class Derivation:
    """Sequence of rewriting steps applied to an expression.

    ``history[0]`` is the initial expression; ``history[k]`` is the result
    after applying the k-th step.  ``steps`` records, per applied step, its
    name and whether it *fired* (changed the expression).

    Pass an ``index_map`` (a :class:`tender.IndexNameMap`) at construction to
    keep index names consistent across all rendering calls on the history.
    """

    def __init__(self, initial, index_map=None):
        self._history = [initial]
        self._steps = []  # (name, fired) per applied step
        self.index_map = index_map

    def step(self, step_fn, *, optional=False, label=None):
        """Apply *step_fn* to the current expression; return *self* for chaining.

        Records whether the step *fired* (changed the expression) under
        *label* (default: the callable's name).  A step that changes nothing
        raises a :class:`NoOpStep` warning unless ``optional=True`` — a
        legitimately conditional step (one applied "in case it helps") should
        say so explicitly.
        """
        before = self._history[-1]
        after = step_fn(before)
        fired = not _core._structural_eq(after, before)
        name = label or getattr(step_fn, "__name__", None) or repr(step_fn)
        if not fired and not optional:
            warnings.warn(
                f"derivation step {name!r} changed nothing (a no-op): either "
                "drop it or mark it step(..., optional=True)",
                NoOpStep,
                stacklevel=2,
            )
        self._history.append(after)
        self._steps.append((name, fired))
        return self

    @property
    def steps(self):
        """Per applied step: ``(name, fired)`` — did it change the expression?"""
        return list(self._steps)

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

    def _repr_html_(self):
        """The derivation as a table, for Jupyter: each step, what it did, and
        the expression it produced.

        A derivation is a narrative, and reading it as a Python repr loses
        exactly the part that matters.  The *fired* column is the one to watch:
        a step that changed nothing is either unnecessary or a sign the
        derivation has stalled — which is how derivations go wrong invisibly
        (vibe 000056).
        """
        rows = [
            "<tr><th style='text-align:left'>#</th>"
            "<th style='text-align:left'>step</th>"
            "<th style='text-align:left'></th>"
            "<th style='text-align:left'>result</th></tr>",
            "<tr><td>0</td><td><i>initial</i></td><td></td>"
            f"<td>${self.latex(0)}$</td></tr>",
        ]
        for k, (name, fired) in enumerate(self._steps, start=1):
            mark = "✓" if fired else "·"
            colour = "" if fired else " style='opacity:.55'"
            rows.append(
                f"<tr{colour}><td>{k}</td><td><code>{name}</code></td>"
                f"<td>{mark}</td><td>${self.latex(k)}$</td></tr>"
            )
        note = ""
        if any(not fired for _, fired in self._steps):
            note = (
                "<div style='opacity:.7;font-size:90%'>· = the step changed "
                "nothing</div>"
            )
        return (
            "<table style='border-collapse:collapse'>"
            + "".join(rows)
            + "</table>"
            + note
        )


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


def eval_eps_concrete(expr):
    """Replace a Levi-Civita symbol with concrete indices by its value.

    ``0`` on any repeated index, else the sign of the permutation (``+1`` even,
    ``-1`` odd).  A symbol with any symbolic index is left unchanged.
    """
    return _d._eval_eps_concrete(expr)


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
    scalar factors are pulled through and the contraction distributes over the
    full additive structure of either side — ``Sum``, ``Difference``, ``Negate``,
    a scalar divisor ``(X/c)``, a scalar-weighted sum ``s·(A+B)``, and summation
    binders — so it fires on indexed / implicitly-summed dyads
    (``(Σ_i e_i⊗e_i):(Σ_j e_j⊗e_j)``) and reduces a whole elastic energy
    ``T ·· ε`` with ``ε = (∇u+(∇u)ᵀ)/2`` down to scalar dots in one pass.  A
    double dot whose sides are not both dyads is left unchanged.
    """
    return _d._expand_double_dot(expr)


def expand_dyad_ops(expr):
    """Expand tr/vec/transpose on dyads by their definition.

    ``tr(a⊗b) → a·b``, ``vec(a⊗b) → a×b``, ``transpose(a⊗b) → b⊗a``; linear over
    sums and negation, scalar factors pulled through, and a symmetric well-known
    (I, δ, g) transposes to itself.  An operation whose operand is not a dyad is
    left in place.
    """
    return _d._expand_dyad_ops(expr)


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
    """Self-preparing fold of equal addends (vibe 000065).

    Canonicalizes first, so terms that are equal only up to dummy-index
    renaming or factor/sign ordering collapse to one normal form, then groups
    identical addends (``X + X → 2X``, ``n·X + X → (n+1)·X``, ``X − X → 0``),
    then restores implicit-sum form.  In particular ``x1 - x2`` reduces to ``0``
    for any algebraically equal ``x1``, ``x2`` — no manual canonicalize needed.
    For the bare structural pass use :func:`fold_equal_addends_structural`.
    """
    return _d._fold_equal_addends(expr)


def fold_equal_addends_structural(expr):
    """Bare structural fold: merge addends written identically only.

    Does NOT rename dummy indices or normalize factor/sign order, so two terms
    equal only after canonicalization are left separate.  Use when the addends
    are already in a common frame; otherwise prefer :func:`fold_equal_addends`.
    """
    return _d._fold_equal_addends_structural(expr)


def collect_terms(expr):
    """Group addends sharing the same tensor (dyad) part, summing coefficients.

    Each addend ``scalar_coeff ⊗ (e_i⊗e_j…)`` is grouped by its non-scalar part;
    the scalar coefficients are added and simplified into one term per distinct
    dyad.  Unlike :func:`fold_equal_addends` (numeric coefficients only) it
    factors an arbitrary scalar, so a curvilinear second gradient's six raw terms
    collapse to one per e_i⊗e_j.
    """
    return _d._collect_terms(expr)


def factor_common(expr):
    """Factor a common scalar factor out of an additive group (vibe 000080).

    The reverse of distribution: ``λ (∇·u) + μ (∇·u) → (λ + μ) (∇·u)``.  Only
    rank-0 non-literal factors are pulled out (they commute, so it is always
    valid); a common numeric coefficient is left to :func:`collect_terms` and a
    common *tensor* factor is already handled there.  Runs bottom-up, so it also
    factors a sum nested inside a gradient: ``∇(λ∇·u + μ∇·u) → ∇((λ+μ)∇·u)``.
    """
    return _d._factor_common(expr)


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


def partial(expr, coord):
    """Partial derivative ∂expr/∂coord (vibe 000069 M2).

    ``coord`` must be a coordinate variable (``tender.coordinate``).  Applies
    linearity, the Leibniz product rule over ``*`` and every contraction, the
    quotient rule over ``/``, and the chain rule over the elementary functions
    and powers.  Only the matching coordinate differentiates to 1; every other
    coordinate and every non-coordinate symbol (reference vectors, parameters,
    literals) is constant.  The result is canonicalized, so e.g.
    ``partial(r * cos(phi), phi)`` is ``-r sin phi``.
    """
    return _d._partial(expr, coord)


def deriv(coord):
    """The unapplied ∂/∂coord operator — the building block of every
    differential operator tender knows (vibes 000077, 000102).

    A first-class :class:`~tender.Expr`, so operators are *built* rather than
    named: pair partials with coefficient vectors and add them up.

        d = td.deriv
        e = chart.physical_frame()
        grad_perp = e.direction(0) * d(x) + e.direction(1) * d(y)   # ∇⊥

    Anything of the form ``Σ_k c_k ⊗ ∂_k`` is a *derivation*: it obeys the
    Leibniz rule by construction, and :func:`apply_operators` will carry that
    out — over any number of factors, for any coefficients, with no rule to
    register and no chart required.

    A ``Deriv`` node — a composable operator that acts on everything to its
    right when multiplied.  ``deriv(x) * f`` builds the (unapplied) product
    ``∂_x f``; :func:`apply_operators` then carries out the differentiation
    (Leibniz).  ``coord`` must be a coordinate variable (``tender.coordinate``).
    """
    return _d._deriv(coord)


def apply_operators(expr):
    """Apply the first-class ∂ operators in ``expr`` by Leibniz (vibe 000077).

    Each unapplied :func:`deriv` operator acts on everything to its right in
    its product term (``∂_x x = 1``, ``∂_x(x·f) = f + x ∂_x f``), operators
    applied rightmost-first, a trailing operator with no operand left bare.
    ``apply_operators(deriv(x) * f)`` is the derivative ``∂_x f``.
    """
    return _d._apply_operators(expr)


def insert_metric(expr, level, target=None):
    """Move summed coordinate indices to ``target`` level, introducing the metric.

    The companion to :func:`contract_metric`, and the "lower both indices" move
    a textbook raising/lowering derivation opens with::

        Σ_m a^m b_m,  target=Level.Upper  →  Σ_m Σ_n g_{mn} a^m b^n
        Σ_m a^m b_m,  target=Level.Lower  →  Σ_m Σ_n g^{mn} a_n b_m

    Every summed coordinate slot at the level opposite ``target`` moves to a
    fresh index at ``target``, and a metric carrying the old and new indices at
    the opposite level restores the balance.  Well-known objects (δ, ε, and
    ``g`` itself) are skipped — they are the currency the move is paid in, not
    what is being moved, which is also what makes the step terminate.
    Orthonormal slots are left alone; the distinction is empty there.
    """
    return _d._insert_metric(expr, level, target)


def contract_metric(expr, target=None):
    """Contract a metric against a summed index — raise, lower, or the inverse pair.

    Where :func:`contract_delta` merely identifies its two indices, ``g`` also
    moves the survivor across the upper/lower divide: the surviving index is
    ``g``'s *other* index, at ``g``'s *other* level.  One operation, read three
    ways::

        Σ_p g^{ip} a_p     →  a^i           raise
        Σ_p g_{ip} a^p     →  a_i           lower
        Σ_p g^{ip} g_{pk}  →  δ^i_k         the inverse pair

    The last falls out of the first two: a ``g`` whose slots straddle the divide
    *is* the Kronecker δ (``g^i_j = e^i·e_j``), so it is normalized to one and
    :func:`contract_delta` can finish.

    The partner index must sit at the level opposite ``g``'s — what Einstein
    summation in an oblique realm demands anyway.  A same-level pair, or a ``g``
    with no partner occurrence, is left exactly as written.
    """
    return _d._contract_metric(expr, target)


def fold_operator(expr, op):
    """Fold a derivation operator's expansion back into the operator.

    The return trip :func:`apply_operators` has no inverse for (vibe 000103's
    operator row).  ``op`` is any expression of the shape ``Σ_k c_k ⊗ ∂_{q_k}``
    — what a frame vector and :func:`deriv` build by hand, and what a Cartesian
    chart's ``∇`` expands to.  Wherever ``expr`` holds the *complete* group of
    addends that applying ``op`` to some operand produced, the group collapses
    back to ``op`` left unapplied::

        f (∂ₓg) i + f (∂_y g) j   →   f ⊗ ((i∂ₓ + j∂_y) ⊗ g)

    You supply ``op``, and with it the claim that this expansion *is* that
    operator: the library cannot know that ``i∂ₓ + j∂_y`` deserves to be read
    back as one thing rather than left as four terms.

    An incomplete group, members disagreeing on the operand or on the factors
    beside them, or a non-scalar factor alongside (where the folded operator
    belongs in the product order would be a guess) all leave the expression
    exactly as written.
    """
    return _d._fold_operator(expr, op)


def simplify_scalars(expr):
    """Targeted scalar-field simplifier (vibe 000069 M3).

    Applies the small set of identities the orthogonal-curvilinear geometry
    pipeline needs, to a fixed point on top of ``canonicalize``: the Pythagorean
    fold ``cos²(u)·C + sin²(u)·C → C``, power cleanup ``x⁰ → 1`` / ``x¹ → x``,
    and ``√(x²ᵏ) → xᵏ`` when ``x`` is known ≥ 0 (a coordinate created with
    ``nonneg=True``).  Finishes in implicit-sum form.
    """
    return _d._simplify_scalars(expr)


def implicitize(expr):
    """Inverse of the implicit-sum convention (vibe 000028/000064 #4).

    Drops each ``explicit_sum`` binder whose index is repeated within a single
    multiplicative term, leaving the contraction implicit (Einstein
    convention) — the user-facing form the derivation steps emit.  An index
    that straddles a ``+`` (a Sum scope boundary, vibe 000052) cannot be left
    implicit and keeps its binder.
    """
    return _d._implicitize(expr)


def simplify(expr):
    """Canonicalize, then strip the materialized sums back to implicit form.

    ``canonicalize`` combines like terms and cancels equal-and-opposite ones
    but materializes every implicit Einstein sum into an ``explicit_sum``;
    ``implicitize`` reverses that last part.  Together they *finish* a
    derivation: a single clean, canonical, implicit-summation result (vibe
    000064 #4).
    """
    return implicitize(canonicalize(expr))


def _check_symmetrisable(expr, name):
    """Transpose swaps *two* slots, so sym/skew mean something only at rank 2.

    Rank 0 is the one other case with an answer, and it is forced: a scalar has
    no slots to swap, so ``sᵀ = s``.  Rank 1 and rank ≥ 3 have none — a vector
    has nothing to swap, and for a rank-3 the question "which pair?" is exactly
    what is missing — so they are refused rather than given a formula that
    reads as if it meant something (found by the `applicable` probe, which
    listed ``sym`` as an option on a scalar; vibe 000106).

    An unknown rank is not refused: the expression may still be well-formed and
    the caller may know more than the inference does.
    """
    rank = expr.rank
    if rank is None or rank in (0, 2):
        return rank
    raise ValueError(
        f"{name}: transpose swaps two slots, so {name} is defined at rank 2 "
        f"(and trivially at rank 0); this expression has rank {rank}"
        + (
            " — a vector has no slots to swap"
            if rank == 1
            else " — say which pair of slots you mean"
        )
    )


def sym(expr):
    """The symmetric part of a rank-2 tensor: ``sym(A) = (A + Aᵀ)/2``.

    A thin builder (vibe 000080 Increment 7A) — the strain ``ε = sym(∇u)`` is
    symmetric by construction.  Recognising the *result* as symmetric (so
    ``sym(A)ᵀ`` folds back) is the separate structural-normalisation work; this
    is just the constructor.

    A scalar is returned unchanged (``sᵀ = s``, so the average is ``s``), and a
    rank-1 or rank ≥ 3 argument raises: see :func:`_check_symmetrisable`.
    """
    if _check_symmetrisable(expr, "sym") == 0:
        return expr
    return (expr + expr.transpose()) / 2


def skew(expr):
    """The antisymmetric part of a rank-2 tensor: ``skew(A) = (A − Aᵀ)/2``.

    Companion to :func:`sym` (vibe 000080 Increment 7A); ``A = sym(A) + skew(A)``.

    A scalar gives **zero** — ``sᵀ = s``, so the difference vanishes — which is
    the asymmetry with :func:`sym` worth knowing: the two do not both reduce to
    the identity at rank 0.
    """
    if _check_symmetrisable(expr, "skew") == 0:
        # `expr - expr`, canonicalized to the literal 0 — a builder does not
        # normally normalise, but there is no information to preserve in a
        # scalar zero and `s - s` would only puzzle the reader.
        return canonicalize(expr - expr)
    return (expr - expr.transpose()) / 2


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


def apply_identity(expr, identity):
    """Apply one *identity* to *expr* — the rule library, one rule at a time.

    The complement of :func:`engine_simplify`, which hands the whole rule set to
    the saturation engine and asks for the best result: here you name the rule
    and the rewrite is the step.  It rewrites the first (deepest-first) subtree
    matching ``identity.lhs`` into the instantiated ``identity.rhs``, and a
    fired result is canonical.  If nothing matches, *expr* comes back
    **unchanged** (the step no-op contract, vibe 000095), so a derivation can
    tell you the identity did not fire instead of silently canonicalizing::

        e = td.apply_identity(a % (b % c), td.rule("bac-cab", ctx))

    *identity* is an :class:`Identity`, from :func:`rule`, :func:`rules`, or
    built by hand.  An ``Identity`` is itself callable, so ``identity(expr)``
    and ``drv.step(identity)`` are the same rewrite spelled shorter; this form
    exists because a step in the catalogue takes its expression first (vibe
    000108).
    """
    return identity(expr)


def rule(name, source):
    """The single identity called *name*, from a context or a rule list.

    *source* is either a :class:`tender.Context` — in which case the rule is
    built from the shipped library — or an iterable of :class:`Identity` to
    pick from, which is how you narrow the choice to rules of your own::

        td.rule("bac-cab", ctx)          # from the library
        td.rule("bac-cab", my_rules)     # from a list you built

    The singular of :func:`rules`, and the form a derivation cites: a rule
    named in a script says which rewrite was taken, where a subscript into a
    list says only where it happened to sit.
    """
    from . import identities as _ident

    if isinstance(source, _core.Context):
        return _ident.rules_for(source, name)[0]
    found = [r for r in source if getattr(r, "name", None) == name]
    if not found:
        available = ", ".join(sorted(getattr(r, "name", "?") for r in source))
        raise ValueError(f"no identity named {name!r}; available: {available}")
    return found[0]


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


def rule_groups():
    """The names of the identity groups the library ships (vibe 000096).

    The library itself lives in :mod:`tender.identities` — plain Python, so
    you can add rules without rebuilding anything.
    """
    from . import identities as _ident

    return _ident.group_names()


def citable_for(name, ctx=None, realm=None, space=None):
    """The rules a derivation of identity *name* may legitimately cite.

    An identity's proof may lean only on identities standing *below* it in
    tender's DAG (:mod:`tender.identities`) — never on itself, never on
    anything that already rests on it.  Passing this to :func:`prove_equal`
    makes a derivation honest by construction instead of by review::

        >>> td.prove_equal(lhs, rhs, td.citable_for("bac-cab", ctx=ctx))

    Returns the ancestors' rules; an identity derived straight from
    definitions (proved by reduction to components) legitimately has none.
    """
    from . import identities as _ident

    if ctx is None:
        ctx = _core.Context()
    kw = {}
    if realm is not None:
        kw["realm"] = realm
    return _ident.citable_for(ctx, name, space=space, **kw)


def rules(*groups, ctx=None, realm=None, space=None):
    """The rules of one or more named identity groups, ready for the verbs.

    Groups are the unit of rule selection: pass the ones a problem needs
    (``rules("cross")``) rather than everything, since rule count is the main
    driver of saturation cost.  All rules are built in *ctx*, so they can be
    matched against any expression from the same context::

        >>> td.prove_equal(lhs, rhs, td.rules("cross", ctx=ctx))

    ``realm`` / ``space`` parameterize the index-level group (``eps_delta``);
    the invariant groups ignore them.  Your own rules are ordinary
    :class:`Identity` objects — just add them to the list.
    """
    from . import identities as _ident

    if ctx is None:
        ctx = _core.Context()
    kw = {}
    if realm is not None:
        kw["realm"] = realm
    out = []
    for name in groups:
        out.extend(_ident.group(ctx, name, space=space, **kw))
    return out


class Budget:
    """How much effort a verb may spend, in units you can reason about.

    Two kinds of cap, and the difference matters:

    **Deterministic** — ``max_passes``, ``max_nodes``.  The same input gives
    the same answer on every machine.  These are what a test suite and CI
    must use: a result that depends on how fast the machine is cannot be
    reproduced or reviewed.

    **Resource** — ``max_seconds``, ``max_bytes``.  What a person actually
    wants to say ("don't spend more than two seconds on this"), and the right
    dial for interactive work — but machine-dependent, so a run that trips one
    here may not trip it there.  ``None`` (the default) means no limit.

    ``max_bytes`` is an **estimate** — e-node count times a per-node figure —
    not a measurement of process memory.  Treat it as a coarse guard, not an
    accounting.

    Set a session-wide default with :func:`set_default_budget`, and override
    per call with ``budget=``::

        td.set_default_budget(td.Budget(max_seconds=10))
        td.prove_equal(lhs, rhs, rules, budget=td.Budget(max_passes=5))
    """

    __slots__ = ("max_passes", "max_nodes", "max_seconds", "max_bytes")

    def __init__(
        self, max_passes=30, max_nodes=10_000, max_seconds=None, max_bytes=None
    ):
        self.max_passes = max_passes
        self.max_nodes = max_nodes
        self.max_seconds = max_seconds
        self.max_bytes = max_bytes

    def replace(self, **changes):
        """A copy with some caps changed."""
        fields = {f: getattr(self, f) for f in self.__slots__}
        unknown = set(changes) - set(fields)
        if unknown:
            raise ValueError(f"unknown budget field(s): {', '.join(sorted(unknown))}")
        fields.update(changes)
        return Budget(**fields)

    def _args(self):
        return (
            self.max_passes,
            self.max_nodes,
            0.0 if self.max_seconds is None else float(self.max_seconds),
            0 if self.max_bytes is None else int(self.max_bytes),
        )

    def __repr__(self):
        caps = [f"max_passes={self.max_passes}", f"max_nodes={self.max_nodes}"]
        if self.max_seconds is not None:
            caps.append(f"max_seconds={self.max_seconds}")
        if self.max_bytes is not None:
            caps.append(f"max_bytes={self.max_bytes}")
        return f"Budget({', '.join(caps)})"


_DEFAULT_BUDGET = Budget()


def set_default_budget(budget):
    """Set the budget the verbs use when none is passed; returns the previous one.

    This is the session-wide default a user sets once.  Per-call ``budget=``
    always wins, so a single expensive derivation can be given more room
    without loosening anything else.
    """
    global _DEFAULT_BUDGET
    previous, _DEFAULT_BUDGET = _DEFAULT_BUDGET, budget
    return previous


def default_budget():
    """The budget the verbs use when none is passed."""
    return _DEFAULT_BUDGET


def _resolve_budget(budget, max_passes, max_nodes):
    """Per-call budget, or the default — with the legacy kwargs still honoured."""
    if budget is not None:
        return budget
    base = _DEFAULT_BUDGET
    changes = {}
    if max_passes is not None:
        changes["max_passes"] = max_passes
    if max_nodes is not None:
        changes["max_nodes"] = max_nodes
    return base.replace(**changes) if changes else base


class BudgetExceeded(UserWarning):
    """Saturation stopped on its budget, so the answer is inconclusive.

    A budget trip is *not* a negative result: the rules may well prove or
    simplify the expression given more room.  Raised as a warning so a
    shortfall can never pass silently for a fixed point.
    """


class ProofResult:
    """The outcome of :func:`prove_equal` — deliberately not a bare bool.

    ``proved`` is the answer; ``status`` says *why*:

    ``"proved"``
        The two sides joined in the e-graph.
    ``"refuted"``
        Expanding both sides into concrete components produced *different*
        results — a real negative, from a decision procedure independent of
        the rules (vibe 000097).  ``refuted`` is the property to test.
    ``"exhausted"``
        The rules ran to a fixed point and the component check could not
        decide either.  If ``components_agree`` is set, the claim looks
        **true** and it is the *rule set* that is incomplete — a very
        different problem from a false claim.
    ``"budget"``
        Stopped early; nothing at all is concluded.
    ``"unsupported"``
        tender could not put the expression in canonical form at all — a
        shape it does not yet handle.  ``detail`` says which; nothing is
        claimed about the mathematics.

    ``fired`` maps rule name → firing count, ``skipped`` lists rules the
    engine could not compile, and ``passes``/``nodes`` size the search.
    """

    def __init__(self, report):
        self.proved = report["proved"]
        self.status = report["status"]
        self.refuted = report["status"] == "refuted"
        #: When ``status`` is ``"unsupported"``: what tender could not do with
        #: the expression.  A statement about the tool, not about the claim.
        self.detail = report.get("detail", "")
        self.components_agree = report.get("components_agree", False)
        self.passes = report["passes"]
        self.nodes = report["nodes"]
        #: Which cap stopped the search ("passes", "nodes", "time", "memory"),
        #: or "" if it was not a budget stop.
        self.stopped_by = report.get("stopped_by", "")
        self.seconds = report.get("seconds", 0.0)
        self.bytes = report.get("bytes", 0)
        self.fired = dict(report["fired"])
        self.skipped = list(report["skipped"])

    def __bool__(self):
        return bool(self.proved)

    def _repr_html_(self):
        """The proof outcome for Jupyter: the verdict, and what produced it."""
        verdict = {
            "proved": ("proved", "#1a7f37"),
            "refuted": ("refuted — the statement is false", "#b42318"),
            "exhausted": ("not proved: the rules were not enough", "#9a6700"),
            "budget": ("inconclusive: stopped on budget", "#9a6700"),
            "unsupported": ("tender could not process this", "#9a6700"),
        }.get(self.status, (self.status, ""))
        parts = [
            f"<b style='color:{verdict[1]}'>{verdict[0]}</b>",
        ]
        if self.status == "unsupported":
            parts.append(f"<div style='opacity:.8'>{self.detail}</div>")
        if self.components_agree:
            parts.append(
                "<div style='opacity:.8'>components agree, so the claim looks "
                "true — it is the rule set that is incomplete</div>"
            )
        if self.fired:
            used = ", ".join(
                f"<code>{k}</code>&nbsp;×{v}" for k, v in self.fired.items()
            )
            parts.append(f"<div>identities used: {used}</div>")
        if self.stopped_by:
            parts.append(
                f"<div style='opacity:.8'>stopped by the "
                f"<b>{self.stopped_by}</b> cap</div>"
            )
        parts.append(
            f"<div style='opacity:.6;font-size:90%'>{self.passes} pass(es), "
            f"{self.nodes} e-nodes, {self.seconds:.3f}s</div>"
        )
        return "".join(parts)

    def __repr__(self):
        detail = f", fired={self.fired}" if self.fired else ""
        if self.status == "unsupported":
            return f"ProofResult(status='unsupported', detail={self.detail!r})"
        if self.components_agree:
            detail += ", components_agree=True (claim looks true; rules "
            detail += "incomplete)"
        return (
            f"ProofResult(proved={self.proved}, status={self.status!r}, "
            f"passes={self.passes}, nodes={self.nodes}{detail})"
        )


def _rule_arrays(rules):
    rules = list(rules)
    return (
        [r.lhs for r in rules],
        [r.rhs for r in rules],
        [r.name or f"rule{i}" for i, r in enumerate(rules)],
    )


def _warn_skipped(skipped, what):
    if skipped:
        warnings.warn(
            f"{what}: {len(skipped)} rule(s) could not be compiled and never "
            f"fired (a multi-term left-hand side has no matcher yet): "
            f"{', '.join(skipped)}",
            UserWarning,
            stacklevel=3,
        )


def prove_equal(lhs, rhs, rules, budget=None, max_passes=None, max_nodes=None):
    """Try to prove ``lhs == rhs`` by equality saturation under *rules*.

    Both sides are saturated together in one e-graph, so rules that rewrite
    either side toward the other suffice — neither has to be driven all the
    way into the other.  If the rules run out without a proof, an independent
    component-expansion check decides the chart-free algebraic fragment, so a
    false claim comes back ``"refuted"`` rather than merely unproved
    (vibe 000097).

    Returns a :class:`ProofResult`; a budget trip also warns
    :class:`BudgetExceeded`, because "not proved within budget" must never be
    mistaken for "not equal".
    """
    lhss, rhss, names = _rule_arrays(rules)
    budget = _resolve_budget(budget, max_passes, max_nodes)
    result = ProofResult(
        _d._prove_equal(lhs, rhs, lhss, rhss, names, *budget._args())
    )
    _warn_skipped(result.skipped, "prove_equal")
    if result.status == "budget":
        warnings.warn(
            f"prove_equal stopped on its {result.stopped_by} budget after "
            f"{result.passes} pass(es) / {result.nodes} nodes / "
            f"{result.seconds:.3f}s — inconclusive, NOT a disproof; retry with "
            f"a larger budget",
            BudgetExceeded,
            stacklevel=2,
        )
    return result


#: Named extraction intents (vibe 000097).  "Simplest" is not a property of
#: the algebra — it is what the user is trying to achieve — so the cost is a
#: parameter.  A large weight means "minimize this first, then size".
PREFER = {
    # Contract the ε's away, even though the δ-expansion is the larger form
    # (vibe 000046).  The historical default.
    "fewest_eps": {"eps": 1_000_000},
    # Plain size: no thumb on any scale.
    "smallest": {"eps": 1},
    # Get rid of cross products — this is what makes the *expansion* of
    # bac-cab the preferred reading of the very same saturated graph.
    "fewest_crosses": {"eps": 1_000, "cross": 1_000_000},
    # Prefer forms without the identity tensor (fold I·x → x, I··A → tr A).
    "fewest_identities": {"eps": 1_000, "identity": 1_000_000},
}


def engine_simplify(
    expr,
    rules,
    prefer="fewest_eps",
    cost=None,
    budget=None,
    max_passes=None,
    max_nodes=None,
):
    """Saturate *expr* under *rules* and return the best form found.

    "Best" is *your* choice, not a fixed notion of simplicity: pass ``prefer``
    to name an intent (see :data:`PREFER` — ``"fewest_eps"`` (default),
    ``"smallest"``, ``"fewest_crosses"``, ``"fewest_identities"``), or
    ``cost`` for raw per-kind weights (keys: ``node``, ``eps``, ``cross``,
    ``delta``, ``identity``, ``unary``, ``div``).  A large weight means
    "minimize this first, then size".

        >>> td.engine_simplify(a % (b % c), rules, prefer="fewest_crosses")

    Because the cost governs only *extraction*, re-reading the same rule set
    under another intent costs one extraction, not another saturation.

    Returns ``(expr, report)``; on a budget trip the best form found so far is
    still returned, with a :class:`BudgetExceeded` warning and
    ``report["complete"] is False``.
    """
    if cost is None:
        try:
            cost = PREFER[prefer]
        except KeyError:
            raise ValueError(
                f"unknown intent {prefer!r}; available: {', '.join(PREFER)} "
                f"(or pass cost={{...}} for raw weights)"
            ) from None
    lhss, rhss, names = _rule_arrays(rules)
    budget = _resolve_budget(budget, max_passes, max_nodes)
    out, report = _d._engine_simplify(
        expr, lhss, rhss, names, *budget._args(), dict(cost)
    )
    report = dict(report)
    report["fired"] = dict(report["fired"])
    report["skipped"] = list(report["skipped"])
    if report.get("unsupported"):
        warnings.warn(
            f"engine_simplify could not process this expression: "
            f"{report['unsupported']} — returning it unchanged",
            UserWarning,
            stacklevel=2,
        )
        return out, report
    _warn_skipped(report["skipped"], "engine_simplify")
    if not report["complete"]:
        warnings.warn(
            f"engine_simplify stopped on its {report['stopped_by']} budget "
            f"after {report['passes']} pass(es) / {report['nodes']} nodes / "
            f"{report['seconds']:.3f}s — the result is the best form found so "
            f"far, not a fixed point",
            BudgetExceeded,
            stacklevel=2,
        )
    return out, report


def structural_eq(a, b):
    """Deep structural equality of two expression trees."""
    return _core._structural_eq(a, b)


def algebraic_eq(a, b):
    """Algebraic equality: ``structural_eq`` of the canonical forms (theory T0),
    falling back to checking that ``simplify_scalars(a - b)`` is the literal 0 —
    so fraction shapes that T0 keeps apart (``x/r + y/r`` vs ``(x+y)/r``) compare
    equal (vibe 000074)."""
    return _core._algebraic_eq(a, b)


def at(expr, path, step):
    """Apply *step* to only the subexpression at *path*, splicing the result back.

    *path* is a ``list[int]`` of child selectors from the root (see
    :meth:`Expr.find` / :meth:`Expr.addends` to obtain one); *step* is any
    ``(Expr) -> Expr`` callable — a built-in step, an ``expand_in_basis``
    closure, ``apply_identity(...)``, etc.  This retargets *any* step to one
    occurrence, so e.g. only one ``I`` in ``a × I × b`` expands::

        p = expr.find(kind="Identity")[0]
        out = td.at(expr, p, lambda s: tb.expand_in_basis(s, frame, cov))

    Paths address one specific tree, so canonicalize *before* selecting and do
    not canonicalize between selecting and applying (a reshaping step would
    invalidate the path).
    """
    return expr.rewrite_at(path, step)
