"""tender — tensor algebra library for computational mechanics."""

from ._tender import (
    # Types
    Rational,
    Expr,
    RationalConst,
    NamedConst,
    SymbolicVar,
    Parameter,
    Sum,
    Scale,
    TensorProduct,
    IdentityTensor,
    LeviCivitaTensor,
    Trace,
    Contract,
    DoubleContract,
    DoubleContractReversed,
    CrossProduct,
    NamedTensor,
    Product,
    FunctionApply,
    Pow,
    MaterialDeriv,
    PatternVar,
    Gradient,
    Divergence,
    Rotor,
    Integral,
    Domain,
    SurfaceDomain,
    VolumeDomain,
    CoordSystem,
    State,
    DerivationStep,
    Derivation,
    Identity,
    Nabla,
    # Expression factories
    tensor,
    scalar,
    parameter,
    named,
    rational,
    make_pattern_var,
    # Algebraic operations
    tp,
    dot,
    ddot,
    ddot2,
    cross,
    trace,
    # Scalar functions
    exp,
    log,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    atan2,
    sinh,
    cosh,
    tanh,
    sqrt,
    pow,
    # Differentiation
    deriv,
    dt,
    ddt,
    # Coordinate systems
    grad,
    make_direct_basis_cs,
    # Symbolic differential operators
    gradient,
    divergence,
    rot,
    # Integral / domain
    make_surface_domain,
    make_volume_domain,
    integral,
    # Derivation rendering
    show,
    show_final,
    # Step factories
    simplify_identity_step,
    expand_step,
    expand_poly_step,
    substitute_step,
    diff_step,
    apply_integration_by_parts_step,
    apply_divergence_theorem_step,
    localize_step,
    collect_step,
    apply_identity,
    find_matches,
    apply_identity_auto,
    matches_at_root,
    _find_and_rewrite_all,
    _capture_step,
    declare_symmetric,
    declare_skew_symmetric,
    simplify_basis_dot_step,
    collect_zero_terms_step,
    reassemble_from_components_step,
    collect_repeated_sum_step,
    reassemble_vector_step,
    reassemble_dot_step,
    # IndexedSum node
    IndexedSum,
    make_indexed_sum,
    # SymBasisVec node
    SymBasisVec,
    make_sym_basis_vec,
    # Singleton getters (private)
    _identity_singleton,
    _levi_civita_singleton,
    _time_param_singleton,
    _wcs_singleton,
    _cylindrical_cs_singleton,
    _spherical_cs_singleton,
)

# Module-level singletons
I = _identity_singleton()
eps = _levi_civita_singleton()
t = _time_param_singleton()
wcs = _wcs_singleton()
cylindrical_cs = _cylindrical_cs_singleton()
spherical_cs = _spherical_cs_singleton()
nabla = Nabla()

_INDEX_LETTERS = ["i", "j", "k", "l", "m", "n"]


def _abstract_index_letters(rank, first_letter=None):
    first = first_letter or "i"
    result = [first]
    for ltr in _INDEX_LETTERS:
        if len(result) >= rank:
            break
        if ltr not in result:
            result.append(ltr)
    for ltr in "abcde":
        if len(result) >= rank:
            break
        if ltr not in result:
            result.append(ltr)
    return result[:rank]


def _abstract_comp_sym(base_sym, cov_list, letters):
    """Build component symbol with grouped indices: "A^ij_k", "A^i_j", etc."""
    result = base_sym
    current_sep = None
    for cov, ltr in zip(cov_list, letters):
        sep = "^" if cov else "_"
        if sep == current_sep:
            result += ltr
        else:
            result += sep + ltr
            current_sep = sep
    return result


def expand_in_basis_step(tensor_expr, cs, covariant=True, abstract=False, letter=None):
    """Create a DerivationStep that replaces tensor_expr with its basis expansion.

    Concrete mode (default, abstract=False):
      rank-1 covariant:    a  →  a^1 e_1 + a^2 e_2 + a^3 e_3
      rank-1 contravariant: a →  a_1 e^1 + a_2 e^2 + a_3 e^3
      rank-2 covariant:    A  →  Σ_{i,j} A^{ij} e_i ⊗ e_j

    Abstract mode (abstract=True):
      rank-1 covariant:   a → a^i e_i  (TensorProduct with SymBasisVec)
      rank-1 contravariant: a → a_j e^j
      rank-2 all-up:    A → A^{ij} e_i ⊗ e_j   (covariant=[True, True])
      rank-2 mixed:     A → A^{i}_{j} e_i ⊗ e^j (covariant=[True, False])

    In abstract mode, simplify_basis_dot_step collapses rank-1 dot products
    directly to an IndexedSum (e.g. a^i b_i) without 9-term expansion.
    For rank ≥ 2, abstract expansion is for display; simplification of
    contractions falls back to the concrete expansion path.

    covariant: bool (applied to all indices) or list[bool] (one per index).
    letter: index letter(s) for abstract mode.
            str  → used as the single letter (rank 1) or first letter (rank > 1),
                   remaining letters filled from "i","j","k","l",...
            list → used directly as the full letter list (length must equal rank).
            None → defaults to "i" for covariant, "j" for contravariant (rank 1),
                   or "i","j","k",... for rank > 1.
            Ignored in concrete mode.

    When expanding multiple tensors in the same expression use distinct letter
    sets to avoid clashing dummy indices, e.g. letter=["i","j"] for one tensor
    and letter=["k","l"] for another.
    """
    sym = tensor_expr.symbol
    r = tensor_expr.rank
    dim = cs.dim

    if abstract:
        if r < 1:
            raise ValueError(
                f"expand_in_basis_step: abstract mode requires rank ≥ 1, got {r}"
            )
        cov_list = ([covariant] * r) if isinstance(covariant, bool) else list(covariant)
        if len(cov_list) != r:
            raise ValueError(
                f"expand_in_basis_step: covariant list length {len(cov_list)} "
                f"does not match tensor rank {r}"
            )
        # Resolve index letters from the letter argument
        if isinstance(letter, list):
            letters = letter
            if len(letters) != r:
                raise ValueError(
                    f"expand_in_basis_step: letter list length {len(letters)} "
                    f"does not match tensor rank {r}"
                )
        elif isinstance(letter, str):
            letters = _abstract_index_letters(r, letter)
        elif r == 1 and not cov_list[0]:
            letters = ["j"]  # rank-1 contravariant default
        else:
            letters = _abstract_index_letters(r, "i")
        comp_sym = _abstract_comp_sym(sym, cov_list, letters)
        expr = tensor(comp_sym)
        for cov, ltr in zip(cov_list, letters):
            expr = expr * make_sym_basis_vec(cs, ltr, not cov)
        return substitute_step(tensor_expr, expr)

    if r == 1:
        if covariant:
            comps = [tensor(f"{sym}^{i + 1}") for i in range(dim)]
            vecs = [cs.basis(i) for i in range(dim)]
        else:
            comps = [tensor(f"{sym}_{i + 1}") for i in range(dim)]
            vecs = [cs.cobasis(i) for i in range(dim)]
        terms = [c * v for c, v in zip(comps, vecs)]
    elif r == 2:
        if covariant:
            def _fmt(i, j):
                return f"{sym}^{{{i + 1}{j + 1}}}"
        else:
            def _fmt(i, j):
                return f"{sym}_{{{i + 1}{j + 1}}}"
        terms = [
            tensor(_fmt(i, j)) * (cs.basis(i) * cs.basis(j))
            for i in range(dim)
            for j in range(dim)
        ]
    else:
        raise ValueError(
            f"expand_in_basis_step: unsupported rank {r} "
            f"(only rank 1 and 2 are supported)"
        )

    expansion = terms[0]
    for t in terms[1:]:
        expansion = expansion + t
    return substitute_step(tensor_expr, expansion)


def doc(entry, format="latex"):
    """Render an Identity (or future Theorem) as LaTeX, plain text, or Jupyter math.

    Parameters
    ----------
    entry : Identity
        The identity to document.
    format : str
        ``"latex"`` (default) — returns a compilable LaTeX snippet.
        ``"plain"`` — returns ASCII text.
        ``"jupyter"`` — displays in Jupyter via IPython.display.Math.
    """
    name = entry.name
    lhs_tex = entry.lhs.latex()
    rhs_tex = entry.rhs.latex()

    if format == "plain":
        return f"[{name}]  {entry.lhs.python()}  =  {entry.rhs.python()}"

    tex = f"\\textbf{{{name}:}}\n\\[\n  {lhs_tex} = {rhs_tex}\n\\]"

    if format == "jupyter":
        try:
            from IPython.display import Math, display
            display(Math(lhs_tex + " = " + rhs_tex))
            return
        except ImportError:
            pass  # fall through to latex

    return tex


def show_jupyter(history):
    """Display a derivation history as formatted LaTeX equations in a Jupyter cell.

    Each step is rendered as a displayed equation with its step label on the
    left, matching the layout produced by :func:`to_latex_document`.

    Parameters
    ----------
    history : list[State]
        The sequence of states returned by :meth:`Derivation.apply`.
    """
    from IPython.display import display, Latex

    blocks = []
    for state in history:
        label = state.label or "initial"
        blocks.append(
            r"\[" + _label_to_math(label) + r"\quad " + state.expr.latex() + r"\]"
        )
    display(Latex("\n".join(blocks)))


def search_apply(target, expr, rules=None, timeout=5.0):
    """Find and apply ``target``, preceded by any necessary preparation rewrites.

    Performs a breadth-first search over sub-expression rewrite steps drawn
    from ``rules``.  Returns a complete list of :class:`DerivationStep` objects
    that, when applied to ``expr`` in order, produce the final result with
    ``target`` applied.  The last step is always the application of ``target``.

    Parameters
    ----------
    target : Identity
        The identity to apply.
    expr : Expr
        Starting expression.
    rules : list[Identity] or None
        Rule library for the BFS.  ``None`` (default) uses all identities in
        ``tender.lib``.
    timeout : float
        Wall-clock time limit in seconds (fractions allowed, e.g. ``0.5``).

    Returns
    -------
    list[DerivationStep]
        Complete steps including the final application of ``target``.
        ``Derivation(steps).apply(State(expr))[-1].expr`` is the result.

    Raises
    ------
    TimeoutError
        No sequence found within ``timeout`` seconds.
    RuntimeError
        Search space exhausted (all reachable expressions visited).
    """
    import time
    from collections import deque

    if rules is None:
        from tender.lib import ALL
        rules = ALL

    def _try_target(e):
        """Return capture step applying target anywhere in e, or None."""
        matches = list(_find_and_rewrite_all(target, e))
        if matches:
            new_e, name = matches[0]
            return _capture_step(name, new_e)
        return None

    direct = _try_target(expr)
    if direct is not None:
        return [direct]

    deadline = time.monotonic() + timeout
    visited = {expr.latex()}
    queue = deque([(expr, [])])

    while queue:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"search_apply: no preparation sequence for '{target.name}' "
                f"found within {timeout:g}s")

        current, steps = queue.popleft()

        for rule in rules:
            for item in _find_and_rewrite_all(rule, current):
                new_expr, step_name = item
                key = new_expr.latex()
                if key in visited:
                    continue
                visited.add(key)
                step = _capture_step(step_name, new_expr)
                new_steps = steps + [step]

                final = _try_target(new_expr)
                if final is not None:
                    return new_steps + [final]

                queue.append((new_expr, new_steps))

    raise RuntimeError(
        f"search_apply: search space exhausted for '{target.name}' — "
        f"no sequence found within the visited rule applications")


def prove_equal_by_components(lhs_expr, rhs_expr, lhs_steps, rhs_steps):
    """Prove lhs_expr == rhs_expr by expanding both to component form.

    Applies ``lhs_steps`` to ``lhs_expr`` and ``rhs_steps`` to ``rhs_expr``,
    then compares the terminal expressions.  Comparison is order-insensitive for
    sums and commutative for scalar products (Product nodes).

    Parameters
    ----------
    lhs_expr, rhs_expr : Expr
        The two invariant expressions to compare.
    lhs_steps, rhs_steps : list[DerivationStep]
        Derivation steps that reduce each side to component form.

    Returns
    -------
    tuple[list[State], list[State]]
        ``(lhs_history, rhs_history)`` on success.

    Raises
    ------
    ValueError
        If the terminal component forms are not equal.
    """
    lhs_history = Derivation(lhs_steps).apply(State(lhs_expr))
    rhs_history = Derivation(rhs_steps).apply(State(rhs_expr))
    lhs_normal = _normalize_component_form(lhs_history[-1].expr)
    rhs_normal = _normalize_component_form(rhs_history[-1].expr)
    if lhs_normal != rhs_normal:
        raise ValueError(
            "Expressions do not reduce to the same component form:\n"
            f"  lhs: {lhs_history[-1].expr.python()}\n"
            f"  rhs: {rhs_history[-1].expr.python()}"
        )
    return lhs_history, rhs_history


def _normalize_component_form(expr):
    """Return a canonical string for component-form equality checks.

    Sorts the terms of any Sum (order-insensitive) and treats Product as
    commutative (sorted factors), so that ``a^k b_k`` and ``b_k a^k``
    compare equal.

    IndexedSum nodes are normalised letter-invariantly: ``a^i b_i`` and
    ``a^j b_j`` are considered equal, and factor order is ignored so that
    ``a^i b_i`` and ``b_i a^i`` compare equal.
    """
    if isinstance(expr, IndexedSum):
        parts = sorted([(expr.lhs_sym, expr.lhs_sep), (expr.rhs_sym, expr.rhs_sep)])
        return "IndexedSum[" + ",".join(f"{s}{p}" for s, p in parts) + "]"
    if isinstance(expr, Sum):
        terms = sorted(_normalize_component_form(t) for t in expr.terms)
        return "Sum[" + ",".join(terms) + "]"
    if isinstance(expr, Product):
        factors = sorted([
            _normalize_component_form(expr.lhs),
            _normalize_component_form(expr.rhs),
        ])
        return "Prod[" + "*".join(factors) + "]"
    return expr.python()


def _label_to_math(label):
    """Render a step label as a LaTeX math-mode fragment.

    Plain identifiers (not part of a \\command) are wrapped in \\mathrm{} so
    they appear upright.  LaTeX commands (\\partial, etc.) are passed through
    unchanged.  Returns a string suitable for use inside \\[...\\].
    """
    import re

    if "\\" not in label:
        # No LaTeX commands — use text mode directly.
        # Escape ^ and _ so they render literally inside \text{} in display math.
        safe = label.replace("_", r"\_").replace("^", r"\^{}")
        return r"\text{[" + safe + r":] }"

    # Match \command sequences first (pass through), then bare alpha runs
    # (wrap in \mathrm{}).  The alternation consumes \partial as one token,
    # preventing "artial" from being re-matched as a separate identifier.
    def _repl(m):
        s = m.group(0)
        return s if s.startswith("\\") else r"\mathrm{" + s + "}"

    math_label = re.sub(r"\\[A-Za-z]+|[A-Za-z]+", _repl, label)
    return r"\text{[}" + math_label + r"\text{:] }"


def to_latex_document(history, title=None):
    """Generate a standalone compilable LaTeX document from a derivation history.

    Each state in history is rendered as a display equation prefixed with
    the step label.  Requires amsmath and amssymb.

    Example::

        history = Derivation([...]).apply(State(expr))
        tex = to_latex_document(history, title="PVW derivation")
        with open("derivation.tex", "w") as f:
            f.write(tex)
    """
    lines = [
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{amsmath,amssymb}",
        r"\begin{document}",
    ]
    if title:
        lines.append(r"\section*{" + title + "}")
    for state in history:
        label = state.label or "initial"
        lines.append(
            r"\[" + _label_to_math(label) + r"\quad " + state.expr.latex() + r"\]"
        )
    lines.append(r"\end{document}")
    return "\n".join(lines)


__all__ = [
    # Types
    "Rational", "Expr", "RationalConst", "NamedConst", "SymbolicVar",
    "Parameter", "Sum", "Scale", "TensorProduct", "IdentityTensor",
    "LeviCivitaTensor", "Trace", "Contract", "DoubleContract",
    "DoubleContractReversed", "CrossProduct", "NamedTensor", "Product",
    "FunctionApply", "Pow", "MaterialDeriv", "PatternVar",
    "Gradient", "Divergence", "Rotor", "Integral",
    "Domain", "SurfaceDomain", "VolumeDomain", "CoordSystem",
    "State", "DerivationStep", "Derivation", "Identity", "Nabla",
    # Singletons
    "I", "eps", "t", "wcs", "cylindrical_cs", "spherical_cs", "nabla",
    # Expression factories
    "tensor", "scalar", "parameter", "named", "rational", "make_pattern_var",
    # Algebraic operations
    "tp", "dot", "ddot", "ddot2", "cross", "trace",
    # Scalar functions
    "exp", "log", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "sqrt", "pow",
    # Differentiation
    "deriv", "dt", "ddt",
    # Coordinate systems
    "grad", "make_direct_basis_cs",
    # Symbolic differential operators
    "gradient", "divergence", "rot",
    # Integral / domain
    "make_surface_domain", "make_volume_domain", "integral",
    # Derivation rendering
    "show", "show_final", "show_jupyter",
    # LaTeX document export
    "to_latex_document",
    # Documentation
    "doc",
    # Step factories
    "simplify_identity_step", "expand_step", "expand_poly_step",
    "substitute_step", "diff_step",
    "apply_integration_by_parts_step", "apply_divergence_theorem_step",
    "localize_step", "collect_step",
    "expand_in_basis_step", "simplify_basis_dot_step",
    "collect_zero_terms_step", "reassemble_from_components_step",
    "collect_repeated_sum_step", "reassemble_vector_step", "reassemble_dot_step",
    "IndexedSum", "make_indexed_sum",
    "SymBasisVec", "make_sym_basis_vec",
    "prove_equal_by_components",
    "apply_identity", "find_matches", "apply_identity_auto", "matches_at_root",
    "search_apply",
    "declare_symmetric", "declare_skew_symmetric",
]
