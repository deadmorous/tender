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


def _label_to_math(label):
    """Render a step label as a LaTeX math-mode fragment.

    Plain identifiers (not part of a \\command) are wrapped in \\mathrm{} so
    they appear upright.  LaTeX commands (\\partial, etc.) are passed through
    unchanged.  Returns a string suitable for use inside \\[...\\].
    """
    import re

    if "\\" not in label:
        # No LaTeX commands — use text mode directly.
        return r"\text{[" + label + r":] }"

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
    "apply_identity", "find_matches", "apply_identity_auto", "matches_at_root",
    "search_apply",
    "declare_symmetric", "declare_skew_symmetric",
]
