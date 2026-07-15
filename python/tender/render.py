"""Jupyter display helpers for tender expressions."""

import html as _html

from ._core import IndexNameMap, render_latex as _render_latex


def display(expr, map=None):
    """Render expr and display it in a Jupyter cell using IPython.display.Math.

    Parameters
    ----------
    expr : tender.Expr
    map  : tender.IndexNameMap or None
        Reuse an existing map to keep index names consistent across cells.
        A fresh map is created if None.

    Returns
    -------
    IPython.display.Math
        The display object; Jupyter renders it automatically when it is the
        last expression in a cell.
    """
    from IPython.display import Math  # noqa: PLC0415

    if map is None:
        map = IndexNameMap()
    return Math(_render_latex(expr, map))


class LabeledExpr:
    """A path-labeled view of an expression, for reading off selection paths.

    Renders the whole expression, plus a legend mapping the **path** of each
    selected part to that part's own LaTeX — so you can see which path to pass
    to :meth:`Expr.at` / :meth:`Expr.rewrite_at` / :func:`tender.derivation.at`.
    Because each part is rendered on its own, the legend is exact regardless of
    how the pretty renderer folds the whole (``Δ``, subtraction, flattening) and
    regardless of hash-cons sharing.

    In Jupyter the object displays as the expression above a table; in a
    terminal ``print`` / ``str`` gives a plain-text table.  ``.latex`` is the
    whole-expression LaTeX and ``.legend`` is a ``list[(path, sub_latex)]``.

    ``which`` selects which nodes are listed: ``"all"`` every node, ``"atoms"``
    the leaves, ``"tensors"`` every named tensor, ``"wellknown"`` the I/δ/ε/g,
    ``"terms"`` the top-level ``+``/``−`` addends.
    """

    def __init__(self, expr, which="all", map=None):
        self.expr = expr
        self.which = which
        self._map = map if map is not None else IndexNameMap()
        # Render the whole first so index names are allocated, then each part
        # with the same map — sub-renders reuse those names (consistency).
        self.latex = _render_latex(expr, self._map)
        paths = expr.addends() if which == "terms" else expr.paths(which)
        self.legend = [(list(p), expr.at(p).latex(self._map)) for p in paths]

    def _repr_html_(self):
        rows = "".join(
            "<tr>"
            f"<td style='text-align:right;padding:2px 10px;font-family:monospace'>{p}</td>"
            f"<td style='padding:2px 10px'>\\({_html.escape(sub)}\\)</td>"
            "</tr>"
            for p, sub in self.legend
        )
        return (
            f"<div>\\({_html.escape(self.latex)}\\)</div>"
            "<table style='border-collapse:collapse;margin-top:6px'>"
            "<tr><th style='text-align:right;padding:2px 10px'>path</th>"
            "<th style='text-align:left;padding:2px 10px'>part</th></tr>"
            f"{rows}</table>"
        )

    def _repr_latex_(self):
        # Fallback for front-ends that only honour LaTeX: the whole expression.
        return f"${self.latex}$"

    def __str__(self):
        width = max((len(str(p)) for p, _ in self.legend), default=4)
        lines = [self.latex, ""]
        lines += [f"{str(p):<{width}}  {sub}" for p, sub in self.legend]
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()


def labeled(expr, which="all", map=None):
    """A path-labeled view of *expr* (see :class:`LabeledExpr`).

    Return it as the last value in a Jupyter cell to display the expression
    with a path→part legend, or ``print`` it in a terminal::

        print(t.render.labeled(expr, which="wellknown"))

    Then pass a path you read off to ``expr.at(path)`` / ``td.at(expr, path, step)``.
    """
    return LabeledExpr(expr, which, map)


def to_latex_document(exprs, title=None):
    """Render a sequence of expressions as a standalone LaTeX document.

    Each expression occupies one display equation.

    Parameters
    ----------
    exprs : iterable of (label, Expr) pairs
    title : str or None

    Returns
    -------
    str
        A compilable LaTeX source string.
    """
    lines = [
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{amsmath,amssymb,cancel}",
        r"\begin{document}",
    ]
    if title:
        lines.append(r"\section*{" + title + "}")
    shared_map = IndexNameMap()
    for label, expr in exprs:
        tex = _render_latex(expr, shared_map)
        lines.append(
            r"\[ \text{" + str(label) + r":}\quad " + tex + r" \]"
        )
    lines.append(r"\end{document}")
    return "\n".join(lines)
