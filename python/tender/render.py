"""Jupyter display helpers for tender expressions."""

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
