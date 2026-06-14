"""Deriving δ^i_i = 3 and δ^i_j δ^i_j = 3 in 3D oblique space.

Shows the full derivation history for two classic contractions:

1. The trace of Kronecker delta:   sum_i δ^i_i = 3
2. The Frobenius-style contraction: sum_i sum_j δ^i_j δ^i_j = 3

Steps applied in both cases:
  unroll_sums         — expand ExplicitSum with concrete values
  eval_delta_concrete — evaluate δ(a,b) to 1 (a==b) or 0 (a≠b)
  fold_arithmetic     — constant-fold the resulting scalar Sum tree
"""

import pathlib

import tender
import tender.derivation as td
from tender import Level, Realm

sp = tender.space_3d  # {1, 2, 3}

# ---------------------------------------------------------------------------
# Helper: render a derivation history as a LaTeX align* block
# ---------------------------------------------------------------------------

def derivation_to_latex(drv, labels=None):
    """Return a LaTeX align* environment showing each step of *drv*.

    *labels* is an optional list of strings (one per step after the initial)
    describing what was applied.
    """
    lines = []
    for k, expr in enumerate(drv.history):
        latex = expr.latex()
        if k == 0:
            lines.append(f"  &= {latex}")
        else:
            label = labels[k - 1] if labels and k - 1 < len(labels) else ""
            comment = rf"\quad\text{{{label}}}" if label else ""
            lines.append(f"  &= {latex}{comment}")
    return "\\begin{align*}\n" + " \\\\\n".join(lines) + "\n\\end{align*}"


step_labels = [
    "unroll sum",
    "evaluate $\\delta$ on concrete indices",
    "fold arithmetic",
]

# ---------------------------------------------------------------------------
# 1. Trace: sum_i δ^i_i = 3
# ---------------------------------------------------------------------------

ctx1 = tender.Context()
i1 = ctx1.alloc_index()

trace_expr = tender.explicit_sum(
    i1,
    tender.delta(Realm.Oblique, sp, Level.Upper, Level.Lower, i1, i1),
    ctx=ctx1,
)

trace_drv = td.Derivation(trace_expr)
trace_drv.step(td.unroll_sums).step(td.eval_delta_concrete).step(td.fold_arithmetic)

assert trace_drv.current.latex() == "3", f"expected 3, got {trace_drv.current.latex()}"
print("δ^i_i derivation:")
for k, e in enumerate(trace_drv.history):
    label = f"  [{step_labels[k-1]}]" if k > 0 else "  [initial]"
    print(f"  step {k}: {e.latex()}{label}")
print()

trace_block = derivation_to_latex(trace_drv, step_labels)

# ---------------------------------------------------------------------------
# 2. Contraction: sum_i sum_j δ^i_j δ^i_j = 3
# ---------------------------------------------------------------------------

ctx2 = tender.Context()
i2 = ctx2.alloc_index()
j2 = ctx2.alloc_index()

d1 = tender.delta(Realm.Oblique, sp, Level.Upper, Level.Lower, i2, j2, ctx=ctx2)
d2 = tender.delta(Realm.Oblique, sp, Level.Upper, Level.Lower, i2, j2, ctx=ctx2)
contract_expr = tender.explicit_sum(
    i2,
    tender.explicit_sum(j2, d1 * d2, ctx=ctx2),
    ctx=ctx2,
)

contract_drv = td.Derivation(contract_expr)
contract_drv.step(td.unroll_sums).step(td.eval_delta_concrete).step(td.fold_arithmetic)

assert contract_drv.current.latex() == "3", f"expected 3, got {contract_drv.current.latex()}"
print("δ^i_j δ^i_j derivation:")
for k, e in enumerate(contract_drv.history):
    label = f"  [{step_labels[k-1]}]" if k > 0 else "  [initial]"
    print(f"  step {k}: {e.latex()}{label}")
print()

contract_block = derivation_to_latex(contract_drv, step_labels)

# ---------------------------------------------------------------------------
# Write standalone LaTeX document
# ---------------------------------------------------------------------------

doc = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\begin{document}

\section*{Deriving $\delta^i{}_i = 3$ and $\delta^i{}_j\,\delta^i{}_j = 3$}

The Kronecker delta $\delta^i{}_j$ in 3D oblique space satisfies two
elementary contraction identities, both derived below by three uniform
rewriting steps:
\begin{enumerate}
  \item \textbf{Unroll sums} — replace each explicit sum over a concrete
    index space by a binary sum tree of the summand at each concrete value.
  \item \textbf{Evaluate $\delta$ on concrete indices} — replace
    $\delta^a{}_b$ (concrete $a,b$) by $1$ when $a=b$ and $0$ otherwise.
  \item \textbf{Fold arithmetic} — reduce the resulting constant
    expression to a single rational number.
\end{enumerate}

\subsection*{1.\quad Trace: $\displaystyle\sum_i \delta^i{}_i = 3$}

""" + trace_block + r"""

\subsection*{2.\quad Self-contraction: $\displaystyle\sum_i\sum_j \delta^i{}_j\,\delta^i{}_j = 3$}

""" + contract_block + r"""

\end{document}
"""

out_dir = pathlib.Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "delta_trace.tex"
out_path.write_text(doc)

print(f"Written : {out_path}")
print(f"Compile : pdflatex -output-directory out {out_path}")
