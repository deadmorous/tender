"""Expanding ε_ijk and deriving Kronecker-delta identities.

Demonstrates the three new derivation steps:

  expand_eps         — replace ε_ijk with its 6-term cofactor expansion
  fold_sums          — fold a concrete N-addend cycle back into an ExplicitSum
  contract_delta     — contract Σ_m δ^m_k δ^m_l into δ_kl

Two derivations are shown:

1. ε(1,2,3) = +1  (concrete evaluation via expand_eps)
2. Σ_m δ^m_k δ^m_l = δ_kl  (fold_sums + contract_delta)
"""

import pathlib

import tender
import tender.derivation as td
from tender import Level, Realm

sp = tender.space_3d


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def derivation_to_latex(drv, labels=None):
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


# ---------------------------------------------------------------------------
# 1. Concrete evaluation of ε(1,2,3) = +1
# ---------------------------------------------------------------------------

eps_123 = tender.levi_civita(
    Realm.Oblique, sp,
    [Level.Lower, Level.Lower, Level.Lower],
    [1, 2, 3],
)

drv1 = td.Derivation(eps_123)
drv1.step(td.expand_eps).step(td.eval_delta_concrete).step(td.fold_arithmetic)

labels1 = [
    "expand $\\varepsilon$ as cofactor determinant",
    "evaluate $\\delta$ on concrete indices",
    "fold arithmetic",
]

assert drv1.current.latex() == "1", f"expected 1, got {drv1.current.latex()}"
print("ε(1,2,3) derivation:")
for k, e in enumerate(drv1.history):
    tag = f"  [{labels1[k-1]}]" if k > 0 else "  [initial]"
    print(f"  step {k}: {e.latex()}{tag}")
print()

block1 = derivation_to_latex(drv1, labels1)

# ---------------------------------------------------------------------------
# 2. Contraction: Σ_m δ^m_k δ^m_l = δ_kl
# ---------------------------------------------------------------------------

ctx2 = tender.Context()
k2 = ctx2.alloc_index()
l2 = ctx2.alloc_index()

def delta_upper(v, idx):
    return tender.delta(Realm.Oblique, sp, Level.Upper, Level.Lower, v, idx, ctx=ctx2)

concrete_sum = (
    delta_upper(1, k2) * delta_upper(1, l2)
    + delta_upper(2, k2) * delta_upper(2, l2)
    + delta_upper(3, k2) * delta_upper(3, l2)
)

drv2 = td.Derivation(concrete_sum)
drv2.step(td.fold_sums).step(td.contract_delta)

labels2 = [
    "fold concrete sum into $\\sum_m$",
    r"contract $\sum_m \delta^m{}_k\,\delta^m{}_l \to \delta_{kl}$",
]

result_latex = drv2.current.latex()
print("Σ_m δ^m_k δ^m_l = δ_kl derivation:")
for k, e in enumerate(drv2.history):
    tag = f"  [{labels2[k-1]}]" if k > 0 else "  [initial]"
    print(f"  step {k}: {e.latex()}{tag}")
print()

block2 = derivation_to_latex(drv2, labels2)

# ---------------------------------------------------------------------------
# Write LaTeX document
# ---------------------------------------------------------------------------

doc = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\begin{document}

\section*{Expanding $\varepsilon_{ijk}$ and contracting Kronecker deltas}

The Levi-Civita symbol in 3D oblique space can be written as a
$3\times3$ determinant of Kronecker deltas (cofactor expansion):
%
\[
  \varepsilon_{ijk}
  = \det\!\begin{pmatrix}
      \delta^1{}_i & \delta^1{}_j & \delta^1{}_k \\
      \delta^2{}_i & \delta^2{}_j & \delta^2{}_k \\
      \delta^3{}_i & \delta^3{}_j & \delta^3{}_k
    \end{pmatrix}.
\]

\subsection*{1.\quad Concrete evaluation: $\varepsilon_{123} = +1$}

""" + block1 + r"""

\subsection*{2.\quad Contraction identity: $\displaystyle\sum_m \delta^m{}_k\,\delta^m{}_l = \delta_{kl}$}

Starting from the explicit concrete sum $\delta^1{}_k\delta^1{}_l
+ \delta^2{}_k\delta^2{}_l + \delta^3{}_k\delta^3{}_l$:

""" + block2 + r"""

\end{document}
"""

out_dir = pathlib.Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "eps_delta.tex"
out_path.write_text(doc)

print(f"Written : {out_path}")
print(f"Compile : pdflatex -output-directory out {out_path}")
