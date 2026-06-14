"""Expanding ε_ijk and deriving Kronecker-delta identities.

Demonstrates the derivation steps:

  expand_eps         — replace ε_ijk with its 6-term cofactor expansion
  expand_products    — distribute tensor products over sums (expand brackets)
  fold_sums          — fold a concrete N-addend cycle back into an ExplicitSum
  contract_delta     — contract Σ_m δ^m_k δ^m_l into δ_kl

Four derivations are shown:

1. ε(1,2,3) = +1         — concrete evaluation via expand_eps
2. Σ_m δ^m_k δ^m_l = δ_kl — fold_sums + contract_delta
3. Σ_{ij} ε^{ijk} ε_{ijl} = 2δ^k_l   — full two-index automated derivation
4. Σ_i ε^{ijk} ε_{iml}               — one-index: expands to 12 delta terms
"""

import pathlib

import tender
import tender.derivation as td
from tender import Level, Realm

sp = tender.space_3d


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def derivation_to_latex(drv, labels=None, steps=None):
    """Render selected derivation steps as LaTeX dmath* blocks.

    Each step occupies its own dmath* environment (from the breqn package),
    which auto-breaks long equations across lines.

    labels : labels[k] is the annotation for drv.history[k+1]
    steps  : list of history indices to include (None = all steps)
    """
    history = drv.history
    if steps is None:
        steps = list(range(len(history)))

    blocks = []
    first = True
    for k in steps:
        eq_latex = history[k].latex()
        if first:
            body = eq_latex
            first = False
        else:
            label = labels[k - 1] if labels and k - 1 < len(labels) else ""
            comment = rf"\quad\text{{{label}}}" if label else ""
            body = f"= {eq_latex}{comment}"
        blocks.append(f"\\begin{{dmath*}}\n  {body}\n\\end{{dmath*}}")
    return "\n".join(blocks)


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
print("Derivation 1: ε(1,2,3) = +1")
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
    r"contract $\sum_m \delta^m_k\,\delta^m_l \to \delta_{kl}$",
]

print("Derivation 2: Σ_m δ^m_k δ^m_l = δ_kl")
for k, e in enumerate(drv2.history):
    tag = f"  [{labels2[k-1]}]" if k > 0 else "  [initial]"
    print(f"  step {k}: {e.latex()}{tag}")
print()

block2 = derivation_to_latex(drv2, labels2)


# ---------------------------------------------------------------------------
# 3. Two-index contraction: Σ_{ij} ε^{ijk} ε_{ijl} = 2δ^k_l
# ---------------------------------------------------------------------------

ctx3 = tender.Context()
i3 = ctx3.alloc_index()
j3 = ctx3.alloc_index()
k3 = ctx3.alloc_index()
l3 = ctx3.alloc_index()

eps3a = tender.levi_civita(
    Realm.Oblique, sp,
    [Level.Upper, Level.Upper, Level.Upper], [i3, j3, k3], ctx=ctx3)
eps3b = tender.levi_civita(
    Realm.Oblique, sp,
    [Level.Lower, Level.Lower, Level.Lower], [i3, j3, l3], ctx=ctx3)

ei3 = tender.explicit_sum(j3,
        tender.explicit_sum(i3, eps3a * eps3b, ctx=ctx3),
        ctx=ctx3)

drv3 = td.Derivation(ei3)

labels3 = [
    "expand both $\\varepsilon$ symbols",
    "unroll $\\sum_i$",
    "unroll $\\sum_j$",
    "distribute products over sums",
    "evaluate $\\delta$ on concrete indices",
    "fold arithmetic (incl.\\ $(-A)(-B)=AB$)",
    "fold first 3-cycle into $\\sum_p$",
    "fold second 3-cycle into $\\sum_p$",
    r"contract both $\sum_p \delta^p_k\,\delta^p_l \to \delta^k_l$",
]

(drv3
 .step(td.expand_eps)
 .step(td.unroll_sums)
 .step(td.unroll_sums)
 .step(td.expand_products)
 .step(td.eval_delta_concrete)
 .step(td.fold_arithmetic)
 .step(td.fold_sums)
 .step(td.fold_sums)
 .step(td.contract_delta))

print("Derivation 3: Σ_{ij} ε^{ijk} ε_{ijl} = 2δ^k_l")
for k, e in enumerate(drv3.history):
    tag = f"  [{labels3[k-1]}]" if k > 0 else "  [initial]"
    print(f"  step {k}: {e.latex()[:100]}{tag}")
print(f"  result: {drv3.current.latex()}")
print()

# For the PDF, skip the large intermediate steps (1–5: unrolled sums and
# distributed products contain hundreds of delta symbols).  Show only the
# initial expression, the state after arithmetic folding, and the final result.
block3 = derivation_to_latex(drv3, labels3, steps=[0, 6, 7, 9])


# ---------------------------------------------------------------------------
# 4. One-index contraction: Σ_i ε^{ijk} ε_{iml} = δ^j_m δ^k_l - δ^j_l δ^k_m
#    Automated steps expand to 12 concrete delta products.
#    Folding into two abstract delta pairs requires a theorem step
#    of the form Σ_p δ^p_A δ^p_B = δ_AB (future work).
# ---------------------------------------------------------------------------

ctx4 = tender.Context()
i4 = ctx4.alloc_index()
j4 = ctx4.alloc_index()
k4 = ctx4.alloc_index()
m4 = ctx4.alloc_index()
l4 = ctx4.alloc_index()

eps4a = tender.levi_civita(
    Realm.Oblique, sp,
    [Level.Upper, Level.Upper, Level.Upper], [i4, j4, k4], ctx=ctx4)
eps4b = tender.levi_civita(
    Realm.Oblique, sp,
    [Level.Lower, Level.Lower, Level.Lower], [i4, m4, l4], ctx=ctx4)

ei4 = tender.explicit_sum(i4, eps4a * eps4b, ctx=ctx4)

drv4 = td.Derivation(ei4)

labels4 = [
    "expand both $\\varepsilon$ symbols",
    "unroll $\\sum_i$ (concrete values $i=1,2,3$)",
    "distribute products over sums",
    "evaluate $\\delta$ on concrete $i$",
    "fold arithmetic (12 concrete delta-pair terms remain)",
]

(drv4
 .step(td.expand_eps)
 .step(td.unroll_sums)
 .step(td.expand_products)
 .step(td.eval_delta_concrete)
 .step(td.fold_arithmetic))

print("Derivation 4: Σ_i ε^{ijk} ε_{iml} (expands to 12 delta-pair terms)")
for k, e in enumerate(drv4.history):
    tag = f"  [{labels4[k-1]}]" if k > 0 else "  [initial]"
    print(f"  step {k}: {e.latex()[:100]}{tag}")
print("  (= δ^j_m δ^k_l - δ^j_l δ^k_m; final folding needs theorem Σ_p δ^p_A δ^p_B = δ_AB)")
print()

# Show only the initial expression and the final 12-term delta expansion;
# the intermediate unrolled/distributed forms are too large for the page.
block4 = derivation_to_latex(drv4, labels4, steps=[0, 5])


# ---------------------------------------------------------------------------
# Write LaTeX document
# ---------------------------------------------------------------------------

doc = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{breqn}   % automatic line-breaking of long display equations
\begin{document}

\section*{Levi-Civita symbol and Kronecker-delta identities}

The Levi-Civita symbol in 3D oblique space can be written as a
$3\times3$ determinant of Kronecker deltas (cofactor expansion):
%
\[
  \varepsilon_{ijk}
  = \det\!\begin{pmatrix}
      \delta^1_i & \delta^1_j & \delta^1_k \\
      \delta^2_i & \delta^2_j & \delta^2_k \\
      \delta^3_i & \delta^3_j & \delta^3_k
    \end{pmatrix}.
\]

\subsection*{1.\quad Concrete evaluation: $\varepsilon_{123} = +1$}

""" + block1 + r"""

\subsection*{2.\quad Contraction identity:
  $\displaystyle\sum_m \delta^m_k\,\delta^m_l = \delta_{kl}$}

Starting from the explicit concrete sum
$\delta^1_k\delta^1_l + \delta^2_k\delta^2_l + \delta^3_k\delta^3_l$:

""" + block2 + r"""

\subsection*{3.\quad Two-index contraction:
  $\displaystyle\sum_{i,j} \varepsilon^{ijk} \varepsilon_{ijl} = 2\delta^k_l$}

Steps: expand both $\varepsilon$ symbols, unroll $\sum_i$ and $\sum_j$
concretely, distribute tensor products over sums, evaluate concrete
$\delta$s, and fold arithmetic (including $(-A)(-B)=AB$).
The intermediate expressions contain hundreds of delta symbols and are
omitted; we resume after arithmetic folding:

""" + block3 + r"""

\subsection*{4.\quad One-index contraction:
  $\displaystyle\sum_i \varepsilon^{ijk} \varepsilon_{iml}$}

Expanding $\varepsilon$, unrolling $\sum_i$ concretely, distributing
products, evaluating concrete $\delta$s, and folding arithmetic yields
12 Kronecker-delta terms representing
$\delta^j_m\delta^k_l - \delta^j_l\delta^k_m$.
Closing this to two abstract delta pairs requires a theorem
$\displaystyle\sum_p \delta^p_A \delta^p_B = \delta_{AB}$ (future work):

""" + block4 + r"""

\noindent\textbf{Expected result:}\quad
$\displaystyle\sum_i\varepsilon^{ijk}\varepsilon_{iml}
  = \delta^j_m\delta^k_l - \delta^j_l\delta^k_m$.

\end{document}
"""

out_dir = pathlib.Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "eps_delta.tex"
out_path.write_text(doc)

print(f"Written : {out_path}")
print(f"Compile : pdflatex -output-directory out {out_path}")
