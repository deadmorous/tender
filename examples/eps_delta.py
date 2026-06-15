"""Expanding ε_ijk and deriving Kronecker-delta identities.

Demonstrates the derivation steps:

  expand_eps         — replace ε_ijk with its 6-term cofactor expansion
  expand_products    — distribute tensor products over sums (expand brackets)
  fold_sums          — fold a concrete N-addend cycle back into an ExplicitSum
  contract_delta     — contract Σ_m δ^m_k δ^m_l into δ_kl
  fold_equal_addends — collect equal addends: X + X → 2X
  contract_eps_pair  — contract Σ ε ε directly to the generalized Kronecker delta

Four derivations are shown:

1. ε(1,2,3) = +1         — concrete evaluation via expand_eps
2. Σ_m δ^m_k δ^m_l = δ_kl — fold_sums + contract_delta
3. Σ_{ij} ε^{ijk} ε_{ijl} = 2δ^k_l   — full two-index automated derivation
4. Σ_i ε^{ijk} ε_{iml} = δ^j_m δ^k_l − δ^j_l δ^k_m  — one-index, closed
   symbolically by contract_eps_pair (no concrete-WCS unrolling).  See
   vibe 000035 for why the older 12-term concrete route needed a "creative
   step" and this one does not.
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

    If drv.index_map is set, it is used for all latex() calls so that
    index names stay consistent across all steps.
    """
    history = drv.history
    imap = drv.index_map
    if steps is None:
        steps = list(range(len(history)))

    blocks = []
    first = True
    for k in steps:
        eq_latex = drv.latex(k) if imap else history[k].latex()
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

imap2 = tender.IndexNameMap()
imap2.assign(k2, "k")
imap2.assign(l2, "l")

def delta_upper(v, idx):
    return tender.delta(Realm.Oblique, sp, Level.Upper, Level.Lower, v, idx, ctx=ctx2)

concrete_sum = (
    delta_upper(1, k2) * delta_upper(1, l2)
    + delta_upper(2, k2) * delta_upper(2, l2)
    + delta_upper(3, k2) * delta_upper(3, l2)
)

drv2 = td.Derivation(concrete_sum, index_map=imap2)
drv2.step(td.fold_sums).step(td.contract_delta)

labels2 = [
    "fold concrete sum into $\\sum_m$",
    r"contract $\sum_m \delta^m_k\,\delta^m_l \to \delta_{kl}$",
]

print("Derivation 2: Σ_m δ^m_k δ^m_l = δ_kl")
for k, e in enumerate(drv2.history):
    tag = f"  [{labels2[k-1]}]" if k > 0 else "  [initial]"
    print(f"  step {k}: {drv2.latex(k)}{tag}")
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

imap3 = tender.IndexNameMap()
imap3.assign(i3, "i")
imap3.assign(j3, "j")
imap3.assign(k3, "k")
imap3.assign(l3, "l")

eps3a = tender.levi_civita(
    Realm.Oblique, sp,
    [Level.Upper, Level.Upper, Level.Upper], [i3, j3, k3], ctx=ctx3)
eps3b = tender.levi_civita(
    Realm.Oblique, sp,
    [Level.Lower, Level.Lower, Level.Lower], [i3, j3, l3], ctx=ctx3)

ei3 = tender.explicit_sum(j3,
        tender.explicit_sum(i3, eps3a * eps3b, ctx=ctx3),
        ctx=ctx3)

drv3 = td.Derivation(ei3, index_map=imap3)

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
    "fold equal addends: $\\delta^k_l + \\delta^k_l \\to 2\\delta^k_l$",
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
 .step(td.contract_delta)
 .step(td.fold_equal_addends))

print("Derivation 3: Σ_{ij} ε^{ijk} ε_{ijl} = 2δ^k_l")
for k, e in enumerate(drv3.history):
    tag = f"  [{labels3[k-1]}]" if k > 0 else "  [initial]"
    print(f"  step {k}: {drv3.latex(k)[:100]}{tag}")
print(f"  result: {drv3.latex(-1)}")
print()

# For the PDF, skip the large intermediate steps (1–5: unrolled sums and
# distributed products contain hundreds of delta symbols).  Show only the
# initial expression, the state after arithmetic folding, the two fold_sums
# steps, the contracted result, and the final collapsed result.
block3 = derivation_to_latex(drv3, labels3, steps=[0, 6, 7, 9, 10])


# ---------------------------------------------------------------------------
# 4. One-index contraction: Σ_i ε^{ijk} ε_{iml} = δ^j_m δ^k_l - δ^j_l δ^k_m
#    Closed symbolically in a single step by contract_eps_pair (the generalized
#    Kronecker delta).  No concrete-WCS unrolling, no "creative step" — see
#    vibe 000035.  The older concrete route (expand_eps + unroll + ...) instead
#    stalls at 12 delta-pair terms; it is shown below for contrast.
# ---------------------------------------------------------------------------

ctx4 = tender.Context()
i4 = ctx4.alloc_index()
j4 = ctx4.alloc_index()
k4 = ctx4.alloc_index()
m4 = ctx4.alloc_index()
l4 = ctx4.alloc_index()

imap4 = tender.IndexNameMap()
imap4.assign(i4, "i")
imap4.assign(j4, "j")
imap4.assign(k4, "k")
imap4.assign(m4, "m")
imap4.assign(l4, "l")

eps4a = tender.levi_civita(
    Realm.Oblique, sp,
    [Level.Upper, Level.Upper, Level.Upper], [i4, j4, k4], ctx=ctx4)
eps4b = tender.levi_civita(
    Realm.Oblique, sp,
    [Level.Lower, Level.Lower, Level.Lower], [i4, m4, l4], ctx=ctx4)

ei4 = tender.explicit_sum(i4, eps4a * eps4b, ctx=ctx4)

# --- symbolic closure: one step ---
drv4 = td.Derivation(ei4, index_map=imap4)
labels4 = [
    r"contract $\varepsilon\varepsilon$ "
    r"(generalized Kronecker $\delta$)",
]
drv4.step(td.contract_eps_pair)

expected4 = (
    r"\delta^{j}_{m} \, \delta^{k}_{l} - \delta^{j}_{l} \, \delta^{k}_{m}"
)
assert drv4.latex(-1) == expected4, f"got {drv4.latex(-1)}"

print("Derivation 4: Σ_i ε^{ijk} ε_{iml} = δ^j_m δ^k_l - δ^j_l δ^k_m")
for k, e in enumerate(drv4.history):
    tag = f"  [{labels4[k-1]}]" if k > 0 else "  [initial]"
    print(f"  step {k}: {drv4.latex(k)}{tag}")
print()

block4 = derivation_to_latex(drv4, labels4)

# --- contrast: the concrete route stalls at 12 delta-pair terms ---
drv4_long = td.Derivation(ei4, index_map=imap4)
labels4_long = [
    "expand both $\\varepsilon$ symbols",
    "unroll $\\sum_i$ (concrete values $i=1,2,3$)",
    "distribute products over sums",
    "evaluate $\\delta$ on concrete $i$",
    "fold arithmetic (12 concrete delta-pair terms remain)",
]
(drv4_long
 .step(td.expand_eps)
 .step(td.unroll_sums)
 .step(td.expand_products)
 .step(td.eval_delta_concrete)
 .step(td.fold_arithmetic))

print("Derivation 4 (concrete route, for contrast): stalls at 12 delta terms")
for k, e in enumerate(drv4_long.history):
    tag = f"  [{labels4_long[k-1]}]" if k > 0 else "  [initial]"
    print(f"  step {k}: {drv4_long.latex(k)[:100]}{tag}")
print()

# Show only the initial expression and the final 12-term delta expansion;
# the intermediate unrolled/distributed forms are too large for the page.
block4_long = derivation_to_latex(drv4_long, labels4_long, steps=[0, 5])


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

The product of two Levi-Civita symbols is a generalized Kronecker delta, so
the contraction closes in a \emph{single} symbolic step
(\texttt{contract\_eps\_pair}), with no concrete-WCS unrolling and no
``creative'' add/subtract:

""" + block4 + r"""

\medskip\noindent
\textbf{For contrast}, the older concrete route --- expand $\varepsilon$,
unroll $\sum_i$ over $i=1,2,3$, distribute, evaluate concrete $\delta$s, fold
arithmetic --- stalls at 12 Kronecker-delta terms.  These cannot be folded
back into $\delta^j_m\delta^k_l - \delta^j_l\delta^k_m$ without re-introducing
terms that arithmetic folding cancelled (see vibe 000035):

""" + block4_long + r"""

\end{document}
"""

out_dir = pathlib.Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "eps_delta.tex"
out_path.write_text(doc)

print(f"Written : {out_path}")
print(f"Compile : pdflatex -output-directory out {out_path}")
