"""Levi-Civita symbol products: eps-delta identity and scalar contractions.

Demonstrates:
  1. Abstract derivation of ε_{ijk}ε_{isp} = δ_{js}δ_{kp} - δ_{jp}δ_{ks}
     (derivation_1) via contract_eps_pair_step.
  2. Specialising to ε_{ijk}ε_{ijp} = 2δ_{kp} (derivation_2).
  3. Full contraction ε_{ijk}ε_{ijk} = 6 (derivation_3).

Note on the abstract index framework
-------------------------------------
tender uses *abstract* indices (symbolic IDs) without a fixed dimension.
Kronecker deltas fold to 1 when both index IDs are identical
(make_kronecker_delta(j,j) → 1), so "δ_{jj}" evaluates to 1, not dim.
Derivations 2 and 3 therefore use a *concrete* 3-D WCS expansion of ε to
obtain the correct numerical results.
"""

import pathlib

from tender import (
    wcs, rational, trace, ddot,
    alloc_index_id, make_levi_civita_symbol,
    contract_eps_pair_step, contract_kronecker_step,
    expand_levi_civita_first_step, simplify_basis_dot_step,
    collect_zero_terms_step, reassemble_from_components_step,
    State, Derivation, show, show_final,
    to_latex_document,
)

cs = wcs


# ---------------------------------------------------------------------------
# Concrete WCS expansion of ε as a sum of 6 signed rank-3 basis products
# ---------------------------------------------------------------------------

def _lc_sign(i, j, k):
    """Levi-Civita sign for indices (i,j,k) ∈ {0,1,2}."""
    if len({i, j, k}) < 3:
        return 0
    vals = [i, j, k]
    inv = sum(1 for a in range(3) for b in range(a + 1, 3) if vals[a] > vals[b])
    return 1 if inv % 2 == 0 else -1


def eps_wcs_expansion(cs):
    """Return Σ_{ijk} ε_{ijk} e^i⊗e^j⊗e^k expanded over the concrete 3-D WCS."""
    terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = _lc_sign(i, j, k)
                if s != 0:
                    ei = cs.basis(i)
                    ej = cs.basis(j)
                    ek = cs.basis(k)
                    terms.append(rational(s) * ei * ej * ek)
    result = terms[0]
    for t in terms[1:]:
        result = result + t
    return result


# ===========================================================================
# derivation_1 — abstract formula: ε_{ijk}ε_{isp} = δ_{js}δ_{kp} − δ_{jp}δ_{ks}
# ===========================================================================

i_id = alloc_index_id()
j_id = alloc_index_id()
k_id = alloc_index_id()
s_id = alloc_index_id()
p_id = alloc_index_id()

lcs_ijk = make_levi_civita_symbol([i_id, j_id, k_id], [False, False, False])
lcs_isp = make_levi_civita_symbol([i_id, s_id, p_id], [False, False, False])

eps_pair = lcs_ijk * lcs_isp

derivation_1 = Derivation([
    contract_eps_pair_step(),
])
history_1 = derivation_1.apply(State(eps_pair))

print("=" * 70)
print("derivation_1 : ε_{ijk}ε_{isp} = δ_{js}δ_{kp} − δ_{jp}δ_{ks}")
print("=" * 70)
print(show(history_1))

# ===========================================================================
# derivation_2 — ε_{ijk}ε_{ijp} = 2δ_{kp}
# ===========================================================================

result_2d = [[0] * 3 for _ in range(3)]
for k in range(3):
    for p in range(3):
        for i in range(3):
            for j in range(3):
                result_2d[k][p] += _lc_sign(i, j, k) * _lc_sign(i, j, p)

print()
print("=" * 70)
print("derivation_2 : Σ_{ij} ε_{ijk}ε_{ijp} = 2δ_{kp}")
print("=" * 70)
print("Result matrix [k][p]:")
for k in range(3):
    print(" ", result_2d[k])
print("→ This is 2 * Identity (2δ_{kp})")

# ===========================================================================
# derivation_3 — ε_{ijk}ε_{ijk} = 6
# ===========================================================================

eps_wcs = eps_wcs_expansion(cs)
expr_3 = trace(ddot(eps_wcs, eps_wcs))

derivation_3 = Derivation([
    simplify_basis_dot_step(cs),
    collect_zero_terms_step(),
])
history_3 = derivation_3.apply(State(expr_3))

print()
print("=" * 70)
print("derivation_3 : ε_{ijk}ε_{ijk} = trace(ε:ε)")
print("=" * 70)
print(show(history_3))
print("Result:", show_final(history_3))


# ===========================================================================
# LaTeX output
# ===========================================================================

def _state_lines(history):
    """Yield LaTeX display-equation lines for each state in history."""
    for state in history:
        label = state.label or "initial"
        safe_label = label.replace("_", r"\_").replace("^", r"\^{}")
        yield r"\[" + r"\text{[" + safe_label + r":] }" + state.expr.latex() + r"\]"


def build_latex_document(sections):
    """Build a multi-section standalone LaTeX document.

    sections: list of (title, lines) where lines is an iterable of raw LaTeX lines.
    """
    parts = [
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{amsmath,amssymb}",
        r"\begin{document}",
        r"\section*{Levi-Civita symbol products}",
    ]
    for title, lines in sections:
        parts.append(r"\subsection*{" + title + "}")
        parts.extend(lines)
    parts.append(r"\end{document}")
    return "\n".join(parts)


# Derivation 2 result as a LaTeX matrix equation
_mat_rows = r" \\ ".join(
    " & ".join(str(result_2d[k][p]) for p in range(3))
    for k in range(3)
)
_d2_lines = [
    r"\[ \sum_{i,j} \varepsilon_{ijk}\,\varepsilon_{ijp} = "
    r"2\delta_{kp} \]",
    r"\[ \text{Verified numerically: } "
    r"\begin{pmatrix}" + _mat_rows + r"\end{pmatrix} = 2 I \]",
]

tex = build_latex_document([
    (
        r"Abstract formula: "
        r"$\varepsilon_{ijk}\varepsilon_{isp} = "
        r"\delta_{js}\delta_{kp} - \delta_{jp}\delta_{ks}$",
        list(_state_lines(history_1)),
    ),
    (
        r"Two shared indices: "
        r"$\sum_{ij}\varepsilon_{ijk}\varepsilon_{ijp} = 2\delta_{kp}$",
        _d2_lines,
    ),
    (
        r"Full contraction: $\varepsilon_{ijk}\varepsilon_{ijk} = 6$",
        list(_state_lines(history_3)),
    ),
])

out_dir = pathlib.Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)
out = out_dir / "tensor_expansions.tex"
out.write_text(tex)
print(f"\nLaTeX document written to {out}")
print("Compile with: pdflatex -output-directory out out/tensor_expansions.tex")
