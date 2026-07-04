"""Equilibrium of a continuous medium in cylindrical coordinates (vibe 000073).

A worked, end-to-end reproduction of the classic hand derivation: take the
balance law ∇·T + f = 0 for a rank-2 stress field T, represent T in the
cylindrical physical frame e_r, e_θ, e_z, and let tender compute the divergence
— components *and* moving basis vectors differentiated together, via the frame
connection ∂_θ e_r = e_θ, ∂_θ e_θ = −e_r.  The three scalar balance equations
fall out and match the standard textbook form.

  0. Cylindrical chart      x = r cosθ,  y = r sinθ,  z = z
  1. Physical frame         e_r, e_θ, e_z  (orthonormal, derived from the map)
  2. Frame connection       ∂_j e_i         (how the frame turns)
  3. Stress in the frame     T = Σ T_ij e_i ⊗ e_j                         (eq 4)
  4. Divergence              ∇·T,  projected onto e_r, e_θ, e_z           (eq 7)
  5. Cross-check             vs the standard cylindrical equilibrium equations
  6. Boiler formula          r-only, no shear  →  ∂_r T_rr + (T_rr−T_θθ)/r + f_r

This is "Route B": T is expanded in the basis explicitly (step 3), then the
operator differentiates that expansion.  Writes a LaTeX summary to ``out/``.
"""

import pathlib

import tender as t
import tender.basis as tb
import tender.derivation as td

ws = t.Workspace()
ctx = ws.ctx

report: list[tuple[str, list[tuple[str, str]]]] = []


def show(title, rows):
    print(f"\n{title}")
    print("-" * len(title))
    latex_rows = []
    for label, value in rows:
        text = value.latex() if hasattr(value, "latex") else str(value)
        print(f"  {label:34s} {text}")
        latex_rows.append((label, text))
    report.append((title, latex_rows))


# ---------------------------------------------------------------------------
# 0. The cylindrical chart
# ---------------------------------------------------------------------------

WCS = ws.wcs()
r, th, z = ws.coords("r", r"\theta", "z", nonneg=("r",))
cyl = ws.chart(WCS, [r, th, z], [r * t.cos(th), r * t.sin(th), z])

# ---------------------------------------------------------------------------
# 1. The physical orthonormal frame e_r, e_θ, e_z
# ---------------------------------------------------------------------------

frame = cyl.physical_frame()
e = [frame.direction(k) for k in range(3)]
names = ["r", r"\theta", "z"]
show(
    "1. Physical frame e_i = g_i / h_i",
    [("e_r", e[0]), ("e_θ", e[1]), ("e_z", e[2]),
     ("orthonormal?", str(frame.is_orthonormal))],
)


def field_latex(components):
    """Σ_i c_i e_i as LaTeX, skipping zero components."""
    parts = []
    for comp, nm in zip(components, names):
        s = td.simplify_scalars(comp).latex()
        if s == "0":
            continue
        vec = rf"\mathbf{{e}}_{{{nm}}}"
        parts.append(vec if s == "1" else rf"\left({s}\right)\,{vec}")
    return " + ".join(parts).replace("+ -", "-") if parts else "0"


# ---------------------------------------------------------------------------
# 2. How the frame turns: the connection ∂_j e_i
# ---------------------------------------------------------------------------

show(
    "2. Frame connection ∂_j e_i (only θ is nonzero)",
    [("∂_θ e_r", field_latex(cyl.connection_coefficients(0, 1))),  # = e_θ
     ("∂_θ e_θ", field_latex(cyl.connection_coefficients(1, 1))),  # = −e_r
     ("∂_r e_r", field_latex(cyl.connection_coefficients(0, 0)))],  # = 0
)

# ---------------------------------------------------------------------------
# 3. The stress field, represented in the cylindrical frame  (eq 4)
# ---------------------------------------------------------------------------

# T is an abstract rank-2 field; expand_in_basis writes it on the frame as
# Σ T_ij e_i ⊗ e_j, minting the physical components T_ij as *fields* of the
# coordinates — so the divergence can differentiate them (vibe 000073).
T = ws.field("T", 2)  # depends on r, θ, z
T_cyl = td.canonicalize(td.unroll_sums(tb.expand_in_basis(T, frame, tb.Variance.Covariant)))
show("3. Stress in the frame  T = Σ T_ij e_i e_j", [("T", T_cyl)])


def reduce_scalar(x):
    """Contract a scalar built from frame dots down to a plain expression."""
    x = tb.simplify_basis_dot(td.expand_products(x), frame)
    return td.simplify_scalars(td.fold_arithmetic(td.eval_delta_concrete(td.canonicalize(x))))


def comp(vec, i):
    """The e_i scalar component of an invariant vector."""
    return reduce_scalar(vec @ e[i])


def Tij(i, j):
    """The physical stress component T_ij = e_i · T · e_j."""
    return reduce_scalar(e[i] @ T_cyl @ e[j])


# ---------------------------------------------------------------------------
# 4. The divergence ∇·T, projected onto the frame  (eq 7)
# ---------------------------------------------------------------------------

div_T = cyl.div(T_cyl)
div_r, div_th, div_z = (comp(div_T, i) for i in range(3))
show(
    "4. ∇·T, per frame component (eq 7)",
    [("(∇·T)_r", div_r), ("(∇·T)_θ", div_th), ("(∇·T)_z", div_z)],
)

# ---------------------------------------------------------------------------
# 5. Cross-check against the standard cylindrical equilibrium equations
# ---------------------------------------------------------------------------

# The textbook divergence of a rank-2 field (contracting the first index):
#   (∇·T)_r = ∂_r T_rr + (1/r)∂_θ T_θr + ∂_z T_zr + (T_rr − T_θθ)/r
#   (∇·T)_θ = ∂_r T_rθ + (1/r)∂_θ T_θθ + ∂_z T_zθ + (T_rθ + T_θr)/r
#   (∇·T)_z = ∂_r T_rz + (1/r)∂_θ T_θz + ∂_z T_zz + T_rz/r
def d(comp_expr, coord):
    return td.simplify_scalars(td.partial(comp_expr, coord))


expect_r = d(Tij(0, 0), r) + d(Tij(1, 0), th) / r + d(Tij(2, 0), z) + (Tij(0, 0) - Tij(1, 1)) / r
expect_th = d(Tij(0, 1), r) + d(Tij(1, 1), th) / r + d(Tij(2, 1), z) + (Tij(0, 1) + Tij(1, 0)) / r
expect_z = d(Tij(0, 2), r) + d(Tij(1, 2), th) / r + d(Tij(2, 2), z) + Tij(0, 2) / r

# div_r combines its terms over the common denominator r, while the textbook
# form is written term-by-term; they are algebraically equal, so compare the
# simplified difference (algebraic_eq alone compares canonical form, which keeps
# the two fraction shapes distinct).
def same(a, b):
    return td.simplify_scalars(a - b).latex() == "0"


assert same(div_r, expect_r), "r-component disagrees with textbook"
assert same(div_th, expect_th), "θ-component disagrees with textbook"
assert same(div_z, expect_z), "z-component disagrees with textbook"
print("\n[assert] ∇·T matches the standard cylindrical equilibrium equations ✓")

# Note the θ-component carries (T_rθ + T_θr)/r.  For a *symmetric* stress
# (T_θr = T_rθ) this is the classic 2 T_rθ / r shear term.

# ---------------------------------------------------------------------------
# 6. The boiler formula: axisymmetric, no shear, radial load only  (eq 9)
# ---------------------------------------------------------------------------

# Assume T and f depend on r only and there is no shear.  The θ- and
# z-equations vanish; the r-equation is the boiler formula.  With only radial
# dependence, ∂_θ T_θr = ∂_z T_zr = 0, so:
boiler = td.simplify_scalars(d(Tij(0, 0), r) + (Tij(0, 0) - Tij(1, 1)) / r)
show(
    "6. Boiler formula (radial equilibrium)",
    [("∂_r T_rr + (T_rr − T_θθ)/r + f_r = 0", boiler)],
)

# ---------------------------------------------------------------------------
# Write a standalone LaTeX summary
# ---------------------------------------------------------------------------

sections = []
for title, rows in report:
    items = "\n".join(
        rf"  \item {label} \quad $\displaystyle {text}$"
        if any(c in text for c in "\\^_{}")
        else rf"  \item {label} \quad \texttt{{{text}}}"
        for label, text in rows
    )
    heading = title.split(". ", 1)[-1]
    sections.append(
        rf"\subsection*{{{heading}}}" + "\n"
        r"\begin{itemize}" + "\n" + items + "\n" + r"\end{itemize}"
    )

doc = (
    r"\documentclass{article}" "\n"
    r"\usepackage[utf8]{inputenc}" "\n"
    r"\usepackage{amsmath,amssymb}" "\n"
    r"\begin{document}" "\n\n"
    r"\section*{Cylindrical equilibrium $\nabla\cdot T + f = 0$}" "\n\n"
    "Generated by \\texttt{cyl\\_equilibrium.py} (vibe 000073).\n\n"
    + "\n\n".join(sections)
    + "\n\n" r"\end{document}" "\n"
)

out_dir = pathlib.Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "cyl_equilibrium.tex"
out_path.write_text(doc)
print(f"\nWritten : {out_path}")
