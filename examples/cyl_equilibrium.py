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
  3. Stress in the frame     T = Σ T_ij e_i ⊗ e_j  (symmetric)            (eq 4)
  4. Divergence              ∇·T,  per frame component                    (eq 7)
  5. Balance law             ∇·T + f = 0,  three scalar equations         (eq 8)
  6. Axisymmetric, no shear  →  one radial equation                       (eq 9)
  7. Boiler formula          thin pipe under pressure  →  T_θθ ≈ R p / d  (eq 10)

Route A: the abstract symmetric field T goes straight into ``cyl.div``, which
expands it in the frame under the hood and differentiates; ``cyl.components``
surfaces the scalar equations (step 4 is ``cyl.components(cyl.div(T))``).  Step 3
shows the same expansion explicitly for the reader.  Writes a LaTeX summary to
``out/``.
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

# T is an abstract *symmetric* rank-2 stress field.  expand_in_basis writes it
# on the frame as Σ T_ij e_i ⊗ e_j, minting the physical components T_ij as
# *fields* of the coordinates (so ∇ can differentiate them) that inherit the
# symmetry, so T_θr folds to T_rθ (vibe 000073).
T = ws.field("T", 2, symmetric=True)  # T_ij = T_ji, depends on r, θ, z
T_cyl = td.canonicalize(td.unroll_sums(tb.expand_in_basis(T, frame, tb.Variance.Covariant)))
show("3. Stress in the frame  T = Σ T_ij e_i e_j", [("T", T_cyl)])

# ---------------------------------------------------------------------------
# 4. The divergence ∇·T, per frame component  (eq 7)  — Route A
# ---------------------------------------------------------------------------

# Route A: hand the *abstract* field T straight to the operator.  cyl.div
# expands it in the chart's frame under the hood, then differentiates the
# components AND the moving basis vectors (via the connection ∂_θ e_i), returning
# the invariant result.  cyl.components surfaces the three scalar components —
# no manual expand / project / reduce pipeline.
div_r, div_th, div_z = cyl.components(cyl.div(T))
show(
    "4. ∇·T, per frame component (eq 7)",
    [("(∇·T)_r", div_r), ("(∇·T)_θ", div_th), ("(∇·T)_z", div_z)],
)

# Cross-check against the standard cylindrical divergence of a symmetric stress:
#   (∇·T)_r = ∂_r T_rr + (1/r)∂_θ T_rθ + ∂_z T_rz + (T_rr − T_θθ)/r
#   (∇·T)_θ = ∂_r T_rθ + (1/r)∂_θ T_θθ + ∂_z T_θz + 2 T_rθ/r
#   (∇·T)_z = ∂_r T_rz + (1/r)∂_θ T_θz + ∂_z T_zz + T_rz/r
# cyl.components on the rank-2 field is the physical component matrix
# Tc[i][j] = e_i·T·e_j (vibe 000074), symmetry folded (Tc[1][0] *is* T_rθ);
# algebraic_eq sees through fraction shapes, so the check is direct.
Tc = cyl.components(T)
d = td.partial

assert td.algebraic_eq(
    div_r,
    d(Tc[0][0], r) + d(Tc[0][1], th) / r + d(Tc[0][2], z)
    + (Tc[0][0] - Tc[1][1]) / r)
assert td.algebraic_eq(
    div_th,
    d(Tc[0][1], r) + d(Tc[1][1], th) / r + d(Tc[1][2], z) + 2 * Tc[0][1] / r)
assert td.algebraic_eq(
    div_z,
    d(Tc[0][2], r) + d(Tc[1][2], th) / r + d(Tc[2][2], z) + Tc[0][2] / r)
print("\n[assert] ∇·T matches the standard cylindrical equilibrium equations ✓")

# ---------------------------------------------------------------------------
# 5. The balance law  ∇·T + f = 0,  three scalar equations  (eq 8)
# ---------------------------------------------------------------------------

# The volume load is a vector field; cyl.components surfaces its frame components
# f_r, f_θ, f_z.  The balance equations are (∇·T)_i + f_i = 0.
f = ws.field("f", 1)  # body force per unit volume
f_r, f_th, f_z = cyl.components(f)
show(
    "5. Balance law  ∇·T + f = 0  (eq 8)",
    [(f"({lbl}) = 0", td.simplify_scalars(dv + fi))
     for lbl, dv, fi in (("e_r", div_r, f_r),
                         ("e_θ", div_th, f_th),
                         ("e_z", div_z, f_z))],
)

# ---------------------------------------------------------------------------
# 6. Axisymmetric, no shear, radial load only  →  one equation  (eq 9)
# ---------------------------------------------------------------------------

# Assume T and f depend on r only, with no shear (T_rθ = T_θz = T_rz = 0) and
# f_θ = f_z = 0.  The θ- and z-equations vanish; only the radial one survives.
# Rebuild the fields with r-only dependence and read the r-equation.
T_r = ws.field("T", 2, deps=[r], symmetric=True)
f_rr = cyl.components(ws.field("f", 1, deps=[r]))[0]
radial = td.simplify_scalars(cyl.components(cyl.div(T_r))[0] + f_rr)
show(
    "6. Axisymmetric radial equilibrium (eq 9)",
    [("∂_r T_rr + (T_rr − T_θθ)/r + f_r = 0", radial)],
)

# ---------------------------------------------------------------------------
# 7. The boiler formula: a thin pipe under internal pressure  (eq 10)
# ---------------------------------------------------------------------------

# A pipe of inner radius R and thickness d ≪ R, pressure p inside, no body force
# (f_r = 0).  Then eq (9) is ∂_r T_rr + (T_rr − T_θθ)/r = 0.  T_rr runs from −p at
# r = R to 0 at r = R + d, so ∂_r T_rr ≈ p/d ≈ const.  With r/d ≫ 1 the (r/d)·p
# term dominates T_rr ∈ [−p, 0], giving the hoop stress T_θθ ≈ R p / d.
show(
    "7. Boiler formula (eq 10)",
    [("thin pipe, pressure p", r"T_{\theta\theta} \approx \dfrac{R\,p}{d}")],
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
