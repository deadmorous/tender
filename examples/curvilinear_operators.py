"""Deriving curvilinear ∇ / div / rot / Δ from a coordinate mapping (vibe 000069).

The whole point of vibe 000069: hand a chart its coordinate *mapping* and let the
library derive everything else — the moving frame, the metric, and the
differential operators — with no hand-supplied scale factors or Christoffel
symbols.  This example walks the cylindrical chart from its mapping all the way
to the textbook operators, then shows the same machinery handle spherical.

Cylindrical mapping:  x = r cos θ,  y = r sin θ,  z = z.

  0. The mapping            R = r cos θ · i + r sin θ · j + z · k
  1. Tangent basis          g_i = ∂R/∂q^i                         (M1 + M2)
  2. Metric & scale factors  g_ij,  h_i = √(g_ii)                 (M3 + M4)
  3. Physical frame          e_i = g_i / h_i  (a real tender Basis) (M4)
  4. How the frame turns     ∂_j e_i = Σ_k γ^k_{ij} e_k            (M5)
  5. Operators               ∇f, div v, Δf, rot v                  (M6)
  6. Same machinery, spherical

Writes a standalone LaTeX summary to ``out/``.
"""

import pathlib

import tender
import tender.basis as tb
import tender.chart as tc
import tender.derivation as td

ctx = tender.Context()

# Each (title, [(label, expr_or_text)]) tuple becomes a section of the report.
report: list[tuple[str, list[tuple[str, str]]]] = []


def show(title, rows):
    """Print a section to stdout and record it for the LaTeX report."""
    print(f"\n{title}")
    print("-" * len(title))
    latex_rows = []
    for label, value in rows:
        text = value.latex() if hasattr(value, "latex") else str(value)
        print(f"  {label:30s} {text}")
        latex_rows.append((label, text))
    report.append((title, latex_rows))


def is_zero(e):
    return td.algebraic_eq(e, tender.scalar(0, ctx=ctx))


def field_latex(components, names):
    """A physical vector field Σ_i c_i e_i as LaTeX, skipping zero components."""
    parts = []
    for comp, name in zip(components, names):
        s = comp.latex()
        if is_zero(comp):
            continue
        vec = rf"\mathbf{{e}}_{{{name}}}"
        if s == "1":
            parts.append(vec)
        elif s == "-1":
            parts.append("-" + vec)
        else:
            parts.append(rf"\left({s}\right)\,{vec}")
    return " + ".join(parts).replace("+ -", "-") if parts else "0"


# ---------------------------------------------------------------------------
# 0. The mapping → the position vector R
# ---------------------------------------------------------------------------

# A chart targets an orthonormal Cartesian reference frame (here WCS i, j, k) and
# is given its coordinate variables and the Cartesian components x^a = f^a(q).
# r ≥ 0 is the one bit of domain the simplifier needs (for √(r²) = r).
cart = tb.wcs(ctx)
r = tender.coordinate("r", chart_id=1, slot=0, nonneg=True, ctx=ctx)
th = tender.coordinate(r"\theta", chart_id=1, slot=1, ctx=ctx)
z = tender.coordinate("z", chart_id=1, slot=2, ctx=ctx)

cyl = tc.CoordinateChart(
    cart,
    [r, th, z],
    [r * tender.cos(th), r * tender.sin(th), z],  # x, y, z
)
names = ["r", r"\theta", "z"]

R = cyl.radius_vector()
show(
    "0. The mapping  ->  position vector R",
    [("R = x i + y j + z k", R)],
)

# ---------------------------------------------------------------------------
# 1. Tangent (holonomic) basis  g_i = ∂R/∂q^i
# ---------------------------------------------------------------------------

# Differentiating R coordinate-by-coordinate is the scalar-field differentiation
# engine at work (∂_θ(r cos θ) = −r sin θ, and i, j, k are constant → 0).
g = [cyl.tangent_vector(i) for i in range(3)]
show(
    "1. Tangent basis  g_i = dR/dq^i",
    [("g_r", g[0]), ("g_θ", g[1]), ("g_z", g[2])],
)

# ---------------------------------------------------------------------------
# 2. Metric and scale factors  g_ij,  h_i = √(g_ii)
# ---------------------------------------------------------------------------

# g_θθ = r² needs cos²θ + sin²θ → 1; h_θ = √(r²) → r needs r ≥ 0.  Both folds are
# the targeted scalar simplifier, fired automatically.
g_rr = cyl.metric_component(0, 0)
g_tt = cyl.metric_component(1, 1)
g_rt = cyl.metric_component(0, 1)
h = [cyl.scale_factor(i) for i in range(3)]

show(
    "2. Metric and scale factors",
    [
        ("g_rr", g_rr),
        ("g_θθ", g_tt),
        ("g_rθ  (off-diagonal -> 0)", g_rt),
        ("h_r,  h_θ,  h_z", f"{h[0].latex()},\\quad {h[1].latex()},\\quad {h[2].latex()}"),
    ],
)
assert is_zero(g_rt)
assert td.algebraic_eq(g_tt, r**2)
assert td.algebraic_eq(h[1], r)  # h_θ = r

# ---------------------------------------------------------------------------
# 3. Physical orthonormal frame  e_i = g_i / h_i  (a real tender Basis)
# ---------------------------------------------------------------------------

frame = cyl.physical_frame()
e = [frame.direction(i) for i in range(3)]
show(
    "3. Physical orthonormal frame  e_i = g_i / h_i",
    [
        ("e_r", e[0]),
        ("e_θ", e[1]),
        ("e_z", e[2]),
        ("frame is a Basis of dim", str(frame.dim)),
        ("orthonormal?", str(frame.is_orthonormal)),
    ],
)

# ---------------------------------------------------------------------------
# 4. How the frame turns:  ∂_j e_i = Σ_k γ^k_{ij} e_k
# ---------------------------------------------------------------------------

# The connection (rotation) coefficients re-express each ∂e in the local frame.
# These are what the operators below use; they are derived, not tabulated.
dphi_er = cyl.connection_coefficients(0, 1)  # ∂_θ e_r
dphi_et = cyl.connection_coefficients(1, 1)  # ∂_θ e_θ
show(
    "4. How the frame turns  (∂_θ e_i in the local frame)",
    [
        ("∂_θ e_r", field_latex(dphi_er, names)),  # = e_θ
        ("∂_θ e_θ", field_latex(dphi_et, names)),  # = −e_r
        ("∂_r e_r  (radial -> 0)", field_latex(cyl.connection_coefficients(0, 0), names)),
    ],
)

# ---------------------------------------------------------------------------
# 5. Differential operators  (∇ = Σ_i (1/h_i) e_i ∂_{q^i})
# ---------------------------------------------------------------------------

# Each operator is ∇ applied intrinsically in the chart's own frame (vibe
# 000071) and returns an invariant tensor — the result stays on e_r, e_θ, e_z,
# with no trigonometry and no return to WCS.  The position vector on the frame is
# R = r e_r + z e_z; grad raises the rank by one, so ∇R = Σ_i e_i ⊗ e_i = I.
R_frame = r * e[0] + z * e[2]  # the same R, expressed intrinsically
grad_R = cyl.gradient(R_frame)  # ∇R = I
grad_theta = cyl.gradient(th)  # ∇θ = (1/r) e_θ
grad_r2 = cyl.gradient(r**2)  # ∇r² = 2r e_r

# div, Δ and rot on concrete fields, matching the textbook curvilinear formulas.
v_radial = r * e[0]  # r e_r
v_swirl = r * e[1]  # r e_θ
div_radial = cyl.divergence(v_radial)  # ∇·(r e_r) = 2
lap_r2 = cyl.laplacian(r**2)  # Δ(r²) = 4
rot_swirl = cyl.rot(v_swirl)  # ∇×(r e_θ) = 2 e_z


def simp(expr):
    """Distribute ⊗ over + and simplify, for clean display/comparison."""
    return td.simplify_scalars(td.expand_products(expr))


show(
    "5. Differential operators",
    [
        ("∇R  (rank-2 identity I)", simp(grad_R)),
        ("∇θ", simp(grad_theta)),
        ("∇(r²)", simp(grad_r2)),
        ("∇·(r e_r)", div_radial),
        ("Δ(r²)", lap_r2),
        ("∇×(r e_θ)", simp(rot_swirl)),
    ],
)
# ∇R is the identity tensor I: the intrinsic operator folds the resolution
# Σ_i e_i ⊗ e_i (in the chart's own frame) back to I (vibe 000071).
assert td.structural_eq(grad_R, tender.identity(ctx))
assert td.algebraic_eq(div_radial, tender.scalar(2, ctx=ctx))
assert td.algebraic_eq(lap_r2, tender.scalar(4, ctx=ctx))
assert td.algebraic_eq(simp(rot_swirl), simp(tender.scalar(2, ctx=ctx) * e[2]))

# ---------------------------------------------------------------------------
# 6. The same machinery, spherical:  x = r sinθ cosφ, y = r sinθ sinφ, z = r cosθ
# ---------------------------------------------------------------------------

rs = tender.coordinate("r", chart_id=2, slot=0, nonneg=True, ctx=ctx)
ts = tender.coordinate(r"\theta", chart_id=2, slot=1, ctx=ctx)
ps = tender.coordinate(r"\phi", chart_id=2, slot=2, ctx=ctx)
sph = tc.CoordinateChart(
    tb.wcs(ctx),
    [rs, ts, ps],
    [
        rs * tender.sin(ts) * tender.cos(ps),
        rs * tender.sin(ts) * tender.sin(ps),
        rs * tender.cos(ts),
    ],
)

show(
    "6. Same machinery, spherical",
    [
        ("h_φ = √(g_φφ)", sph.scale_factor(2)),  # = r sin θ
        ("∂_φ e_φ", field_latex(sph.connection_coefficients(2, 2), ["r", r"\theta", r"\phi"])),
        ("Δ(r²)", sph.laplacian(rs**2)),  # = 6
    ],
)
assert td.algebraic_eq(sph.scale_factor(2), rs * tender.sin(ts))
assert td.algebraic_eq(sph.laplacian(rs**2), tender.scalar(6, ctx=ctx))

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
    # LaTeX section titles drop the leading "N. ".
    heading = title.split(". ", 1)[-1]
    sections.append(
        rf"\subsection*{{{heading}}}" + "\n"
        r"\begin{itemize}" + "\n" + items + "\n" + r"\end{itemize}"
    )

doc = (
    r"\documentclass{article}"
    "\n"
    r"\usepackage[utf8]{inputenc}"
    "\n"
    r"\usepackage{amsmath,amssymb}"
    "\n"
    r"\begin{document}"
    "\n\n"
    r"\section*{Curvilinear $\nabla$ from a coordinate mapping}"
    "\n\n"
    "Generated by \\texttt{curvilinear\\_operators.py} (vibe 000069).\n\n"
    + "\n\n".join(sections)
    + "\n\n"
    r"\end{document}"
    "\n"
)

out_dir = pathlib.Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "curvilinear_operators.tex"
out_path.write_text(doc)

print(f"\nWritten : {out_path}")
print(f"Compile : pdflatex -output-directory out {out_path}")
