"""Kronecker delta in 3D oblique space.

Constructs δ^i_j (upper index i, lower index j) in the 3D spatial index
space with Oblique realm, then writes a standalone LaTeX document to out/.
"""

import pathlib

import tender
from tender import IndexNameMap, Level, Realm

# ---------------------------------------------------------------------------
# Build the expression
# ---------------------------------------------------------------------------

sp = tender.space_3d
i = tender.alloc_index()
j = tender.alloc_index()

delta = tender.delta(Realm.Oblique, sp, Level.Upper, Level.Lower, i, j)

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

index_map = IndexNameMap()
tex_expr = tender.render_latex(delta, index_map)

print("Expression  :", tex_expr)
print("Name for i  :", index_map.lookup(i))
print("Name for j  :", index_map.lookup(j))

# ---------------------------------------------------------------------------
# Write standalone LaTeX document
# ---------------------------------------------------------------------------

doc = (
    r"\documentclass{article}"
    "\n"
    r"\usepackage[utf8]{inputenc}"
    "\n"
    r"\usepackage{amsmath,amssymb}"
    "\n"
    r"\begin{document}"
    "\n\n"
    r"\section*{Kronecker delta (oblique 3D)}"
    "\n\n"
    "The Kronecker delta with one upper and one lower index in the oblique\n"
    "3D index space:\n"
    r"\["
    "\n"
    "  " + tex_expr + "\n"
    r"\]"
    "\n\n"
    r"\end{document}"
    "\n"
)

out_dir = pathlib.Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "kronecker_delta.tex"
out_path.write_text(doc)

print(f"\nWritten : {out_path}")
print(f"Compile : pdflatex -output-directory out {out_path}")
