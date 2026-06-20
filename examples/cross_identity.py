"""Deriving the invariant identity  a × I = I × a  for any vector a.

The identity tensor I resolves to the dyad I = Σ_m e_m ⊗ e^m, so a × I is a
vector crossed with a dyad.  We expand in the World Cartesian System and apply
`simplify_basis_cross`, which distributes the cross over the dyad and turns each
basis-vector cross into a Levi-Civita term; `canonicalize` finishes.  Both a × I
and I × a reduce to the *same* coordinate expression, proving the identity.
Writes a standalone LaTeX derivation to out/.
"""

import pathlib

import tender
import tender.basis as tb
import tender.derivation as td
from tender import IndexNameMap

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

ctx = tender.Context()
frame = tb.wcs(ctx)  # orthonormal frame i, j, k
a = tender.tensor("a", rank=1, ctx=ctx)
I = tender.identity(ctx=ctx)


# ---------------------------------------------------------------------------
# Reduce a cross-with-identity expression to its coordinate form
# ---------------------------------------------------------------------------


def reduce(expr):
    """expand → simplify_basis_cross (distributes + ε) → canonicalize."""
    expr = tb.expand_in_basis(expr, frame, tb.Variance.Covariant)
    expr = tb.simplify_basis_cross(expr, frame)
    return td.canonicalize(expr)


imap = IndexNameMap()

# Stage the left-hand side as a small derivation.
stages = [("a \\times I", a % I)]
s1 = tb.expand_in_basis(a % I, frame, tb.Variance.Covariant)
stages.append(("\\text{expand } I = \\textstyle\\sum_m e_m \\otimes e^m", s1))
s2 = td.canonicalize(tb.simplify_basis_cross(s1, frame))
stages.append(
    ("\\text{distribute over the dyad},\\ e_i \\times e_m \\to \\varepsilon e^k", s2)
)

lhs = reduce(a % I)
rhs = reduce(I % a)

print("a × I reduced through the basis:")
for label, expr in stages:
    print(f"  {label:55s}  {expr.latex(imap)}")
print()
print("a × I  ->", lhs.latex(imap))
print("I × a  ->", rhs.latex(imap))
print("algebraic_eq(a×I, I×a):", td.algebraic_eq(lhs, rhs))

assert td.algebraic_eq(lhs, rhs), "a × I should equal I × a"

# ---------------------------------------------------------------------------
# Write a standalone LaTeX derivation document
# ---------------------------------------------------------------------------

lines = " \\\\\n".join(
    f"  {label} &: \\quad {expr.latex(imap)}" for label, expr in stages
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
    r"\section*{A cross-with-identity identity: $a \times I = I \times a$}"
    "\n\n"
    "The identity tensor resolves to the dyad $I = \\sum_m e_m \\otimes e^m$, so\n"
    "$a \\times I$ is a vector crossed with a dyad.  Reducing it in the World\n"
    "Cartesian System:\n"
    r"\begin{align*}"
    "\n" + lines + "\n"
    r"\end{align*}"
    "\n\n"
    "The same reduction applied to $I \\times a$ yields the identical\n"
    r"\(" + rhs.latex(imap) + r"\)"
    ", so\n"
    r"\[ a \times I = I \times a. \]"
    "\n\n"
    r"\end{document}"
    "\n"
)

out_dir = pathlib.Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "cross_identity.tex"
out_path.write_text(doc)

print(f"\nWritten : {out_path}")
print(f"Compile : pdflatex -output-directory out {out_path}")
