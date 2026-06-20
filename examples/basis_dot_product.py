"""Dot product through a basis: deriving ``a · b = b · a`` from first principles.

Expands two invariant vectors in the World Cartesian System, turns the basis
dot products into Kronecker deltas, contracts them, and shows that ``a · b`` and
``b · a`` reduce to the *same* scalar coordinate contraction ``Σ_i a_i b_i`` —
so the dot product commutes.  Writes a standalone LaTeX derivation to ``out/``.
"""

import pathlib

import tender
import tender.basis as tb
import tender.derivation as td
from tender import IndexNameMap

# ---------------------------------------------------------------------------
# Build the invariant expression and the coordinate frame
# ---------------------------------------------------------------------------

ctx = tender.Context()
frame = tb.wcs(ctx)  # orthonormal frame i, j, k

a = tender.tensor("a", rank=1, ctx=ctx)
b = tender.tensor("b", rank=1, ctx=ctx)


# ---------------------------------------------------------------------------
# Reduce an invariant to its scalar coordinate form in WCS
# ---------------------------------------------------------------------------


def reduce_in_basis(expr):
    """a · b  →  Σ_i a_i b_i, through the basis bridge."""
    expr = tb.expand_in_basis(expr, frame, tb.Variance.Covariant)
    expr = tb.simplify_basis_dot(expr, frame)
    expr = td.canonicalize(expr)
    expr = td.unroll_sums(expr)
    expr = td.eval_delta_concrete(expr)
    expr = td.fold_arithmetic(expr)
    return td.fold_sums(td.canonicalize(expr))


# A shared index-name map keeps dummy indices named consistently across steps.
imap = IndexNameMap()

# Show the stages of a · b as a derivation.
ab = a @ b
stages = [("a \\cdot b", ab)]
ab1 = tb.expand_in_basis(ab, frame, tb.Variance.Covariant)
stages.append(("\\text{expand in basis}", ab1))
ab2 = td.canonicalize(tb.simplify_basis_dot(ab1, frame))
stages.append(("\\text{basis dots} \\to \\delta", ab2))
ab_final = reduce_in_basis(ab)
stages.append(("\\text{contract } \\delta", ab_final))

ba_final = reduce_in_basis(b @ a)

print("a · b reduces through the WCS basis:")
for label, expr in stages:
    print(f"  {label:30s}  {expr.latex(imap)}")
print()
print("a · b  ->", ab_final.latex(imap))
print("b · a  ->", ba_final.latex(imap))
print("algebraic_eq(a·b, b·a):", td.algebraic_eq(ab_final, ba_final))

assert td.algebraic_eq(ab_final, ba_final), "a · b should equal b · a"

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
    r"\section*{Dot product through a basis: $a \cdot b = b \cdot a$}"
    "\n\n"
    "Expanding $a$ and $b$ in the World Cartesian System and contracting the\n"
    "Kronecker deltas reduces $a \\cdot b$ to a scalar coordinate sum:\n"
    r"\begin{align*}"
    "\n" + lines + "\n"
    r"\end{align*}"
    "\n\n"
    "The same reduction applied to $b \\cdot a$ gives\n"
    r"\(" + ba_final.latex(imap) + r"\)"
    ", the identical expression (the coordinates are scalars and commute), so\n"
    r"\[ a \cdot b = b \cdot a. \]"
    "\n\n"
    r"\end{document}"
    "\n"
)

out_dir = pathlib.Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "basis_dot_product.tex"
out_path.write_text(doc)

print(f"\nWritten : {out_path}")
print(f"Compile : pdflatex -output-directory out {out_path}")
