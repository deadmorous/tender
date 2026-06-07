"""Proof that a·b = b·a from first principles (Phase 13.2 exit criterion).

The argument uses only:
    - bilinearity of the dot product (via expand_step)
    - the basis evaluation rule e_i · e^j = δ_i^j (simplify_basis_dot_step)
    - WCS component expansion (expand_in_basis_step)

No commutativity is assumed; the result follows from the fact that the final
component sum a^1 b_1 + a^2 b_2 + a^3 b_3 is symmetric in a and b.
"""

import pathlib
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'python'))

import tender
from tender import (
    tensor, dot, wcs,
    State, Derivation,
    expand_step, expand_in_basis_step,
    simplify_basis_dot_step,
    show, to_latex_document,
)

# Vectors
a = tensor("a", 1)
b = tensor("b", 1)
cs = wcs

# Starting expression
expr = dot(a, b)

# Derivation steps
steps = [
    # a → a^1 e_1 + a^2 e_2 + a^3 e_3  (expand covariant components)
    expand_in_basis_step(a, cs, covariant=True),
    # b → b_1 e^1 + b_2 e^2 + b_3 e^3  (expand contravariant components)
    expand_in_basis_step(b, cs, covariant=False),
    # distribute · over the sums — produces 9 terms (a^i e_i)·(b_j e^j)
    expand_step(),
    # e_i·e^j → δ_ij and zero terms are dropped automatically
    simplify_basis_dot_step(cs),
]

history = Derivation(steps).apply(State(expr))
print(show(history))

result = history[-1].expr
print("\nFinal result:")
print(" ", result.latex())
print("\nThe expression a^1 b_1 + a^2 b_2 + a^3 b_3 is manifestly symmetric")
print("in the components of a and b, proving a·b = b·a. ∎")

tex = to_latex_document(
    history,
    title="Dot-product commutativity — proof from first principles",
)
out_dir = pathlib.Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)
out = out_dir / "dot_commutativity.tex"
out.write_text(tex)
print(f"\nLaTeX document written to {out}")
print("Compile with: pdflatex -output-directory out out/dot_commutativity.tex")
