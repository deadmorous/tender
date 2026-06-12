"""
BAC-CAB identity — definition and manual/automatic application
==============================================================

The BAC-CAB rule is the vector triple product identity:

    a × (b × c)  =  b (a·c) − c (a·b)

This example shows how to:
  1. Define a named algebraic identity using pattern variables.
  2. Apply it manually to a concrete expression (explicit mapping).
  3. Apply it automatically via find_matches / apply_identity_auto.

Run:
    source examples/env.sh          # sets PYTHONPATH
    python examples/bac_cab.py
"""

import pathlib

from tender import (
    tensor,
    make_pattern_var,
    cross,
    dot,
    Identity,
    apply_identity,
    find_matches,
    apply_identity_auto,
    doc,
    State,
    Derivation,
    show,
    to_latex_document,
    Contract,
    Sum,
)

# ---------------------------------------------------------------------------
# Define the BAC-CAB identity using pattern variables
# ---------------------------------------------------------------------------
# Pattern variables act as typed placeholders. constrain_rank(1) means the
# variable stands for any rank-1 (vector) expression.

a = make_pattern_var("a"); a.constrain_rank(1)
b = make_pattern_var("b"); b.constrain_rank(1)
c = make_pattern_var("c"); c.constrain_rank(1)

# LHS: a × (b × c)
# cross() is a binary free function — nesting is unambiguous by construction.
lhs = cross(a, cross(b, c))

# RHS: b(a·c) − c(a·b)
# a·c and a·b are scalars (rank 0); multiplying a vector by a scalar via *
# produces a tensor product, which equals scalar scaling here.
rhs = b * dot(a, c) - c * dot(a, b)

bac_cab = Identity("BAC-CAB", lhs, rhs)
print(f"Identity defined: {bac_cab.name}")
print(f"  LHS: {bac_cab.lhs.latex()}")
print(f"  RHS: {bac_cab.rhs.latex()}")
print()

# ---------------------------------------------------------------------------
# Apply the identity to a concrete expression
# ---------------------------------------------------------------------------
u = tensor("\\mathbf{u}", 1)
v = tensor("\\mathbf{v}", 1)
w = tensor("\\mathbf{w}", 1)

# Target: u × (v × w)
expr = cross(u, cross(v, w))

step = apply_identity(bac_cab, {a: u, b: v, c: w})
history = Derivation([step]).apply(State(expr))

print(show(history))

# ---------------------------------------------------------------------------
# Structural check on the result
# ---------------------------------------------------------------------------
result = history[-1].expr

# Result is a Sum of two terms.
assert isinstance(result, Sum), \
    f"Expected Sum, got {type(result).__name__}"
assert len(result.terms) == 2, \
    f"Expected 2 terms, got {len(result.terms)}"

# Both terms must contain contractions (the dot products a·c and a·b).
def has_contract(e):
    if isinstance(e, Contract):
        return True
    if hasattr(e, "expr"):        # Scale
        return has_contract(e.expr)
    if hasattr(e, "lhs"):         # TensorProduct, DoubleContract, …
        return has_contract(e.lhs) or has_contract(e.rhs)
    return False

assert all(has_contract(t) for t in result.terms), \
    "Expected each term to contain a dot product (Contract node)"

print("BAC-CAB identity applied and verified.")

# ---------------------------------------------------------------------------
# Automatic targeting via find_matches / apply_identity_auto
# ---------------------------------------------------------------------------
matches = find_matches(bac_cab, expr)
assert len(matches) == 1, f"Expected exactly 1 match, got {len(matches)}"

auto_step = apply_identity_auto(bac_cab, expr)
auto_history = Derivation([auto_step]).apply(State(expr))
assert auto_history[-1].expr.latex() == history[-1].expr.latex(), \
    "Auto-targeting produced a different result than manual targeting"

print("Auto-targeting verified: same result as manual application.")

# ---------------------------------------------------------------------------
# doc() — render the identity as LaTeX
# ---------------------------------------------------------------------------
print()
print(doc(bac_cab, format="plain"))
print()
print(doc(bac_cab))

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
# BAC-CAB has been promoted from an asserted identity to a proved theorem.
import tender.lib.theorems as _thm
print()
print(f"Library theorem (proved from first principles): {_thm.bac_cab.name}")

# ---------------------------------------------------------------------------
# Write a compilable LaTeX document
# ---------------------------------------------------------------------------
tex = to_latex_document(
    history,
    title="BAC-CAB identity — application",
)
out_dir = pathlib.Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)
out = out_dir / "bac_cab.tex"
out.write_text(tex)
print(f"\nLaTeX document written to {out}")
print("Compile with: pdflatex -output-directory out out/bac_cab.tex")
