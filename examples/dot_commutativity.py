"""Proof that a·b = b·a from first principles (Phase 13.2 / 13.6).

Two proofs are shown:

1. **Indexed-sum proof** (Phase 13.6, Part 1): expand a·b to component form
   a^1 b_1 + a^2 b_2 + a^3 b_3, then collapse to Einstein notation a^i b_i.

2. **Short proof** (Phase 13.6, Part 3): expand both a·b and b·a to component
   form independently, then confirm they are equal via prove_equal_by_components.
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
    collect_repeated_sum_step,
    prove_equal_by_components,
    show, to_latex_document,
)

# Vectors
a = tensor("a", 1)
b = tensor("b", 1)
cs = wcs

# ── Proof 1: indexed-sum notation ────────────────────────────────────────────
steps = [
    # a → a^1 e_1 + a^2 e_2 + a^3 e_3  (covariant components)
    expand_in_basis_step(a, cs, covariant=True),
    # b → b_1 e^1 + b_2 e^2 + b_3 e^3  (contravariant components)
    expand_in_basis_step(b, cs, covariant=False),
    # distribute · over the sums
    expand_step(),
    # e_i·e^j → δ_ij; zero terms dropped automatically
    simplify_basis_dot_step(cs),
    # a^1 b_1 + a^2 b_2 + a^3 b_3  →  a^i b_i  (Einstein notation)
    collect_repeated_sum_step(cs),
]

history = Derivation(steps).apply(State(dot(a, b)))
print("=== Proof 1: indexed-sum notation ===")
print(show(history))

# ── Proof 2: equality via component form ──────────────────────────────────────
common_steps = [
    expand_in_basis_step(a, cs, covariant=True),
    expand_in_basis_step(b, cs, covariant=False),
    expand_step(),
    simplify_basis_dot_step(cs),
]
# For b·a we swap the expansion convention so both sides yield a^k b_k terms:
# b → b_1 e^1 + ...  (covariant=False),  a → a^1 e_1 + ...  (covariant=True)
rhs_steps = [
    expand_in_basis_step(b, cs, covariant=False),
    expand_in_basis_step(a, cs, covariant=True),
    expand_step(),
    simplify_basis_dot_step(cs),
]

lhs_hist, rhs_hist = prove_equal_by_components(
    dot(a, b), dot(b, a), common_steps, rhs_steps
)
print("\n=== Proof 2: a·b = b·a (component equality) ===")
print(f"  a·b  expands to:  {lhs_hist[-1].expr.latex()}")
print(f"  b·a  expands to:  {rhs_hist[-1].expr.latex()}")
print("  Component forms are equal (up to scalar commutativity).  ∎")

# ── LaTeX output ──────────────────────────────────────────────────────────────
tex = to_latex_document(
    history,
    title="Dot-product commutativity — proof from first principles",
)
out_dir = pathlib.Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)
out = out_dir / "dot_commutativity.tex"
out.write_text(tex)
print(f"\nLaTeX document written to {out}")
