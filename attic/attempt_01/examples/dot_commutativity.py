"""Proof that a·b = b·a from first principles (Phase 13.6).

Two proofs are shown using Einstein summation — no 9-term expansions.

1. **Abstract-index proof**: expand a·b using symbolic basis vectors
   a^i e_i and b_j e^j, then collapse directly to a^i b_i.

2. **Component equality proof**: expand both a·b and b·a to IndexedSum
   form independently and confirm equality via prove_equal_by_components.
"""

import pathlib
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'python'))

import tender
from tender import (
    tensor, dot, wcs,
    State, Derivation,
    expand_in_basis_step,
    simplify_basis_dot_step,
    contract_kronecker_step,
    prove_equal_by_components,
    show, to_latex_document,
)

# Vectors
a = tensor("a", 1)
b = tensor("b", 1)
cs = wcs

# ── Proof 1: abstract-index notation ─────────────────────────────────────────
steps = [
    # a → a^i e_i  (covariant components, abstract index i)
    expand_in_basis_step(a, cs, covariant=True, abstract=True),
    # b → b_j e^j  (contravariant components, abstract index j)
    expand_in_basis_step(b, cs, covariant=False, abstract=True),
    # (a^i e_i)·(b_j e^j)  →  a^i b_j δ_i^j
    simplify_basis_dot_step(cs),
    # a^i b_j δ_i^j  →  a^i b_i  (contract the Kronecker delta)
    contract_kronecker_step(),
]

history = Derivation(steps).apply(State(dot(a, b)))
print("=== Proof 1: abstract-index notation ===")
print(show(history))

# ── Proof 2: equality via component form ──────────────────────────────────────
lhs_steps = [
    expand_in_basis_step(a, cs, covariant=True, abstract=True),
    expand_in_basis_step(b, cs, covariant=False, abstract=True),
    simplify_basis_dot_step(cs),
    contract_kronecker_step(),
]
rhs_steps = [
    # swap to contravariant b · covariant a so both sides yield b_i a^i → same normal form
    expand_in_basis_step(b, cs, covariant=False, abstract=True),
    expand_in_basis_step(a, cs, covariant=True, abstract=True),
    simplify_basis_dot_step(cs),
    contract_kronecker_step(),
]

lhs_hist, rhs_hist = prove_equal_by_components(
    dot(a, b), dot(b, a), lhs_steps, rhs_steps
)
print("\n=== Proof 2: a·b = b·a (component equality) ===")
print(f"  a·b  expands to:  {lhs_hist[-1].expr.latex()}")
print(f"  b·a  expands to:  {rhs_hist[-1].expr.latex()}")
print("  Component forms are equal (up to Einstein index convention).  ∎")

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
