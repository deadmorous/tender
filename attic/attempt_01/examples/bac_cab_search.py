"""
BAC-CAB via automatic rewrite search
=====================================

`search_apply` finds the preparatory sequence of rewrites needed before a
target identity can be applied, then applies it.

This example shows:
  1. Direct case: u×(v×w) — BAC-CAB already matches, one step.
  2. Swapped case: (v×w)×u — anti-commutativity needed first, two steps.

Run:
    source examples/env.sh          # sets PYTHONPATH
    python examples/bac_cab_search.py
"""

import pathlib

from tender import (
    tensor,
    cross,
    dot,
    tp,
    make_pattern_var,
    Identity,
    State,
    Derivation,
    show,
    search_apply,
    to_latex_document,
)

# Define BAC-CAB as a local Identity for use as a rewrite rule.
# (The library proves this as a formal theorem in tender.lib.theorems.)
_a = make_pattern_var("a"); _a.constrain_rank(1)
_b = make_pattern_var("b"); _b.constrain_rank(1)
_c = make_pattern_var("c"); _c.constrain_rank(1)
bac_cab = Identity("BAC-CAB", cross(_a, cross(_b, _c)), tp(_b, dot(_a, _c)) - tp(_c, dot(_a, _b)))

u = tensor("\\mathbf{u}", 1)
v = tensor("\\mathbf{v}", 1)
w = tensor("\\mathbf{w}", 1)

# ---------------------------------------------------------------------------
# Direct case
# ---------------------------------------------------------------------------
expr_direct = cross(u, cross(v, w))
steps_direct = search_apply(bac_cab, expr_direct)
history_direct = Derivation(steps_direct).apply(State(expr_direct))

print("Direct case:  u × (v × w)")
print(show(history_direct))
assert len(steps_direct) == 1

# ---------------------------------------------------------------------------
# Swapped case — anti-commutativity rearranges the operands first
# ---------------------------------------------------------------------------
expr_swapped = cross(cross(v, w), u)
steps_swapped = search_apply(bac_cab, expr_swapped)
history_swapped = Derivation(steps_swapped).apply(State(expr_swapped))

print("Swapped case: (v × w) × u")
print(show(history_swapped))
assert len(steps_swapped) == 2
assert steps_swapped[0].name == "apply(cross-anticomm)"
assert steps_swapped[1].name == "apply(BAC-CAB)"

print("Both cases verified.")

# ---------------------------------------------------------------------------
# Write a compilable LaTeX document for the swapped-case derivation
# ---------------------------------------------------------------------------
tex = to_latex_document(
    history_swapped,
    title="BAC-CAB via anti-commutativity",
)
out_dir = pathlib.Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)
out = out_dir / "bac_cab_search.tex"
out.write_text(tex)
print(f"\nLaTeX document written to {out}")
print("Compile with: pdflatex -output-directory out out/bac_cab_search.tex")
