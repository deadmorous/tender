"""Field & operator usability showcase (vibe 000070).

Demonstrates the usability layer added in vibe 000070:

  * the context-bound Workspace facade and coordinate minting (P1/P2);
  * ∇R folding to the identity tensor I, and ∇×(R×I) to −2I (P3/P4/P6);
  * abstract tensor *fields* that differentiate instead of vanishing (P7);
  * first-class, composable ∇ and ∂_q operators (P8).

Run:  PYTHONPATH=python python3 examples/field_operators.py
"""

import tender as t
import tender.derivation as td
from tender.operators import nabla, d, laplacian


def show(title, rows):
    print(f"\n{title}")
    for label, expr in rows:
        text = expr.latex() if hasattr(expr, "latex") else str(expr)
        print(f"  {label:30s} {text}")


# --- P1/P2: a terse, context-bound preamble (no ctx threading, no slots) ----
ws = t.Workspace()
WCS = ws.wcs()
x, y, z = ws.coords("x", "y", "z")
cart = ws.chart(WCS, [x, y, z], [x, y, z])
I = ws.identity()
R = cart.radius_vector()

# --- P3/P4/P6: resolution of identity folds, fields with I reduce -----------
grad_R = cart.gradient(R)  # ∇R = I (folded from Σ_k e_k⊗e_k)
rot_RxI = cart.rot(R % I)  # ∇×(R×I) = I − 3I = −2I
show(
    "1. Resolution of identity (P3/P4/P6)",
    [("∇R", grad_R), ("∇×(R×I)", rot_RxI)],
)
assert td.structural_eq(grad_R, I)
assert td.structural_eq(rot_RxI, td.canonicalize(t.scalar(-2, ctx=ws.ctx) * I))

# --- P7: abstract tensor fields differentiate (no longer constant) ----------
T = cart.field("T", 2)  # a rank-2 field on all coordinates
v = cart.field("v", 1)
fx = t.field("f", 0, deps=[x], ctx=ws.ctx)  # scalar field f(x) only
show(
    "2. Tensor fields (P7)",
    [
        ("div T", cart.divergence(T)),
        ("div v", cart.divergence(v)),
        ("∂_x f(x)", td.partial(fx, x)),
        ("∂_y f(x)  (= 0)", td.partial(fx, y)),
    ],
)
assert not td.algebraic_eq(cart.divergence(T), t.scalar(0, ctx=ws.ctx))
assert td.algebraic_eq(td.partial(fx, y), t.scalar(0, ctx=ws.ctx))

# --- P8: first-class, composable ∇ and ∂_q ----------------------------------
g = cart.field("g", 0)
lap_expr = nabla @ (nabla * g)  # symbolic ∇·∇g
show(
    "3. First-class operators (P8)",
    [
        ("∇g            (symbolic)", nabla * g),
        ("∇·∇g          (symbolic)", lap_expr),
        ("Δg            (symbolic)", laplacian(g)),
        ("∇·∇g          (evaluated)", lap_expr.evaluate(cart)),
        ("∂_x g         (built)", (d(x) * g).evaluate(cart)),
    ],
)
# Δ is a citable atom that agrees with ∇·∇ by construction (decision 3).
assert td.structural_eq(laplacian(g).evaluate(cart), lap_expr.evaluate(cart))

# A custom operator built from ∇: the directional derivative v·∇.  (v·∇)R = v.
vc = (
    t.scalar(2, ctx=ws.ctx) * WCS.basis(0)
    + t.scalar(3, ctx=ws.ctx) * WCS.basis(1)
    + WCS.basis(2)
)
dir_R = (nabla.along(vc) * R).evaluate(cart)
show("4. Custom operator v·∇ (P8)", [("(v·∇)R   (v = 2i+3j+k)", dir_R)])
assert td.algebraic_eq(dir_R, vc)

print("\nAll field/operator showcase assertions passed.")
