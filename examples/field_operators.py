"""Field & operator usability showcase (vibe 000070 / 000071).

Demonstrates:

  * the context-bound Workspace facade and coordinate minting (P1/P2, vibe 70);
  * intrinsic differential operators in the chart's own frame — ∇R = I, and the
    second gradient of a radial field with no trigonometry (vibe 71);
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
r, th, z = ws.coords("r", r"\theta", "z", nonneg=["r"])
cyl = ws.chart(WCS, [r, th, z], [r * t.cos(th), r * t.sin(th), z])
I = ws.identity()

# --- vibe 71: intrinsic operators in the chart's own frame ------------------
fb = cyl.physical_frame()  # e_r, e_θ, e_z with a known connection
e_r, e_th, e_z = (fb.direction(i) for i in range(3))
R = r * e_r + z * e_z  # the position vector on the frame

grad_R = cyl.grad(R)  # ∇R = I (Σ_i e_i⊗e_i, in the frame)
f = t.field("f", 0, deps=[r], ctx=ws.ctx)  # a radial field f(r)
grad_f = cyl.grad(f)  # ∇f = f' e_r
grad_grad_f = cyl.grad(grad_f)  # ∇∇f — no trig, all in e_r, e_θ
show(
    "1. Intrinsic operators, in the chart frame (vibe 71)",
    [
        ("∇R", grad_R),
        ("∇f(r)", grad_f),
        ("∇∇f(r)", grad_grad_f),
        ("∇×(r e_θ)", cyl.rot(r * e_th)),  # = 2 e_z
    ],
)
assert td.structural_eq(grad_R, I)
assert td.structural_eq(
    td.canonicalize(cyl.rot(r * e_th)),
    td.canonicalize(t.scalar(2, ctx=ws.ctx) * e_z),
)
# The second gradient stays on e_r, e_θ (no cos/sin, no i, j, k):
assert "cos" not in grad_grad_f.latex() and "mathbf{i}" not in grad_grad_f.latex()

# A Cartesian chart for the abstract-field and first-class-operator sections.
x, y, zc = ws.coords("x", "y", "z")
cart = ws.chart(WCS, [x, y, zc], [x, y, zc])

# --- P7: abstract tensor fields differentiate (no longer constant) ----------
T = cart.field("T", 2)  # a rank-2 field on all coordinates
v = cart.field("v", 1)
fx = t.field("f", 0, deps=[x], ctx=ws.ctx)  # scalar field f(x) only
show(
    "2. Tensor fields (P7)",
    [
        ("div T", cart.div(T)),
        ("div v", cart.div(v)),
        ("∂_x f(x)", td.partial(fx, x)),
        ("∂_y f(x)  (= 0)", td.partial(fx, y)),
    ],
)
assert not td.algebraic_eq(cart.div(T), t.scalar(0, ctx=ws.ctx))
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
cfb = cart.physical_frame()
ex, ey, ez = (cfb.direction(i) for i in range(3))
Rc = x * ex + y * ey + zc * ez  # position vector on the Cartesian frame
vc = t.scalar(2, ctx=ws.ctx) * ex + t.scalar(3, ctx=ws.ctx) * ey + ez
dir_R = (nabla.along(vc) * Rc).evaluate(cart)
show("4. Custom operator v·∇ (P8)", [("(v·∇)R   (v = 2 e_x + 3 e_y + e_z)", dir_R)])
assert td.algebraic_eq(dir_R, vc)

print("\nAll field/operator showcase assertions passed.")
