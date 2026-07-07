"""Strain compatibility, the operator way (vibe 000077–000078, gap D).

The strain incompatibility of a symmetric strain field ε is

    inc ε = ∇×(∇×ε)ᵀ ,

and compatibility is inc ε = 0.  The point of this example is *how* it is built
and reduced:

  1. With the first-class chart-free ∇ operator (vibe 000078), inc ε is written
     coordinate-free — `∇ × (∇ × ε)ᵀ` — with **ε abstract**.
  2. `expand_nabla` lowers each ∇ to the *free-index* frame form e_i ∂_i, so
     inc ε becomes the interior  Σ_{i,j} e_i × (e_j × ∂_i∂_j ε)ᵀ  — still abstract
     in ε (only second derivatives ∂_i∂_j ε appear, never ε_xy components).  This
     is the "expand ∇ only, keep ε abstract, commit the ∂'s to the left"
     derivation, done natively.
  3. Applying the a×B×c identity to that interior and reassembling the frame-
     indexed ∂'s back into the invariant operators yields the closed identity

         inc ε = −∇∇θ + Δθ·I − (∇∇··ε)·I − Δε + 2(∇∇·ε)ˢ ,   θ = tr ε.

This script builds (1)–(2) natively, states the closed identity (3), and then
*verifies* it against inc ε component-by-component — in a Cartesian frame and,
the curvilinear endpoint, in a cylindrical frame — using tender's differential
operators.  Both sides are coordinate-free tensors, so the identity holds in
every frame; tender confirms it in both.

Run:  python examples/strain_compatibility.py
"""

import tender as t
import tender.derivation as td


def _is_zero(chart, e):
    """True if the scalar field `e` reduces to 0 on the chart (basis-expand it,
    then run the curvilinear scalar simplifier to a fixed point)."""
    return td.simplify_scalars(td.canonicalize(chart.expand(e))).latex() == "0"


def closed_identity_matrix(chart, eps):
    """The RHS −∇∇θ + Δθ·I − (∇∇··ε)·I − Δε + 2(∇∇·ε)ˢ as a 3×3 component matrix
    on the chart's physical frame, assembled from the invariant operators.  The
    symmetric part 2(∇∇·ε)ˢ = ∇∇·ε + (∇∇·ε)ᵀ is taken at the matrix level."""
    theta = t.tr(eps)
    gg = chart.components(chart.grad(chart.grad(theta)))   # ∇∇θ
    de = chart.components(chart.div(chart.grad(eps)))       # Δε = ∇·∇ε
    gd = chart.components(chart.grad(chart.div(eps)))       # ∇∇·ε = ∇(∇·ε)
    lap = chart.laplacian(theta)                            # Δθ
    dd = chart.div(chart.div(eps))                          # ∇∇··ε
    out = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            r = (gg[i][j] * (-1)) + (de[i][j] * (-1)) + gd[i][j] + gd[j][i]
            if i == j:
                r = r + lap - dd
            out[i][j] = r
    return out


def verify(chart, eps, label):
    inc = chart.components(chart.rot(chart.rot(eps).transpose()))  # inc ε
    rhs = closed_identity_matrix(chart, eps)
    ok = all(
        _is_zero(chart, chart.expand(inc[i][j]) - chart.expand(rhs[i][j]))
        for i in range(3)
        for j in range(3)
    )
    status = "✓ all 9 components equal" if ok else "✗ MISMATCH"
    print(f"   {label}: inc ε == −∇∇θ + Δθ·I − (∇∇··ε)·I − Δε + 2(∇∇·ε)ˢ   {status}")
    assert ok, f"closed identity failed in {label}"


def main():
    ws = t.Workspace()
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    nabla = t.nabla(ctx=ws.ctx)

    # ---- 1. chart-free, ε abstract ------------------------------------------
    inc = nabla % (nabla % eps).transpose()
    print("1. Coordinate-free:  inc ε =", inc.latex())
    print("   ε components present?", "varepsilon_{" in inc.latex(), "(abstract)")

    # ---- 2. expand ∇ only, keep ε abstract ----------------------------------
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    interior = cart.expand_nabla(inc)
    print("\n2. expand_nabla → free-index interior  Σ e_i × (e_j × ∂_i∂_j ε)ᵀ")
    print("   ε components present?", "varepsilon_{" in interior.latex(), "(still abstract)")
    print("   second derivatives ∂∂ present?", "partial" in interior.latex())

    # ---- 3. the closed identity, stated -------------------------------------
    print("\n3. a×B×c reduction + reassembly of the ∂-indices gives the closed form")
    print("   inc ε = −∇∇θ + Δθ·I − (∇∇··ε)·I − Δε + 2(∇∇·ε)ˢ ,   θ = tr ε")

    # ---- 4. verify the identity, Cartesian ----------------------------------
    print("\n4. Verification (component-by-component, via the chart operators):")
    verify(cart, eps, "Cartesian ")

    # ---- 5. the curvilinear endpoint: cylindrical ---------------------------
    ws2 = t.Workspace()
    eps2 = ws2.field(r"\varepsilon", 2, symmetric=True)
    r, th, zc = ws2.coords("r", r"\theta", "z", nonneg=("r",))
    cyl = ws2.chart(ws2.wcs(), [r, th, zc],
                    [r * t.cos(th), r * t.sin(th), zc])
    verify(cyl, eps2, "Cylindrical")

    print("\nCompatibility inc ε = 0 is thus the vanishing of the closed-form")
    print("right-hand side — the same tensor equation in any frame.")


if __name__ == "__main__":
    main()
