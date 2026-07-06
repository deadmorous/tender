"""Strain compatibility, the operator way (vibe 000077, gap D).

The strain incompatibility of a symmetric strain field ε is

    inc ε = ∇×(∇×ε)ᵀ ,

and compatibility is inc ε = 0.  The point of this example is *how* it is built:
with the first-class ∇ operator (vibe 000077), ∇ expands over the frame while
**ε stays abstract** — no component explosion.  This is the "expand ∇ only, keep
ε abstract, commit the ∂'s to the left" derivation, done natively.

Run:  python examples/strain_compatibility.py
"""

import tender as t
import tender.basis as tb
import tender.chart as tc
import tender.derivation as td


def main():
    ctx = t.Context()
    ref = tb.wcs(ctx)
    x = t.coordinate("x", chart_id=7, slot=0, ctx=ctx)
    y = t.coordinate("y", chart_id=7, slot=1, ctx=ctx)
    z = t.coordinate("z", chart_id=7, slot=2, ctx=ctx)
    cart = tc.CoordinateChart(ref, [x, y, z], [x, y, z])
    eps = t.field(r"\varepsilon", 2, symmetric=True, ctx=ctx)

    # The first-class ∇ = Σ_i e_i ∂_{q^i} — one composable, inspectable operator.
    nabla = cart.nabla()
    print("∇ =", nabla.latex())

    # ∇×ε: ∇ expands to i∂_x + j∂_y + k∂_z; ε stays a symbol (∂_a ε), NOT
    # exploded into components.
    curl = td.apply_operators(nabla % eps)
    print("\n∇×ε =", curl.latex())

    # inc ε = ∇×(∇×ε)ᵀ — still abstract: only second derivatives ∂_a∂_b ε appear.
    inc = td.apply_operators(nabla % curl.transpose())
    print("\ninc ε = ∇×(∇×ε)ᵀ has the shape Σ_{a,b} u_a × (u_b × ∂_a∂_b ε)ᵀ,")
    print("        with ε abstract throughout (no ε_xy components):")
    print("   ε components present?", "varepsilon_{" in inc.latex())

    # Sanity: the operator ∇ reproduces the classical operators.  Applied with
    # the matching product and expanded into the frame, ∇·v == chart.div(v):
    v = t.field("v", 1, ctx=ctx)
    div_via_operator = cart.expand(td.apply_operators(nabla @ v))
    print("\n∇·v via the operator == chart.div(v):",
          td.algebraic_eq(div_via_operator, cart.div(v)))

    # Next step (the "gap D" reduction): apply the a×B×c identity to the interior
    # u_a × (u_b × ∂_a∂_b ε)ᵀ and reassemble Σ u_a ∂_a … back into the invariant
    # operators ∇∇θ, Δε, (∇∇·ε)ˢ, ∇∇··ε — yielding the closed identity
    #   inc ε = −∇∇θ + Δθ·I − (∇∇··ε)I − Δε + 2(∇∇·ε)ˢ ,   θ = tr ε.


if __name__ == "__main__":
    main()
