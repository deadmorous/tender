"""Building a chart tender does not name: parabolic cylindrical coordinates.

`Workspace` names the standard charts — `cartesian_chart()`,
`cylindrical_chart()`, `spherical_chart()`, `polar_chart()` — and those cover
most work.  This example is the other case: a coordinate system tender has
never heard of, built from nothing but its **embedding**.

That is the whole interface.  You supply the map from your coordinates to the
world Cartesian frame, and everything else is *derived* — tangent basis,
metric, scale factors, the physical orthonormal frame, the connection
coefficients that say how that frame turns, and hence grad / div / rot / Δ.
Nothing is tabulated and nothing is hand-entered, so a chart you invent this
morning has the same standing as cylindrical.

Parabolic cylindrical coordinates (σ, τ, z):

    x = ½(σ² − τ²),    y = σ τ,    z = z

Their level surfaces are confocal parabolic cylinders — the natural
coordinates for a flow past a semi-infinite plate, or a crack tip.  The known
geometry is h_σ = h_τ = √(σ² + τ²), h_z = 1, which the chart should reproduce
without being told.

Run:  python examples/custom_chart.py
"""

import tender as t
import tender.derivation as td

ws = t.Workspace()

# ---------------------------------------------------------------------------
# 1. The coordinates and the embedding — the only thing supplied by hand
# ---------------------------------------------------------------------------

# `coords` mints the coordinate atoms.  `nonneg` matters: it is what licenses
# √(x²) → x, so a coordinate you know to be non-negative should say so or its
# scale factors will not simplify.  Here σ and τ range over all reals, so
# neither is declared non-negative — and we will see the consequence below.
sigma, tau, z = ws.coords(r"\sigma", r"\tau", "z")

parabolic = ws.chart(
    ws.wcs(),
    [sigma, tau, z],
    [
        (sigma**2 - tau**2) / t.scalar(2, ctx=ws.ctx),  # x = ½(σ² − τ²)
        sigma * tau,                                    # y = σ τ
        z,                                              # z = z
    ],
)

print("Parabolic cylindrical coordinates:  x = ½(σ²−τ²),  y = στ,  z = z")
print()

# ---------------------------------------------------------------------------
# 2. Everything else is derived
# ---------------------------------------------------------------------------

print("Derived geometry")
print("----------------")
for i, name in enumerate((r"\sigma", r"\tau", "z")):
    g = parabolic.tangent_vector(i)
    print(f"  g_{name:8s} = {td.simplify_scalars(g).latex()}")
print()

metric = [parabolic.metric_component(i, i) for i in range(3)]
for i, name in enumerate((r"\sigma", r"\tau", "z")):
    print(f"  g_{name}{name} = {td.simplify_scalars(metric[i]).latex()}")

# Off-diagonal metric entries vanish: the coordinates are orthogonal.  That is
# a *result* here, not an assumption — tender computed g_στ = ∂R/∂σ · ∂R/∂τ.
off = td.simplify_scalars(parabolic.metric_component(0, 1))
print(f"  g_στ  = {off.latex()}   (orthogonal — derived, not assumed)")
print()

# The scale factors h_i = √(g_ii).  h_σ = h_τ = √(σ²+τ²): tender leaves the
# square root standing, because σ and τ were not declared non-negative and
# √(σ²+τ²) is not σ+τ.  The known answer is exactly this expression.
for i, name in enumerate((r"\sigma", r"\tau", "z")):
    h = td.simplify_scalars(parabolic.scale_factor(i))
    print(f"  h_{name:8s} = {h.latex()}")
print()

# ---------------------------------------------------------------------------
# 3. …including the differential operators
# ---------------------------------------------------------------------------

# A chart built five lines ago differentiates like any other: the moving frame
# and its connection come from the embedding, so grad/div/rot/Δ are available
# immediately.
f = parabolic.field("f", 0)
grad_f = parabolic.components(parabolic.grad(f))
print("Differential operators on the new chart")
print("---------------------------------------")
for comp, name in zip(grad_f, (r"\sigma", r"\tau", "z")):
    print(f"  (∇f)_{name:8s} = {td.simplify_scalars(comp).latex()}")

# A sanity check with a known answer: ∇z = e_z, since h_z = 1 and z is the
# third coordinate.
grad_z = parabolic.grad(z)
e_z = parabolic.physical_frame().direction(2)
assert td.algebraic_eq(grad_z, e_z), "∇z should be the unit vector e_z"
print()
print("  [assert] ∇z = e_z ✓")

print()

# ---------------------------------------------------------------------------
# 4. A limitation worth knowing before you bring your own chart
# ---------------------------------------------------------------------------
#
# ∇R = I holds in every chart, and tender confirms it for the named ones.
# Here it does *not* come out — and the reason is not the geometry, which is
# correct above, but scalar simplification.  This chart's scale factor is a
# surd, √(σ²+τ²), so the position vector's components arrive as
#
#     R_σ = (σ³ + στ²) / (2√(σ²+τ²))
#
# whose reduction to σ√(σ²+τ²)/2 needs the caller to factor σ(σ²+τ²) and
# cancel (σ²+τ²)/√(σ²+τ²) = √(σ²+τ²).  `simplify_scalars` does not do surd
# cancellation, so the expression stays in an unreduced form and the identity
# cannot be recognised.
#
# The named charts hide this: their scale factors are r, r sinθ, 1 — no surds
# survive.  A chart of your own is quite likely to produce them, so expect to
# meet this, and expect the *geometry* to be right even where the *display*
# is unreduced.
R_sigma = parabolic.components(parabolic.position())[0]
print("Known limitation: surds in the scale factors")
print("--------------------------------------------")
print(f"  R_σ = {td.simplify_scalars(R_sigma).latex()}")
print("  …should reduce to σ√(σ²+τ²)/2; simplify_scalars does not cancel")
print("  (σ²+τ²)/√(σ²+τ²), so ∇R = I is not recognised on this chart.")
print("  The geometry above is nonetheless correct.")
print()
print("Nothing above was tabulated: the embedding is the whole input.")
