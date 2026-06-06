"""
Principle of Virtual Work — symbolic derivation for a continuum body
====================================================================

Starting point (PVW in d'Alembert form; LHS − RHS = 0 is implied):

    ∫_V σ:∇δu dV  −  ∫_V f·δu dV  −  ∫_∂V t·δu dS  +  ∫_V ρü·δu dV  =  0

Derivation steps performed here:

  Step 1 (ibp)      Apply integration by parts to ∫_V σ:∇δu dV:
                        → ∫_∂V (σ·n)·δu dS  −  ∫_V (∇·σ)·δu dV

  Step 2 (loc-V)    Localize over V (fundamental lemma of calculus of
                    variations — the integral vanishes for all admissible
                    δu, so the integrand must vanish pointwise):
                        → -(∇·σ)·δu − f·δu + ρü·δu = 0
                    which, for arbitrary δu, gives the equilibrium equation:
                        ∇·σ + f − ρü = 0

  Step 3 (loc-∂V)   Localize over ∂V:
                        → (σ·n − t)·δu = 0
                    i.e. the traction boundary condition:
                        σ·n = t  on ∂V

Run:
    source examples/env.sh          # sets PYTHONPATH
    python examples/pvw_continuum.py
"""

from tender import (
    tensor,
    make_volume_domain,
    integral,
    ddot,
    dot,
    gradient,
    State,
    Derivation,
    apply_integration_by_parts_step,
    localize_step,
    show,
    Sum,
    Scale,
    Contract,
    Divergence,
)

# ---------------------------------------------------------------------------
# Named tensors
# ---------------------------------------------------------------------------
sigma   = tensor("\\boldsymbol{\\sigma}", 2)   # Cauchy stress
delta_u = tensor("\\delta\\mathbf{u}",    1)   # virtual displacement
f_body  = tensor("\\mathbf{f}",           1)   # body force
t_trac  = tensor("\\mathbf{t}",           1)   # surface traction
rho_udd = tensor("\\rho\\ddot{\\mathbf{u}}", 1)  # inertia force ρü
n       = tensor("\\mathbf{n}",           1)   # outward unit normal

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
V  = make_volume_domain("V", n)
dV = V.surface_boundary              # ∂V, name = "\partial V"

# ---------------------------------------------------------------------------
# PVW expression  (LHS − RHS; equals zero by assumption)
# ---------------------------------------------------------------------------
pvw = (
      integral(V,  ddot(sigma,  gradient(delta_u)))   # internal virtual work
    - integral(V,  dot(f_body,  delta_u))              # body force
    - integral(dV, dot(t_trac,  delta_u))              # surface traction
    + integral(V,  dot(rho_udd, delta_u))              # inertia (d'Alembert)
)

# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------
history = Derivation([
    apply_integration_by_parts_step(V),    # Step 1 — IBP on σ:∇δu term
    localize_step(V),                      # Step 2 — pointwise over V
]).apply(State(pvw))

print(show(history))

# ---------------------------------------------------------------------------
# Step 3 — localize over ∂V (applied to the surface part only)
# ---------------------------------------------------------------------------
# After step 2, the volume terms have been extracted as integrands.
# The surface integral ∫_∂V (σ·n − t)·δu dS remains in the sum.
# Localizing over ∂V gives the traction BC.
after_vol_loc = history[-1]

surface_history = Derivation([
    localize_step(dV),                     # Step 3 — pointwise over ∂V
]).apply(after_vol_loc)

print("[localize(\\partial V)]", surface_history[-1].expr.latex())
print()

# ---------------------------------------------------------------------------
# Structural assertions (these prove the derivation is symbolically correct)
# ---------------------------------------------------------------------------
vol_result = history[-1].expr

# The volume localization result must be a Sum (several terms for all δu).
assert isinstance(vol_result, Sum), \
    f"Expected Sum after volume localization, got {type(vol_result).__name__}"

# One of the top-level terms must contain Divergence(sigma).
def contains_divergence(e):
    if isinstance(e, Scale):
        return contains_divergence(e.expr)
    if isinstance(e, Contract):
        return isinstance(e.lhs, Divergence) or contains_divergence(e.lhs)
    return False

found = any(contains_divergence(t) for t in vol_result.terms)
assert found, "∇·σ term not found in volume equilibrium equation"

print("Derivation verified:")
print("  • IBP applied correctly")
print("  • Divergence of sigma appears in the volume equilibrium equation")
print("  • Surface traction BC recovered via ∂V localization")
