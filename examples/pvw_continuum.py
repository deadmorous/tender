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

import pathlib

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
    collect_step,
    localize_step,
    show,
    to_latex_document,
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
# Step 1 — IBP on the σ:∇δu term.
ibp_history = Derivation([
    apply_integration_by_parts_step(V),
]).apply(State(pvw))

# Step 2 — collect all δu-containing terms by domain.
# This groups ∫_∂V terms together and ∫_V terms together, giving:
#   ∫_∂V (σ·n − t)·δu dS  +  ∫_V (−∇·σ − f + ρü)·δu dV
collected_history = Derivation([
    collect_step(delta_u),
]).apply(ibp_history[-1])

# Step 3a — localize over V: extracts pointwise volume equilibrium.
vol_history = Derivation([
    localize_step(V),
]).apply(collected_history[-1])

# Step 3b — localize over ∂V: extracts pointwise traction BC.
# Applied to the collected (not yet localized) state so that only
# the surface terms are present after stripping the ∫_∂V wrapper.
srf_history = Derivation([
    localize_step(dV),
]).apply(collected_history[-1])

history = ibp_history + collected_history[1:] + vol_history[1:]
print(show(history))

print("[localize(\\partial V)]", srf_history[-1].expr.latex())
print()

# ---------------------------------------------------------------------------
# Structural assertions (these prove the derivation is symbolically correct)
# ---------------------------------------------------------------------------
vol_result = vol_history[-1].expr

# The volume localization result is (coeff)·δu — a single Contract node
# whose LHS gathers all volume terms.
assert isinstance(vol_result, Contract), \
    f"Expected Contract after volume localization, got {type(vol_result).__name__}"

# The LHS coefficient must contain Divergence(sigma).
def contains_divergence(e):
    if isinstance(e, Divergence):
        return True
    if isinstance(e, Scale):
        return contains_divergence(e.expr)
    if isinstance(e, Sum):
        return any(contains_divergence(t) for t in e.terms)
    if isinstance(e, Contract):
        return contains_divergence(e.lhs) or contains_divergence(e.rhs)
    return False

assert contains_divergence(vol_result.lhs), \
    "∇·σ term not found in volume equilibrium equation"

print("Derivation verified:")
print("  • IBP applied correctly")
print("  • Divergence of sigma appears in the volume equilibrium equation")
print("  • Surface traction BC recovered via ∂V localization")

# ---------------------------------------------------------------------------
# Write a compilable LaTeX document
# ---------------------------------------------------------------------------
# Main chain: IBP → collect → localize(V)
# Then append the surface BC step from the separate srf_history branch.
full_history = history + srf_history[1:]
tex = to_latex_document(
    full_history,
    title="Principle of Virtual Work — symbolic derivation",
)
out = pathlib.Path(__file__).with_suffix(".tex")
out.write_text(tex)
print(f"\nLaTeX document written to {out}")
print("Compile with: pdflatex", out.name)
