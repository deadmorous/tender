"""Equilibrium in spherical coordinates: ∇·T for a symmetric stress field.

The spherical companion of challenge 000007: the three physical components of
∇·T must match the standard textbook equations (with the sinθ connection
terms).
"""

from challenges import harness

CHALLENGE = harness.declare(
    title="∇·T in spherical coordinates = textbook equilibrium equations",
    tier="E",
    source="Lurie, Theory of Elasticity, appendix of coordinate formulas",
)


@harness.level("L1", expected=False, reason="not yet attempted")
def test_divergence_matches_textbook_equations():
    harness.todo(
        "sph.components(sph.div(T)) vs the three textbook spherical "
        "equilibrium equations (transcribe the reference formulas carefully)"
    )
