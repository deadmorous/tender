"""Lagrange's equation of the plane pendulum, from δ∫L dt = 0.

    L = ½ m l² φ̇² + m g l cos φ      ⟹      m l² φ̈ + m g l sin φ = 0

The payoff challenge of vibe 000110: everything up to and including δL is
reachable today (challenge 000025 certifies it), and the last two moves are
not.  Getting from δS = 0 to the equation needs

  * the **definite integral** ∫_{t₀}^{t₁} … dt over a named domain
    (vibe 000110 I4 — the same node M5B item 1 needs for cross-section
    resultants), and
  * **integration by parts in time** plus the fundamental lemma of the calculus
    of variations (I5): the boundary term dies because δφ vanishes at the ends,
    and ∫ X δφ dt = 0 for arbitrary δφ gives X = 0.

Enumerated red, not hidden: this is the roadmap entry for I4/I5.
"""

from challenges import harness

CHALLENGE = harness.declare(
    title="m l² φ̈ + m g l sin φ = 0 from δ∫L dt = 0",
    tier="E",
    source="Gantmacher, Lectures in Analytical Mechanics §2 (pendulum)",
)


@harness.level(
    "L1", expected=False, reason="needs the ∫ node (vibe 000110 I4)"
)
def test_verified():
    harness.todo(
        "the action ∫L dt has no representation yet — vibe 000110 I4 "
        "(definite integral over a named domain)"
    )


@harness.level(
    "L2", expected=False, reason="needs by parts + the fundamental lemma (I5)"
)
def test_performed():
    harness.todo(
        "δ∫L dt = 0 → Euler–Lagrange needs integration by parts in time and "
        "the fundamental lemma — vibe 000110 I5"
    )
