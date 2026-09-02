"""Lagrange's equation of the plane pendulum, from δ∫L dt = 0.

    L = ½ m l² φ̇² + m g l cos φ      ⟹      m l² φ̈ + m g l sin φ = 0

The payoff challenge of vibe 000110: everything up to and including δL is
reachable today (challenge 000025 certifies it), and the last two moves are
not.  Getting from δS = 0 to the equation needs

  * the **definite integral** ∫_{t₀}^{t₁} … dt over a named domain, and
  * **integration by parts in time** plus the fundamental lemma of the calculus
    of variations: the boundary term dies because δφ vanishes at the ends, and
    ∫ X δφ dt = 0 for arbitrary δφ gives X = 0.

Both moved to **vibe 000111** with the continuum arc, which needs the same node
for cross-section resultants and needs it first.  Enumerated red, not hidden:
this is that vibe's roadmap entry.

Worth noting what this red does *not* block: the same equation is reachable
without any integral, by d'Alembert–Lagrange — virtual work of the active and
inertial forces, concluded by equating the coefficient of an arbitrary δφ
(vibe 000110 I7).  For finitely many degrees of freedom the integral buys
Hamilton's *route*, not the answer.
"""

from challenges import harness

CHALLENGE = harness.declare(
    title="m l² φ̈ + m g l sin φ = 0 from δ∫L dt = 0",
    tier="E",
    source="Gantmacher, Lectures in Analytical Mechanics §2 (pendulum)",
)


@harness.level(
    "L1", expected=False, reason="needs the ∫ node (vibe 000111)"
)
def test_verified():
    harness.todo(
        "the action ∫L dt has no representation yet — vibe 000111 "
        "(definite integral over a named domain)"
    )


@harness.level(
    "L2", expected=False, reason="needs by parts + the fundamental lemma (vibe 000111)"
)
def test_performed():
    harness.todo(
        "δ∫L dt = 0 → Euler–Lagrange needs integration by parts in time and "
        "the fundamental lemma — vibe 000111"
    )
