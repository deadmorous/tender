"""The convective-acceleration identity: (u·∇)u = ∇(u²/2) − u×(∇×u).

The workhorse of fluid mechanics, and ∇(u·v) behind it.
"""

from challenges import harness

CHALLENGE = harness.declare(
    title="(u·∇)u = ∇(u²/2) − u×(∇×u)",
    tier="C",
    source="Kochin, Vector Calculus §applications",
)


@harness.level("L1", expected=False, reason="not yet attempted")
def test_verified_in_components():
    harness.todo(
        "components of (u·∇)u (nabla.along) vs ∇(u·u/2) − u×(∇×u) in a "
        "Cartesian chart"
    )
