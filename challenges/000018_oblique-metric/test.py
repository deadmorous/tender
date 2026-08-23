"""Oblique basis: raising and lowering with the metric g_ij.

In a non-orthonormal basis, a·b = g_ij a^i b^j = a^i b_i — components with
co- and contravariant indices related through the metric and its inverse,
with g_ij g^jk = δ_i^k.
"""

from challenges import harness

CHALLENGE = harness.declare(
    title="a·b = g_ij a^i b^j in an oblique basis",
    tier="B",
    source="Lurie, Theory of Elasticity, tensor-calculus appendix",
)


@harness.level("L1", expected=False, reason="oblique-metric reduction not yet attempted")
def test_dot_product_through_the_metric():
    harness.todo(
        "expand a·b in an oblique basis (make_oblique_basis) and verify the "
        "g_ij contraction, incl. g_ij g^jk = δ_i^k"
    )
