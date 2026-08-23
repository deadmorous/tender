"""Round-trip: reassemble(expand_in_basis(x)) == x over a battery of shapes.

Expanding an invariant into frame coordinates and reassembling must be the
identity for every expressible shape of rank ≤ 2 (dots, crosses, dyads,
traces, sums with scalar weights).  A systematic battery, not a hand-picked
pair — the acceptance test for the coordinate bridge.
"""

from challenges import harness

CHALLENGE = harness.declare(
    title="reassemble ∘ expand_in_basis = id (shape battery)",
    tier="B",
    source="vibes 000053/000061/000063 (reassembly engine)",
)


@harness.level("L1", expected=False, reason="battery not yet built")
def test_round_trip_battery():
    harness.todo(
        "generate the rank ≤ 2 shape battery and assert the expand→reassemble "
        "round-trip on every shape"
    )
