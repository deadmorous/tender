"""The exact-sequence identities: ∇×(∇f) = 0 and ∇·(∇×u) = 0.

Curl of a gradient and divergence of a curl vanish identically — verified as
one-call operator compositions in both a Cartesian and a cylindrical chart
(the cylindrical case exercises the moving-frame connection).
"""

import tender as t
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="∇×∇f = 0 and ∇·(∇×u) = 0, any chart",
    tier="C",
    source="Borisenko–Tarapov §vector calculus identities",
)


def _charts():
    ws = t.Workspace()
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    ws2 = t.Workspace()
    r, th, zc = ws2.coords("r", r"\theta", "z", nonneg=("r",))
    cyl = ws2.chart(ws2.wcs(), [r, th, zc], [r * t.cos(th), r * t.sin(th), zc])
    return (("Cartesian", ws, cart), ("cylindrical", ws2, cyl))


@harness.level("L2")
def test_curl_of_gradient_vanishes():
    for label, ws, chart in _charts():
        f = ws.field("f", 0)
        result = chart.rot(chart.grad(f))
        show(f"∇×∇f, {label}", result)
        for k, comp in enumerate(chart.components(result)):
            harness.assert_chart_zero(chart, comp, f"(∇×∇f)_{k}, {label}")


@harness.level("L2")
def test_divergence_of_curl_vanishes():
    for label, ws, chart in _charts():
        u = ws.field("u", 1)
        result = chart.div(chart.rot(u))
        show(f"∇·(∇×u), {label}", result)
        harness.assert_chart_zero(chart, result, f"∇·(∇×u), {label}")
