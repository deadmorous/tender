"""Certification harness for tender challenges (vibes 000093 M0 / 000094).

A challenge lives in ``challenges/NNNNNN_descriptive-name/`` with a fixed
``test.py`` and a free-form ``meta/``.  Its ``test.py`` declares itself and
marks each test with the certification level it certifies::

    from challenges import harness

    CHALLENGE = harness.declare(
        title="bac-cab: a×(b×c) = b(a·c) − c(a·b)",
        tier="A",
        source="Lurie, Theory of Elasticity, tensor-calculus appendix",
    )

    @harness.level("L1")
    def test_verified():      # endpoint confirmed by component check
        ...

    @harness.level("L2", expected=False, reason="needs the M3 verb surface")
    def test_performed():     # direct-notation derivation, public surface only
        ...

Levels: **L1** = the claim is verified (typically by reduction to components
and ``algebraic_eq``); **L2** = the derivation is *performed* in direct
notation on the documented public surface, with every step motivated — no
trial-and-error plumbing.  A challenge's certified level is the highest level
whose tests all genuinely pass.  A not-yet-reachable level is a strict xfail
(``expected=False``): the red stays enumerated, and an unexpected pass turns
CI red until the marker is removed by an explicit promotion commit.

Human readability is a requirement (vibe 000094): state the claim in the
docstring, keep the body reading as the derivation, and narrate key
expressions with :func:`show` — visible under ``pytest -s`` and in failure
reports.
"""

import pytest

L1 = "L1"
L2 = "L2"
TIERS = {
    "A": "Algebra (chart-free)",
    "B": "Basis / coordinates",
    "C": "Vector calculus (invariant ∇)",
    "D": "Curvilinear",
    "E": "Mechanics endpoints",
}


def declare(*, title, tier, source=None):
    """The challenge's machine-readable metadata (module-level ``CHALLENGE``)."""
    assert tier in TIERS, f"unknown tier {tier!r}"
    return {"title": title, "tier": tier, "source": source}


def level(lvl, *, expected=True, reason=""):
    """Mark a test with the certification level it certifies.

    ``expected=False`` declares the level not yet reachable: the test is a
    *strict* xfail — it must fail today, and starts failing CI the day it
    unexpectedly passes, forcing an explicit promotion commit.
    """
    assert lvl in (L1, L2), f"unknown level {lvl!r}"

    def deco(fn):
        fn = pytest.mark.challenge_level(lvl)(fn)
        if not expected:
            fn = pytest.mark.xfail(
                strict=True, reason=reason or f"{lvl} not reachable yet"
            )(fn)
        return fn

    return deco


def todo(what):
    """Body of a not-yet-attempted test (always under ``expected=False``):
    fails with a message naming the missing work, so the xfail report reads
    as a roadmap entry."""
    pytest.fail(f"not yet attempted: {what}")


# ---------------------------------------------------------------------------
# Narration and assertion helpers
# ---------------------------------------------------------------------------


def show(label, value):
    """Narrate an expression (or any value) to the test output."""
    text = value.latex() if hasattr(value, "latex") else str(value)
    print(f"  {label:44s} {text}")


def assert_algebraic_eq(lhs, rhs, claim=""):
    """Assert algebraic equality, reporting both sides rendered on failure."""
    import tender.derivation as td

    if not td.algebraic_eq(lhs, rhs):
        pytest.fail(
            (f"{claim}: " if claim else "")
            + "expressions are not algebraically equal\n"
            + f"  lhs: {lhs.latex()}\n  rhs: {rhs.latex()}"
        )


def assert_chart_zero(chart, expr, claim=""):
    """Assert a scalar field reduces to 0 on *chart* (component check)."""
    import tender.derivation as td

    reduced = td.simplify_scalars(td.canonicalize(chart.expand(expr)))
    if reduced.latex() != "0":
        pytest.fail(
            (f"{claim}: " if claim else "")
            + f"does not reduce to 0 on the chart; got {reduced.latex()}"
        )


def assert_components_equal(chart, lhs, rhs, claim=""):
    """Assert two rank-1 invariants agree component-by-component on *chart*."""
    for i in range(3):
        li, ri = chart.components(lhs)[i], chart.components(rhs)[i]
        assert_chart_zero(
            chart, chart.expand(li) - chart.expand(ri), f"{claim} [component {i}]"
        )
