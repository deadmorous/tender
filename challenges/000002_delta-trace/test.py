"""Kronecker-delta contractions: δ^i_i = 3 and δ^i_j δ^i_j = 3 (3D).

Performed by the three uniform steps — unroll the sum, evaluate δ on concrete
indices, fold the arithmetic — with the full history narrated.
"""

import tender
import tender.derivation as td
from tender import Level, Realm

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="δ-contractions: δ^i_i = 3, δ^i_j δ^i_j = 3",
    tier="A",
    source="Borisenko–Tarapov §Kronecker delta; examples/delta_trace.py",
)

sp = tender.space_3d


def _run(expr, claim):
    drv = td.Derivation(expr)
    drv.step(td.unroll_sums).step(td.eval_delta_concrete).step(td.fold_arithmetic)
    for k, e in enumerate(drv.history):
        show(f"{claim} step {k}", e)
    return drv.current


@harness.level("L2")
def test_trace_of_delta_is_3():
    """δ^i_i (implicit sum over i) folds to the space dimension, 3."""
    ctx = tender.Context()
    i = ctx.alloc_index()
    expr = tender.explicit_sum(
        i, tender.delta(Realm.Oblique, sp, Level.Upper, Level.Lower, i, i), ctx=ctx
    )
    assert _run(expr, "δ^i_i").latex() == "3"


@harness.level("L2")
def test_delta_self_contraction_is_3():
    """δ^i_j δ^i_j (summed over both) also folds to 3."""
    ctx = tender.Context()
    i, j = ctx.alloc_index(), ctx.alloc_index()
    d1 = tender.delta(Realm.Oblique, sp, Level.Upper, Level.Lower, i, j, ctx=ctx)
    d2 = tender.delta(Realm.Oblique, sp, Level.Upper, Level.Lower, i, j, ctx=ctx)
    expr = tender.explicit_sum(i, tender.explicit_sum(j, d1 * d2, ctx=ctx), ctx=ctx)
    assert _run(expr, "δ^i_j δ^i_j").latex() == "3"
