"""Levi-Civita pair contractions to the generalized Kronecker delta.

    Σ_i  ε^{ijk} ε_{iml}  =  δ^j_m δ^k_l − δ^j_l δ^k_m
    Σ_ij ε^{ijk} ε_{ijl}  =  2 δ^k_l

Performed symbolically by `contract_eps_pair` — one step, no concrete-WCS
unrolling, no "creative step" (vibe 000035).
"""

import tender
import tender.derivation as td
from tender import Level, Realm

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="ε-pairs: Σ_i ε^ijk ε_iml = δδ−δδ, Σ_ij ε^ijk ε_ijl = 2δ",
    tier="A",
    source="Lurie, Theory of Elasticity, tensor-calculus appendix; "
    "examples/eps_delta.py",
    proves=["eps-delta-1", "eps-delta-2"],
)

sp = tender.space_3d


def _eps_pair(ctx, upper_ids, lower_ids):
    up = tender.levi_civita(
        Realm.Oblique, sp, [Level.Upper] * 3, upper_ids, ctx=ctx
    )
    lo = tender.levi_civita(
        Realm.Oblique, sp, [Level.Lower] * 3, lower_ids, ctx=ctx
    )
    return up * lo


@harness.level("L2")
def test_one_index_contraction():
    """Σ_i ε^{ijk} ε_{iml} closes to δ^j_m δ^k_l − δ^j_l δ^k_m in one step."""
    ctx = tender.Context()
    i, j, k, m, l = (ctx.alloc_index() for _ in range(5))
    imap = tender.IndexNameMap()
    for idx, nm in zip((i, j, k, m, l), "ijkml"):
        imap.assign(idx, nm)

    expr = tender.explicit_sum(i, _eps_pair(ctx, [i, j, k], [i, m, l]), ctx=ctx)
    result = td.contract_eps_pair(expr)
    show("Σ_i ε^ijk ε_iml", expr.latex(imap))
    show("contract_eps_pair", result.latex(imap))
    expected = r"\delta^{j}_{m} \, \delta^{k}_{l} - \delta^{j}_{l} \, \delta^{k}_{m}"
    assert result.latex(imap) == expected, result.latex(imap)


@harness.level("L2")
def test_two_index_contraction():
    """Σ_ij ε^{ijk} ε_{ijl} closes to 2 δ^k_l in one step."""
    ctx = tender.Context()
    i, j, k, l = (ctx.alloc_index() for _ in range(4))
    imap = tender.IndexNameMap()
    for idx, nm in zip((i, j, k, l), "ijkl"):
        imap.assign(idx, nm)

    expr = tender.explicit_sum(
        j,
        tender.explicit_sum(i, _eps_pair(ctx, [i, j, k], [i, j, l]), ctx=ctx),
        ctx=ctx,
    )
    result = td.contract_eps_pair(expr)
    show("Σ_ij ε^ijk ε_ijl", expr.latex(imap))
    show("contract_eps_pair", result.latex(imap))
    assert result.latex(imap) == r"2 \, \delta^{k}_{l}", result.latex(imap)
