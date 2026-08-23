"""Elastic energy density: T ·· ε = λ (tr ε)² + 2μ ε ·· ε.

For the isotropic Hooke stress T = λ tr(ε) I + 2με of a symmetric strain ε,
the double contraction T··ε is the (doubled) energy density; in components,

    T ·· ε  =  λ ε_ii ε_jj + 2μ ε_ij ε_ij .

L1 verifies by expanding both in the frame and reducing the double dots
(the vibe-000091 route).  L2 (future) derives it invariantly — I··ε = tr ε
plus dyadic distributivity, the M2 engine's material.
"""

import tender as t
import tender.basis as tb
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="T··ε = λ(tr ε)² + 2μ ε··ε (energy density)",
    tier="A",
    source="Lurie, Theory of Elasticity §elastic energy; vibe 000091",
)


@harness.level("L1")
def test_reduces_to_the_textbook_quadratic_form():
    ws = t.Workspace()
    ctx = ws.ctx
    lam = t.tensor(r"\lambda", 0, ctx=ctx)
    mu = t.tensor(r"\mu", 0, ctx=ctx)
    I = t.identity(ctx)
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    frame = cart.physical_frame()

    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    T = lam * t.tr(eps) * I + 2 * mu * eps

    def reduce(e):
        e = tb.expand_in_basis(e, frame, tb.Variance.Covariant)
        e = td.expand_double_dot(e)
        e = tb.simplify_basis_dot(e, frame)
        e = td.canonicalize(e)
        e = tb.simplify_basis_dot(e, frame)  # dots exposed by canonicalize
        e = td.contract_delta(td.canonicalize(e))
        return td.simplify(e)

    lhs = reduce(T // eps)
    rhs = reduce(lam * t.tr(eps) * t.tr(eps) + 2 * mu * (eps // eps))
    show("T··ε reduced", lhs)
    show("λ(tr ε)² + 2μ ε··ε reduced", rhs)
    harness.assert_algebraic_eq(lhs, rhs, "energy density")
    assert r"\varepsilon_{ii} \, \varepsilon_{jj}" in lhs.latex()


@harness.level(
    "L2",
    expected=False,
    reason="canon cannot state T··ε: a ⊗-product inside a double-dot operand "
    "throws 'awaits fence distribution' (vibe 000096 increment 3, open)",
)
def test_performed_invariantly():
    """Blocked on a canonicalization limit, not on a missing identity.

    The rule this needs — `A··I = tr A`, the `double_dot` group — exists and
    fires.  What fails is stating the problem at all: `T = λ tr(ε) I + 2με`
    puts a ⊗-product (`λ ⊗ tr(ε) ⊗ I`) inside the double-dot's operand, and
    `encapsulate` rejects a nested ⊗ there with "awaits fence distribution".
    So `prove_equal` does not return a negative — it raises, which is also a
    verb-surface defect worth fixing in M3: a goal-directed call should never
    surface a canon-internal error message.

    Promote this by teaching canon to distribute that fence; the identity
    side of it is already done.
    """
    ws = t.Workspace()
    ctx = ws.ctx
    lam = t.tensor(r"\lambda", 0, ctx=ctx)
    mu = t.tensor(r"\mu", 0, ctx=ctx)
    I = t.identity(ctx)
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    T = lam * t.tr(eps) * I + 2 * mu * eps

    result = td.prove_equal(
        T // eps,
        lam * t.tr(eps) * t.tr(eps) + 2 * mu * (eps // eps),
        td.rules("double_dot", "dyadic", ctx=ctx),
    )
    assert result.proved
