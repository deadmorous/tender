"""Elastic energy density: T ·· ε = λ (tr ε)² + 2μ ε ·· ε.

**Constitutive assumption: Hooke's law for a linearly elastic, ISOTROPIC
material** — the stress depends on the strain through just two moduli, the
Lamé constants λ and μ:

    T = λ tr(ε) I + 2μ ε ,        ε symmetric.

Isotropy is what makes the two-constant form legitimate: a general linear
elastic solid needs a rank-4 stiffness `T = C ·· ε` with up to 21 independent
moduli, and the identity below is *false* for such a material.  The
anisotropic case needs the rank-4 route and is not this challenge.

The double contraction T··ε is then twice the strain energy density; in
components,

    T ·· ε  =  λ ε_ii ε_jj + 2μ ε_ij ε_ij .

L1 verifies by expanding both in the frame and reducing the double dots
(the vibe-000091 route).  L2 derives it invariantly — I··ε = tr ε plus
dyadic distributivity — and is currently blocked, see below.
"""

import tender as t
import tender.basis as tb
import tender.derivation as td

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="T··ε = λ(tr ε)² + 2μ ε··ε (energy density)",
    tier="A",
    proves="ddot-identity",
    source="Lurie, Theory of Elasticity §elastic energy; vibe 000091",
)


@harness.level("L1")
def test_reduces_to_the_textbook_quadratic_form():
    ws = t.Workspace()
    ctx = ws.ctx
    lam = t.tensor(r"\lambda", 0, ctx=ctx)
    mu = t.tensor(r"\mu", 0, ctx=ctx)
    I = t.identity(ctx)
    cart, (x, y, z) = ws.cartesian_chart()
    frame = cart.physical_frame()

    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    T = lam * t.tr(eps) * I + 2 * mu * eps  # isotropic Hooke's law

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


@harness.level("L2")
def test_performed_invariantly():
    """T··ε = λ(tr ε)² + 2μ ε··ε, with ε abstract and no frame in sight.

    This was blocked for two milestones, and the blocker was never the
    identity — `A··I = tr A` has been in the library since M2 and fires.
    What failed was *stating* the problem: `T = λ tr(ε) I + 2με` puts a
    ⊗-product inside a double-dot operand, and `distribute_contraction`
    covered `·` and `×` but not the double dots, so the scalars were never
    floated out and canon met a ⊗ it could not encapsulate (vibe 000101).

    With that gap closed the proof is one call — and the L1 test below, which
    has to expand into a frame and reduce the double dots by hand, shows what
    it replaces.
    """
    ws = t.Workspace()
    ctx = ws.ctx
    lam = t.tensor(r"\lambda", 0, ctx=ctx)
    mu = t.tensor(r"\mu", 0, ctx=ctx)
    I = t.identity(ctx)
    eps = ws.field(r"\varepsilon", 2, symmetric=True)
    T = lam * t.tr(eps) * I + 2 * mu * eps  # isotropic Hooke's law

    result = td.prove_equal(
        T // eps,
        lam * t.tr(eps) * t.tr(eps) + 2 * mu * (eps // eps),
        td.rules("double_dot", "dyadic", ctx=ctx),
    )
    show("prove_equal(T··ε, λ(tr ε)² + 2μ ε··ε)", repr(result))
    assert result.proved
    assert result.fired.get("ddot-identity") == 1
