"""The ways to write a rotation — each one *verified*, not asserted.

An abstract `ws.rotation("P")` says that P is orthogonal.  These say which
orthogonal tensor it is:

    Q = ws.reflection("Q", n)         Q = I − 2 n⊗n                 improper
    P = ws.rotation("P", n, θ)        P = n⊗n + (I−n⊗n)cos θ + (n×I)sin θ
    F = ws.frame_rotation("F", e, E)  F = Σ e_i ⊗ E_i

Both come back as *named symbols* carrying the property, with their formula
registered as a defining identity — so the algebra runs on one letter and
unfolds only where the formula is wanted.  That is not just convenience: a
property lives on a leaf, and every one of these forms is a sum.

**The stamp is earned.**  Each constructor reduces `X·Xᵀ` and `Xᵀ·X` and
refuses to name anything that does not reach `I`.  The reduction is directed,
not a saturation — measured, `prove_equal` on the turn tensor exhausts memory
while the same facts applied as directed rewrites close in four rounds
(vibe 000102's "a transformation, not a pattern", once more).

What no verification here can check is the **sign**: `X·Xᵀ = I` holds for a
reflection as much as for a rotation, and tender has no determinant.  `proper=`
is the user's assertion, recorded as one (vibe 000110 I5, Q5).

Five rules arrived with this increment, four of them the axial-vector bridge
that makes a skew tensor `a × I` manipulable at all — which is how vibe 000110
writes an angular velocity, so I6 rests on them directly.
"""

import tender as t
import tender.derivation as td
import tender.rotation as tr

from challenges import harness
from challenges.harness import show

CHALLENGE = harness.declare(
    title="rotation forms: reflection and turn tensor, verified when built",
    tier="A",
    source="Zhilin, angle-free rotation tensors; vibe 000110 I5",
    proves=[
        "skew-transpose",
        "skew-dot",
        "skew-dot-left",
        "skew-product",
        "cross-self",
    ],
)


def _setup():
    ws = t.Workspace()
    n = ws.vector("n", unit=True)
    theta = ws.tensor(r"\theta", rank=0)
    return ws, n, theta, ws.identity()


@harness.level("L1")
def test_the_axial_vector_rules_hold():
    """The four skew-tensor facts, checked by components with no rules.

    A skew tensor is written `a × I` in this project, never as a standalone Ω,
    so these are how an angular velocity is manipulated at all.
    """
    ws, n, theta, I = _setup()
    a, b = ws.tensor("a", rank=1), ws.tensor("b", rank=1)
    claims = [
        ("(a×I)ᵀ = −(a×I)", (a % I).transpose(), -(a % I)),
        ("(a×I)·b = a×b", (a % I) @ b, a % b),
        ("b·(a×I) = b×a", b @ (a % I), b % a),
        ("(a×I)·(b×I) = b⊗a − (a·b)I", (a % I) @ (b % I), b * a - (a @ b) * I),
        ("a×a = 0", a % a, t.scalar(0, ctx=ws.ctx)),
    ]
    for label, lhs, rhs in claims:
        result = td.prove_equal(lhs, rhs, [])
        show(label, f"{result.status}, components_agree={result.components_agree}")
        assert result.components_agree, f"{label} does not hold in components"


@harness.level("L2")
def test_each_rule_proves_its_own_statement():
    ws, n, theta, I = _setup()
    a, b = ws.tensor("a", rank=1), ws.tensor("b", rank=1)
    rules = td.rules("rotation", ctx=ws.ctx) + [td.rule("cross-self", ws.ctx)]
    for lhs, rhs in [
        ((a % I).transpose(), -(a % I)),
        ((a % I) @ b, a % b),
        (b @ (a % I), b % a),
        ((a % I) @ (b % I), b * a - (a @ b) * I),
        (a % a, t.scalar(0, ctx=ws.ctx)),
    ]:
        result = td.prove_equal(lhs, rhs, rules)
        assert result.proved and result.fired, f"{lhs} = {rhs}: {result.status}"


@harness.level("L2")
def test_the_reflection_is_verified_when_built():
    """I − 2 n⊗n is orthogonal, and improper — the second is not checked.

    The proof needs `n·n = 1` (from the unit declaration) and the identity
    tensor on both sides of a dot; nothing about reflections in particular.
    """
    ws, n, theta, I = _setup()
    Q = ws.reflection("Q", n)
    show("Q unfolds to", ws.definition(Q).rhs)
    assert td.prove_equal(Q @ Q.transpose(), I, []).proved
    assert td.prove_equal(Q.transpose() @ Q, I, []).proved
    assert ("Q", "orthogonal", False) in ws.ctx.constrained_symbols()


@harness.level("L2")
def test_the_turn_tensor_is_verified_when_built():
    """n⊗n + (I − n⊗n) cos θ + (n × I) sin θ reduces to I against its transpose.

    The one that exercises the whole increment: the four skew rules, `n·n = 1`,
    `a × a = 0`, and — because it is a *scalar* fact and therefore a step
    rather than a rule — the Pythagorean identity from `simplify_scalars`.
    """
    ws, n, theta, I = _setup()
    P = ws.rotation("P", n, theta)
    show("P unfolds to", ws.definition(P).rhs)
    reduced = tr.reduce_orthogonality(ws.ctx, P_formula(ws, n, theta))
    show("P·Pᵀ reduces to", reduced)
    harness.assert_algebraic_eq(reduced, I, "turn tensor is orthogonal")
    assert ("P", "orthogonal", True) in ws.ctx.constrained_symbols()


def P_formula(ws, n, theta):
    I = ws.identity()
    nn = n * n
    return (nn + (I - nn) * t.cos(theta) + (n % I) * t.sin(theta)) @ (
        nn + (I - nn) * t.cos(theta) + (n % I) * t.sin(theta)
    ).transpose()


@harness.level("L2")
def test_a_form_that_is_not_orthogonal_is_refused():
    """The stamp is earned: a wrong form is refused with its residual.

    Soundness bought by refusing to check would be no better than asserting.
    """
    import pytest

    ws, n, theta, I = _setup()
    A = ws.tensor("A", rank=2)
    with pytest.raises(ValueError) as excinfo:
        ws.orthogonal_from("R", I - A)
    show("refusal", str(excinfo.value)[:70])
    assert "not orthogonal" in str(excinfo.value)
    assert not any(s[0] == "R" for s in ws.ctx.constrained_symbols())


@harness.level("L2")
def test_constructed_rotations_compose():
    """A product of a rotation and a reflection is orthogonal, improper.

    Nothing declares the composite; it follows from the two symbols' own rules
    plus the transpose group (vibe 000110 I4b).  The *sign* is the part no rule
    computes — which is why `proper=` exists at all.
    """
    ws, n, theta, I = _setup()
    Q = ws.reflection("Q", n)
    P = ws.rotation("P", n, theta)
    result = td.prove_equal(
        (P @ Q) @ ((P @ Q).transpose()), I, td.rules("transpose", "dyadic", ctx=ws.ctx)
    )
    show("(P·Q)·(P·Q)ᵀ = I", f"{result.status}, fired={result.fired}")
    assert result.proved


@harness.level("L2")
def test_the_frame_pair_form_is_verified_when_built():
    """P = Σ e_i ⊗ E_i, the tensor carrying one frame onto another.

    This was the increment's red until M3 was fixed.  It reduces to the
    identity *written out on the frame* — `i⊗i + j⊗j + k⊗k` — and two things
    had to be true for that to be recognised as `I`: the indexed `e₁` and the
    value symbol `i` had to be one atom (M3), and the reduction had to be
    allowed to use the frame's own completeness, which is knowledge no identity
    about symbols can carry.

    Its sign is the one the library settles for itself: two frames of the same
    orientation give a rotation, opposite ones a reflection.
    """
    ws, n, theta, I = _setup()
    E = ws.wcs()
    F = ws.frame_rotation("F", E, E)
    show("F unfolds to", ws.definition(F).rhs)
    assert td.prove_equal(F @ F.transpose(), I, []).proved
    assert ("F", "orthogonal", True) in ws.ctx.constrained_symbols()
