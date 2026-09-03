"""The rotation forms of vibe 000110 I5 — constructors that verify."""

import pytest

import tender as t
import tender.derivation as td
import tender.rotation as tr


def _ws():
    ws = t.Workspace()
    return ws, ws.vector("n", unit=True), ws.tensor(r"\theta", rank=0)


def test_reflection_is_stamped_improper():
    ws, n, _ = _ws()
    Q = ws.reflection("Q", n)
    assert ("Q", "orthogonal", False) in ws.ctx.constrained_symbols()
    assert td.prove_equal(Q @ Q.transpose(), ws.identity(), []).proved


def test_turn_tensor_is_stamped_proper():
    ws, n, theta = _ws()
    P = ws.turn("P", n, theta)
    assert ("P", "orthogonal", True) in ws.ctx.constrained_symbols()
    assert td.prove_equal(P @ P.transpose(), ws.identity(), []).proved


def test_a_constructed_rotation_unfolds_to_its_formula():
    ws, n, theta = _ws()
    Q = ws.reflection("Q", n)
    unfolded = td.apply_identity(Q, ws.definition(Q))
    expected = ws.identity() - 2 * (n * n)
    assert td.algebraic_eq(unfolded, expected)


def test_an_abstract_rotation_has_no_definition():
    ws, _, _ = _ws()
    with pytest.raises(ValueError, match="no definition"):
        ws.definition(ws.rotation("S"))


def test_a_form_that_is_not_orthogonal_is_refused_and_not_declared():
    ws, n, _ = _ws()
    A = ws.tensor("A", rank=2)
    with pytest.raises(ValueError, match="not orthogonal"):
        ws.orthogonal_from("R", ws.identity() - A)
    assert not any(s[0] == "R" for s in ws.ctx.constrained_symbols())


def test_a_reflection_needs_its_axis_to_be_a_unit_vector():
    # Without n·n = 1 the form simply is not orthogonal, and the verification
    # says so rather than stamping it anyway.
    ws = t.Workspace()
    m = ws.tensor("m", rank=1)  # not declared unit
    with pytest.raises(ValueError, match="not orthogonal"):
        ws.reflection("Q", m)


def test_verification_reduces_a_known_form_to_the_identity():
    ws, n, _ = _ws()
    Q_form = ws.identity() - 2 * (n * n)
    assert td.algebraic_eq(
        tr.reduce_orthogonality(ws.ctx, Q_form @ Q_form.transpose()),
        ws.identity(),
    )


def test_constructed_rotations_compose_without_being_told():
    ws, n, theta = _ws()
    P = ws.turn("P", n, theta)
    Q = ws.reflection("Q", n)
    result = td.prove_equal(
        (P @ Q) @ ((P @ Q).transpose()),
        ws.identity(),
        td.rules("transpose", "dyadic", ctx=ws.ctx),
    )
    assert result.proved
