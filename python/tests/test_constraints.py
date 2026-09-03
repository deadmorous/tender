"""Declared symbol constraints — unit vectors and orthogonal tensors.

Vibe 000110 I4.  The surface is two workspace factories; what they buy is that
the constraint is in force everywhere the symbol appears, and *only* where that
symbol appears.
"""

import pytest

import tender as t
import tender.derivation as td
import tender.identities as ti


def test_declaration_registers_and_stamps():
    ws = t.Workspace()
    P = ws.rotation("P")
    n = ws.vector("n", unit=True)
    plain = ws.vector("v")
    assert ws.ctx.constrained_symbols() == [
        ("P", "orthogonal", True),
        ("n", "unit", True),
    ]
    assert P.rank == 2 and n.rank == 1 and plain.rank == 1


def test_improper_is_recorded_as_such():
    ws = t.Workspace()
    ws.orthogonal("Q", proper=False)
    assert ws.ctx.constrained_symbols() == [("Q", "orthogonal", False)]


def test_minted_rules_are_per_symbol():
    ws = t.Workspace()
    ws.rotation("P")
    ws.vector("n", unit=True)
    assert sorted(r.name for r in ti.constraint_rules(ws.ctx)) == [
        "P-orthogonal",
        "P-orthogonal-T",
        "P-transport",
        "P-transport-reduced",
        "n-unit",
    ]
    assert ti.constraint_rules(t.Workspace().ctx) == []


def test_constraints_are_in_force_without_being_passed():
    ws = t.Workspace()
    P, I = ws.rotation("P"), ws.identity()
    assert td.prove_equal(P @ P.transpose(), I, []).proved
    assert td.prove_equal(P.transpose() @ P, I, []).proved


def test_the_licence_belongs_to_the_symbol_not_the_shape():
    # The test that fails if a minted rule is a schema over any tensor.
    ws = t.Workspace()
    ws.rotation("P")
    A, I = ws.tensor("A", rank=2), ws.identity()
    assert td.prove_equal(A @ A.transpose(), I, []).refuted


def test_a_unit_vector_does_not_make_every_vector_unit():
    ws = t.Workspace()
    ws.vector("n", unit=True)
    c = ws.tensor("c", rank=1)
    assert td.prove_equal(c @ c, t.scalar(1, ctx=ws.ctx), []).refuted


def test_a_conditional_claim_is_undecided_rather_than_refuted():
    # The component expansion cannot represent a quadratic constraint, so it
    # must abstain instead of answering (vibe 000110 M1).
    ws = t.Workspace()
    P = ws.rotation("P")
    a, b = ws.vector("a"), ws.vector("b")
    result = td.prove_equal((P @ a) @ (P @ b), a @ b, [])
    assert not result.refuted
    assert not result.components_agree


def test_declaring_a_second_symbol_keeps_the_first():
    ws = t.Workspace()
    P, Q, I = ws.rotation("P"), ws.rotation("Q"), ws.identity()
    assert td.prove_equal(P @ P.transpose(), I, []).proved
    assert td.prove_equal(Q @ Q.transpose(), I, []).proved


def test_expressions_carry_their_context():
    ws = t.Workspace()
    P = ws.rotation("P")
    assert P.ctx.constrained_symbols() == [("P", "orthogonal", True)]


def test_an_unknown_constraint_kind_is_rejected():
    ws = t.Workspace()
    with pytest.raises(ValueError):
        t.constrained_tensor("Z", 2, "sideways", ctx=ws.ctx)
