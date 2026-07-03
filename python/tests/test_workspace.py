"""Tests for the context-bound Workspace facade (vibe 000070 P1/P2/P5)."""

import pytest

import tender as t
import tender.derivation as td


def test_workspace_preamble_is_terse():
    # The whole preamble, with no ctx threading and no restated chart_id/slot.
    ws = t.Workspace()
    WCS = ws.wcs()
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(WCS, [x, y, z], [x, y, z])
    R = cart.radius_vector()
    # grad R = I still folds through the facade-built chart.
    assert td.structural_eq(cart.grad(R), ws.identity())


def test_coords_auto_slots_and_chart_id():
    ws = t.Workspace()
    x, y, z = ws.coords("x", "y", "z")
    # Distinct coordinates differentiate independently: ∂_x x = 1, ∂_x y = 0.
    assert td.algebraic_eq(td.partial(x, x), t.scalar(1, ctx=ws.ctx))
    assert td.algebraic_eq(td.partial(y, x), t.scalar(0, ctx=ws.ctx))
    # Two coord sets get different chart_ids, so they are independent.
    a, b = ws.coords("a", "b")
    assert td.algebraic_eq(td.partial(a, x), t.scalar(0, ctx=ws.ctx))


def test_chart_coords_accessor():
    ws = t.Workspace()
    WCS = ws.wcs()
    coords = ws.coords("x", "y", "z")
    cart = ws.chart(WCS, coords, coords)
    # The chart hands its coordinate atoms back (P1).
    got = cart.coords
    assert len(got) == 3
    assert all(td.structural_eq(a, b) for a, b in zip(got, coords))


def test_nonneg_coords_simplify_scale_factor():
    ws = t.Workspace()
    WCS = ws.wcs()
    r, th, z = ws.coords("r", "\\theta", "z", nonneg=["r"])
    cyl = ws.chart(WCS, [r, th, z], [r * t.cos(th), r * t.sin(th), z])
    # div(r e_r) = 2 needs √(r²) → r, which the nonneg bit on r licenses.
    v = r * cyl.physical_frame().direction(0)
    assert td.algebraic_eq(cyl.div(v), t.scalar(2, ctx=ws.ctx))


def test_field_factory_via_workspace():
    ws = t.Workspace()
    x, y, z = ws.coords("x", "y", "z")
    cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
    T = ws.field("T", 2)
    assert not td.algebraic_eq(cart.div(T), t.scalar(0, ctx=ws.ctx))


def test_multiletter_name_error_suggests_latex():
    # P5: a plain multi-letter coordinate name is rejected with a message that
    # points at the LaTeX form.
    with pytest.raises(ValueError) as ei:
        t.coordinate("phi")
    assert "\\phi" in str(ei.value)
