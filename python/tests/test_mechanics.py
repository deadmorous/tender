"""tender.mechanics — time, the chain, d/dt and δ (vibe 000110)."""

import pytest

import tender as t
import tender.derivation as td


def test_time_mints_a_chain_with_decorated_names():
    ws = t.Workspace()
    tm = ws.time("t")
    q, qd, qdd = tm.coordinate("q")
    assert str(tm.t) == "t"
    assert (str(q), str(qd), str(qdd)) == ("q", r"\dot{q}", r"\ddot{q}")
    assert str(tm.variation_of(qd)) == r"\delta{\dot{q}}"


def test_chain_closes_one_order_beyond_what_it_returns():
    # d/dt of the last *returned* member is its successor, not a silent zero.
    ws = t.Workspace()
    tm = ws.time()
    q, qd, qdd = tm.coordinate("q", orders=2)
    assert str(td.apply_operators(tm.ddt() * qdd)) == r"\dddot{q}"


def test_greek_base_name_decorates():
    ws = t.Workspace()
    tm = ws.time()
    phi, phid = tm.coordinate(r"\phi", orders=1)
    assert str(phid) == r"\dot{\phi}"
    assert str(tm.variation_of(phid)) == r"\delta{\dot{\phi}}"


def test_operators_see_coordinates_minted_later():
    # The operators are built afresh, so mint first and take them at use.
    ws = t.Workspace()
    tm = ws.time()
    q, qd = tm.coordinate("q", orders=1)
    r, rd = tm.coordinate("r", orders=1)
    assert td.algebraic_eq(td.apply_operators(tm.ddt() * r), rd)
    assert td.algebraic_eq(td.apply_operators(tm.variation() * r),
                           tm.variation_of(r))


def test_several_coordinates_do_not_cross_contaminate():
    ws = t.Workspace()
    tm = ws.time()
    q, qd = tm.coordinate("q", orders=1)
    r, rd = tm.coordinate("r", orders=1)
    L = tm.field("L", 0, deps=[q, r, tm.t])
    dL = td.apply_operators(tm.ddt() * L)
    text = str(dL)
    assert r"\dot{q}" in text and r"\dot{r}" in text
    assert r"\delta" not in text     # variations are not deps of L


def test_field_requires_explicit_dependence():
    ws = t.Workspace()
    tm = ws.time()
    tm.coordinate("q")
    with pytest.raises(ValueError) as ei:
        tm.field("L", 0, deps=None)
    assert "declare its dependence" in str(ei.value)


def test_orders_capped_and_names_unique():
    ws = t.Workspace()
    tm = ws.time()
    tm.coordinate("q")
    with pytest.raises(ValueError):
        tm.coordinate("q")                 # already minted
    with pytest.raises(ValueError):
        tm.coordinate("r", orders=3)       # LaTeX has no fourth dot


def test_variation_of_rejects_a_non_chain_object():
    ws = t.Workspace()
    tm = ws.time()
    q, qd, qdd = tm.coordinate("q")
    with pytest.raises(ValueError):
        tm.variation_of(tm.variation_of(q))   # δδq is not a thing
    with pytest.raises(ValueError):
        tm.variation_of(tm.t)


def test_variation_without_coordinates_refuses():
    ws = t.Workspace()
    tm = ws.time()
    with pytest.raises(ValueError):
        tm.variation()


def test_time_passes_through_an_abstract_nabla():
    # Vibe 000110 I3: time is not a coordinate of space, so ∂ₜ∇ = 0.
    ws = t.Workspace()
    cart, (x, y, z) = ws.cartesian_chart()
    tm = ws.time()
    u = ws.field("u", 1, deps=[x, y, z, tm.t])
    nab = ws.nabla()
    dt = td.deriv(tm.t)
    assert td.algebraic_eq(
        td.apply_operators(dt * (nab * u)),
        nab * td.apply_operators(dt * u),
    )


def test_a_spatial_coordinate_does_not_get_that_licence():
    ws = t.Workspace()
    cyl, (r, th, z) = ws.cylindrical_chart()
    u = ws.field("u", 1)
    with pytest.raises(ValueError):
        td.apply_operators(td.deriv(r) * (ws.nabla() * u))
