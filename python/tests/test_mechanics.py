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


# ---- spin: the angular velocity of a derivation (vibe 000110 I6) --------


def _turning():
    ws = t.Workspace()
    tm = ws.time("t")
    q, qd = tm.coordinate("q", orders=1)
    return ws, tm, q, qd, tm.rotation("P", deps=[q])


def test_a_turning_rotation_is_a_field_and_a_constrained_symbol():
    ws, tm, q, qd, P = _turning()
    assert ("P", "orthogonal", True) in ws.ctx.constrained_symbols()
    # A field: it differentiates.  A rotation: P·Pᵀ is I with no rules passed.
    assert not td.algebraic_eq(
        td.apply_operators(tm.ddt() * P), t.scalar(0, ctx=ws.ctx)
    )
    assert td.prove_equal(P @ P.transpose(), ws.identity(), []).proved


def test_the_spin_is_skew_under_both_derivations():
    ws, tm, q, qd, P = _turning()
    rules = tm.constraint_rules()
    for operator in (tm.ddt(), tm.variation()):
        spin = tm.spin(P, operator)
        assert td.prove_equal(spin.transpose(), -spin, rules).proved


def test_the_two_spins_differ_only_in_the_derivation():
    ws, tm, q, qd, P = _turning()
    virtual = tm.spin(P, tm.variation())
    substituted = virtual.replace_at(virtual.find(name=r"\delta{q}")[0], qd)
    assert td.algebraic_eq(substituted, tm.spin(P))


def test_a_rotation_of_time_alone_has_no_variation():
    ws = t.Workspace()
    tm = ws.time("t")
    tm.coordinate("q", orders=1)
    P = tm.rotation("P")
    assert td.algebraic_eq(
        td.apply_operators(tm.variation() * P), t.scalar(0, ctx=ws.ctx)
    )
    # …and no rule is minted about it: a rule concerning 0 is noise.
    assert not any(r.name.endswith("-delta") for r in tm.constraint_rules())


def test_a_moving_unit_vector_gets_its_differentiated_constraint():
    ws, tm, q, qd, P = _turning()
    n = tm.unit_field("n", deps=[q])
    ndot = td.apply_operators(tm.ddt() * n)
    assert td.prove_equal(
        n @ ndot, t.scalar(0, ctx=ws.ctx), tm.constraint_rules()
    ).proved


def test_the_angular_velocity_is_minus_half_the_vector_invariant():
    ws, tm, q, qd, P = _turning()
    omega = tm.angular_velocity(P)
    expected = t.scalar(t.Rational(-1, 2), ctx=ws.ctx) * tm.spin(P).vec()
    assert td.algebraic_eq(omega, expected)


def test_poisson_is_derived_and_returns_a_citable_rule():
    ws, tm, q, qd, P = _turning()
    rule = tm.poisson(P, tm.variation())
    assert rule.name.endswith("poisson")
    assert td.algebraic_eq(
        rule.lhs, td.apply_operators(tm.variation() * P)
    )
    assert td.algebraic_eq(rule.rhs, tm.angular_velocity(P, tm.variation()) % P)


def test_poisson_carries_a_frame_vector():
    ws = t.Workspace()
    tm = ws.time("t")
    P = tm.rotation("P")
    E = ws.wcs()
    e0 = P @ E.direction(0)
    rules = (
        td.rules("rotation", "transpose", "dyadic", ctx=ws.ctx)
        + tm.constraint_rules()
        + [tm.poisson(P)]
    )
    assert td.prove_equal(
        td.apply_operators(tm.ddt() * e0),
        tm.angular_velocity(P) % e0,
        rules,
    ).proved


def test_poisson_refuses_a_symbol_it_cannot_derive_for():
    ws = t.Workspace()
    tm = ws.time("t")
    # A field that is not declared a rotation: no spin, no skewness, no rule.
    F = tm.field("F", 2, deps=[tm.t])
    with pytest.raises(ValueError):
        tm.poisson(F)


def test_a_rule_about_a_rate_does_not_rewrite_the_symbol_itself():
    # ∂-marks are part of identity: `Ṗ → ω × P` must not fire on a bare P.
    # It did, and then on its own output, without end (vibe 000110 I7).
    ws = t.Workspace()
    tm = ws.time("t")
    P = tm.rotation("P")
    tm.angular_velocity(P, name=r"\omega")
    poisson = tm.poisson(P)
    assert td.structural_eq(
        td.apply_identity(td.canonicalize(P), poisson), td.canonicalize(P)
    )
    assert not td.structural_eq(
        td.apply_identity(
            td.canonicalize(td.apply_operators(tm.ddt() * P)), poisson
        ),
        td.canonicalize(td.apply_operators(tm.ddt() * P)),
    )


def test_composed_rotations_add_their_transported_angular_velocities():
    ws = t.Workspace()
    tm = ws.time("t")
    I = ws.identity()
    P, Q = tm.rotation("P"), tm.rotation("Q")
    wP = tm.angular_velocity(P, name=r"\omega")
    wQ = tm.angular_velocity(Q, name=r"\varpi")
    product = P @ Q
    spin = td.apply_operators(tm.ddt() * product) @ product.transpose()
    reduced = tm.reduce(spin, [tm.poisson(P), tm.poisson(Q)])
    assert td.algebraic_eq(
        reduced, td.canonicalize(td.expand_products((wP + P @ wQ) % I))
    )


def test_the_rigid_body_acceleration_has_its_two_named_terms():
    ws = t.Workspace()
    tm = ws.time("t")
    P = tm.rotation("P")
    omega = tm.angular_velocity(P, name=r"\omega")
    poisson = tm.poisson(P)
    rho = ws.tensor(r"\rho", rank=1)
    rC = tm.field("c", 1, deps=[tm.t])
    ap = td.apply_operators

    v = tm.reduce(ap(tm.ddt() * (rC + P @ rho)), [poisson])
    a = tm.reduce(ap(tm.ddt() * v), [poisson])
    arm = P @ rho
    expected = (
        ap(tm.ddt() * ap(tm.ddt() * rC))
        + ap(tm.ddt() * omega) % arm
        + omega % (omega % arm)
    )
    assert td.algebraic_eq(a, expected)


def test_constraints_are_differentiated_per_independent_variable():
    # Not per operator: d/dt P of a two-coordinate rotation is a *sum*, and a
    # rule about a sum has a multi-term LHS the matcher cannot compile.  Each
    # partial spin is skew on its own (vibe 000110 I8).
    ws = t.Workspace()
    tm = ws.time("t")
    q, qd = tm.coordinate("q", orders=1)
    r, rd = tm.coordinate("r", orders=1)
    tm.rotation("P", deps=[q, r])
    names = sorted(rule.name for rule in tm.constraint_rules())
    assert names == ["P-spin-q", "P-spin-r"]


def test_the_variation_of_the_angular_velocity_with_one_coordinate():
    ws = t.Workspace()
    tm = ws.time("t")
    q, qd = tm.coordinate("q", orders=1)
    P = tm.rotation("P", deps=[q])
    ap = td.apply_operators
    omega = tm.angular_velocity(P)
    virtual = tm.angular_velocity(P, tm.variation())
    assert td.algebraic_eq(
        tm.reduce(omega % virtual, rounds=8), t.scalar(0, ctx=ws.ctx)
    )
    assert td.algebraic_eq(
        tm.reduce(ap(tm.variation() * omega), rounds=12),
        tm.reduce(ap(tm.ddt() * virtual) - omega % virtual, rounds=12),
    )


def _rigid():
    ws = t.Workspace()
    tm = ws.time("t")
    q, qd = tm.coordinate("q", orders=1)
    P = tm.rotation("P", deps=[q])
    return ws, tm, q, P, tm.poisson_rules(P)


def test_poisson_is_stated_per_coordinate_with_a_named_axis():
    ws, tm, q, P, rules = _rigid()
    assert [r.name for r in rules] == ["q-poisson"]
    assert td.structural_eq(rules[0].lhs, td.partial(P, q))
    # `q̂ × P`, with the axis named — the formula would rewrite its own RHS.
    assert r"\hat{q}" in str(rules[0].rhs)


def test_the_virtual_displacement_mirrors_the_velocity():
    ws, tm, q, P, rules = _rigid()
    ap = td.apply_operators
    rho = ws.tensor(r"\rho", rank=1)
    rC = tm.field("c", 1, deps=[q])
    r = rC + P @ rho
    axis = ws.field(r"\hat{q}", 1, deps=[q])
    dq = tm.variation_of(q)
    qdot = ap(tm.ddt() * q)
    assert td.algebraic_eq(
        tm.reduce(ap(tm.variation() * r), rules, rounds=10),
        dq * td.partial(rC, q) + dq * (axis % (P @ rho)),
    )
    assert td.algebraic_eq(
        tm.reduce(ap(tm.ddt() * r), rules, rounds=10),
        qdot * td.partial(rC, q) + qdot * (axis % (P @ rho)),
    )


def test_coefficients_splits_a_virtual_work():
    ws, tm, q, P, rules = _rigid()
    dq = tm.variation_of(q)
    a, b = ws.tensor("a", rank=1), ws.tensor("b", rank=1)
    work = dq * (a @ b) + dq * (b @ a)
    out = tm.coefficients(work)
    assert list(out) == [r"\delta{q}"]
    assert td.algebraic_eq(out[r"\delta{q}"], td.canonicalize(a @ b + b @ a))


def test_coefficients_refuses_what_the_lemma_does_not_cover():
    ws, tm, q, P, rules = _rigid()
    dq = tm.variation_of(q)
    with pytest.raises(ValueError, match="linear"):
        tm.coefficients(dq * dq)
    with pytest.raises(ValueError, match="variations"):
        tm.coefficients(ws.tensor("s", rank=0))
