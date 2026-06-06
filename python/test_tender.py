"""Phase 12 exit criterion tests for the tender Python bindings."""

import pytest
import tender
from tender import (
    I, eps, t, wcs, nabla,
    tensor, scalar, parameter,
    dot, ddot, cross, tp, trace,
    sin, cos, exp, sqrt,
    deriv, dt,
    grad,
    gradient, divergence, rot,
    make_surface_domain, make_volume_domain, integral,
    State, Derivation,
    simplify_identity_step, expand_step, substitute_step,
    apply_integration_by_parts_step, localize_step, collect_step,
    show,
    Rational,
    Sum, Scale, Contract, DoubleContract, CrossProduct,
    IdentityTensor, LeviCivitaTensor,
)


# ===========================================================================
# Singletons
# ===========================================================================

def test_identity_tensor_importable():
    assert I is not None
    assert I.rank == 2


def test_levi_civita_importable():
    assert eps is not None
    assert eps.rank == 3


def test_wcs_importable():
    assert wcs is not None
    assert wcs.dim == 3


def test_wcs_basis_vectors():
    ei = wcs.i
    ej = wcs.j
    ek = wcs.k
    assert ei is not None and ei.rank == 1
    assert ej is not None and ej.rank == 1
    assert ek is not None and ek.rank == 1


def test_wcs_basis_property():
    bv = wcs.basis_vectors
    assert len(bv) == 3
    assert all(e.rank == 1 for e in bv)


def test_time_param_importable():
    assert t is not None
    assert t.rank == 0


def test_nabla_importable():
    assert nabla is not None


# ===========================================================================
# Expression construction and rank
# ===========================================================================

def test_tensor_rank():
    v = tensor("v", 1)
    assert v.rank == 1


def test_scalar_rank():
    f = scalar("f")
    assert f.rank == 0


def test_tensor_default_rank():
    x = tensor("x")
    assert x.rank == 0


def test_rational_type():
    r = Rational(3, 2)
    assert r.num == 3
    assert r.den == 2
    assert abs(r.to_double() - 1.5) < 1e-12


# ===========================================================================
# Operators
# ===========================================================================

def test_add_rank():
    u = tensor("u", 1)
    v = tensor("v", 1)
    s = u + v
    assert s.rank == 1
    assert isinstance(s, Sum)


def test_sub_rank():
    u = tensor("u", 1)
    v = tensor("v", 1)
    d = u - v
    assert d.rank == 1


def test_neg():
    v = tensor("v", 1)
    neg_v = -v
    assert neg_v.rank == 1
    assert isinstance(neg_v, Scale)


def test_mul_int():
    v = tensor("v", 1)
    s = 3 * v
    assert s.rank == 1
    assert isinstance(s, Scale)


def test_rmul_int():
    v = tensor("v", 1)
    s = v * 2
    assert s.rank == 1


def test_tensor_product_rank():
    u = tensor("u", 1)
    v = tensor("v", 1)
    tp_result = u * v
    assert tp_result.rank == 2


def test_contract_rank():
    u = tensor("u", 1)
    v = tensor("v", 1)
    c = u @ v
    assert c.rank == 0
    assert isinstance(c, Contract)


def test_double_contract_rank():
    A = tensor("A", 2)
    B = tensor("B", 2)
    dc = A // B
    assert dc.rank == 0
    assert isinstance(dc, DoubleContract)


def test_double_contract_reversed_rank():
    A = tensor("A", 2)
    B = tensor("B", 2)
    dcr = A ** B
    assert dcr.rank == 0


def test_cross_product_rank():
    a = tensor("a", 1)
    b = tensor("b", 1)
    cp = a % b
    assert cp.rank == 1
    assert isinstance(cp, CrossProduct)


def test_cross_product_chaining_raises():
    a = tensor("a", 1)
    b = tensor("b", 1)
    c = tensor("c", 1)
    ab = a % b
    with pytest.raises(Exception):
        _ = ab % c


def test_cross_free_function_allows_nesting():
    # cross() is binary and unambiguous — nesting must work for identity patterns
    a = tensor("a", 1)
    b = tensor("b", 1)
    c = tensor("c", 1)
    result = cross(a, cross(b, c))
    assert result.rank == 1


def test_truediv_int():
    v = tensor("v", 1)
    half_v = v / 2
    assert half_v.rank == 1
    assert isinstance(half_v, Scale)


def test_scalar_pow():
    x = scalar("x")
    x2 = x ** 2
    assert x2.rank == 0


# ===========================================================================
# Latex rendering
# ===========================================================================

def test_repr_latex_returns_dollars():
    v = tensor("v", 1)
    r = v._repr_latex_()
    assert r.startswith("$$") and r.endswith("$$")


def test_identity_latex():
    latex = I.latex()
    assert "mathbf" in latex or "I" in latex


def test_levi_civita_latex():
    latex = eps.latex()
    assert "varepsilon" in latex or "epsilon" in latex


def test_wcs_basis_latex():
    assert wcs.i.latex() != ""
    assert wcs.j.latex() != ""
    assert wcs.k.latex() != ""


# ===========================================================================
# Free functions
# ===========================================================================

def test_dot_function():
    u = tensor("u", 1)
    v = tensor("v", 1)
    c = dot(u, v)
    assert c.rank == 0


def test_cross_function():
    a = tensor("a", 1)
    b = tensor("b", 1)
    c = cross(a, b)
    assert c.rank == 1


def test_ddot_function():
    A = tensor("A", 2)
    B = tensor("B", 2)
    d = ddot(A, B)
    assert d.rank == 0


def test_trace_function():
    A = tensor("A", 2)
    tr = trace(A)
    assert tr.rank == 0


def test_sin_cos():
    x = scalar("x")
    s = sin(x)
    assert s.rank == 0
    assert "sin" in s.latex()


def test_exp():
    x = scalar("x")
    e = exp(x)
    assert e.rank == 0


def test_sqrt():
    x = scalar("x")
    r = sqrt(x)
    assert r.rank == 0


# ===========================================================================
# Differentiation
# ===========================================================================

def test_deriv_of_param():
    x = parameter("x")
    dx = deriv(x, x)
    assert dx.rank == 0


def test_dt():
    f = tensor("f", 0)
    dft = dt(f)
    assert dft.rank == 0


def test_deriv_of_independent():
    x = parameter("x")
    y = scalar("y")
    dy_dx = deriv(x, y)
    # y doesn't depend on x → zero
    assert "0" in dy_dx.latex() or dy_dx.latex() == "0"


# ===========================================================================
# Coordinate system gradient
# ===========================================================================

def test_wcs_grad_rank():
    x = wcs.coord(0)
    g = grad(x, wcs)
    # grad of a scalar is rank 1
    assert g.rank == 1


def test_wcs_is_orthonormal():
    assert wcs.is_orthonormal


# ===========================================================================
# Symbolic differential operators
# ===========================================================================

def test_gradient_node_rank():
    v = tensor("v", 1)
    gv = gradient(v)
    assert gv.rank == 2


def test_divergence_node_rank():
    v = tensor("v", 1)
    dv = divergence(v)
    assert dv.rank == 0


def test_rotor_node_rank():
    v = tensor("v", 1)
    rv = rot(v)
    assert rv.rank == 1


def test_divergence_scalar_raises():
    f = scalar("f")
    with pytest.raises(Exception):
        divergence(f)


def test_rotor_scalar_raises():
    f = scalar("f")
    with pytest.raises(Exception):
        rot(f)


def test_nabla_matmul_gives_divergence():
    v = tensor("v", 1)
    dv = nabla @ v
    assert dv.rank == 0


def test_nabla_mul_gives_gradient():
    v = tensor("v", 1)
    gv = nabla * v
    assert gv.rank == 2


def test_nabla_mod_gives_rotor():
    v = tensor("v", 1)
    rv = nabla % v
    assert rv.rank == 1


# ===========================================================================
# Domains and integrals
# ===========================================================================

def test_volume_domain():
    n = tensor("n", 1)
    V = make_volume_domain("V", n)
    assert V.name == "V"
    assert V.outward_normal is n


def test_surface_domain_auto_created():
    n = tensor("n", 1)
    V = make_volume_domain("V", n)
    bdy = V.surface_boundary
    assert bdy is not None
    assert "V" in bdy.name


def test_integral_rank():
    n = tensor("n", 1)
    V = make_volume_domain("V", n)
    f = scalar("f")
    I_expr = integral(V, f)
    assert I_expr.rank == 0


def test_integral_vector_rank():
    n = tensor("n", 1)
    V = make_volume_domain("V", n)
    v = tensor("v", 1)
    I_expr = integral(V, v)
    assert I_expr.rank == 1


# ===========================================================================
# State and Derivation
# ===========================================================================

def test_state_construction():
    f = scalar("f")
    s = State(f)
    assert s.expr is f
    assert s.label == ""


def test_derivation_simplify_identity():
    I_expr = I
    v = tensor("v", 1)
    Iv = I_expr @ v
    s0 = State(Iv)
    history = Derivation([simplify_identity_step()]).apply(s0)
    assert len(history) == 2
    result = history[-1].expr
    assert result is v


def test_derivation_history_labels():
    f = scalar("f")
    s0 = State(f)
    step = substitute_step(f, scalar("g"))
    history = Derivation([step]).apply(s0)
    assert len(history) == 2
    assert history[0].label == ""
    assert "substitute" in history[1].label


def test_derivation_concatenation():
    f = scalar("f")
    s0 = State(f)
    d1 = Derivation([simplify_identity_step()])
    d2 = Derivation([expand_step()])
    d3 = d1 + d2
    history = d3.apply(s0)
    assert len(history) == 3


def test_show_returns_string():
    f = scalar("f")
    s0 = State(f)
    history = Derivation([simplify_identity_step()]).apply(s0)
    text = show(history)
    assert isinstance(text, str)
    assert len(text) > 0


# ===========================================================================
# Integration by parts (mini PVW)
# ===========================================================================

def test_ibp_mini():
    n = tensor("n", 1)
    V = make_volume_domain("V", n)
    dV = V.surface_boundary
    sigma = tensor("sigma", 2)
    delta_u = tensor("delta_u", 1)

    pvw = integral(V, ddot(sigma, gradient(delta_u)))
    s0 = State(pvw)
    step = apply_integration_by_parts_step(V)
    history = Derivation([step]).apply(s0)
    assert len(history) == 2
    result = history[-1].expr
    # Result is a Sum containing both a surface integral and a volume integral
    assert isinstance(result, Sum)
    assert result.rank == 0


def test_localize_strips_integral():
    n = tensor("n", 1)
    V = make_volume_domain("V", n)
    f = scalar("f")
    I_expr = integral(V, f)
    s0 = State(I_expr)
    step = localize_step(V)
    history = Derivation([step]).apply(s0)
    result = history[-1].expr
    assert result is f


# ===========================================================================
# Jupyter repr protocol
# ===========================================================================

def test_state_repr_latex():
    v = tensor("v", 1)
    s = State(v)
    r = s._repr_latex_()
    assert "$$" in r


def test_expr_repr_not_empty():
    v = tensor("v", 1)
    assert repr(v) != ""


# ===========================================================================
# collect_step
# ===========================================================================

def test_collect_step_scalar_raises():
    s = tensor("s", 0)
    with pytest.raises(Exception):
        collect_step(s)


def test_collect_step_groups_pointwise():
    from tender import Contract, Sum as TSum
    A = tensor("A", 1)
    B = tensor("B", 1)
    v = tensor("v", 1)
    expr = dot(A, v) + dot(B, v)
    step = collect_step(v)
    result = Derivation([step]).apply(State(expr))[-1].expr
    assert isinstance(result, Contract)
    assert isinstance(result.lhs, TSum)
    assert len(result.lhs.terms) == 2


def test_collect_step_groups_by_domain():
    from tender import Integral, Contract, Sum as TSum
    n = tensor("n", 1)
    V = make_volume_domain("V", n)
    dV = V.surface_boundary
    A = tensor("A", 1)
    B = tensor("B", 1)
    v = tensor("v", 1)
    expr = integral(V, dot(A, v)) + integral(V, dot(B, v))
    step = collect_step(v)
    result = Derivation([step]).apply(State(expr))[-1].expr
    assert isinstance(result, Integral)
    assert isinstance(result.integrand, Contract)
    assert isinstance(result.integrand.lhs, TSum)


def test_collect_step_separates_domains():
    from tender import Sum as TSum
    n = tensor("n", 1)
    V = make_volume_domain("V", n)
    dV = V.surface_boundary
    A = tensor("A", 1)
    B = tensor("B", 1)
    v = tensor("v", 1)
    expr = integral(V, dot(A, v)) + integral(dV, dot(B, v))
    step = collect_step(v)
    result = Derivation([step]).apply(State(expr))[-1].expr
    assert isinstance(result, TSum)
    assert len(result.terms) == 2


def test_collect_step_name():
    v = tensor("v", 1)
    step = collect_step(v)
    assert step.name == "collect(v)"


def test_collect_then_localize_pvw_pattern():
    """IBP → collect(δu) → localize(V) gives clean volume equilibrium."""
    from tender import Integral, Contract
    n  = tensor("n", 1)
    V  = make_volume_domain("V", n)
    dV = V.surface_boundary
    f  = tensor("f", 1)
    g  = tensor("g", 1)
    du = tensor("du", 1)
    # Simulate IBP result: ∫_∂V f·du dS + ∫_V g·du dV
    expr = integral(dV, dot(f, du)) + integral(V, dot(g, du))
    history = Derivation([
        collect_step(du),
        localize_step(V),
    ]).apply(State(expr))
    result = history[-1].expr
    # After localize(V), surface integral remains and volume term is pointwise
    from tender import Sum as TSum
    assert isinstance(result, TSum)
    terms = result.terms
    assert any(isinstance(t, Integral) for t in terms)
    assert any(isinstance(t, Contract) for t in terms)
