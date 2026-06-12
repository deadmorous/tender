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
    apply_identity, find_matches, apply_identity_auto, search_apply,
    declare_symmetric, declare_skew_symmetric,
    Identity, make_pattern_var,
    show, doc,
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

# ===========================================================================
# Phase 13 — automatic identity matching and standard library
# ===========================================================================

def test_find_matches_basic():
    a = make_pattern_var("a"); a.constrain_rank(1)
    b = make_pattern_var("b"); b.constrain_rank(1)
    c = make_pattern_var("c"); c.constrain_rank(1)
    u = tensor("u", 1); v = tensor("v", 1); w = tensor("w", 1)
    lhs = cross(a, cross(b, c))
    id_ = Identity("test", lhs, a)
    expr = cross(u, cross(v, w))
    matches = find_matches(id_, expr)
    assert len(matches) == 1
    assert matches[0][a] is u
    assert matches[0][b] is v
    assert matches[0][c] is w


def test_find_matches_no_match():
    a = make_pattern_var("a"); a.constrain_rank(1)
    u = tensor("u", 0)  # scalar — won't match rank-1
    id_ = Identity("test", a, a)
    matches = find_matches(id_, u)
    assert len(matches) == 0


def test_apply_identity_auto_bac_cab():
    a = make_pattern_var("a"); a.constrain_rank(1)
    b = make_pattern_var("b"); b.constrain_rank(1)
    c = make_pattern_var("c"); c.constrain_rank(1)
    bac_cab = Identity("BAC-CAB", cross(a, cross(b, c)), tp(b, dot(a, c)) - tp(c, dot(a, b)))
    u = tensor("u", 1); v = tensor("v", 1); w = tensor("w", 1)
    expr = cross(u, cross(v, w))
    step = apply_identity_auto(bac_cab, expr)
    result = Derivation([step]).apply(State(expr))[-1].expr
    from tender import Sum as TSum
    assert isinstance(result, TSum)
    assert len(result.terms) == 2


def test_apply_identity_auto_no_match_raises():
    a = make_pattern_var("a"); a.constrain_rank(1)
    u = tensor("u", 0)  # scalar
    id_ = Identity("test", a, a)
    with pytest.raises(Exception):
        apply_identity_auto(id_, u)


def test_doc_plain():
    a = make_pattern_var("a"); a.constrain_rank(1)
    id_ = Identity("my-id", a, a)
    result = doc(id_, format="plain")
    assert "my-id" in result


def test_doc_latex():
    a = make_pattern_var("a"); a.constrain_rank(1)
    id_ = Identity("my-id", a, a)
    result = doc(id_, format="latex")
    assert "my-id" in result
    assert r"\[" in result


def test_standard_library_epsilon():
    from tender.lib.identities.epsilon import ALL
    # bac_cab and anti_commutativity were promoted to proved theorems in
    # tender.lib.theorems (Phase 13.9) and removed from the asserted-identities
    # list.
    assert len(ALL) == 0


def test_standard_library_identity_tensor():
    from tender.lib.identities.identity_tensor import (
        identity_dot_vec, identity_double_contract, ALL)
    assert identity_dot_vec.name == "identity-dot-vec"
    assert len(ALL) == 2


def test_standard_library_functions():
    from tender.lib.identities.functions import exp_product, ALL
    assert exp_product.name == "exp-product"
    assert len(ALL) == 2


def test_declare_symmetric():
    A = tensor("A", 2)
    declare_symmetric(A)
    a = make_pattern_var("a"); a.constrain_rank(2); a.constrain_symmetric()
    id_ = Identity("test", a, a)
    matches = find_matches(id_, A)
    assert len(matches) == 1


def test_declare_skew_symmetric():
    W = tensor("W", 2)
    declare_skew_symmetric(W)
    a = make_pattern_var("a"); a.constrain_rank(2); a.constrain_skew_symmetric()
    id_ = Identity("test", a, a)
    matches = find_matches(id_, W)
    assert len(matches) == 1


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
    assert step.name == "collect(\\mathbf{v})"


def test_collect_then_localize_pvw_pattern():
    """collect(du) → localize(V) gives only the volume integrand; ∂V discarded."""
    from tender import Contract
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
    # localize(V) discards the ∂V term; result is just g·du
    assert isinstance(result, Contract)
    assert result.lhs is g
    assert result.rhs is du


def _make_bac_cab_identity():
    """Create a local BAC-CAB Identity for use as a rewrite rule in tests."""
    a = make_pattern_var("a"); a.constrain_rank(1)
    b = make_pattern_var("b"); b.constrain_rank(1)
    c = make_pattern_var("c"); c.constrain_rank(1)
    return (
        Identity("BAC-CAB", cross(a, cross(b, c)), tp(b, dot(a, c)) - tp(c, dot(a, b))),
        a, b, c,
    )


def test_apply_identity_wrong_expression_raises():
    # Regression: apply_identity must validate the binding against the actual
    # expression, not silently substitute into the wrong one.
    # BAC-CAB LHS: a×(b×c).  Using mapping {a:u, b:v, c:w} on (v×w)×u
    # (operands swapped) must raise, not produce a silently wrong result.
    bac_cab, a, b, c = _make_bac_cab_identity()
    u = tensor("u", 1); v = tensor("v", 1); w = tensor("w", 1)

    step = apply_identity(bac_cab, {a: u, b: v, c: w})

    correct = cross(u, cross(v, w))
    wrong   = cross(cross(v, w), u)

    # Correct expression: no error
    Derivation([step]).apply(State(correct))

    # Swapped expression: must raise
    with pytest.raises(Exception):
        Derivation([step]).apply(State(wrong))


def test_search_apply_direct():
    # When target already applies, search_apply returns one step (the application).
    bac_cab, *_ = _make_bac_cab_identity()
    u = tensor("u", 1); v = tensor("v", 1); w = tensor("w", 1)
    expr = cross(u, cross(v, w))
    steps = search_apply(bac_cab, expr, timeout=2.0)
    assert len(steps) == 1
    history = Derivation(steps).apply(State(expr))
    result = history[-1].expr.latex()
    # BAC-CAB: a×(b×c) = b(a·c) - c(a·b) → v(u·w) - w(u·v)
    assert r"\cdot" in result


def test_search_apply_needs_anticomm():
    # cross(cross(v,w), u) needs anti-commutativity before BAC-CAB applies.
    bac_cab, *_ = _make_bac_cab_identity()
    u = tensor("u", 1); v = tensor("v", 1); w = tensor("w", 1)
    expr = cross(cross(v, w), u)
    steps = search_apply(bac_cab, expr, timeout=5.0)
    assert len(steps) >= 2
    history = Derivation(steps).apply(State(expr))
    result = history[-1].expr.latex()
    assert r"\cdot" in result


def test_search_apply_exhausted():
    # An unreachable target raises RuntimeError.
    bac_cab, *_ = _make_bac_cab_identity()
    u = tensor("u", 1); v = tensor("v", 1); w = tensor("w", 1)
    # A plain tensor product: no cross products, so BAC-CAB can never apply.
    expr = dot(u, v)
    with pytest.raises(RuntimeError):
        search_apply(bac_cab, expr, rules=[], timeout=1.0)


# ===========================================================================
# Phase 13.2 — basis expansion steps
# ===========================================================================

def test_expand_in_basis_step_rank1_covariant():
    from tender import expand_in_basis_step, Sum, TensorProduct
    a = tensor("a", 1)
    step = expand_in_basis_step(a, wcs, covariant=True)
    result = step.apply(State(a)).expr
    assert isinstance(result, Sum)
    assert len(result.terms) == 3
    for term in result.terms:
        assert isinstance(term, TensorProduct)
        assert term.lhs.rank == 0   # scalar component
        assert term.rhs.rank == 1   # basis vector


def test_expand_in_basis_step_rank1_contravariant():
    from tender import expand_in_basis_step, Sum, TensorProduct
    a = tensor("a", 1)
    step = expand_in_basis_step(a, wcs, covariant=False)
    result = step.apply(State(a)).expr
    assert isinstance(result, Sum)
    assert len(result.terms) == 3


def test_expand_in_basis_step_rank2():
    from tender import expand_in_basis_step, Sum
    A = tensor("A", 2)
    step = expand_in_basis_step(A, wcs, covariant=True)
    result = step.apply(State(A)).expr
    assert isinstance(result, Sum)
    assert len(result.terms) == 9   # 3x3 components


def test_expand_in_basis_step_bad_rank_raises():
    from tender import expand_in_basis_step
    T = tensor("T", 3)
    with pytest.raises(ValueError):
        expand_in_basis_step(T, wcs)


def test_simplify_basis_dot_step_diagonal():
    from tender import simplify_basis_dot_step, RationalConst
    e0 = wcs.basis(0)
    expr = dot(e0, e0)
    step = simplify_basis_dot_step(wcs)
    result = step.apply(State(expr)).expr
    assert isinstance(result, RationalConst)
    assert result.value == Rational(1)


def test_simplify_basis_dot_step_off_diagonal():
    from tender import simplify_basis_dot_step, RationalConst
    e0 = wcs.basis(0)
    e1 = wcs.basis(1)
    expr = dot(e0, e1)
    step = simplify_basis_dot_step(wcs)
    result = step.apply(State(expr)).expr
    assert isinstance(result, RationalConst)
    assert result.value == Rational(0)


def test_collect_zero_terms_step():
    from tender import collect_zero_terms_step, rational, Sum
    a = tensor("a", 1)
    b = tensor("b", 1)
    zero = rational(0)
    # Build sum manually: a + 0 + b + 0
    expr = a + zero + b + zero
    step = collect_zero_terms_step()
    result = step.apply(State(expr)).expr
    assert isinstance(result, Sum)
    assert len(result.terms) == 2


def test_reassemble_from_components_step_identity():
    from tender import reassemble_from_components_step, IdentityTensor, tp
    e0, e1, e2 = wcs.basis(0), wcs.basis(1), wcs.basis(2)
    identity_sum = tp(e0, e0) + tp(e1, e1) + tp(e2, e2)
    step = reassemble_from_components_step(wcs)
    result = step.apply(State(identity_sum)).expr
    assert isinstance(result, IdentityTensor)


def test_dot_commutativity_pipeline():
    """Full a·b = b·a derivation via component expansion in WCS."""
    from tender import expand_in_basis_step, simplify_basis_dot_step
    from tender import collect_zero_terms_step, expand_step, Sum, Product

    a = tensor("a", 1)
    b = tensor("b", 1)
    expr = dot(a, b)

    steps = [
        expand_in_basis_step(a, wcs, covariant=True),
        expand_in_basis_step(b, wcs, covariant=False),
        expand_step(),
        simplify_basis_dot_step(wcs),
        collect_zero_terms_step(),
    ]
    history = Derivation(steps).apply(State(expr))
    result = history[-1].expr

    # Result should be a sum of 3 scalar products a^i * b_i
    assert isinstance(result, Sum)
    assert len(result.terms) == 3
    for term in result.terms:
        assert isinstance(term, Product)
        assert term.lhs.rank == 0
        assert term.rhs.rank == 0


def test_definitions_library():
    from tender.lib.identities.definitions import (
        tp_contract_right, tp_contract_left, trace_dyad,
        identity_vec, cross_def, scalar_comm, ALL,
    )
    assert len(ALL) == 8  # added eps_anticomm, eps_delta in Phase 13.9
    names = {id_.name for id_ in ALL}
    assert "tp-contract-right" in names
    assert "scalar-comm" in names
    assert "cross-def" in names
    assert "eps-anticomm" in names
    assert "eps-delta" in names


def test_from_derivation_pattern_vars():
    """Identity.from_derivation with pattern_vars lifts concrete tensors to PatternVars."""
    from tender import tensor, dot, Identity, PatternVar, State
    a = tensor("a", 1)
    b = tensor("b", 1)
    lhs_expr = dot(a, b)
    rhs_expr = dot(b, a)
    history = [State(lhs_expr), State(rhs_expr)]
    id_ = Identity.from_derivation("dot-comm-test", history, pattern_vars=[a, b])
    assert id_.name == "dot-comm-test"
    # Sub-nodes of lhs Contract should be PatternVar instances.
    lhs_co = id_.lhs
    assert isinstance(lhs_co.lhs, PatternVar), "lhs of dot should be PatternVar"
    assert isinstance(lhs_co.rhs, PatternVar), "rhs of dot should be PatternVar"
    assert lhs_co.lhs.rank == 1
    assert lhs_co.lhs.symbol == "a"


def test_from_derivation_no_pattern_vars_is_unchanged():
    """from_derivation without pattern_vars preserves concrete tensors."""
    from tender import tensor, dot, Identity, PatternVar, State
    a = tensor("a", 1)
    b = tensor("b", 1)
    lhs_expr = dot(a, b)
    rhs_expr = dot(b, a)
    history = [State(lhs_expr), State(rhs_expr)]
    id_ = Identity.from_derivation("raw", history)
    # No substitution: lhs is the original Contract, sub-nodes are NamedTensors.
    assert id_.lhs is lhs_expr
    assert id_.rhs is rhs_expr


def test_vectors_library():
    """Phase 13.5 vectors module loads with correct identities."""
    from tender.lib.identities.vectors import (
        dot_comm, cross_self_zero, cross_anticomm,
        double_cross, jacobi, ALL,
    )
    assert len(ALL) == 5  # cross_anticomm now included (Phase 13.9)
    names = {id_.name for id_ in ALL}
    assert "dot-comm" in names
    assert "cross-self-zero" in names
    assert "cross-anticomm" in names
    assert "double-cross" in names
    assert "jacobi" in names
    assert cross_anticomm.name == "cross-anticomm"


def test_vectors_latex_forms():
    """Check LaTeX rendering of key vector identities."""
    from tender.lib.identities.vectors import dot_comm, cross_self_zero, double_cross
    assert "cdot" in dot_comm.lhs.latex()
    assert dot_comm.rhs.latex() == dot_comm.lhs.latex()[::-1] or \
        "cdot" in dot_comm.rhs.latex()
    # cross_self_zero rhs should be 0
    assert cross_self_zero.rhs.latex() == "0"
    # double_cross lhs should contain times
    assert "times" in double_cross.lhs.latex()


def test_full_identity_library_count():
    """Library has at least 16 identities after Phase 13.5."""
    from tender.lib.identities import ALL
    assert len(ALL) >= 16
    names = {id_.name for id_ in ALL}
    assert "dot-comm" in names
    assert "double-cross" in names
    assert "jacobi" in names


# ===========================================================================
# SymBasisVec and abstract-index expansion (Phase 13.6)
# ===========================================================================

def test_sym_basis_vec_latex():
    from tender import make_sym_basis_vec, SymBasisVec
    # index_id 0 → letter "i", index_id 1 → letter "j" (both alone → "i")
    sbv_basis = make_sym_basis_vec(wcs, 0, False)
    sbv_cobasis = make_sym_basis_vec(wcs, 0, True)
    assert isinstance(sbv_basis, SymBasisVec)
    assert sbv_basis.latex() == r"\mathbf{e}_{i}"
    assert sbv_cobasis.latex() == r"\mathbf{e}^{i}"


def test_sym_basis_vec_properties():
    from tender import make_sym_basis_vec, SymBasisVec
    sbv = make_sym_basis_vec(wcs, 0, True)
    assert sbv.index_id == 0
    assert sbv.is_cobasis is True
    assert sbv.rank == 1


def test_abstract_basis_expansion_covariant():
    """expand_in_basis_step with abstract=True produces a single TP(AbstractComp, SBV)."""
    from tender import (
        expand_in_basis_step, TensorProduct, AbstractComp, SymBasisVec
    )

    a = tensor("a", 1)
    step = expand_in_basis_step(a, wcs, covariant=True, abstract=True)
    result = Derivation([step]).apply(State(a))[-1].expr

    assert isinstance(result, TensorProduct)
    assert isinstance(result.lhs, AbstractComp)
    assert result.lhs.base_sym == "a"
    # indices: [(id, True)] meaning upper (covariant) index
    assert len(result.lhs.indices) == 1
    assert result.lhs.indices[0][1] is True
    assert isinstance(result.rhs, SymBasisVec)
    assert result.rhs.is_cobasis is False
    # latex should render as "a^{i} \mathbf{e}_{i}" style
    assert r"\mathbf{e}_{" in result.rhs.latex()


def test_abstract_basis_expansion_contravariant():
    from tender import (
        expand_in_basis_step, TensorProduct, AbstractComp, SymBasisVec
    )

    b = tensor("b", 1)
    step = expand_in_basis_step(b, wcs, covariant=False, abstract=True)
    result = Derivation([step]).apply(State(b))[-1].expr

    assert isinstance(result, TensorProduct)
    assert isinstance(result.lhs, AbstractComp)
    assert result.lhs.base_sym == "b"
    # indices: [(id, False)] meaning lower (contravariant) index
    assert result.lhs.indices[0][1] is False
    assert isinstance(result.rhs, SymBasisVec)
    assert result.rhs.is_cobasis is True


def test_abstract_basis_expansion_rank_error():
    from tender import expand_in_basis_step
    import pytest
    # rank 0 (scalar) has no indices to expand
    s = tensor("s", 0)
    with pytest.raises(ValueError, match="abstract mode requires rank"):
        expand_in_basis_step(s, wcs, abstract=True)


def test_abstract_basis_expansion_rank2_covariant():
    """Abstract rank-2 all-up: A → TP(TP(AbstractComp, SBV_i), SBV_j)."""
    from tender import expand_in_basis_step, SymBasisVec, TensorProduct, AbstractComp
    A = tensor("A", 2)
    step = expand_in_basis_step(A, wcs, covariant=True, abstract=True)
    # TP structure: TP(TP(comp, SBV_i), SBV_j)
    result = Derivation([step]).apply(State(A))[-1].expr
    assert isinstance(result, TensorProduct)
    sbv_j = result.rhs
    assert isinstance(sbv_j, SymBasisVec)
    assert not sbv_j.is_cobasis
    inner = result.lhs
    assert isinstance(inner, TensorProduct)
    sbv_i = inner.rhs
    assert isinstance(sbv_i, SymBasisVec)
    assert not sbv_i.is_cobasis
    comp = inner.lhs
    assert isinstance(comp, AbstractComp)
    assert comp.base_sym == "A"
    # Two upper indices
    assert len(comp.indices) == 2
    assert comp.indices[0][1] is True
    assert comp.indices[1][1] is True


def test_abstract_basis_expansion_rank2_mixed():
    """Abstract rank-2 mixed: A → TP(TP(AbstractComp, SBV_i), SBV^j)."""
    from tender import expand_in_basis_step, SymBasisVec, TensorProduct, AbstractComp
    A = tensor("A", 2)
    step = expand_in_basis_step(A, wcs, covariant=[True, False], abstract=True)
    result = Derivation([step]).apply(State(A))[-1].expr
    assert isinstance(result, TensorProduct)
    sbv_j = result.rhs
    assert isinstance(sbv_j, SymBasisVec)
    assert sbv_j.is_cobasis  # covariant=False → cobasis
    comp = result.lhs.lhs
    assert isinstance(comp, AbstractComp)
    assert comp.base_sym == "A"
    assert comp.indices[0][1] is True   # first: upper
    assert comp.indices[1][1] is False  # second: lower


def test_abstract_basis_expansion_rank2_covariant_list_mismatch():
    """covariant list length != rank raises ValueError."""
    from tender import expand_in_basis_step
    import pytest
    A = tensor("A", 2)
    with pytest.raises(ValueError, match="covariant list length"):
        expand_in_basis_step(A, wcs, covariant=[True], abstract=True)


def test_abstract_comp_sym_latex():
    """sym_to_latex renders multi-index component symbols correctly."""
    from tender import tensor
    # Force a latex call through NamedTensor.latex() (rank 0, uses sym_to_latex)
    n = tensor("A^i_j", 0)
    assert n.latex() == "A^{i}_{j}"
    n2 = tensor("A^ij", 0)
    assert n2.latex() == "A^{ij}"


def test_abstract_dot_simplify():
    """Abstract-index dot product: a^i e_i · b_j e^j δ_j^i → a^i b_i."""
    from tender import (
        expand_in_basis_step, simplify_basis_dot_step, contract_kronecker_step,
        Product, AbstractComp,
    )
    a = tensor("a", 1)
    b = tensor("b", 1)
    steps = [
        expand_in_basis_step(a, wcs, covariant=True, abstract=True),
        expand_in_basis_step(b, wcs, covariant=False, abstract=True),
        simplify_basis_dot_step(wcs),
        contract_kronecker_step(),
    ]
    result = Derivation(steps).apply(State(dot(a, b)))[-1].expr

    assert isinstance(result, Product)
    assert isinstance(result.lhs, AbstractComp)
    assert isinstance(result.rhs, AbstractComp)
    assert result.lhs.base_sym == "a"
    assert result.rhs.base_sym == "b"
    # latex renders as "a^{i} b_{i}" (enrich assigns "i" to the shared index_id)
    assert result.latex() == "a^{i} b_{i}"


def test_abstract_dot_derivation_steps():
    """Proof 1 in dot_commutativity: exactly 4 states (initial + 3 steps)."""
    from tender import expand_in_basis_step, simplify_basis_dot_step

    a = tensor("a", 1)
    b = tensor("b", 1)
    steps = [
        expand_in_basis_step(a, wcs, covariant=True, abstract=True),
        expand_in_basis_step(b, wcs, covariant=False, abstract=True),
        simplify_basis_dot_step(wcs),
    ]
    history = Derivation(steps).apply(State(dot(a, b)))
    assert len(history) == 4


def test_prove_equal_by_components_abstract():
    """prove_equal_by_components works with abstract contracted endpoints."""
    from tender import (
        expand_in_basis_step, simplify_basis_dot_step,
        prove_equal_by_components, contract_kronecker_step,
        Product, AbstractComp,
    )
    a = tensor("a", 1)
    b = tensor("b", 1)

    lhs_steps = [
        expand_in_basis_step(a, wcs, covariant=True, abstract=True),
        expand_in_basis_step(b, wcs, covariant=False, abstract=True),
        simplify_basis_dot_step(wcs),
        contract_kronecker_step(),
    ]
    rhs_steps = [
        expand_in_basis_step(b, wcs, covariant=False, abstract=True),
        expand_in_basis_step(a, wcs, covariant=True, abstract=True),
        simplify_basis_dot_step(wcs),
        contract_kronecker_step(),
    ]
    lhs_hist, rhs_hist = prove_equal_by_components(
        dot(a, b), dot(b, a), lhs_steps, rhs_steps
    )
    assert isinstance(lhs_hist[-1].expr, Product)
    assert isinstance(rhs_hist[-1].expr, Product)
    # Both sides reduce to the same base symbols (letter-invariant)
    assert lhs_hist[-1].expr.latex() == "a^{i} b_{i}"
    # rhs: b is expanded first (contravariant) then a (covariant) → b_{i} a^{i}
    assert rhs_hist[-1].expr.latex() == "b_{i} a^{i}"


# ---------------------------------------------------------------------------
# Phase 13.8: KroneckerDelta, LeviCivitaSymbol, contract_kronecker_step
# ---------------------------------------------------------------------------

def test_kronecker_delta_equal_ids_folds_to_one():
    """KroneckerDelta with equal IDs folds to RationalConst(1)."""
    from tender import make_kronecker_delta, RationalConst
    kd = make_kronecker_delta(5, 5)
    assert isinstance(kd, RationalConst)
    assert kd.python() == "Rational(1)"


def test_kronecker_delta_distinct_ids():
    """KroneckerDelta with distinct IDs stays as KroneckerDelta."""
    from tender import make_kronecker_delta, KroneckerDelta
    kd = make_kronecker_delta(0, 1)
    assert isinstance(kd, KroneckerDelta)
    assert kd.lower_id == 0
    assert kd.upper_id == 1


def test_kronecker_delta_latex():
    """KroneckerDelta renders with subscript/superscript."""
    from tender import make_kronecker_delta
    kd = make_kronecker_delta(0, 1)
    assert kd.latex() == r"\delta_{i}^{j}"


def test_kronecker_delta_python():
    """KroneckerDelta python() round-trips correctly."""
    from tender import make_kronecker_delta, KroneckerDelta
    kd = make_kronecker_delta(2, 7)
    assert kd.python() == "make_kronecker_delta(2, 7)"


def test_levi_civita_symbol_all_lower():
    """LeviCivitaSymbol with all-lower indices renders simply."""
    from tender import make_levi_civita_symbol, LeviCivitaSymbol
    lcs = make_levi_civita_symbol([0, 1, 2], [False, False, False])
    assert isinstance(lcs, LeviCivitaSymbol)
    assert lcs.latex() == r"\varepsilon_{ijk}"


def test_levi_civita_symbol_all_upper():
    """LeviCivitaSymbol with all-upper indices renders simply."""
    from tender import make_levi_civita_symbol, LeviCivitaSymbol
    lcs = make_levi_civita_symbol([0, 1, 2], [True, True, True])
    assert lcs.latex() == r"\varepsilon^{ijk}"


def test_levi_civita_symbol_mixed_indices():
    """LeviCivitaSymbol with mixed indices uses dot-placeholder scheme."""
    from tender import make_levi_civita_symbol
    # index 0 upper, 1 lower, 2 lower
    lcs = make_levi_civita_symbol([0, 1, 2], [True, False, False])
    assert lcs.latex() == r"\varepsilon^{i{\cdot}{\cdot}}_{{\cdot}jk}"


def test_levi_civita_symbol_python():
    """LeviCivitaSymbol python() round-trips correctly."""
    from tender import make_levi_civita_symbol
    lcs = make_levi_civita_symbol([0, 1, 2], [False, False, False])
    assert lcs.python() == "make_levi_civita_symbol([0, 1, 2], [False, False, False])"


def test_contract_kronecker_step_basic():
    """contract_kronecker_step: δ_i^j a^i b_j → a^i b_i (via dot path)."""
    from tender import (
        expand_in_basis_step, simplify_basis_dot_step,
        contract_kronecker_step, KroneckerDelta, Product,
    )
    a = tensor("a", 1)
    b = tensor("b", 1)
    # After expand+simplify we get Product(Product(AC_a, AC_b), KroneckerDelta)
    intermediate = Derivation([
        expand_in_basis_step(a, wcs, covariant=True, abstract=True),
        expand_in_basis_step(b, wcs, covariant=False, abstract=True),
        simplify_basis_dot_step(wcs),
    ]).apply(State(dot(a, b)))[-1].expr
    assert isinstance(intermediate, Product)
    assert isinstance(intermediate.rhs, KroneckerDelta)
    # contract_kronecker_step removes the delta
    result = Derivation([contract_kronecker_step()]).apply(State(intermediate))[-1].expr
    assert isinstance(result, Product)
    assert not isinstance(result.rhs, KroneckerDelta)
    assert result.latex() == "a^{i} b_{i}"


def test_contract_kronecker_step_full_dot_path():
    """End-to-end: a·b → a^i b_j δ_i^j → a^i b_i."""
    from tender import (
        expand_in_basis_step, simplify_basis_dot_step,
        contract_kronecker_step, Product, AbstractComp,
    )
    a = tensor("a", 1)
    b = tensor("b", 1)
    steps = [
        expand_in_basis_step(a, wcs, covariant=True, abstract=True),
        expand_in_basis_step(b, wcs, covariant=False, abstract=True),
        simplify_basis_dot_step(wcs),
        contract_kronecker_step(),
    ]
    result = Derivation(steps).apply(State(dot(a, b)))[-1].expr
    assert isinstance(result, Product)
    assert isinstance(result.lhs, AbstractComp)
    assert isinstance(result.rhs, AbstractComp)
    assert result.lhs.base_sym == "a"
    assert result.rhs.base_sym == "b"
    # Shared index ID — same slot in both factors
    assert result.lhs.indices[0][0] == result.rhs.indices[0][0]
    assert result.latex() == "a^{i} b_{i}"


def test_theorems_import():
    """tender.lib.theorems imports cleanly and all theorems are proved."""
    import tender.lib.theorems as thm
    assert hasattr(thm, "dot_commutativity")
    assert hasattr(thm, "cross_anticommutativity")
    assert hasattr(thm, "bac_cab")
    assert hasattr(thm, "eps_delta_theorem")
    assert thm.dot_commutativity.name == "dot_commutativity"
    assert thm.cross_anticommutativity.name == "cross_anticommutativity"
    assert thm.bac_cab.name == "bac_cab"
    assert thm.eps_delta_theorem.name == "eps_delta"


def test_expand_levi_civita_first_step():
    """expand_levi_civita_first_step replaces only the first eps, not both."""
    from tender import (
        expand_levi_civita_first_step, LeviCivitaTensor, LeviCivitaSymbol,
        TensorProduct,
    )
    a = tensor("a", 1)
    b = tensor("b", 1)
    c = tensor("c", 1)
    expr = ddot(eps, tp(a, ddot(eps, tp(b, c))))

    # One application should replace the outer eps only.
    step = expand_levi_civita_first_step(wcs)
    result = Derivation([step]).apply(State(expr))[-1].expr

    # The outer eps should be gone (no LeviCivitaTensor at top-level).
    # The inner eps is still a LeviCivitaTensor.
    lct_nodes = []
    def _collect(e):
        if isinstance(e, LeviCivitaTensor):
            lct_nodes.append(e)
        if isinstance(e, TensorProduct):
            _collect(e.lhs); _collect(e.rhs)
        if hasattr(e, "lhs") and not isinstance(e, TensorProduct):
            _collect(e.lhs); _collect(e.rhs)
    _collect(result)
    # Exactly one LeviCivitaTensor remains (the inner one).
    assert len(lct_nodes) == 1


def test_contract_eps_pair_step():
    """contract_eps_pair_step applies the ε-δ formula for two LCS sharing a dummy."""
    from tender import (
        contract_eps_pair_step, make_levi_civita_symbol, alloc_index_id,
        Sum, Scale,
    )
    # Build ε_{ijk} ε_{ilm}: dummy = i (shared at position 0 in both)
    i = alloc_index_id()
    j = alloc_index_id()
    k = alloc_index_id()
    l = alloc_index_id()
    m = alloc_index_id()
    lcs1 = make_levi_civita_symbol([i, j, k], [False, False, False])
    lcs2 = make_levi_civita_symbol([i, l, m], [False, False, False])
    prod = lcs1 * lcs2  # TensorProduct (both rank-0)

    result = Derivation([contract_eps_pair_step()]).apply(State(prod))[-1].expr
    # Result should be a Sum of two terms (δδ - δδ form).
    assert isinstance(result, Sum)
    assert len(result.terms) == 2


def test_eps_delta_theorem_lhs_rhs():
    """eps_delta_theorem has the expected lhs and rhs forms."""
    from tender.lib.theorems import eps_delta_theorem
    assert "eps" in eps_delta_theorem.lhs.python()
    assert "eps" in eps_delta_theorem.lhs.python()
    # RHS should contain tp and dot
    rhs_py = eps_delta_theorem.rhs.python()
    assert "tp(" in rhs_py
