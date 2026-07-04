"""Scalar fields (vibe 000069 M1): coordinate variables, elementary functions,
powers — construction, rendering, and the canonicalize round trip."""

import pytest

import tender as t
import tender.derivation as td


def test_coordinate_renders_as_letter():
    ctx = t.Context()
    r = t.coordinate("r", ctx=ctx)
    assert r.latex() == "r"


def test_scalar_function_rendering():
    ctx = t.Context()
    phi = t.coordinate(r"\varphi", ctx=ctx)
    assert t.cos(phi).latex() == r"\cos\left(\varphi\right)"
    assert t.sin(phi).latex() == r"\sin\left(\varphi\right)"


def test_power_and_sqrt_rendering():
    ctx = t.Context()
    r = t.coordinate("r", ctx=ctx)
    assert (r**2).latex() == "r^{2}"
    assert t.sqrt(r**2).latex() == r"\sqrt{r^{2}}"


def test_polar_x_component():
    ctx = t.Context()
    r = t.coordinate("r", ctx=ctx)
    phi = t.coordinate(r"\varphi", ctx=ctx)
    x = r * t.cos(phi)
    assert x.latex() == r"r \, \cos\left(\varphi\right)"


def test_canonicalize_round_trips():
    ctx = t.Context()
    r = t.coordinate("r", ctx=ctx)
    phi = t.coordinate(r"\varphi", ctx=ctx)
    x = r * t.cos(phi)
    c = td.canonicalize(x)
    # Canonical form is idempotent.
    assert td.canonicalize(c).latex() == c.latex()


def test_like_terms_collect_over_scalar_fields():
    ctx = t.Context()
    r = t.coordinate("r", ctx=ctx)
    phi = t.coordinate(r"\varphi", ctx=ctx)
    term = r * t.cos(phi)
    folded = td.canonicalize(2 * term + term)
    expected = td.canonicalize(3 * term)
    assert folded.latex() == expected.latex()


def test_pythagorean_pair_survives_unsimplified():
    # cos² + sin² stays a two-term sum until the M3 simplifier folds it to 1.
    ctx = t.Context()
    phi = t.coordinate(r"\varphi", ctx=ctx)
    s = td.canonicalize(t.cos(phi) ** 2 + t.sin(phi) ** 2)
    assert "\\cos" in s.latex() and "\\sin" in s.latex()


# ---- partial differentiation (vibe 000069 M2) -------------------------------


def test_partial_coordinate_and_constant():
    ctx = t.Context()
    r = t.coordinate("r", slot=0, ctx=ctx)
    phi = t.coordinate(r"\varphi", slot=1, ctx=ctx)
    assert td.partial(r, r).latex() == "1"
    assert td.partial(phi, r).latex() == "0"
    # A reference vector is constant.
    i = t.tensor("i", rank=1, ctx=ctx)
    assert td.partial(i, r).latex() == "0"


def test_partial_chain_rule():
    ctx = t.Context()
    phi = t.coordinate(r"\varphi", ctx=ctx)
    assert td.partial(t.sin(phi), phi).latex() == r"\cos\left(\varphi\right)"


def test_partial_rejects_non_coordinate():
    ctx = t.Context()
    r = t.coordinate("r", ctx=ctx)
    a = t.tensor("a", rank=0, ctx=ctx)
    with pytest.raises(ValueError):
        td.partial(r, a)


def test_simplify_pythagorean():
    ctx = t.Context()
    phi = t.coordinate(r"\varphi", ctx=ctx)
    assert td.simplify_scalars(t.cos(phi) ** 2 + t.sin(phi) ** 2).latex() == "1"


def test_simplify_pythagorean_squared():
    # (sin²θ+cos²θ)² expanded — no linear pair survives, so the power fold closes
    # it.  This is the coefficient left by a rank-2 express WCS round-trip.
    ctx = t.Context()
    th = t.coordinate(r"\theta", ctx=ctx)
    s, c = t.sin(th), t.cos(th)
    e = s**4 + t.scalar(2) * s**2 * c**2 + c**4
    assert td.simplify_scalars(e).latex() == "1"


def test_simplify_pythagorean_cubed():
    ctx = t.Context()
    th = t.coordinate(r"\theta", ctx=ctx)
    s, c = t.sin(th), t.cos(th)
    e = s**6 + t.scalar(3) * s**4 * c**2 + t.scalar(3) * s**2 * c**4 + c**6
    assert td.simplify_scalars(e).latex() == "1"


def test_simplify_lone_cos_square_not_inflated():
    # The guarded power fold must not rewrite a lone cos²θ to the larger 1−sin²θ.
    ctx = t.Context()
    th = t.coordinate(r"\theta", ctx=ctx)
    assert td.simplify_scalars(t.cos(th) ** 2).latex() == r"\cos\left(\theta\right)^{2}"


def test_simplify_combines_fractions_over_common_denominator():
    # A/r² + B/r → (A + B r)/r²: split fraction coefficients acquire one shape,
    # so algebraically-equal coefficients become structurally equal.
    ctx = t.Context()
    r = t.coordinate("r", nonneg=True, ctx=ctx)
    A = t.field("A", 0, ctx=ctx)
    B = t.field("B", 0, ctx=ctx)
    split = A / r**2 + B / r
    combined = (A + B * r) / r**2
    assert td.simplify_scalars(split).latex() == td.simplify_scalars(combined).latex()


def test_simplify_combines_fractions_numeric_denominator():
    # A/2 + B/r brings in a numeric least-common-denominator (2r).
    ctx = t.Context()
    r = t.coordinate("r", nonneg=True, ctx=ctx)
    A = t.field("A", 0, ctx=ctx)
    B = t.field("B", 0, ctx=ctx)
    lhs = A / t.scalar(2) + B / r
    rhs = (A * r + t.scalar(2) * B) / (t.scalar(2) * r)
    assert td.simplify_scalars(lhs).latex() == td.simplify_scalars(rhs).latex()


def test_simplify_metric_component():
    ctx = t.Context()
    r = t.coordinate("r", slot=0, nonneg=True, ctx=ctx)
    phi = t.coordinate(r"\varphi", slot=1, ctx=ctx)
    g = r**2 * t.sin(phi) ** 2 + r**2 * t.cos(phi) ** 2
    assert td.simplify_scalars(g).latex() == "r^{2}"


def test_simplify_root_of_square_needs_nonneg():
    ctx = t.Context()
    r = t.coordinate("r", slot=0, nonneg=True, ctx=ctx)
    s = t.coordinate("s", slot=0, ctx=ctx)  # sign unknown
    assert td.simplify_scalars(t.sqrt(r**2)).latex() == "r"
    assert td.simplify_scalars(t.sqrt(s**2)).latex() == r"\sqrt{s^{2}}"


def test_simplify_scale_factor():
    ctx = t.Context()
    r = t.coordinate("r", slot=0, nonneg=True, ctx=ctx)
    phi = t.coordinate(r"\varphi", slot=1, ctx=ctx)
    h_phi = t.sqrt(r**2 * t.sin(phi) ** 2 + r**2 * t.cos(phi) ** 2)
    assert td.simplify_scalars(h_phi).latex() == "r"


def test_partial_polar_tangent_vectors():
    ctx = t.Context()
    r = t.coordinate("r", slot=0, ctx=ctx)
    phi = t.coordinate(r"\varphi", slot=1, ctx=ctx)
    i = t.tensor("i", rank=1, ctx=ctx)
    j = t.tensor("j", rank=1, ctx=ctx)
    R = r * t.cos(phi) * i + r * t.sin(phi) * j

    g_r = td.canonicalize(t.cos(phi) * i + t.sin(phi) * j)
    assert td.algebraic_eq(td.partial(R, r), g_r)

    g_phi = td.canonicalize(-(r * t.sin(phi) * i) + r * t.cos(phi) * j)
    assert td.algebraic_eq(td.partial(R, phi), g_phi)
