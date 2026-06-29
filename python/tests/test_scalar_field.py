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
