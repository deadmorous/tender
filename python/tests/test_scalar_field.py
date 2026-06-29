"""Scalar fields (vibe 000069 M1): coordinate variables, elementary functions,
powers — construction, rendering, and the canonicalize round trip."""

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
