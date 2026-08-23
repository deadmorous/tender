"""tender.identities — the standard identity library, as data you can extend.

Rules live here, in Python, so the set can grow without rebuilding anything
(vibe 000096): a rule is just a :class:`tender.derivation.Identity`, and your
own rules are first-class alongside these.  Copy any factory below as a
template::

    import tender as t, tender.derivation as td

    ctx = t.Context()
    u, v = (t.tensor(n, rank=1, ctx=ctx) for n in "uv")
    my_rule = td.Identity("swap", u % v, -(v % u))
    td.prove_equal(lhs, rhs, td.rules("cross", ctx=ctx) + [my_rule])

Rules are grouped, and groups are the unit of selection: pass the groups a
problem needs rather than everything, since rule count is the main driver of
saturation cost.

Two conventions matter when writing a rule, both learned the hard way:

**Pattern variables are slot-less abstract tensors** (vibe 000051): a
``t.tensor("u", rank=1)`` in a rule's left-hand side binds any whole factor.
Well-known tensors (``I``, δ, ε) and index-slotted tensors stay literal.

**Variable names are load-bearing.**  Canon sorts a *symmetric* contraction
chain (``:``, ``··``, and ``·`` between rank-1 operands) by tensor name, and
the matcher compares chain factors positionally — so a rule whose LHS is such
a chain fires only on targets whose own names sort the same way, and the
variable's name silently decides which those are.  Measured case: ``X··I =
tr X`` fires with a variable named ``B`` and not one named ``X``.  Until AC
chain matching lands, test any such rule across a spread of target names —
:func:`tender.derivation.prove_equal` reports which rules actually fired, so
an inert rule is visible rather than mysterious.
"""

import tender as _t

from .derivation import Identity

__all__ = [
    "group",
    "group_names",
    "all_rules",
    # eps_delta
    "delta_contraction",
    "delta_trace",
    "eps_delta_1",
    "eps_delta_2",
    # cross
    "bac_cab",
    "cross_identity",
    "cross_removal",
    "lagrange",
    # dyadic
    "trace_cyclic",
    "identity_dot",
]

_U = _t.Level.Upper
_L = _t.Level.Lower


def _level_for(realm, requested):
    """The index level a rule should be spelled with (vibe 000047).

    In the Orthonormal realm upper and lower are interchangeable, so the
    library spells every Orthonormal index **lower**; other realms keep the
    requested level.  This is not cosmetic: matching is level-exact and
    canonicalize deliberately does not coerce levels, so a rule and the
    targets it must match have to be spelled identically.
    """
    return _L if realm == _t.Realm.Orthonormal else requested


def _var(ctx, name, rank):
    """A subtree pattern variable: a slot-less abstract tensor of `rank`."""
    return _t.tensor(name, rank=rank, ctx=ctx)


# ---- eps_delta group: index-level δ / ε contractions ----------------------


def delta_contraction(ctx, space=None, realm=_t.Realm.Oblique):
    """Σ_p δ^p_a δ^p_b = δ_ab (any space)."""
    space = space or _t.space_3d

    def d(la, lb, a, b):
        return _t.delta(
            realm, space, _level_for(realm, la), _level_for(realm, lb), a, b, ctx=ctx
        )

    p, a, b = (ctx.alloc_index() for _ in range(3))
    return Identity(
        "delta-contraction",
        _t.explicit_sum(p, d(_U, _L, p, a) * d(_U, _L, p, b), ctx=ctx),
        d(_L, _L, a, b),
    )


def delta_trace(ctx, space=None, realm=_t.Realm.Oblique):
    """Σ_p δ^p_p = dim(space)."""
    space = space or _t.space_3d
    p = ctx.alloc_index()
    lhs = _t.explicit_sum(
        p,
        _t.delta(
            realm, space, _level_for(realm, _U), _level_for(realm, _L), p, p, ctx=ctx
        ),
        ctx=ctx,
    )
    dim = 3 if space is _t.space_3d else (2 if space is _t.space_2d else 4)
    return Identity("delta-trace", lhs, _t.scalar(dim, ctx=ctx))


def eps_delta_1(ctx, realm=_t.Realm.Oblique):
    """Σ_i ε^ijk ε_ilm = δ^j_l δ^k_m − δ^j_m δ^k_l (3D)."""
    sp = _t.space_3d

    def eps(level, x, y, z):
        el = _level_for(realm, level)
        return _t.levi_civita(realm, sp, [el, el, el], [x, y, z], ctx=ctx)

    def d(a, b):
        return _t.delta(
            realm, sp, _level_for(realm, _U), _level_for(realm, _L), a, b, ctx=ctx
        )

    i, j, k, l, m = (ctx.alloc_index() for _ in range(5))
    return Identity(
        "eps-delta-1",
        _t.explicit_sum(i, eps(_U, i, j, k) * eps(_L, i, l, m), ctx=ctx),
        d(j, l) * d(k, m) - d(j, m) * d(k, l),
    )


def eps_delta_2(ctx, realm=_t.Realm.Oblique):
    """Σ_ij ε^ijk ε_ijl = 2 δ^k_l (3D)."""
    sp = _t.space_3d

    def eps(level, x, y, z):
        el = _level_for(realm, level)
        return _t.levi_civita(realm, sp, [el, el, el], [x, y, z], ctx=ctx)

    i, j, k, l = (ctx.alloc_index() for _ in range(4))
    lhs = _t.explicit_sum(
        i, _t.explicit_sum(j, eps(_U, i, j, k) * eps(_L, i, j, l), ctx=ctx), ctx=ctx
    )
    rhs = _t.scalar(2, ctx=ctx) * _t.delta(
        realm, sp, _level_for(realm, _U), _level_for(realm, _L), k, l, ctx=ctx
    )
    return Identity("eps-delta-2", lhs, rhs)


# ---- cross group: invariant cross-product identities ----------------------


def bac_cab(ctx):
    """a × (b × c) = b (a·c) − c (a·b) — the vector triple product.

    Fires only on a genuine rank-1 triple product: with a rank-2 middle
    operand the crosses reassociate around the fence (vibe 000055) and the
    identity does not hold — see :func:`cross_removal` for that case.
    """
    u, v, w = (_var(ctx, n, 1) for n in "uvw")
    return Identity("bac-cab", u % (v % w), v * (u @ w) - w * (u @ v))


def cross_identity(ctx):
    """a × I = I × a."""
    u = _var(ctx, "u", 1)
    I = _t.identity(ctx=ctx)
    return Identity("cross-identity", u % I, I % u)


def cross_removal(ctx):
    """a × (b × I) = b ⊗ a − (a·b) I — cross removal against I (Zhilin).

    The rank-2 companion of bac-cab, and the motivating case of vibe 000056:
    the inner cross is a vector-with-dyad fence, so bac-cab itself does not
    (and must not) fire.
    """
    u, v = (_var(ctx, n, 1) for n in "uv")
    I = _t.identity(ctx=ctx)
    return Identity("cross-removal", u % (v % I), v * u - (u @ v) * I)


def lagrange(ctx):
    """(a × b) · (c × d) = (a·c)(b·d) − (a·d)(b·c)."""
    p, q, r, s = (_var(ctx, n, 1) for n in "pqrs")
    return Identity(
        "lagrange", (p % q) @ (r % s), (p @ r) * (q @ s) - (p @ s) * (q @ r)
    )


# ---- dyadic group: rank-2 algebra ----------------------------------------


def trace_cyclic(ctx):
    """tr(A · B) = tr(B · A)."""
    a, b = (_var(ctx, n, 2) for n in "UW")
    return Identity("trace-cyclic", _t.tr(a @ b), _t.tr(b @ a))


def identity_dot(ctx):
    """I · a = a."""
    u = _var(ctx, "u", 1)
    return Identity("identity-dot", _t.identity(ctx=ctx) @ u, u)


# ---- groups ---------------------------------------------------------------
#
# Rules that canonicalization already decides are deliberately absent: tr(a⊗b)
# = a·b, vec(a⊗b) = a×b, (a⊗b)ᵀ = b⊗a, the dyad double-dots, (Aᵀ)ᵀ = A,
# a·b = b·a and tr(I) = n are all proved with **zero** rules (vibe 000096
# increment 2).  Shipping them would be inert decoration that looks like
# coverage.  The double-dot-with-I rules are absent for the opposite reason:
# no spelling of them is name-robust until AC chain matching lands.

_GROUPS = {
    "eps_delta": (delta_contraction, delta_trace, eps_delta_1, eps_delta_2),
    "cross": (bac_cab, cross_identity, cross_removal, lagrange),
    "dyadic": (trace_cyclic, identity_dot),
}


def group_names():
    """The names of every group in the library."""
    return list(_GROUPS)


def group(ctx, name, realm=_t.Realm.Oblique, space=None):
    """The rules of one named group; raises ``ValueError`` if unknown.

    ``realm`` / ``space`` parameterize the index-level group (``eps_delta``);
    the invariant groups ignore them.
    """
    try:
        factories = _GROUPS[name]
    except KeyError:
        raise ValueError(
            f"unknown identity group {name!r}; available: {', '.join(_GROUPS)}"
        ) from None
    out = []
    for factory in factories:
        if factory in (delta_contraction, delta_trace):
            out.append(factory(ctx, space=space, realm=realm))
        elif factory in (eps_delta_1, eps_delta_2):
            out.append(factory(ctx, realm=realm))
        else:
            out.append(factory(ctx))
    return out


def all_rules(ctx, realm=_t.Realm.Oblique, space=None):
    """Every group concatenated — for exploration and benchmarking.

    Prefer naming the groups a problem needs: rule count is the main driver
    of saturation cost.
    """
    out = []
    for name in _GROUPS:
        out.extend(group(ctx, name, realm=realm, space=space))
    return out
