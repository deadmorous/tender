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

**Symmetric chains are AC-matched.**  Canon puts a *commutative* binary
contraction's operands in name order (``:``, ``··``, and ``·`` between rank-1
operands), so a pattern variable's own name once decided which targets a rule
fired on — ``X··I = tr X`` matched targets named A–H and silently missed J–Z.
The matcher now tries the swapped order for exactly those chains that canon
may reorder (vibe 000096 increment 3), so rules are name-independent while a
*directional* contraction (``A·b``, ``C:ε``) is still matched strictly.
Fire-test new rules anyway: :func:`tender.derivation.prove_equal` reports
which rules actually fired, so an inert rule is visible rather than
mysterious.
"""

import tender as _t

from .derivation import Identity

__all__ = [
    # the DAG
    "IdentityNode",
    "register",
    "node",
    "names",
    "nodes",
    "ancestors",
    "descendants",
    "citable_for",
    "check_acyclic",
    "depth",
    "rules_for",
    "AXIOM",
    "DERIVED",
    # groups (a labelling over the DAG)
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
    # double_dot
    "ddot_identity",
    # leibniz
    "grad_product",
    "div_cross",
    "curl_curl",
    "div_scaled",
    "curl_scaled",
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


# ---- double_dot group: double contractions -------------------------------


def ddot_identity(ctx):
    """A ·· I = tr A — the identity tensor's double contraction is the trace.

    Name-robust only since AC chain matching landed (vibe 000096 increment
    3): canon puts a commutative binary contraction's operands in name order,
    so before that this rule fired on targets named A–H and silently missed
    J–Z, whatever spelling it was given.
    """
    a = _var(ctx, "B", 2)
    return Identity("ddot-identity", a // _t.identity(ctx=ctx), _t.tr(a))


# ---- dyadic group: rank-2 algebra ----------------------------------------


def trace_cyclic(ctx):
    """tr(A · B) = tr(B · A)."""
    a, b = (_var(ctx, n, 2) for n in "UW")
    return Identity("trace-cyclic", _t.tr(a @ b), _t.tr(b @ a))


def identity_dot(ctx):
    """I · a = a."""
    u = _var(ctx, "u", 1)
    return Identity("identity-dot", _t.identity(ctx=ctx) @ u, u)


# ---- leibniz group: ∇ acting on a product ---------------------------------
#
# These need a *workspace-bound* ∇, so unlike the algebraic rules they build
# their operator from the context.  Only the forms canon can state are here:
# a ⊗-product inside a contraction operand (`∇·(f u)`, `∇×(f u)`) is rejected
# with "awaits fence distribution", which is the same gap that blocks
# challenge 000010 — see vibe 000101.


def _nabla(ctx):
    return _t.nabla(ctx=ctx)


def _field(ctx, name, rank):
    """A field pattern variable: slot-less, so it binds any whole factor."""
    return _t.field(name, rank, ctx=ctx)


def grad_product(ctx):
    """∇(f g) = f ∇g + g ∇f — Leibniz for the gradient of a scalar product."""
    nab = _nabla(ctx)
    f, g = (_field(ctx, n, 0) for n in "FG")
    return Identity("grad-product", nab * (f * g), f * (nab * g) + g * (nab * f))


def div_cross(ctx):
    """∇·(u×v) = v·(∇×u) − u·(∇×v)."""
    nab = _nabla(ctx)
    u, v = (_field(ctx, n, 1) for n in "UV")
    return Identity("div-cross", nab @ (u % v), v @ (nab % u) - u @ (nab % v))


def div_scaled(ctx):
    """∇·(f u) = f (∇·u) + u·∇f — Leibniz for the divergence of a scaled vector.

    Statable only since the fence fix (vibe 000101): the operand `f⊗u` holds
    no ∇ itself, so canon used to reject it, even though floating the scalar
    out is precisely the error this rule exists to correct.
    """
    nab = _nabla(ctx)
    f = _field(ctx, "F", 0)
    u = _field(ctx, "U", 1)
    return Identity("div-scaled", nab @ (f * u), f * (nab @ u) + u @ (nab * f))


def curl_scaled(ctx):
    """∇×(f u) = f (∇×u) − u×∇f."""
    nab = _nabla(ctx)
    f = _field(ctx, "F", 0)
    u = _field(ctx, "U", 1)
    return Identity("curl-scaled", nab % (f * u), f * (nab % u) - u % (nab * f))


def curl_curl(ctx):
    """∇×(∇×u) = ∇(∇·u) − Δu — the double-curl identity."""
    nab = _nabla(ctx)
    u = _field(ctx, "U", 1)
    return Identity("curl-curl", nab % (nab % u), nab * (nab @ u) - nab @ (nab * u))


# ---- the identity DAG -----------------------------------------------------
#
# An identity is normally *derived*, and its derivation is exactly what a
# challenge is; a derivation in turn uses other, already-derived identities.
# Those uses are dependencies, and the dependency relation makes a directed
# acyclic graph (vibe 000097).
#
# The graph lives here, in the library, because it is what a *user* needs:
# which identities exist, what each rests on, and which ones may legitimately
# be cited when deriving something new.  It records each node's proof
# obligation as inert data — a challenge id — and never imports the challenge
# suite; the development harness reads this registry and checks that the
# obligations are met.  So the library ships the graph, the suite satisfies it,
# and neither depends on the other's internals.
#
# `kind` distinguishes the two ways a node earns its place:
#
#   "axiom"    a defining property, taken as given (Σ_p δ^p_a δ^p_b = δ_ab
#              *is* what δ means).  No proof obligation.
#   "derived"  a theorem, which owes a derivation.  `proof` names the
#              challenge that supplies it, and `cites` the identities that
#              derivation is allowed to lean on.
#
# A derivation may cite exactly its node's *ancestors* — never the node
# itself, never anything downstream — which is what makes circularity a
# detectable bug instead of a matter of judgement.  Note that a challenge's
# L1 test (reduction to components) is a proof *from definitions*: it needs no
# citations at all, which is why several nodes below are `derived` with an
# empty `cites`.

class IdentityNode:
    """One identity in the DAG: how to build it, and what it rests on."""

    __slots__ = ("name", "factory", "kind", "cites", "summary", "proof", "tags")

    def __init__(self, name, factory, kind, cites, summary, proof, tags):
        self.name = name
        self.factory = factory
        self.kind = kind
        self.cites = tuple(cites)
        self.summary = summary
        self.proof = proof
        self.tags = tuple(tags)

    def __repr__(self):
        return (
            f"IdentityNode({self.name!r}, kind={self.kind!r}, "
            f"cites={list(self.cites)}, proof={self.proof!r})"
        )


_NODES = {}

AXIOM = "axiom"
DERIVED = "derived"


def register(
    name, factory, *, kind=DERIVED, cites=(), summary="", proof=None, tags=()
):
    """Add an identity to the DAG — yours are first-class alongside the shipped ones.

    ``factory`` is ``ctx -> Identity``; ``cites`` names the identities its
    derivation leans on (they must already be registered, which is what makes
    a cycle impossible to create); ``proof`` optionally references the
    challenge that derives it.  Re-registering a name replaces it.

        >>> ti.register("my-rule", lambda ctx: td.Identity(...),
        ...             cites=["bac-cab"], summary="…")
    """
    if kind not in (AXIOM, DERIVED):
        raise ValueError(f"kind must be {AXIOM!r} or {DERIVED!r}, got {kind!r}")
    for dep in cites:
        if dep not in _NODES:
            raise ValueError(
                f"identity {name!r} cites {dep!r}, which is not registered "
                f"(register dependencies first — this is what keeps the graph "
                f"acyclic by construction)"
            )
    if kind == AXIOM and cites:
        raise ValueError(f"axiom {name!r} cannot cite anything: {list(cites)}")
    node = IdentityNode(name, factory, kind, cites, summary, proof, tags)
    _NODES[name] = node
    return node


def node(name):
    """The :class:`IdentityNode` called *name*; raises ``ValueError`` if unknown."""
    try:
        return _NODES[name]
    except KeyError:
        raise ValueError(
            f"unknown identity {name!r}; registered: {', '.join(sorted(_NODES))}"
        ) from None


def names():
    """Every registered identity name, in registration (dependency) order."""
    return list(_NODES)


def ancestors(name):
    """Every identity *name* transitively rests on — what its proof may cite."""
    seen = set()
    stack = list(node(name).cites)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(node(current).cites)
    return seen


def descendants(name):
    """Every identity that transitively rests on *name*."""
    return {n for n in _NODES if name in ancestors(n)}


def check_acyclic():
    """Raise ``ValueError`` if the DAG has a cycle.

    Registration already prevents one (a node may only cite names already
    present), so this is a belt-and-braces check for a registry mutated by
    other means.
    """
    for name in _NODES:
        if name in ancestors(name):
            raise ValueError(f"identity {name!r} transitively cites itself")


def depth(name):
    """How far *name* stands above the axioms — 0 for an axiom or a node
    derived straight from definitions."""
    cites = node(name).cites
    return 0 if not cites else 1 + max(depth(c) for c in cites)


def _build(n, ctx, realm, space):
    """Instantiate one node's Identity in `ctx`."""
    factory = n.factory
    if factory in (delta_contraction, delta_trace):
        return factory(ctx, space=space, realm=realm)
    if factory in (eps_delta_1, eps_delta_2):
        return factory(ctx, realm=realm)
    return factory(ctx)


def rules_for(ctx, *identity_names, realm=_t.Realm.Oblique, space=None):
    """Build the named identities as rules, ready for the verbs."""
    return [_build(node(n), ctx, realm, space) for n in identity_names]


def citable_for(ctx, name, realm=_t.Realm.Oblique, space=None):
    """The rules a derivation of *name* may legitimately use: its ancestors.

    Deriving an identity from itself — or from anything that already rests on
    it — is circular, so those are exactly the rules left out.  Passing this
    to :func:`tender.derivation.prove_equal` makes a proof honest by
    construction rather than by review.
    """
    order = [n for n in _NODES if n in ancestors(name)]
    return rules_for(ctx, *order, realm=realm, space=space)


# ---- groups: a labelling over the DAG, for convenient selection -----------
#
# Rules that canonicalization already decides are deliberately absent: tr(a⊗b)
# = a·b, vec(a⊗b) = a×b, (a⊗b)ᵀ = b⊗a, the dyad double-dots, (Aᵀ)ᵀ = A,
# a·b = b·a and tr(I) = n are all proved with **zero** rules (vibe 000096
# increment 2).  Shipping them would be inert decoration that looks like
# coverage.  `A··(b⊗c) = c·A·b` is absent for a different reason: canon cannot
# yet *state* it — a nested ⊗ inside a contraction operand throws "awaits
# fence distribution" (vibe 000096 increment 3, still open).


def group_names():
    """The names of every group (tag) in the library."""
    seen = []
    for n in _NODES.values():
        for tag in n.tags:
            if tag not in seen:
                seen.append(tag)
    return seen


def group(ctx, name, realm=_t.Realm.Oblique, space=None):
    """The rules tagged *name*; raises ``ValueError`` if the tag is unknown."""
    tagged = [n.name for n in _NODES.values() if name in n.tags]
    if not tagged:
        raise ValueError(
            f"unknown identity group {name!r}; available: "
            f"{', '.join(group_names())}"
        )
    return rules_for(ctx, *tagged, realm=realm, space=space)


def all_rules(ctx, realm=_t.Realm.Oblique, space=None):
    """Every registered identity — for exploration and benchmarking.

    Prefer naming the groups a problem needs, or :func:`citable_for` when
    deriving: rule count is the main driver of saturation cost.
    """
    return rules_for(ctx, *_NODES, realm=realm, space=space)


# ---- the shipped graph ----------------------------------------------------
#
# Registration order is dependency order: a node may only cite names already
# registered, so the graph cannot be given a cycle.

register(
    "delta-contraction", delta_contraction, kind=AXIOM, tags=("eps_delta",),
    summary="Σ_p δ^p_a δ^p_b = δ_ab — the defining property of δ",
)
register(
    "delta-trace", delta_trace, kind=AXIOM, tags=("eps_delta",),
    summary="Σ_p δ^p_p = dim(space)",
)
register(
    "identity-dot", identity_dot, kind=AXIOM, tags=("dyadic",),
    summary="I · a = a — the defining property of the identity tensor",
)
register(
    "eps-delta-1", eps_delta_1, tags=("eps_delta",), proof="000003",
    summary="Σ_i ε^ijk ε_ilm = δ^j_l δ^k_m − δ^j_m δ^k_l",
)
register(
    "eps-delta-2", eps_delta_2, tags=("eps_delta",), proof="000003",
    summary="Σ_ij ε^ijk ε_ijl = 2 δ^k_l",
)
register(
    "bac-cab", bac_cab, tags=("cross",), proof="000001",
    cites=("eps-delta-1",),
    summary="a × (b × c) = b (a·c) − c (a·b)",
)
register(
    "lagrange", lagrange, tags=("cross",), proof="000014",
    cites=("eps-delta-1", "delta-contraction"),
    summary="(a × b) · (c × d) = (a·c)(b·d) − (a·d)(b·c)",
)
register(
    "cross-identity", cross_identity, tags=("cross",), proof="000005",
    summary="a × I = I × a",
)
register(
    "cross-removal", cross_removal, tags=("cross",), proof="000015",
    summary="a × (b × I) = b ⊗ a − (a·b) I",
)
register(
    "trace-cyclic", trace_cyclic, tags=("dyadic",), proof="000016",
    summary="tr(A · B) = tr(B · A)",
)
register(
    "grad-product", grad_product, tags=("leibniz",), proof="000013",
    summary="∇(f g) = f ∇g + g ∇f",
)
register(
    "div-cross", div_cross, tags=("leibniz",), proof="000013",
    summary="∇·(u×v) = v·(∇×u) − u·(∇×v)",
)
register(
    "div-scaled", div_scaled, tags=("leibniz",), proof="000013",
    summary="∇·(f u) = f (∇·u) + u·∇f",
)
register(
    "curl-scaled", curl_scaled, tags=("leibniz",), proof="000013",
    summary="∇×(f u) = f (∇×u) − u×∇f",
)
register(
    "curl-curl", curl_curl, tags=("leibniz",), proof="000012",
    summary="∇×(∇×u) = ∇(∇·u) − Δu",
)
register(
    "ddot-identity", ddot_identity, tags=("double_dot",), proof="000010",
    summary="A ·· I = tr A",
)


def nodes():
    """Every :class:`IdentityNode`, in registration (dependency) order."""
    return list(_NODES.values())


check_acyclic()
