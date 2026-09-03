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


def cross_self(ctx):
    """a × a = 0.

    Canon knows the antisymmetry — `a×b + b×a` folds to 0 — but not its
    degenerate case, because the canonical ordering of a cross has nothing to
    swap when the two operands are already equal.  So the fact is a rule, and
    it is not decoration: without it the turn tensor's `P·Pᵀ` reduced to
    `I + (…)(n × n)` and stopped one step from the answer (vibe 000110 I5).
    """
    u = _var(ctx, "u", 1)
    return Identity("cross-self", u % u, _t.scalar(0, ctx=ctx))


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
    """I · X = X, at any rank.

    The variable is deliberately *unranked*: `I·a = a` for a vector and
    `I·A = A` for a tensor are the same defining property, and a rank-1 gate
    left `P·I·Pᵀ` stuck one step from `I` (vibe 000110 I4b).
    """
    u = _var(ctx, "u", None)
    return Identity("identity-dot", _t.identity(ctx=ctx) @ u, u)


def identity_dot_right(ctx):
    """X · I = X, at any rank — the companion of :func:`identity_dot`.

    Canon does not commute a contraction chain, so `I·X` and `X·I` are two
    shapes and one rule cannot cover both.  Missing it left the reflection
    tensor `(I − 2n⊗n)` stuck one step from `I`, on the term `n·I`
    (vibe 000110 I5).
    """
    u = _var(ctx, "u", None)
    return Identity("identity-dot-right", u @ _t.identity(ctx=ctx), u)


# ---- transpose group: moving a transpose through the algebra --------------
#
# The five facts a derivation about orthogonal tensors leans on constantly
# (vibe 000110 I5/I6), and the reason they were missing was invisible until
# I0: every one of them used to come back `refuted`, so the gap read as a
# wrong answer rather than as an absent rule.
#
# Two neighbours are deliberately *not* here: `(a⊗b)ᵀ = b⊗a` and
# `(A+B)ᵀ = Aᵀ+Bᵀ` are decided by canon with no rules at all (measured), so
# registering them would add saturation cost and never fire.


def transpose_product(ctx):
    """(A · B)ᵀ = Bᵀ · Aᵀ — the workhorse; conjugation is built from it."""
    a, b = (_var(ctx, n, 2) for n in "UW")
    return Identity(
        "transpose-product", (a @ b).transpose(), b.transpose() @ a.transpose()
    )


def transpose_trace(ctx):
    """tr(Aᵀ) = tr(A)."""
    a = _var(ctx, "U", 2)
    return Identity("transpose-trace", _t.tr(a.transpose()), _t.tr(a))


def transpose_adjoint(ctx):
    """(A · u) · v = u · (Aᵀ · v) — the defining property, as a move.

    What every orthogonality argument runs on: it is how `(P·a)·(P·b) = a·b`
    reaches `a·(Pᵀ·P)·b` and thence `a·b`.
    """
    a = _var(ctx, "U", 2)
    u, v = (_var(ctx, n, 1) for n in "uw")
    return Identity("transpose-adjoint", (a @ u) @ v, u @ (a.transpose() @ v))


def transpose_dot_left(ctx):
    """a · Aᵀ = A · a — a vector crosses a transposed tensor by dropping it."""
    a = _var(ctx, "U", 2)
    u = _var(ctx, "u", 1)
    return Identity("transpose-dot-left", u @ a.transpose(), a @ u)


def transpose_vec(ctx):
    """vec(Aᵀ) = −vec(A); a symmetric tensor therefore has none.

    With `(a⊗b)ᵀ = b⊗a` and `vec(a⊗b) = a×b`, transposing reverses every
    dyad's cross product.  Wanted by the axial-vector bridge, where a skew
    tensor is exactly one that equals minus its transpose.
    """
    a = _var(ctx, "U", 2)
    return Identity("transpose-vec", _t.vec(a.transpose()), -_t.vec(a))


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


# ---- rotation group: the skew tensor `a × I` and its axial vector ---------
#
# A skew tensor is written `a × I` here, never as a standalone Ω (vibe 000110
# I6), so these four are how an angular velocity is manipulated at all: they
# are what turns `Ṗ·Pᵀ` into `ω × P`.  All four were measured true and absent
# before being written down — the component procedure agreed, and no rule
# fired.


def skew_transpose(ctx):
    """(a × I)ᵀ = −(a × I) — the skewness, as a rewrite."""
    u = _var(ctx, "u", 1)
    return Identity(
        "skew-transpose", (u % _t.identity(ctx=ctx)).transpose(),
        -(u % _t.identity(ctx=ctx)),
    )


def skew_dot(ctx):
    """(a × I)·b = a × b — the axial vector acting to the right."""
    u, v = (_var(ctx, n, 1) for n in "uw")
    return Identity("skew-dot", (u % _t.identity(ctx=ctx)) @ v, u % v)


def skew_dot_left(ctx):
    """b·(a × I) = b × a — acting to the left, hence the reversed cross.

    Not a restatement of :func:`skew_dot`: canon does not commute a
    contraction, so the two sides of a skew tensor are two shapes.  The sign
    follows from skewness — `b·T = Tᵀ·b = −T·b`.
    """
    u, v = (_var(ctx, n, 1) for n in "uw")
    return Identity("skew-dot-left", v @ (u % _t.identity(ctx=ctx)), v % u)


def skew_decomposition(ctx):
    """½(A − Aᵀ) = −½ (A_×) × I — the skew part *is* the vector invariant.

    Unconditional, for every rank-2 A, which is what makes it useful: a skew
    tensor is its own skew part, so `S = −½ (S_×) × I` follows without a
    skewness hypothesis to encode.  That is the step from "the spin is skew" to
    "the spin is ω × I", and hence to `Ṗ = ω × P` (vibe 000110 I6b).

    The −½ is the library's own convention, not a textbook's: `(a × I)_× = −2a`
    here, measured on a concrete vector.
    """
    a = _var(ctx, "U", 2)
    half = _t.scalar(_t.Rational(1, 2), ctx=ctx)
    eye = _t.identity(ctx=ctx)
    return Identity(
        "skew-decomposition",
        (a - a.transpose()) * half,
        (_t.scalar(_t.Rational(-1, 2), ctx=ctx) * _t.vec(a)) % eye,
    )


def axial_to_skew(ctx):
    """(−½ A_×) × I = ½(A − Aᵀ) — :func:`skew_decomposition` read the other way.

    One theorem, two directed rules, because a rewrite library has directions
    and both are wanted: the first *extracts* an axial vector from a tensor,
    this one *consumes* one.  It is the step that turns `ω × I` back into the
    spin it came from, and hence the one that gets `Ṗ = ω × P` (vibe 000110
    I6b).
    """
    a = _var(ctx, "U", 2)
    half = _t.scalar(_t.Rational(1, 2), ctx=ctx)
    return Identity(
        "axial-to-skew",
        (_t.scalar(_t.Rational(-1, 2), ctx=ctx) * _t.vec(a))
        % _t.identity(ctx=ctx),
        (a - a.transpose()) * half,
    )


def skew_dot_tensor(ctx):
    """(a × I)·B = a × B for a rank-2 B — the rank-2 form of :func:`skew_dot`.

    Two rules rather than one because the rank matters to the matcher, and it
    matters to the mathematics too: the rank-1 form ends a chain, this one
    continues it.
    """
    u = _var(ctx, "u", 1)
    b = _var(ctx, "W", 2)
    return Identity("skew-dot-tensor", (u % _t.identity(ctx=ctx)) @ b, u % b)


def cross_dot_assoc(ctx):
    """(a × B)·c = a × (B·c), for a *rank-2* B.

    Canon flattens a chain of one operator — `(A·B)·c` and `A·(B·c)` are one
    form — but not across two, and rightly: with a rank-1 middle operand the
    two groupings are not both well-formed, since `b·c` is a scalar and `a × `
    a scalar is nothing.  Associativity here is rank-conditional, which a
    flattening keyed on the operator cannot decide, so it is a rule (vibe
    000110 I6b).
    """
    u, w = (_var(ctx, n, 1) for n in "uw")
    b = _var(ctx, "W", 2)
    return Identity("cross-dot-assoc", (u % b) @ w, u % (b @ w))


def skew_product(ctx):
    """(a × I)·(b × I) = b⊗a − (a·b) I.

    The composition of two skew tensors, and the identity behind the
    commutator that relates δω to the virtual rotation.
    """
    u, v = (_var(ctx, n, 1) for n in "uw")
    eye = _t.identity(ctx=ctx)
    return Identity(
        "skew-product",
        (u % eye) @ (v % eye),
        v * u - (u @ v) * eye,
    )


# ---- constrained symbols: rules a *context* declares (vibe 000110 I4) -----


def constraint_rules(ctx):
    """The rules the declared constraints of *ctx* amount to.

    A constraint is not an optional fact about a problem — it is what the
    symbol *means* — so these are supplied to the verbs automatically rather
    than passed by hand.  Declaring nothing costs nothing.

        ws.rotation("P")      →   P·Pᵀ → I,  Pᵀ·P → I
        ws.vector("n", unit=True)  →   n·n → 1

    Per symbol, rather than one schema over a property-restricted pattern
    variable: a derivation has a handful of rotations, and this needs no
    matcher work.  Composition needs no rule of its own — `(P·Q)·(P·Q)ᵀ`
    reaches `I` through these plus the `transpose` group.
    """
    out = []
    for name, kind, proper in ctx.constrained_symbols():
        # Built through the constrained factory, not `tensor`: the rule's atom
        # must carry the constraint trait, or the matcher reads it as a pattern
        # variable and the rule becomes "for any X, X·Xᵀ = I" (measured — it
        # proved the orthogonality of every tensor in sight).
        if kind == "unit":
            n = _t.constrained_tensor(name, 1, "unit", ctx=ctx)
            out.append(
                Identity(f"{name}-unit", n @ n, _t.scalar(1, ctx=ctx))
            )
        elif kind == "orthogonal":
            p = _t.constrained_tensor(
                name, 2, "orthogonal", proper=proper, ctx=ctx
            )
            eye = _t.identity(ctx=ctx)
            out.append(
                Identity(f"{name}-orthogonal", p @ p.transpose(), eye)
            )
            out.append(
                Identity(
                    f"{name}-orthogonal-T", p.transpose() @ p, eye
                )
            )
    return out


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
    "cross-self", cross_self, tags=("cross",), proof="000031",
    summary="a × a = 0 — the degenerate case of antisymmetry",
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
    "identity-dot-right", identity_dot_right, kind=AXIOM, tags=("dyadic",),
    summary="a · I = a — the identity tensor on the right",
)
register(
    "transpose-product", transpose_product, tags=("dyadic", "transpose"),
    proof="000029",
    summary="(A·B)ᵀ = Bᵀ·Aᵀ — transposing reverses a contraction chain",
)
register(
    "transpose-trace", transpose_trace, tags=("dyadic", "transpose"),
    proof="000029",
    summary="tr(Aᵀ) = tr(A)",
)
register(
    "transpose-adjoint", transpose_adjoint, tags=("dyadic", "transpose"),
    proof="000029",
    summary="(A·u)·v = u·(Aᵀ·v) — the defining property of the transpose",
)
register(
    "transpose-dot-left", transpose_dot_left, tags=("dyadic", "transpose"),
    proof="000029",
    summary="a·Aᵀ = A·a",
)
register(
    "transpose-vec", transpose_vec, tags=("dyadic", "transpose"),
    proof="000029",
    summary="vec(Aᵀ) = −vec(A)",
)
register(
    "skew-transpose", skew_transpose, tags=("rotation",), proof="000031",
    summary="(a × I)ᵀ = −(a × I) — a skew tensor is minus its transpose",
)
register(
    "skew-dot", skew_dot, tags=("rotation",), proof="000031",
    summary="(a × I)·b = a × b — the axial vector acting to the right",
)
register(
    "skew-dot-left", skew_dot_left, tags=("rotation",), proof="000031",
    summary="b·(a × I) = b × a — acting to the left",
)
register(
    "skew-product", skew_product, tags=("rotation",), proof="000031",
    summary="(a × I)·(b × I) = b⊗a − (a·b) I",
)
register(
    "skew-decomposition", skew_decomposition, tags=("rotation",),
    proof="000033",
    summary="½(A − Aᵀ) = −½ (A_×) × I — the skew part is the vector invariant",
)
register(
    "axial-to-skew", axial_to_skew, tags=("rotation",), proof="000033",
    summary="(−½ A_×) × I = ½(A − Aᵀ) — the decomposition, consumed",
)
register(
    "skew-dot-tensor", skew_dot_tensor, tags=("rotation",), proof="000033",
    summary="(a × I)·B = a × B for rank-2 B",
)
register(
    "cross-dot-assoc", cross_dot_assoc, tags=("rotation",), proof="000033",
    summary="(a × B)·c = a × (B·c) for rank-2 B",
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
