"""tender.mechanics — time, generalized coordinates, d/dt and the variation δ.

The first increment group of the applied-mechanics arc (vibe 000093 M5A item 1,
brief in vibe 000110).  Nothing here is a new algebraic mechanism: time is an
ordinary coordinate, a generalized coordinate and its rates are ordinary
coordinate atoms, and both operators are ordinary expressions of the form
``Σ_k c_k ∂_k`` — the *derivation* shape of vibe 000102, which
:func:`tender.derivation.apply_operators` already carries out by Leibniz over
any number of factors::

    d/dt  =  ∂_t + q̇ ∂_q + q̈ ∂_q̇ + …
    δ     =  δq ∂_q + δq̇ ∂_q̇ + …

What this module adds is the *consistency* of that construction, and it is not
a formality.  δ and d/dt commute only if the variations ride the time chain
too — ``d/dt δq = δq̇`` — which means the d/dt operator has to carry the
variation chain as well as the coordinate chain.  Assembled by hand the two
operators come out non-commuting on the first try (measured, vibe 000110); the
factory below owns that invariant so no derivation has to.

    tm = ws.time("t")
    q, qd, qdd = tm.coordinate("q", orders=2)     # q, \\dot{q}, \\ddot{q}
    L = tm.field("L", 0, deps=[q, qd, tm.t])
    ddt, delta = tm.ddt(), tm.variation()

    td.apply_operators(ddt * L)     # ∂_t L + (∂_q L) q̇ + (∂_q̇ L) q̈
    td.apply_operators(delta * L)   # (∂_q L) δq + (∂_q̇ L) δq̇
"""

from . import derivation as _td
from ._core import Rational as _Rational

__all__ = ["Time"]

# The accent per derivative order.  LaTeX (and MathJax) stop at three dots,
# and so does the chain: order 2 is the last a user may ask for, leaving
# \dddot for the closing member below.
_ACCENT = {0: "{b}", 1: "\\dot{{{b}}}", 2: "\\ddot{{{b}}}", 3: "\\dddot{{{b}}}"}

MAX_ORDERS = 2


class Time:
    """Time, the generalized coordinates that move with it, and δ.

    Built by :meth:`tender.Workspace.time`, not directly.  Owns one coordinate
    group (a chart id with no geometry — time and configuration coordinates are
    independent variables, not points of a space with a metric), and hands out
    the two operators built over everything minted so far.
    """

    def __init__(self, ws, name="t"):
        self._ws = ws
        self._chart_id = ws._next_chart_id
        ws._next_chart_id += 1
        self._slot = 0
        #: The time coordinate itself.
        self.t = self._mint(name)
        self._chains = {}  # base name -> [q, q̇, …] incl. the closing member
        # name -> (kind, rank, deps, proper) for the constrained symbols minted
        # on this chain (vibe 000110 I6): their *differentiated* constraints
        # need the dependence, which the context's registry does not record.
        self._constrained = {}
        # (rotation, derivation) -> the named angular velocity, when one was
        # minted (vibe 000110 I7).
        self._axial = {}
        # (rotation, coordinate) -> the named axis of that coordinate's partial
        # rotation (vibe 000110 I8): `∂_c P = ĉ × P`.
        self._axes = {}
        self._vars = {}  # base name -> [δq, δq̇, …], same length

    # ---- minting --------------------------------------------------------

    def _mint(self, name):
        c = self._ws.coordinate(
            name,
            chart_id=self._chart_id,
            slot=self._slot,
            nonspatial=True,
        )
        self._slot += 1
        return c

    def coordinate(self, name, orders=2):
        """Mint a generalized coordinate and its rates: ``q, q̇, q̈``.

        ``name`` is the undotted base (``"q"``, ``"\\phi"``); the rates are its
        decorated names (vibe 000110 I1).  Returns ``orders + 1`` atoms, so
        ``q, qd, qdd = tm.coordinate("q")`` unpacks.  Each member gets a
        variation ``δq``, ``δq̇``, … minted alongside it — see
        :meth:`variation_of`.

        The chain is closed one order beyond what is returned, so that d/dt of
        the last *returned* member is its true successor rather than a silent
        zero.  ``orders`` is capped at 2 because LaTeX runs out of dots.
        """
        if name in self._chains:
            raise ValueError(f"generalized coordinate {name!r} already minted")
        if not 0 <= orders <= MAX_ORDERS:
            raise ValueError(
                f"orders must be between 0 and {MAX_ORDERS} (got {orders}): "
                "the chain is closed one order beyond what it returns, and "
                "LaTeX has no fourth dot"
            )
        names = [_ACCENT[k].format(b=name) for k in range(orders + 2)]
        self._chains[name] = [self._mint(n) for n in names]
        self._vars[name] = [self._mint("\\delta{" + n + "}") for n in names]
        return list(self._chains[name][: orders + 1])

    def variation_of(self, coord):
        """The variation δq of a chain member q (or of a variation: δδq is
        not a thing — this raises for anything not minted as a coordinate)."""
        key = str(coord)
        for base, chain in self._chains.items():
            for k, member in enumerate(chain):
                if str(member) == key:
                    return self._vars[base][k]
        raise ValueError(
            f"{key} is not a generalized coordinate of this time; "
            "variations exist only for coordinates minted by "
            "Time.coordinate()"
        )

    def field(self, name, rank, deps, symmetric=False):
        """A field of the time chain — ``deps`` is *required*.

        A field left to depend on "all coordinates" (the core default) would
        chain through the *variations* too, and ``dL/dt`` would sprout δ terms
        that mean nothing.  So say what it depends on: ``deps=[q, qd, tm.t]``.
        """
        if deps is None:
            raise ValueError(
                "a time-chain field must declare its dependence explicitly "
                "(e.g. deps=[q, qd, tm.t]); an all-coordinates field would "
                "chain through the variations as well, which is never meant. "
                "Use Workspace.field for a field of a spatial chart."
            )
        return self._ws.field(name, rank, deps=deps, symmetric=symmetric)

    def rotation(self, name, deps=None, proper=True):
        """A rotation that *turns*: a rank-2 field, declared orthogonal.

        Both halves are needed and neither is optional.  Without the field
        dependence `d/dt P` is zero; without the constraint the whole spin
        construction collapses, because `∂_t P` and `P` share a name and would
        be two pattern *variables* in one — every rule relating `Ṗ` to `P`
        would bind the same variable twice and never fire (vibe 000110 I6).

        `deps` defaults to time alone, which is the orientation of a body whose
        motion is not yet parameterized.  Pass the generalized coordinates when
        it is — `deps=[q]` — because δ reaches a rotation only through them: a
        rotation that depends on `t` alone has `δP = 0`, and rightly so, since
        the variation varies the configuration and not the clock.
        """
        return self._constrain(name, 2, "orthogonal", deps, proper)

    def unit_field(self, name, deps=None):
        """A unit vector that moves: rank-1 field with `n·n = 1`.

        Its differentiated constraint `n·ṅ = 0` is what
        :meth:`constraint_rules` mints — the same mechanism as the rotation's
        skewness, which is Stepan's observation that the two are one thing.
        """
        return self._constrain(name, 1, "unit", deps, True)

    def _constrain(self, name, rank, kind, deps, proper):
        from . import _core as _c

        deps = [self.t] if deps is None else list(deps)
        self._constrained[name] = (kind, rank, deps, proper)
        return _c.constrained_field(
            name, rank, kind, deps=deps, proper=proper, ctx=self._ws.ctx
        )

    def spin(self, P, operator=None):
        """`D(P)·Pᵀ` — the spin of a rotation under a derivation.

        Skew, and *derivably* so: differentiating `P·Pᵀ = I` gives
        `D(P)·Pᵀ + P·D(P)ᵀ = 0`.  With `operator` omitted the derivation is
        d/dt and the spin is the angular velocity's tensor form; pass
        :meth:`variation` for the virtual rotation's.
        """
        op = self.ddt() if operator is None else operator
        return _td.apply_operators(op * P) @ P.transpose()

    def angular_velocity(self, P, operator=None, name=None):
        """The axial vector of :meth:`spin`: `ω = −½ (D(P)·Pᵀ)_×`.

        The project writes a skew tensor as `w × I`, never as a standalone Ω,
        and `(a × I)_× = −2a` in tender's own conventions — so this factor of
        −½ is the library's, not a textbook's (vibe 000110 M7).

        With a `name`, the vector is minted as a *field* of the same
        coordinates as `P` and the formula is registered as its definition
        (``ws.definition(ω)``), so `ω̇` is one mark rather than the derivative
        of a three-deep expression.  Nothing about the mathematics changes; the
        difference is whether a second derivative stays readable — measured,
        and the reason I7 names it where I6b did not need to.
        """
        half = self._ws.scalar(_Rational(-1, 2))
        formula = half * self.spin(P, operator).vec()
        if name is None:
            return formula
        key = (str(P), "delta" if operator is not None else "dt")
        deps = self._constrained.get(str(P).strip(), (None, None, [self.t], None))[2]
        w = self._ws.field(name, 1, deps=deps)
        self._ws._definitions[name] = (w, formula)
        self._axial[key] = w
        return w

    def poisson(self, P, operator=None):
        """Derive `D(P) = w × P` for a rotation, and hand it back as a rule.

        With `operator` omitted this is `Ṗ = ω × P`, the kinematic content of a
        rotation: the rate of a rotation is its angular velocity crossed into
        it.  Pass :meth:`variation` for `δP = δo × P`, which is the same
        derivation and the reason the virtual rotation needs no machinery of
        its own.

        *Derived, not asserted.*  Three links, each of them the library's own
        work (vibe 000110 I6b):

            w × I   =  D(P)·Pᵀ        the spin is skew, and a skew tensor is
                                      its axial vector crossed into I
            (w × I)·P  =  w × P       `skew-dot-tensor`
            D(P)·Pᵀ·P  =  D(P)        orthogonality

        The returned :class:`~tender.derivation.Identity` is the citable record
        of that, in the shape a later derivation can use — Poisson's formula
        for a moving frame vector follows from it in one step.  Raises if any
        link fails rather than returning a rule that was never proved.
        """
        op = self.ddt() if operator is None else operator
        derivative = _td.apply_operators(op * P)
        key = (str(P), "delta" if operator is not None else "dt")
        named = self._axial.get(key)
        w = self.angular_velocity(P, operator)
        eye = self._ws.identity()
        spin = derivative @ P.transpose()
        rules = (
            _td.rules("rotation", "transpose", "dyadic", ctx=self._ws.ctx)
            + self.constraint_rules()
        )
        by_name = {r.name: r for r in rules}

        # Link 1 is directed rather than saturated, and deliberately so: after
        # `axial-to-skew` the spin's transpose sits inside a *parenthesised*
        # sum, where no rule reaches it (vibe 000100 again), so the sum is
        # distributed before the skewness fires.
        reduced = _td.canonicalize(w % eye)
        # A fixed point, not one pass: `apply_identity` rewrites the first hit,
        # and a rotation of several generalized coordinates has one term per
        # coordinate — each needing `axial-to-skew` and then its own partial
        # skewness (vibe 000110 I8).
        for _ in range(2 * len(rules) + 4):
            previous = reduced
            reduced = _td.apply_identity(reduced, by_name["axial-to-skew"])
            reduced = _td.canonicalize(_td.expand_products(reduced))
            for rule in rules:
                if "-spin-" in rule.name:
                    reduced = _td.apply_identity(reduced, rule)
            reduced = _td.canonicalize(_td.expand_products(reduced))
            if _td.structural_eq(reduced, previous):
                break
        if not _td.algebraic_eq(reduced, spin):
            raise ValueError(
                f"the spin of this rotation does not reduce to its axial "
                f"form: {w} × I gave {reduced}, not {spin}.  Is the symbol "
                "declared orthogonal, and does the derivation reach it?"
            )
        for claim, (lhs, rhs) in {
            "(w × I)·P = w × P": ((w % eye) @ P, w % P),
            "D(P)·Pᵀ·P = D(P)": (spin @ P, derivative),
        }.items():
            if not _td.prove_equal(lhs, rhs, rules).proved:
                raise ValueError(f"link failed: {claim}")
        name = str(P).replace("\\mathbf{", "").replace("}", "")
        # State the rule with the *named* angular velocity when there is one:
        # the fact is the same, and a second derivative of a name is a mark
        # where a second derivative of the formula is a page.
        axial = named if named is not None else w
        return _td.Identity(f"{name}-poisson", derivative, axial % P)

    def reduce(self, expr, extra=(), rounds=16):
        """Reduce *expr* with everything this chain knows, to a fixed point.

        The rules a rotation derivation needs come from three places — the
        `rotation`/`transpose`/`dyadic` groups, the context's own declarations
        (`P·Pᵀ → I` and the transport), and this chain's differentiated
        constraints — and `prove_equal` gathers them for you while a *directed*
        derivation does not.  Pass anything further (a `poisson` rule, say) as
        `extra`.

        Directed rather than saturated for the reason the rotation forms found
        in I5: the reduction has to interleave rewriting with re-expansion, and
        some of what it needs (scalar simplification) is a step, not a rule.
        """
        from . import identities as _ti

        rules = (
            _td.rules("rotation", "transpose", "dyadic", ctx=self._ws.ctx)
            # `a × a = 0` lives in the `cross` group, whose other members
            # (`cross-identity` especially) take a skew tensor out of the shape
            # the rotation rules expect — so it is picked out by name, as the
            # rotation-form verifier does (vibe 000110 I5).
            + [_td.rule("cross-self", self._ws.ctx)]
            + _ti.constraint_rules(self._ws.ctx)
            + self.constraint_rules()
            + list(extra)
        )
        e = _td.canonicalize(_td.expand_products(expr))
        for _ in range(rounds):
            previous = e
            for rule in rules:
                e = _td.apply_identity(e, rule)
            e = _td.canonicalize(_td.expand_products(e))
            if _td.structural_eq(e, previous):
                break
        return e

    def poisson_rules(self, P):
        """Poisson's relation **per coordinate**: `∂_c P = a_c × P`.

        The usable form, and for the same reason the constraints are minted per
        variable: `D(P)` for a rotation of several coordinates is a *sum*, so a
        rule about it has a multi-term left-hand side the matcher cannot
        compile — and even with one coordinate it is a *product* (`δq ∂_q P`),
        which no rule can match inside a contraction chain.  One coordinate at
        a time, each rule a single factor on the left, and the operator-level
        statement is their sum.

        `a_c = −½ (∂_c P·Pᵀ)_×` is the axial vector of that partial spin — the
        **axis about which that coordinate turns the body** — and it is minted
        as a *named* field `ĉ`, with the formula registered as its definition.
        The name is not decoration: the formula contains `∂_c P`, so a rule
        written with it rewrites its own right-hand side, over and over
        (measured: seven times before the reduction was stopped).

        The angular velocity is then `ω = Σ_c ċ ĉ` and the virtual rotation
        `δo = Σ_c δc ĉ`, which is why the two are one construction.
        """
        # Keyed by the declared name, matched structurally rather than by the
        # rendered form: a rank-2 symbol renders bolded.
        from . import _core as _c

        entry = None
        for name, record in self._constrained.items():
            kind, rank, deps, proper = record
            built = _c.constrained_field(
                name, rank, kind, deps=deps, proper=proper, ctx=self._ws.ctx
            )
            if _td.structural_eq(built, P):
                entry = record
                break
        if entry is None:
            raise ValueError(
                f"{P} was not minted on this time chain, so its dependence is "
                "unknown; use tm.rotation(...) to declare a rotation that turns"
            )
        kind, rank, deps, proper = entry
        rules = []
        for coord in deps:
            partial = _td.partial(P, coord)
            if _td.algebraic_eq(partial, self._ws.scalar(0)):
                continue
            formula = self._ws.scalar(_Rational(-1, 2)) * (
                partial @ P.transpose()
            ).vec()
            base = str(coord)
            axis = self._axes.get((str(P), base))
            if axis is None:
                axis = self._ws.field(
                    "\\hat{" + base + "}", 1, deps=deps
                )
                self._ws._definitions["\\hat{" + base + "}"] = (axis, formula)
                self._axes[(str(P), base)] = axis
            tag = base.replace("\\", "")
            rules.append(_td.Identity(f"{tag}-poisson", partial, axis % P))
        return rules

    def coefficients(self, expr):
        """Split an expression *linear in the variations* into their factors.

        The finite-dimensional fundamental lemma, in the only form it needs:
        `δA = Σ_c δc Q_c` with the `δc` arbitrary and independent, so `δA = 0`
        means every `Q_c = 0`.  No integral, no lemma over a domain — for
        finitely many degrees of freedom the conclusion is reached by *equating
        coefficients*, which is why the whole applied-mechanics arc needs no
        integral (vibe 000110, and why vibe 000111 owns that separately).

        Returns ``{variation name: coefficient}``.  A term carrying no
        variation, or two, is an error rather than a silent omission: the first
        means the expression was not a virtual work, the second that it is not
        linear and the lemma does not apply.
        """
        variations = {}
        for chain in self._vars.values():
            for member in chain:
                variations[str(member)] = member
        out = {}
        for path in _td.canonicalize(_td.expand_products(expr)).addends():
            term = _td.canonicalize(_td.expand_products(expr)).at(path)
            found = [
                name for name in variations if term.find(name=name)
            ]
            if len(found) != 1:
                raise ValueError(
                    f"the term {term} carries {len(found)} variations; a "
                    "virtual work must be linear in them, one to a term"
                )
            name = found[0]
            spots = term.find(name=name)
            if len(spots) != 1:
                raise ValueError(
                    f"the term {term} carries {name} more than once, so it is "
                    "not linear in it"
                )
            one = self._ws.scalar(1)
            coefficient = _td.canonicalize(term.replace_at(spots[0], one))
            out[name] = (
                coefficient
                if name not in out
                else _td.canonicalize(out[name] + coefficient)
            )
        return out

    def constraint_rules(self):
        """The *differentiated* constraints of every symbol on this chain.

        Stepan's observation, and the reason these are one mechanism rather
        than two: `n·ṅ = 0` and the skewness of `Ṗ·Pᵀ` are the same statement —
        the derivative of a constraint is a constraint.

        Differentiated **per independent variable**, not per operator, and that
        is not a detail: `d/dt P` for a rotation of two generalized coordinates
        is `q̇ ∂_q P + ṙ ∂_r P`, so a rule about the whole spin is a rule about a
        *sum*, and the moment anything distributes it there is nothing left for
        the rule to match.  Each partial spin is skew in its own right —
        differentiating `P·Pᵀ = I` by one coordinate says so — and a sum of
        skew terms is skew, so the finer statement is both truer and more
        usable (vibe 000110 I8).

            ∂_c(n·n = 1)   ⟹   n·∂_c n = 0
            ∂_c(P·Pᵀ = I)  ⟹   (∂_c P·Pᵀ)ᵀ = −∂_c P·Pᵀ

        The undifferentiated forms come from the context itself and are already
        in force everywhere (vibe 000110 I4); these are what a *moving*
        constrained symbol adds.
        """
        rules = []
        for name, (kind, rank, deps, proper) in self._constrained.items():
            for coord in deps:
                rule = self._differentiated(name, kind, rank, deps, proper, coord)
                if rule is not None:
                    rules.append(rule)
        return rules

    def _differentiated(self, name, kind, rank, deps, proper, coord):
        from . import _core as _c

        symbol = _c.constrained_field(
            name, rank, kind, deps=deps, proper=proper, ctx=self._ws.ctx
        )
        derivative = _td.partial(symbol, coord)
        if _td.algebraic_eq(derivative, self._ws.scalar(0)):
            return None
        tag = str(coord).replace("\\", "")
        if kind == "unit":
            return _td.Identity(
                f"{name}-unit-{tag}", symbol @ derivative, self._ws.scalar(0)
            )
        spin = derivative @ symbol.transpose()
        return _td.Identity(f"{name}-spin-{tag}", spin.transpose(), -spin)

    # ---- the two operators ----------------------------------------------

    def ddt(self):
        """The total time derivative ``d/dt = ∂_t + Σ q̇ ∂_q + Σ δq̇ ∂_δq``.

        Built afresh over everything minted so far, so mint first and take the
        operator at the point of use; an operator taken earlier does not know
        about coordinates minted after it.
        """
        d = _td.deriv
        op = d(self.t)
        for chain in list(self._chains.values()) + list(self._vars.values()):
            for k in range(len(chain) - 1):
                op = op + chain[k + 1] * d(chain[k])
        return op

    def variation(self):
        """The variation ``δ = Σ_k δq_k ∂_{q_k}``, over every chain member.

        A derivation like any other (vibe 000102): ``apply_operators`` carries
        it out, so ``δ(fg) = (δf) g + f (δg)`` needs no rule.  It commutes with
        :meth:`ddt` by construction — the variations are themselves members of
        the time chain there.
        """
        if not self._chains:
            raise ValueError(
                "no generalized coordinates to vary — mint one with "
                "Time.coordinate() first"
            )
        d = _td.deriv
        op = None
        for base, chain in self._chains.items():
            var = self._vars[base]
            for k, member in enumerate(chain):
                term = var[k] * d(member)
                op = term if op is None else op + term
        return op
