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

    def angular_velocity(self, P, operator=None):
        """The axial vector of :meth:`spin`: `ω = −½ (D(P)·Pᵀ)_×`.

        The project writes a skew tensor as `w × I`, never as a standalone Ω,
        and `(a × I)_× = −2a` in tender's own conventions — so this factor of
        −½ is the library's, not a textbook's (vibe 000110 M7).
        """
        half = self._ws.scalar(_Rational(-1, 2))
        return half * self.spin(P, operator).vec()

    def constraint_rules(self):
        """The *differentiated* constraints of every symbol on this chain.

        Stepan's observation, and the reason these are one mechanism rather
        than two: `n·ṅ = 0` and the skewness of `Ṗ·Pᵀ` are the same statement —
        the derivative of a constraint is a constraint.  For every derivation
        in play (d/dt, and δ once coordinates are minted):

            D(n·n = 1)   ⟹   n·D(n) = 0
            D(P·Pᵀ = I)  ⟹   (D(P)·Pᵀ)ᵀ = −D(P)·Pᵀ

        The undifferentiated forms come from the context itself and are already
        in force everywhere (vibe 000110 I4); these are what a *moving*
        constrained symbol adds.
        """
        rules = []
        operators = [("dt", self.ddt())]
        if self._chains:
            operators.append(("delta", self.variation()))
        # Only what this Time minted: the context's registry records a name and
        # a kind, not what the symbol *depends on*, and a rule built with the
        # wrong dependence would be about a different object.
        for name, (kind, rank, deps, proper) in self._constrained.items():
            for tag, op in operators:
                rule = self._differentiated(name, kind, rank, deps, proper, tag, op)
                if rule is not None:
                    rules.append(rule)
        return rules

    def _differentiated(self, name, kind, rank, deps, proper, tag, op):
        from . import _core as _c

        symbol = _c.constrained_field(
            name, rank, kind, deps=deps, proper=proper, ctx=self._ws.ctx
        )
        derivative = _td.apply_operators(op * symbol)
        if _td.algebraic_eq(derivative, self._ws.scalar(0)):
            # The derivation does not reach this symbol — δ of a rotation that
            # depends on time alone, say.  A rule about `0` is noise, not a
            # fact.
            return None
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
