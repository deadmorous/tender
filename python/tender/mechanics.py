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
        self._vars = {}  # base name -> [δq, δq̇, …], same length

    # ---- minting --------------------------------------------------------

    def _mint(self, name):
        c = self._ws.coordinate(
            name, chart_id=self._chart_id, slot=self._slot
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
