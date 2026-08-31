"""tender.explore — a session for user-guided derivations (vibe 000108).

A derivation is a sequence of choices, and the hard part is the choosing:
:mod:`tender.steps` can already say what applies here, but stringing the
answers together by hand means retyping the frame into every call and keeping
the history in your head.  A :class:`Session` holds that history::

    import tender.derivation as td

    s = td.explore(a @ b, gui=False)      # the objects come from your namespace
    print(s.applicable())                 # what bites, and what is not tried
    s.apply("expand_in_basis").apply("reduce_frame")
    print(s.script())                     # the derivation, as code to paste

Three things the session does that a bare sequence of calls does not:

**It fills the extra arguments from your namespace.**  The catalogue records the
*kind* of every extra argument — ``basis``, ``coord``, ``rules``, ``level``,
``op`` — and the scope the session was opened from holds the objects.  So a
``Basis`` sitting in a notebook cell is found and passed to every step that
wants one, and no call repeats it.

**It records the whole tree, and shows one path through it.**  Stepping back and
choosing differently keeps the abandoned branch: it costs nothing to keep, and
:meth:`Session.tried` then says *"you have been here"* even when the same
expression is reached by another route, because the memory is keyed on the
canonical form rather than on the path.

**It emits ordinary tender code.**  Names are how the session and the catalogue
talk; function references are how the emitted script reads, and the argument
names in it are the ones your own namespace uses — which is the same scan that
filled them in the first place.

The interactive surface built on this lives in :mod:`tender.gui`; everything
here works without it, in a plain terminal.
"""

import inspect

from tender import _core
from . import basis as _b
from . import derivation as _d
from . import steps as _ts

__all__ = [
    "Binding",
    "DidNotFire",
    "Node",
    "Session",
    "explore",
    "scan_scope",
    "KINDS",
]


class DidNotFire(Exception):
    """A step chosen in a session changed nothing; the message says why.

    A step that does nothing does not belong in a history, and the reason is
    the step's own (vibe 000106) rather than one inferred from the outside.
    """


# The argument kinds a step can want, and how to recognise an object of that
# kind in a namespace.  Closed list — the same one the catalogue draws `needs`
# from — so a new kind is a deliberate addition in two places, not a guess.
KINDS = ("basis", "coord", "rules", "level", "op")


def _is_coord(v):
    """A coordinate variable is what ``partial`` will differentiate by."""
    try:
        _d.partial(v, v)
    except Exception:
        return False
    return True


def _is_rules(v):
    return (
        isinstance(v, (list, tuple))
        and len(v) > 0
        and all(isinstance(x, _d.Identity) for x in v)
    )


def _is_op(v):
    """An operator is an expression carrying ∂ marks — ``Σ_k c_k ⊗ ∂_{q_k}``."""
    if not isinstance(v, _core.Expr):
        return False
    try:
        return _ts.shape(v)["deriv_marks"] >= 1
    except Exception:
        return False


_RECOGNISE = {
    "basis": lambda v: isinstance(v, _b.Basis),
    "coord": lambda v: isinstance(v, _core.Expr) and _is_coord(v),
    "rules": _is_rules,
    "level": lambda v: isinstance(v, _core.Level),
    "op": _is_op,
}


class Binding:
    """An object from your namespace, the kind it serves, and what you call it.

    The name is not decoration: it is what :meth:`Session.script` writes, so the
    emitted code leans on the preamble rather than on a repr nobody can paste.
    """

    __slots__ = ("kind", "name", "value")

    def __init__(self, kind, name, value):
        self.kind, self.name, self.value = kind, name, value

    def __repr__(self):
        return f"<Binding {self.kind}={self.name}>"


def scan_scope(scope):
    """Every object in *scope* that could serve as a step argument, by kind.

    Returns ``{kind: [Binding, …]}``.  A chart's coordinates are offered under
    their attribute path (``chart.coords[0]``) as well, since a chart is what a
    notebook usually holds rather than the loose coordinates.
    """
    found = {k: [] for k in KINDS}
    seen = {k: set() for k in KINDS}

    def offer(kind, name, value):
        if id(value) in seen[kind]:
            return
        seen[kind].add(id(value))
        found[kind].append(Binding(kind, name, value))

    charts = []
    for name, value in scope.items():
        if name.startswith("_"):
            continue
        if type(value).__name__ == "CoordinateChart":
            charts.append((name, value))
        for kind in KINDS:
            try:
                if _RECOGNISE[kind](value):
                    offer(kind, name, value)
            except Exception:
                continue
    for cname, chart in charts:
        try:
            coords = list(chart.coords)
        except Exception:
            continue
        for i, q in enumerate(coords):
            offer("coord", f"{cname}.coords[{i}]", q)
    return found


def _module_aliases(scope):
    """What the user's namespace calls ``tender.basis`` and friends."""
    alias = {"tender": "tender", "tender.basis": "tb", "tender.derivation": "td"}
    for name, value in scope.items():
        if inspect.ismodule(value) and getattr(value, "__name__", "") in alias:
            alias[value.__name__] = name
    return alias


class Node:
    """One expression in the tree, and the step that produced it."""

    __slots__ = ("expr", "parent", "step", "kwargs", "children")

    def __init__(self, expr, parent=None, step=None, kwargs=None):
        self.expr = expr
        self.parent = parent
        self.step = step
        self.kwargs = dict(kwargs or {})
        self.children = []

    def __repr__(self):
        what = self.step or "start"
        return f"<Node {what}: {self.expr.latex()[:40]}>"


def _key(expr):
    """The visited-set key: the canonical form, so a route back is recognised."""
    try:
        return _d.canonicalize(expr).latex()
    except Exception:
        return expr.latex()


class Session:
    """A derivation in progress: a tree of attempts, and a path through it."""

    def __init__(self, expr, needs=None, scope=None, context=None, name=None):
        """*needs* is ``{kind: object}``, given explicitly; omit it to scan
        *scope* for objects of each kind instead.  *context* is merged over
        either — it is where a keyword argument to :func:`explore` lands.
        """
        self.scope = dict(scope or {})
        self.aliases = _module_aliases(self.scope)
        self.scanned = needs is None
        # Explicit beats inferred, and asking for one excludes the other: a
        # `needs` dict is a statement about what the steps get, so nothing the
        # caller did not name should appear beside it.
        if self.scanned:
            self.bindings = scan_scope(self.scope)
        else:
            self.bindings = {k: [] for k in KINDS}
            for kind, value in needs.items():
                self._offer(kind, value)
        for kind, value in (context or {}).items():
            self._offer(kind, value)
        # One chosen object per kind — the first candidate, which is the only
        # one in the common case; `use` picks another.
        self.context = {k: v[0] for k, v in self.bindings.items() if v}
        for kind, value in (context or {}).items():
            self.context[kind] = self._binding(kind, value)
        self.root = Node(expr)
        self.path = [self.root]
        self.start_name = name or self._name_of(expr, None) or "e"
        self._tried = {}

    def _binding(self, kind, value):
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
        return Binding(kind, self._name_of(value, kind), value)

    def _offer(self, kind, value):
        """Put *value* at the head of the candidates for *kind*."""
        b = self._binding(kind, value)
        rest = [x for x in self.bindings.get(kind, []) if x.value is not value]
        self.bindings[kind] = [b] + rest

    # -- context ----------------------------------------------------------
    def _name_of(self, value, kind):
        for n, v in self.scope.items():
            if v is value and not n.startswith("_"):
                return n
        return kind or "expr"

    @property
    def values(self):
        """The context as the steps see it: ``{kind: object}``."""
        return {k: b.value for k, b in self.context.items()}

    def use(self, kind, value):
        """Choose which object of *kind* the steps get."""
        self._offer(kind, value)
        self.context[kind] = self._binding(kind, value)
        return self

    # -- the path ---------------------------------------------------------
    @property
    def current(self):
        return self.path[-1].expr

    @property
    def steps(self):
        """The derivation as data — ``[(name, kwargs), …]`` — for replay."""
        return [(n.step, dict(n.kwargs)) for n in self.path[1:]]

    def applicable(self, **extra):
        """What applies to the current expression, with the context filled in."""
        return _ts.applicable(self.current, **{**self.values, **extra})

    def why_not(self, step, **extra):
        """Why *step* does nothing here."""
        return _ts.why_not(self.current, step, **{**self.values, **extra})

    def tried(self, expr=None):
        """The step names already tried from *expr* (default: here).

        Keyed on the canonical form, so wandering off and returning by another
        route still reports what was tried.
        """
        return set(self._tried.get(_key(self.current if expr is None else expr), ()))

    def apply(self, step, **extra):
        """Take *step* from here, extending the path.

        Extra keywords override the inferred context — ``target="u"``, say.
        Raises if the step does not fire, because a step that changes nothing
        does not belong in the history.
        """
        name = step if isinstance(step, str) else step.name
        st = _ts.info(name)
        res = st.run(self.current, **{**self.values, **extra})
        self._tried.setdefault(_key(self.current), set()).add(name)
        if not res.fired:
            raise DidNotFire(f"{st.qualified}: {res.reason}")
        kwargs = {k: self.values[k] for k in st.needs if k in self.values}
        kwargs.update({k: v for k, v in extra.items() if k in st.needs + st.options})
        node = Node(res.expr, parent=self.path[-1], step=name, kwargs=kwargs)
        self.path[-1].children.append(node)
        self.path.append(node)
        return self

    def back(self, n=1):
        """Step back up the path; the branch left behind is kept."""
        for _ in range(n):
            if len(self.path) > 1:
                self.path.pop()
        return self

    def goto(self, k):
        """Truncate the shown path to *k* steps (0 = the initial expression)."""
        self.path = self.path[: k + 1]
        return self

    # -- output -----------------------------------------------------------
    # The public module an object of a bound type belongs to, so that an enum
    # is emitted as the user would write it rather than as its bare repr.
    _PUBLIC = {
        "tender._core": "tender",
        "tender._core.basis": "tender.basis",
        "tender._core.derivation": "tender.derivation",
    }

    def _literal(self, value):
        """*value* as source text: your own name for it where there is one."""
        if isinstance(value, (str, bool, int, float)) or value is None:
            return repr(value)
        for name, held in self.scope.items():
            if held is value and not name.startswith("_"):
                return name
        text = repr(value)
        home = self._PUBLIC.get(getattr(type(value), "__module__", ""))
        if home and text.startswith(type(value).__name__ + "."):
            return f"{self.aliases.get(home, home)}.{text}"
        return text

    def _call_text(self, node):
        st = _ts.info(node.step)
        alias = self.aliases.get(st.home, st.home)
        args = ["e"]
        for kind in st.needs:
            if kind in node.kwargs:
                b = self.context.get(kind)
                args.append(
                    b.name
                    if b is not None and b.value is node.kwargs[kind]
                    else self._literal(node.kwargs[kind])
                )
        for kind in st.options:
            if kind in node.kwargs:
                args.append(f"{kind}={self._literal(node.kwargs[kind])}")
        return f"{alias}.{node.step}({', '.join(args)})"

    def script(self):
        """The path so far, as tender code that relies on your preamble."""
        lines = [f"e = {self.start_name}"]
        for node in self.path[1:]:
            lines.append(f"e = {self._call_text(node)}")
        return "\n".join(lines)

    def attempts(self):
        """The whole tree, abandoned branches included."""
        shown = set(id(n) for n in self.path)
        out = []

        def walk(node, depth):
            mark = "→" if id(node) in shown else " "
            label = node.step or "start"
            out.append(f"{mark} {'  ' * depth}{label:28s} {node.expr.latex()[:60]}")
            for child in node.children:
                walk(child, depth + 1)

        walk(self.root, 0)
        return "\n".join(out)

    @property
    def tree(self):
        return self.root

    def __repr__(self):
        return (
            f"<Session {len(self.path) - 1} steps, "
            f"{', '.join(f'{k}={b.name}' for k, b in sorted(self.context.items()))}>"
        )


def _caller_scope(depth):
    frame = inspect.currentframe()
    for _ in range(depth):
        frame = frame.f_back
    scope = dict(frame.f_globals)
    scope.update(frame.f_locals)
    return scope


def explore(
    expr,
    needs=None,
    scope=None,
    gui=None,
    max_height="520px",
    _depth=2,
    **context,
):
    """Open a derivation session on *expr*, and (in a notebook) its surface.

    Say what the steps get, by kind::

        s = td.explore(a @ b, {"basis": frame})     # or basis=frame

    Or leave it out, and the calling scope is searched for an object of each
    kind — convenient in a notebook, where the setup is the cell you already
    wrote, and the reason a frame need not be repeated into every call::

        frame = tb.wcs(ctx)
        s = td.explore(a @ b)      # finds `frame`, passes it to every step

    The two do not mix: a *needs* dict is a statement about what the steps get,
    so nothing you did not name appears beside it.  Keyword arguments
    (``basis=frame``) are the same thing spelled shorter, and win over both.

    *gui* forces the widget on or off; the default builds it when
    ``ipywidgets`` is available and there is a frontend to show it in.  The
    session is returned either way, and outlives the widget — closing the
    output is not a lifecycle event (vibe 000108 §8).  *max_height* is how tall
    the list of steps may grow before it scrolls.
    """
    session = Session(
        expr,
        needs=needs,
        scope=scope if scope is not None else _caller_scope(_depth),
        context=context,
    )
    if gui is False:
        return session
    try:
        from .gui import show
    except ImportError as ex:
        if gui:
            raise
        _no_widget(f"ipywidgets is not installed ({ex})")
        return session
    try:
        show(session, max_height=max_height)
    except Exception as ex:  # noqa: BLE001 - the reason goes to the user
        if gui:
            raise
        _no_widget(str(ex))
    return session


def _no_widget(why):
    """Say why there is no widget, rather than appearing to do nothing.

    A widget needs a browser to draw in, so a terminal gets the session alone —
    which is the whole library, only typed.  Silence would look like a failure;
    the session is not one.
    """
    print(
        f"tender: no derivation widget here — {why}.\n"
        "        The session works without it:\n"
        "          print(s.applicable())    what applies here\n"
        "          s.apply('reduce_frame')  take a step\n"
        "          print(s.why_not('sym'))  why one does nothing\n"
        "          print(s.script())        the derivation, as code"
    )
