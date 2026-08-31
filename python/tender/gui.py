"""tender.gui — the interactive derivation surface (vibe 000108).

A widget for a Jupyter cell, so that choosing the next step in a derivation is
a click rather than a recall exercise::

    import tender.derivation as td
    s = td.explore(a @ b)          # opens the widget, returns the session

What it shows is a :class:`tender.explore.Session`: the expression at every
point of the current path, and under each one a chooser holding the steps that
*actually do something there* — :func:`tender.steps.applicable`, asked of the
expression rather than of memory.  Choosing a step at some point in the list
drops the tail and continues from there; the abandoned branch is kept, and a
step tried before from the same expression is marked ``· tried``.

Steps are shown as ``module.name`` — `chart.expand` is not `derivation.
expand_products`, and the bare word would not say which you were choosing.

Beside each chooser is a **filter**, and it is one lens over all three
categories rather than a search of the list: typing narrows what fires, what was
not tried, and what ran and did nothing.  It matches what is shown, so a module
name reaches a whole module's worth of steps.  Type a name that is not among the
options and you have asked "why not that one?" — so when the pattern leaves a
single non-firing step, its reason appears instead of its name.  The pattern is
a regular expression; a half-typed one falls back to a substring search and
outlines the box rather than complaining.

Three things do not appear, and their absence is the design (vibe 000108 §3):

* **no preamble box** — the session inherits the kernel's namespace, so the
  imports and the variables are already there;
* **no needs fields** — the argument each step wants is found in that namespace
  by kind, and shown once at the top rather than typed per step;
* **no save prompt** — the session is an ordinary Python object that outlives
  the widget, so closing the output decides nothing.

Everything here is assembly: :mod:`tender.steps` supplies the catalogue, what
applies, why something did not, and what a step changed.  The widget adds no
algebra of its own.
"""

import html as _html
import re

import ipywidgets as W
from IPython.display import display

from . import steps as _ts
from .explore import DidNotFire, _shell

__all__ = ["DerivationWidget", "Item", "build", "show"]

_LAYOUT = W.Layout(width="100%")


def _math(latex):
    """The expression, typeset — live DOM rather than a picture (vibe 108 §5)."""
    return W.HTMLMath(
        value=f'<div style="padding:2px 0">$${latex}$$</div>', layout=_LAYOUT
    )


def _tail(qualified):
    """``tender.chart.expand`` → ``chart.expand``.

    The bare name is not enough to choose by: `expand`, `expand_products` and
    `expand_in_basis` are three different moves, and one of them is a plain
    English word.  The module tail is where the difference lives, and it is how
    `describe()` already presents the catalogue.
    """
    return qualified.removeprefix("tender.")


def _named(options, name):
    """The rule called *name* among *options*, or ``None``."""
    return next(
        (r for _, r in options if r is not None and r.name == name), None
    )


def _valid(pattern):
    try:
        re.compile(pattern)
    except re.error:
        return False
    return True


def _matcher(pattern):
    """A step-name test for *pattern* — a regex, or a substring if it is not one.

    A half-typed regex (`contract_(`) is a normal state of a text box, not an
    error to shout about; it falls back to a substring search and the box is
    outlined instead.
    """
    if not pattern:
        return lambda name: True
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return lambda name: pattern.lower() in name.lower()
    return lambda name: rx.search(name) is not None


def _note(text, colour="#666"):
    return W.HTML(
        f'<div style="color:{colour};font-size:90%;padding-left:1.5em">'
        f"{_html.escape(text)}</div>"
    )


class Item:
    """One point on the path, and the widgets that act on it.

    Named rather than indexed: the row's order is a layout decision, and
    nothing else should have to know it.
    """

    __slots__ = (
        "box", "chooser", "filter", "reveal", "target", "note",
        "rules", "rule_note", "rule_row",
    )

    def __init__(self, **parts):
        for name, part in parts.items():
            setattr(self, name, part)


class DerivationWidget:
    """The widget bound to one session.  ``.box`` is the displayable widget."""

    def __init__(self, session, max_height="520px"):
        self.session = session
        # `applicable` runs every catalogued step, so a list of ten items would
        # re-run four hundred of them on each refresh.  A node's expression is
        # immutable, so the answer only depends on it and on the context.
        self._reports = {}
        self.items = []  # one Item per point on the shown path
        self.status = W.HTML()
        self.history = W.VBox(
            layout=W.Layout(width="100%", max_height=max_height, overflow="auto")
        )
        self.code = W.Textarea(
            layout=W.Layout(width="100%", height="140px"),
            disabled=False,
        )
        self._building = False
        self.box = W.VBox(
            [
                self._context_row(),
                self.status,
                self.history,
                self._catalogue_pane(),
                self._why_not_pane(),
                W.HTML("<b>the derivation, as code</b>"),
                self.code,
                _note(
                    "the notebook keeps this code, not the session — run "
                    "s.to_cell() in a cell to put it in one, and the "
                    "derivation re-runs without the widget"
                ),
            ],
            layout=_LAYOUT,
        )
        self.refresh()

    # -- the fixed panes ---------------------------------------------------
    def _context_row(self):
        """What the steps get, by kind — and what nothing was given for."""
        scanned = self.session.scanned
        absent = "— none in scope" if scanned else "— not given"
        cells = []
        for kind, found in sorted(self.session.bindings.items()):
            if cells:
                cells.append(W.HTML("<span>,&nbsp;</span>"))
            if not found:
                # Named anyway: "how do I give a step what it needs?" is a
                # question the panel should answer where it is asked, and an
                # empty kind is the whole answer — put an object of it in the
                # cell, or pass it to `explore`.
                cells.append(
                    W.HTML(
                        f'<code>{kind}</code> <span style="color:#a00">'
                        f"{absent}</span>"
                    )
                )
                continue
            if len(found) == 1:
                cells.append(W.HTML(f"<code>{kind}</code> = {found[0].name}"))
                continue
            dd = W.Dropdown(
                options=[(b.name, b) for b in found],
                value=self.session.context.get(kind, found[0]),
                description=kind,
                layout=W.Layout(width="240px"),
            )

            def choose(change, kind=kind):
                if change["name"] == "value" and not self._building:
                    self.session.use(kind, change["new"].value)
                    self.refresh()

            dd.observe(choose, names="value")
            cells.append(dd)
        return W.VBox(
            [
                W.HBox(cells, layout=W.Layout(flex_flow="row wrap", width="100%")),
                _note(
                    "found in your namespace by kind — name them instead with "
                    "td.explore(expr, {'basis': frame})"
                    if scanned
                    else "as given — omit the dict to search your namespace "
                    "for an object of each kind instead"
                ),
            ],
            layout=_LAYOUT,
        )

    def _catalogue_pane(self):
        """Every step, so that "why not this one?" has something to point at."""
        body = W.HTML(
            f'<pre style="font-size:90%">{_html.escape(_ts.describe())}</pre>'
        )
        pane = W.Accordion(children=[body])
        pane.set_title(0, "all steps")
        pane.selected_index = None
        return pane

    def _why_not_pane(self):
        pick = W.Dropdown(
            options=[("— pick a step —", None)] + [(n, n) for n in _ts.names()],
            value=None,
            layout=W.Layout(width="320px"),
        )
        answer = W.HTML()

        def ask(change):
            if change["name"] != "value" or change["new"] is None:
                return
            answer.value = (
                f'<div style="font-size:90%">'
                f"{_html.escape(self.session.why_not(change['new']))}</div>"
            )

        pick.observe(ask, names="value")
        pane = W.Accordion(children=[W.VBox([pick, answer])])
        pane.set_title(0, "why not …?")
        pane.selected_index = None
        return pane

    # -- the history -------------------------------------------------------
    def _item(self, k):
        session = self.session
        node = session.path[k]
        report = self._report(node)
        tried = session.tried(node.expr)
        chosen = session.path[k + 1].step if k + 1 < len(session.path) else None
        # By name, not by object: a rule rebuilt from the same context is the
        # same rule, and the widget is displaying a derivation rather than
        # comparing pointers.
        taken = (
            session.path[k + 1].kwargs.get("identity")
            if k + 1 < len(session.path)
            else None
        )
        taken_name = taken.name if taken is not None else None

        # (label, name, searchable) — the filter matches what is shown, so
        # typing "chart" narrows to the chart's steps.
        offered = []
        for hit in report:
            shown = _tail(hit.step.qualified)
            label = shown
            if hit.reshapes_only:
                label += "   · reshapes only"
            else:
                label += "   " + ", ".join(
                    f"{key}{v:+d}" for key, v in sorted(hit.change.items())
                )
            if hit.step.name in tried:
                label += "   · tried"
            offered.append((label, hit.step.name, shown))
        if chosen is not None and chosen not in [n for _, n, _ in offered]:
            taken_shown = _tail(_ts.info(chosen).qualified)
            offered.append((f"{taken_shown}   · taken", chosen, taken_shown))

        # A step that wants an identity is not offered like the others: which
        # rule to apply is a question, not a probed move, so it opens a second
        # list rather than being tried behind the scenes (vibe 000108 §11).
        asks = sorted(n for n in _ts.names() if "identity" in _ts.info(n).needs)

        # The other two categories, shown the same way so the filter reaches
        # all three through one pattern.
        blocked = {
            _tail(q): lack
            for q, lack in report.missing.items()
            if q.rsplit(".", 1)[-1] not in asks
        }
        seen = {n for _, n, _ in offered} | set(asks)
        quiet = {
            _tail(_ts.info(n).qualified): n
            for n in sorted(_ts.names())
            if n not in seen
        }
        quiet = {k: v for k, v in quiet.items() if k not in blocked}

        chooser = W.Dropdown(
            options=[("— choose a step —", None)],
            value=None,
            layout=W.Layout(width="440px"),
        )
        note = W.HTML()
        rule_chooser = W.Dropdown(
            options=[("— choose an identity —", None)],
            value=None,
            layout=W.Layout(width="440px"),
        )
        rule_note = W.HTML()
        rule_row = W.VBox(
            [W.HBox([W.HTML("&nbsp;&nbsp;↳&nbsp;"), rule_chooser]), rule_note],
            layout=W.Layout(display="none", width="100%"),
        )
        filt = W.Text(
            placeholder="filter (regex)",
            layout=W.Layout(width="150px"),
        )
        target = W.Text(
            placeholder="target (a name)",
            layout=W.Layout(width="180px", display="none"),
        )
        reveal = W.ToggleButton(
            value=False,
            description="target",
            tooltip="act on one named object only",
            layout=W.Layout(width="80px"),
        )

        def show_for(pattern):
            """Re-aim all three categories at *pattern* — one lens, not three.

            Typing narrows what fires (the chooser), what was not tried, and
            what ran and did nothing — because a name you type and cannot find
            is the moment "why not that one?" gets asked, and the answer should
            be where the question is.
            """
            keep = _matcher(pattern)
            kept = [
                (label, name)
                for label, name, shown in offered
                if keep(shown) or name == chosen
            ]
            asked = [
                (f"{_tail(_ts.info(n).qualified)} …", n)
                for n in asks
                if keep(_tail(_ts.info(n).qualified)) or n == chosen
            ]
            self._building = True
            try:
                chooser.options = (
                    [("— choose a step —", None)]
                    + kept
                    + ([("─" * 24, None)] + asked if asked else [])
                )
                chooser.value = chosen
                if rule_row.layout.display != "none":
                    rule_chooser.options = [
                        ("— choose an identity —", None)
                    ] + [o for o in rule_all if keep(o[1].name)]
                    rule_chooser.value = _named(rule_chooser.options, taken_name)
            finally:
                self._building = False
            filt.layout.border = "" if _valid(pattern) else "1px solid #a00"

            lines = []
            miss = {n: lack for n, lack in blocked.items() if keep(n)}
            if miss:
                lines.append(
                    "not tried: "
                    + "; ".join(
                        f"{n} (needs {', '.join(lack)})"
                        for n, lack in sorted(miss.items())
                    )
                )
            if pattern:
                dead = [shown for shown in quiet if keep(shown)]
                if len(dead) == 1:
                    # Asked of *this* item's expression, not the session's
                    # current one — the two differ whenever you filter above
                    # the working end.
                    lines.append(
                        _ts.why_not(node.expr, quiet[dead[0]], **session.values)
                    )
                elif dead:
                    lines.append("did not fire: " + ", ".join(dead))
                if not kept and not miss and not dead:
                    lines.append("no step matches that")
            note.value = "".join(_note(t).value for t in lines)

        rule_all = []  # (label, Identity) for the rules that fire here

        def open_rules():
            """Probe the rule library — now, because the user just asked.

            The probing is the same as for any step; what is different is when
            it happens.  Every other row was tried behind the scenes; these are
            tried only on request, so a growing rule library costs nothing until
            it is wanted.
            """
            rule_row.layout.display = ""
            if rule_all:
                return
            found = session.identities
            if not found:
                rule_note.value = _note(
                    "no identities to choose from — put a Context (or a rules "
                    "list) in scope, or pass ctx=… to explore",
                    "#a00",
                ).value
                return
            by_name = {r.name: r for r in found}
            probed = _ts.applicable(
                node.expr, steps=_ts.rule_steps(found), **session.values
            )
            for hit in probed:
                label = _tail(hit.step.qualified) + "   " + (
                    ", ".join(f"{k}{v:+d}" for k, v in sorted(hit.change.items()))
                    or "· reshapes only"
                )
                rule_all.append((label, by_name[hit.step.name]))
            missed = sorted(set(by_name) - {r.name for _, r in rule_all})
            lines = []
            if not rule_all:
                lines.append("none of the %d identities match here" % len(found))
            elif missed:
                lines.append("did not match: " + ", ".join(missed))
            rule_note.value = "".join(_note(t).value for t in lines)
            show_for(filt.value)

        def take_rule(change):
            if change["name"] != "value" or self._building:
                return
            if change["new"] is None or change["new"].name == taken_name:
                return
            self._advance(
                k, asks[0], target.value.strip(), identity=change["new"]
            )

        rule_chooser.observe(take_rule, names="value")

        def refilter(change):
            if change["name"] == "value":
                show_for(change["new"])

        filt.observe(refilter, names="value")
        show_for("")
        if chosen in asks:
            # Reading a derivation back: the rule that was taken is shown where
            # it was chosen, not buried in the emitted code.
            open_rules()
            self._building = True
            try:
                rule_chooser.value = _named(rule_chooser.options, taken_name)
            finally:
                self._building = False

        def toggle(change):
            target.layout.display = "" if change["new"] else "none"

        reveal.observe(toggle, names="value")

        def take(change):
            if change["name"] != "value" or self._building:
                return
            name = change["new"]
            if name is None:
                return
            if name in asks:
                open_rules()
                return
            if name == chosen:
                return
            self._advance(k, name, target.value.strip())

        chooser.observe(take, names="value")

        row = [chooser, filt, reveal, target]
        if k:
            drop = W.Button(
                description="↑",
                tooltip="go back to here",
                layout=W.Layout(width="40px"),
            )
            drop.on_click(lambda _b, k=k: self._goto(k))
            row.append(drop)

        box = W.VBox(
            [_math(node.expr.latex()), W.HBox(row), rule_row, note],
            layout=W.Layout(
                border="1px solid #e0e0e0",
                padding="4px",
                margin="2px 0",
                width="100%",
                # A flex child shrinks below its content by default, so a long
                # history squeezed every item until the chooser was unreachable
                # behind a scrollbar of its own.  The list scrolls; the items
                # keep their size.
                flex="0 0 auto",
            ),
        )
        return Item(
            box=box,
            chooser=chooser,
            filter=filt,
            reveal=reveal,
            target=target,
            note=note,
            rules=rule_chooser,
            rule_note=rule_note,
            rule_row=rule_row,
        )

    def _report(self, node):
        session = self.session
        context = tuple(sorted((k, id(b.value)) for k, b in session.context.items()))
        key = (id(node), context)
        if key not in self._reports:
            self._reports[key] = _ts.applicable(node.expr, **session.values)
        return self._reports[key]

    # -- transitions -------------------------------------------------------
    def _advance(self, k, name, target, **extra):
        session = self.session
        session.goto(k)
        if target:
            extra["target"] = target
        try:
            session.apply(name, **extra)
            self.status.value = ""
        except DidNotFire as ex:
            self.status.value = (
                f'<div style="color:#a00;font-size:90%">{_html.escape(str(ex))}</div>'
            )
        except Exception as ex:  # a domain error is an answer too
            self.status.value = (
                f'<div style="color:#a00;font-size:90%">'
                f"{type(ex).__name__}: {_html.escape(str(ex))}</div>"
            )
        self.refresh()

    def _goto(self, k):
        self.session.goto(k)
        self.refresh()

    def refresh(self):
        """Rebuild the list and the code panel from the session."""
        self._building = True
        try:
            self.items = [self._item(k) for k in range(len(self.session.path))]
            self.history.children = tuple(item.box for item in self.items)
            self.code.value = self.session.script()
        finally:
            self._building = False

    def _ipython_display_(self):
        display(self.box)


def _in_kernel():
    """Is there a frontend to show a widget in?"""
    shell = _shell()
    return shell is not None and hasattr(shell, "kernel")


def build(session, max_height="520px"):
    """The widget for *session*, unshown — for embedding in a layout of your own."""
    return DerivationWidget(session, max_height=max_height)


def show(session, max_height="520px"):
    """Build the widget and display it in the current cell.

    Raises ``RuntimeError`` when there is no notebook frontend, so that
    :func:`tender.explore.explore` can fall back to the session alone rather
    than printing a widget repr into a terminal.

    A long derivation eventually wants more room than a cell: raise
    *max_height*, or dock the widget beside the notebook with JupyterLab's
    right-click → *Create New View for Output*, where it does not scroll with
    the cells.
    """
    if not _in_kernel():
        raise RuntimeError("there is no Jupyter frontend to draw it in")
    widget = build(session, max_height=max_height)
    display(widget.box)
    return widget
