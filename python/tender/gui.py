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

import ipywidgets as W
from IPython.display import display

from . import steps as _ts
from .explore import DidNotFire

__all__ = ["DerivationWidget", "build", "show"]

_LAYOUT = W.Layout(width="100%")


def _math(latex):
    """The expression, typeset — live DOM rather than a picture (vibe 108 §5)."""
    return W.HTMLMath(
        value=f'<div style="padding:2px 0">$${latex}$$</div>', layout=_LAYOUT
    )


def _note(text, colour="#666"):
    return W.HTML(
        f'<div style="color:{colour};font-size:90%;padding-left:1.5em">'
        f"{_html.escape(text)}</div>"
    )


class DerivationWidget:
    """The widget bound to one session.  ``.box`` is the displayable widget."""

    def __init__(self, session, max_height="520px"):
        self.session = session
        # `applicable` runs every catalogued step, so a list of ten items would
        # re-run four hundred of them on each refresh.  A node's expression is
        # immutable, so the answer only depends on it and on the context.
        self._reports = {}
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

        options = [("— choose a step —", None)]
        for hit in report:
            label = hit.step.name
            if hit.reshapes_only:
                label += "   · reshapes only"
            else:
                label += "   " + ", ".join(
                    f"{key}{v:+d}" for key, v in sorted(hit.change.items())
                )
            if hit.step.name in tried:
                label += "   · tried"
            options.append((label, hit.step.name))
        if chosen is not None and chosen not in [o[1] for o in options]:
            options.append((f"{chosen}   · taken", chosen))

        chooser = W.Dropdown(
            options=options,
            value=chosen,
            layout=W.Layout(width="460px"),
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

        def toggle(change):
            target.layout.display = "" if change["new"] else "none"

        reveal.observe(toggle, names="value")

        def take(change):
            if change["name"] != "value" or self._building:
                return
            name = change["new"]
            if name is None or name == chosen:
                return
            self._advance(k, name, target.value.strip())

        chooser.observe(take, names="value")

        row = [chooser, reveal, target]
        if k:
            drop = W.Button(
                description="↑",
                tooltip="go back to here",
                layout=W.Layout(width="40px"),
            )
            drop.on_click(lambda _b, k=k: self._goto(k))
            row.append(drop)

        parts = [_math(node.expr.latex()), W.HBox(row)]
        if report.missing:
            parts.append(
                _note(
                    "not tried: "
                    + "; ".join(
                        f"{q.rsplit('.', 1)[-1]} (needs {', '.join(lack)})"
                        for q, lack in sorted(report.missing.items())
                    )
                )
            )
        return W.VBox(
            parts,
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

    def _report(self, node):
        session = self.session
        context = tuple(sorted((k, id(b.value)) for k, b in session.context.items()))
        key = (id(node), context)
        if key not in self._reports:
            self._reports[key] = _ts.applicable(node.expr, **session.values)
        return self._reports[key]

    # -- transitions -------------------------------------------------------
    def _advance(self, k, name, target):
        session = self.session
        session.goto(k)
        extra = {"target": target} if target else {}
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
            self.history.children = tuple(
                self._item(k) for k in range(len(self.session.path))
            )
            self.code.value = self.session.script()
        finally:
            self._building = False

    def _ipython_display_(self):
        display(self.box)


def _in_kernel():
    """Is there a frontend to show a widget in?"""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
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
