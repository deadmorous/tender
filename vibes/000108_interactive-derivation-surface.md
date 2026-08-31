# 000108 An interactive derivation surface

A GUI for user-guided derivations, so that choosing the next step is a click
rather than a recall exercise.  The design is the user's; this note records it,
settles the technology, and works through the points that came out of the
discussion.

Two motivations, and the second is easy to miss:

1. **Speed.**  Even with vibe 000106's catalogue and feedback functions, a
   derivation is typed one call at a time, with the frame re-passed to every one.
2. **It is the instrument vibe 000107 is waiting for.**  That vibe is postponed
   for want of data on how a user *states what they want*, and a session that
   records what was tried, abandoned and aimed at is exactly where that data
   comes from.  Building this first is what unblocks that.

## 1. The GUI, as specified

- **Preamble** — a multi-line text box for setup code, pasted from a notebook:
  imports, and the variables the initial expression depends on.  Raw text, no
  completion, in version 1.  Changing it invalidates the history, which then
  re-evaluates what it can.
- **Derivation history** — linear in version 1 (no branch memory):
  - a line edit for the initial expression;
  - a vertical list of steps, each showing:
    - the **expression, rendered** (an image in version 1; selecting a target
      with the mouse comes later),
    - a **target** line edit (empty = no target),
    - a **needs** line edit, usually naming a preamble variable; a new item
      inherits the previous item's value,
    - a **chooser of applicable steps** — a drop-down with a "nothing selected"
      default, or a row of radio-style buttons when the list is short.
  - Selecting a step drops any history past that item and appends the result.
- **A panel of Python code** representing the current derivation, copyable.

Rationale for the fixed-panel shape, in the user's words: a long history of
notebook cells is *"just too much text for a human, and things user wants to
keep looking at will scroll up."*

## 2. Technology: a Jupyter cell, with `ipywidgets`

Everything needed is already installed — `ipywidgets` 8.1.7, `matplotlib`
3.10.8, JupyterLab 4.4 — and the project already maintains ten notebooks under
a CI cleanliness job, so this is the existing surface rather than a new one.

Three findings settled it:

**Rich HTML alone cannot do this.**  An earlier suggestion in the discussion —
"start with `_repr_html_`, zero dependencies" — is wrong for *this* design.
Choosing a step must call Python to compute the next expression, and HTML in an
output cell has **no channel back to the kernel**.  What pandas shows is static
styled HTML; interactive table libraries sort in JavaScript and never touch the
kernel.  `ipywidgets` exists precisely for the round trip, so it is required
from version 1.

**Fixed panels work, and the scrolling complaint is answered.**  A cell's output
is a bounded box: a `VBox` with `height` and `overflow: auto` scrolls *inside
itself*, not the notebook.  For a genuinely detached panel, JupyterLab's
right-click → *Create New View for Output* docks the widget beside the notebook,
where it never scrolls with the cells.  There is no native modal, and none is
needed: OK/Cancel is a working copy plus two buttons.

**Rendering to an image needs no LaTeX install.**  Measured: `matplotlib`
mathtext renders tender's output as-is — `\varepsilon`, `\mathbf`, `\partial`,
`\frac`, `\mathsf{T}` all included — on four representative expressions from the
corpus, 4/4.  So version 1's "just an image" is a solved problem with a library
already present.

*Escape hatch, if the cell ever feels too cramped:* Panel runs the same code both
as a notebook widget and as a standalone page.  Bigger dependency, not installed,
but the migration would not be a rewrite.

## 3. What the discussion changed

### The preamble box is redundant when launched from a notebook

If the widget is opened from a cell — `td.explore(expr)` — it inherits the
kernel's namespace, which already holds the imports and the variables.  There is
nothing to paste.  The preamble box exists in the sketch because a standalone web
app would need it; choosing Jupyter deletes it.

That is worth noting as an argument for Jupyter *that only appears after the
choice is made*: the same decision removes a panel, a text field, and the
invalidation rule attached to it.  Re-running the launching cell is the
invalidation.

### Needs should be inferred from the launching scope, not typed

The catalogue records the **kind** of every extra argument (`basis`, `coord`,
`rules`, `level`, `op`), and the launching scope holds the objects.  So the
surface can look for a `Basis` in the namespace and fill `basis=` itself:
zero clicks where there is exactly one candidate, a small chooser where there
are several.

This serves the user's constraint directly — *"reduce clicking, limiting setup
to a single copy-pasting… clicking every field creates overhead"* — and it is
only possible because vibe 000106 made the argument kinds data rather than lore.

### Steps blocked for want of a need are a *third* category

The user's point, and it is right: a step whose needs are unmet has not been
tried, so nothing is known about it — and the user may want to supply the
missing thing precisely to find out.  So the chooser wants three groups, not
two:

| group | shown as |
|---|---|
| **fires** — changes the content | the selectable options |
| **not tried** — a need is unmet | listed with *which* needs are missing, so supplying one is an obvious next move |
| **did not fire** | available for `why_not` on demand |

`applicable` already returns a `missing` map, but it is keyed by *kind* and
records only the first unmet kind per step.  For this it should be keyed by
**step**, carrying the full list — a small change, recorded here so it is not
forgotten.

### `why_not` needs the whole catalogue in view

Asking "why not *this* step?" presupposes seeing the step.  So the surface needs
a full list somewhere — which `ts.describe()` already produces, grouped by
category with each step's needs.  The user asked for exactly that: *"it would be
nice to see the list of all steps' needs somewhere."*  One panel, collapsed by
default.

### The click budget

Stating it as a target, since it is the design's real measure: after the
launching cell, **one click per derivation step** in the common case — the
chooser, with needs inferred and no target.  Anything that adds a mandatory
field costs one click on every step of every derivation, so a target selector
should appear on demand rather than stand permanently.

## 4. What the code panel emits

**Ordinary tender code, referencing the preamble by name.**  The panel's job is
to be copy-pasted into a script and then edited, so it should look like what a
person would have written — and like what the challenges already contain:

```python
e = tb.expand_in_basis(e, frame)
e = tb.reduce_frame(e, frame)
e = td.contract_eps_pair(e)
e = tb.reduce_frame(e, frame)
e = tb.reassemble(e, frame)
```

That answers the name-vs-reference question by layer rather than by preference:
**names are how the GUI and the catalogue talk** (`applicable` and `why_not`
speak them, and they serialize); **function references are how the emitted code
reads**, because a pasted script should be navigable, editable, and independent
of the registry.  Both, at the level each belongs to.

Alongside it, the same derivation stays available as *data* —
`s.steps → [(name, kwargs), …]` — for programmatic replay and for vibe 000107's
corpus.  A `Derivation`-style emission (`.step(...)` with the audit trail) is a
second `style=` on the same call, since a few challenges are written that way.

**A pleasant consequence of §3's needs inference:** to write `frame` rather than
a repr of the basis, the emitter must know *what the user's namespace calls it*.
Scanning the launching scope for objects of the right kind gives that name for
free.  One mechanism serves both filling the needs and emitting code that relies
on the preamble.

## 5. Rendering the expression

Not an image.  `ipywidgets.HTMLMath` — *"renders the string as HTML, and render
mathematics"* — takes the LaTeX directly and handles typesetting of
dynamically-updated content, which was the one thing that made MathJax in a
widget look risky.

This is the better choice for a reason beyond fidelity: the deferred feature is
**selecting a target with the mouse**, and that needs the rendered expression to
carry positions.  A picture has none.  Live DOM can, and `tender.render.labeled`
already produces the path→part legend that would drive it — so HTML makes the
deferred feature reachable, where an image would have been a dead end requiring
the display to be rebuilt.

The image path is not wasted, and stays as the fallback: matplotlib mathtext was
measured rendering tender's output 4/4, needs no LaTeX install, and works in any
frontend.  If `HTMLMath` disappoints somewhere, the swap is one function.

## 6. Open questions

None outstanding on the version-1 surface.  What remains is scheduling.

## 7. Branching, settled

The sketch said "linear, truncate the tail"; the exploration model wanted
branches.  Resolved as: **record the whole tree, show one path through it.**

**Recording is free** — the abandoned tail is already computed, and throwing it
away is a deliberate act rather than a saving.  It is also the data vibe 000107
needs most: what a person *rejected* says more about their intent than the route
that worked.

**Displaying it costs almost nothing, if it goes in the right place.**  Not a
tree pane — the chooser at each node already lists the applicable steps, so a
step tried before from this expression is simply marked:

```
○ reduce_frame
○ contract_eps_pair        · tried
○ to_concrete
```

No new widget, and the flag appears exactly where the decision is made.

**Keyed on the expression, not the path.**  The memory is a lookup on
`(canonical form, step name)`, so it fires even when the same state is reached by
a different route — wander off, come back another way, and it still says you have
been here.  A path-keyed history cannot do that, and the canonical form is
already what the search prototype used as its visited-set key.

**Re-selecting a tried step starts fresh**, rather than reinstating the old
subtree.  The user's reason is the decisive one and is about the display, not
about cost: *the list is a path through the tree*, so a reinstated subtree would
have to collapse to a path anyway, and the list would jump forward several items
at once.  Fresh keeps the list honest; the annotation already carries the
knowledge that the ground was covered.

**The label says "tried", not "tried, abandoned".**  A tool has no business
editorialising about how a derivation was arrived at.

**A full view stays available with no GUI at all:** `s.attempts()`, printed in a
cell.  In-flow guidance is free; whole-tree inspection is a cell away.

## 8. On exit: nothing happens

Because this lives in a kernel, closing the widget is not a lifecycle event:

```python
s = td.explore(expr)   # the widget mutates s
s.script()             # the surviving path, as code
s.tree                 # everything, abandoned branches included
```

The session is an ordinary Python object and persists as long as the kernel
does, which is as long as anyone cares.  Nothing is written to disk, there is no
save prompt, and no decision about what to keep.  Persisting for vibe 000107's
corpus becomes `td.explore(expr, record=…)` when that vibe resumes; it is not
version 1's problem.

This is a third dividend of the technology choice, alongside deleting the
preamble box and inferring the needs: a standalone application would need a
session lifecycle, a storage location, and a policy on discarding.  Here there is
nothing to design.

## Status

**Built.**  `tender.explore` (the session), `tender.gui` (the widget),
`td.explore` as the entry point, and `examples/guided_derivation.ipynb` as the
showcase.  31 tests in `python/tests/test_explore.py`; `ipywidgets` is an
optional dependency and the widget tests skip without it.

The two library items the design called for are done:

- `applicable`'s `missing` is now keyed by **step** and carries every unmet
  need, with `Report.blocked_on(kind)` for the other direction (§3);
- `td.explore(expr)` returns a `Session` owning the tree, the shown path, and
  `script()` / `steps` / `attempts()` (§8).

Everything else was assembly, as predicted — the widget contains no algebra,
only `applicable`, `why_not`, `describe` and `explain` wired to three widgets.

### What the build added beyond the design

- **A `DidNotFire` exception.**  `apply` refuses a step that does not fire, and
  the message is the step's own reason.  A history of moves that changed
  nothing is exactly the failure mode `NoOpStep` was introduced to catch, so
  the session should not be able to record one.
- **`Session.use(kind, value)`**, and a chooser row for it.  Two bases in a
  notebook is the normal case in the curvilinear examples, not an edge one.
- **A coordinate is recognised by `partial`, not by a type.**  There is no
  Python-visible predicate separating a coordinate from a scalar; but
  `partial(v, v)` succeeds on exactly the coordinates, which is the same
  question the `coord` steps ask.  Charts contribute their `coords` under an
  attribute path, since a notebook holds the chart rather than loose
  coordinates.
- **`_literal` for emission.**  An enum has to be written the way a person
  writes it — `tb.Variance.Contravariant`, not `Variance.Contravariant` — so
  the emitter maps the binding type's private module to its public alias.
- **The reports are cached per node.**  `applicable` runs forty steps, and the
  list re-renders on every change; a ten-item history would otherwise re-run
  four hundred of them per click.  A node's expression is immutable, so the
  cache key is `(node, context)`.

### What is not built, deliberately

Mouse selection of a target (the reason §5 chose live DOM over an image) and
`record=` for vibe 000107's corpus.  Both were scoped out of version 1 and
neither is blocked by anything here.
