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

## 9. First use, and what it changed

The surface was used in JupyterLab on `a × (a × b)` and reached
`(a·b) a − (a·a) b` in six steps.  Four observations, three of them fixed.

**A widget has no terminal form, and silence was the wrong way to say so.**
`explore` from a script or a plain REPL did nothing at all: `ipywidgets` needs a
browser to draw in, `show` refused, and the refusal was swallowed.  That is the
correct behaviour wrapped in the worst possible presentation — the session
*does* work there, and is the whole library, only typed.  It now says so, and
names the four calls that replace the clicks.  A text-mode driver is a real
possibility and is not this: it would be a second surface with its own idea of
how a list is chosen, not a fallback.

**The list squeezed its items instead of scrolling.**  Past about six steps
every item shrank until its chooser was unreachable behind a scrollbar of its
own.  The cause is flexbox rather than tender: a flex child may shrink below
its content, so a fixed-height column divides the space among its children
instead of overflowing.  `flex: 0 0 auto` on the item is the fix — the *list*
scrolls, the items keep their size.  `max_height` is now a parameter, and the
detached output view (§2) is documented where the question arises rather than
only here.

**"How do I specify a step's needs?" was asked of a panel that half-answered
it.**  The context row showed `basis = wcs` — inferred, correct, and silent
about the four kinds it found nothing for.  Naming those is the answer to the
question: a kind with nothing in scope is now shown as such, with the three ways
to supply one.  The lesson generalises — *the inference is only legible if its
misses are visible too*; an empty row said nothing where a named one says
everything.

**Filtering the chooser by typing** was requested, and the design question it
raised is answered in §10.

**The search should not have been the only way in.**  Inferring the arguments
from the calling scope was §3's headline simplification, and it stays — in a
notebook it is what removes the needs fields and the preamble box.  But it was
*the* mechanism rather than *a* mechanism, and a reader of
`td.explore(a @ b)` had nowhere to look for where the frame came from.  So the
kinds can now be named: `td.explore(expr, {"basis": frame})`, or the shorter
`basis=frame`.

Naming and searching **do not mix**: a `needs` dict is a statement about what
the steps get, so nothing the caller did not name appears beside it.  A dict
that merged with the search would be the worst of the two — explicit in
appearance, implicit in effect, and impossible to read off the call.  The widget
says which way the arguments arrived, so the panel is never ambiguous about it
either.

The search keeps its place as the default because the click budget (§3) is real
and a notebook genuinely holds these objects.  What changed is that it is now
the convenience rather than the contract.

### Version 2, as it now stands

- selecting a target with the mouse (§5, the reason the rendering is live DOM);
- a compact form for items above the current one, so a long derivation does not
  push the working end off the screen;
- `record=` for vibe 000107's corpus (§8).

## 10. The filter is a lens, not a search box

Typing to narrow the chooser was asked for as a combo-box.  What it wanted
designing (§9) was its relation to the three categories of §3 — *fires*, *not
tried*, *did not fire* — and the answer is that it applies to **all three at
once**:

```
[ reduce_frame  coordinates−2 ▾ ]  [ contract_ ]
not tried: partial (needs coord)
did not fire: contract_eps_pair, contract_identity
```

The chooser shows the matching steps that fire; the line under it shows the
matching steps that were not tried, with what they lack; and — only when a
pattern is typed — the matching steps that ran and did nothing.

**The payoff is at one match.**  Typing a step's name and not finding it in the
list *is* the question `why_not` answers, so when the pattern leaves a single
non-firing step, its reason appears in place of its name.  The filter and the
feedback function turn out to be the same gesture: the user narrows to a step
because they expected it, and the expectation is exactly what wants explaining.
It also asks about *that item's* expression rather than the working end, since
filtering above the end is a normal thing to do.

**A drop-down, not a combo-box.**  `Combobox` would be one widget instead of
two, but its matching is the browser's substring search over the option *text* —
and the options carry their fingerprint delta (`coordinates−2`) and their
`· tried` mark, which are exactly what a typist does not want to match against.
A `Text` beside a `Dropdown` filters on the *name* and leaves the labels rich.

**A half-typed regex is a normal state of a text box**, not an error: `contract_(`
falls back to a substring search and outlines the box.  Nothing raises, nothing
is logged, and the next keystroke usually fixes it.

One implementation note worth keeping: rewriting a `Dropdown`'s options fires
its `value` observer, which is how a step gets taken.  Filtering therefore has
to suppress the same guard a rebuild does, and the step already chosen has to
survive the filter, or the widget holds a value that is not among its options.

## 11. Applying one identity: an entry that asks

The rule library is a second vocabulary beside the step catalogue, and the
surface had no way into it.  The obvious move — one chooser row per identity —
is wrong twice over, and the user said so:

**It breaks the list's guarantee.**  Every row in the chooser is there because
the step was *run and did something*; the delta beside it is the evidence.
Sixteen identities added as rows would be unprobed guesses looking exactly like
probed moves, and *everything here does something* would quietly stop being
true for most of the list.

**It pays for what nobody asked for.**  Probing N rules × M items on every
redraw, against N rules once, on request.  The registry grows; the chooser
should not.

So `apply_identity` is **one entry, below a delimiter**, and choosing it does
not take a step — it opens a second list:

```
[ apply_identity …                                    ▾ ]
  ↳ [ bac-cab   nodes+7                               ▾ ]
    did not match: cross-identity, cross-removal, curl-curl, …
```

The second list is annotated exactly like the first, because it *is* the first
mechanism pointed elsewhere: `applicable(expr, steps=…)` now takes a step set,
and `steps.rule_steps(rules)` turns a rule library into one.  That call is the
whole feature for a terminal user — "which identities apply here?" without a
widget — and it is why the two-level display cost almost no new code.

**The identity is a per-item argument, not a session binding.**  Which rule to
apply is a decision per step; binding one into the context would have fought
the requirement that different steps take different identities.  So `identity`
is a kind that nothing in a namespace is ever scanned for — it is chosen where
the step is chosen.

### Where the candidates come from

An `Expr` does not carry its `Context`, and identities are context-bound, so
they cannot be conjured from the expression the widget is looking at.  Hence a
new kind, `ctx`: a `Context` in the namespace (or a `Workspace`, which
contributes `ws.ctx` as a chart contributes its coordinates) yields the whole
shipped library, built once per session.  A bound `rules` list overrides it,
which is how the choice is narrowed to rules of one's own.  With neither, the
second list says so rather than sitting empty — a real absence, not an
oversight.

### Two corrections the build forced

**`apply_identity` had the wrong shape.**  It was `apply_identity(identity) ->
step`, the only member of the library whose signature was `(arg) -> step`
rather than `(expr, …) -> expr` — and it added nothing, since
`apply_identity(r)` is `r`, an `Identity` being callable.  That inconsistency is
exactly what kept it out of the catalogue.  It is now `apply_identity(expr,
identity)` like every other step; the call sites are all shorter for it.

**A rule is cited by name.**  `rs[5]` says where a rule sat in a list; the
emitted script says which rewrite was taken:

```python
e = td.apply_identity(e, td.rule("bac-cab", ws.ctx))
```

`td.rule(name, source)` is the singular of `td.rules`, taking a `Context` or a
list.  The widget compares rules by name for the same reason — a rule rebuilt
from the same context is the same rule, and the panel is displaying a
derivation, not comparing pointers.

## Status

**Built.**  `tender.explore` (the session), `tender.gui` (the widget),
`td.explore` as the entry point, and `examples/guided_derivation.ipynb` as the
showcase.  70 tests in `python/tests/test_explore.py`; `ipywidgets` is an
optional dependency and the widget tests skip without it.  Used, and corrected
by that use — §9.

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

The version-2 list of §9, less the filter (§10, built): mouse selection of a
target (the reason §5 chose live DOM over an image), a compact form for items
above the working end, and `record=` for vibe 000107's corpus.  None is blocked
by anything here.
