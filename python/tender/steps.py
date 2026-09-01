"""tender.steps — the step catalogue.

A derivation is a sequence of steps, and there are more of them than anyone
remembers.  This module is the index: every step, what it does, what category
of work it belongs to, and what it needs besides the expression.

    >>> import tender.steps as ts
    >>> [s.name for s in ts.in_category("bridge") if s.primary]
    ['expand_in_basis', 'reduce_frame', 'to_concrete', 'reassemble']
    >>> print(ts.describe("index"))          # doctest: +SKIP

The categories are what a *user* needs — "how do I get back to direct
notation?" is a question about the **bridge** row — not the module split, which
follows implementation history (`tender.derivation` vs `tender.basis` differ
mostly by whether a step takes a basis).

**Primary vs. demoted.**  A `primary` step is one a derivation reaches for by
name.  The rest are the moves those are built from: still importable, still
supported, but not vocabulary.  The split was measured, not chosen (vibe
000106): across the challenges and examples, 14 of the 15 steps used in real
derivations never appeared alone — they were punctuation between the moves that
mattered.

**`needs`** lists what a step wants besides the expression, drawn from a short
closed list — ``basis``, ``chart``, ``coord``, ``rules``, ``level``, ``op``,
``identity``.
``options`` lists what it will *accept*: ``target`` names a single object to act on,
``variance`` picks co/contravariant.  Both are here so a tool can supply them
from context rather than each caller remembering.

Registration is open: :func:`register` adds your own step, so it appears
alongside the shipped ones — the same choice the identity library makes.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from . import _core
from . import basis as _b
from . import chart as _c
from . import derivation as _d

# Re-exported for callers who want the demoted moves directly.
from .derivation import (  # noqa: F401
    distribute_contraction,
    fold_equal_addends_structural,
    implicitize,
    saturate,
)

CATEGORIES = ("normalise", "bridge", "index", "operators", "engine")

_CATEGORY_BLURB = {
    "normalise": "reshape an expression without changing its content",
    "bridge": "cross between invariant and component form",
    "index": "contract or move indices",
    "operators": "∇ and ∂ — apply them, or fold them back",
    "engine": "rule-driven: the identity library, saturated or one rule at a time",
}


@dataclass(frozen=True)
class StepResult:
    """What a step did: the expression it produced, whether it fired, and why not."""

    expr: Any
    fired: bool
    reason: str

    def __repr__(self):
        return f"<StepResult fired={self.fired} {self.reason[:60]!r}>"


@dataclass(frozen=True)
class Step:
    """One entry in the catalogue."""

    name: str
    category: str
    summary: str
    fn: Callable
    home: str
    needs: tuple = ()
    options: tuple = ()
    primary: bool = False
    # Minimum fingerprint counts for the step to have anything to act on —
    # {"deltas": 1} means "needs at least one δ".  Data, so `why_not` can say
    # *why* rather than just "nothing happened" (vibe 000106).
    wants: dict = field(default_factory=dict)
    # The reporting form, when the step has been taught to explain itself:
    # (expr, **kw) -> (expr, fired, reason).  A step's own account of why it did
    # nothing beats anything inferred from the outside, because the reason lives
    # in its logic (vibe 000106).
    reported: Any = None

    def __call__(self, expr, **ctx):
        """Apply the step, taking its extra arguments from *ctx* by kind."""
        missing = [k for k in self.needs if k not in ctx]
        if missing:
            raise TypeError(
                f"{self.name} needs {missing} — pass them by keyword, e.g. "
                f"{self.name}(expr, {missing[0]}=…)"
            )
        kw = {k: ctx[k] for k in self.needs if k in ctx}
        kw.update({k: ctx[k] for k in self.options if k in ctx})
        return self.fn(expr, **kw)

    def run(self, expr, **ctx):
        """Apply the step and say what happened — the reporting interface.

        Uniform from the start: a step that explains itself supplies the reason,
        and one that does not gets a synthesized one, so callers need not know
        which is which and steps can be taught one at a time.
        """
        missing = [k for k in self.needs if k not in ctx]
        if missing:
            return StepResult(
                expr,
                False,
                f"not tried: it needs {', '.join(missing)}, which the context "
                f"does not have",
            )
        kw = {k: ctx[k] for k in self.needs if k in ctx}
        kw.update({k: ctx[k] for k in self.options if k in ctx})
        if self.reported is not None:
            got, fired, reason = self.reported(expr, **kw)
            return StepResult(got, fired, reason)
        got = self.fn(expr, **kw)
        fired = not _d.structural_eq(got, expr)
        return StepResult(got, fired, "" if fired else self._guess(expr))

    def _guess(self, expr):
        """A reason inferred from outside, for a step that has none of its own."""
        sh = shape(expr)
        short = {k: (v, sh[k]) for k, v in self.wants.items() if sh.get(k, 0) < v}
        if short:
            return "nothing to act on here: " + ", ".join(
                f"{k} (needs {need}, has {got})"
                for k, (need, got) in sorted(short.items())
            )
        return (
            "it ran and changed nothing; its pattern is not present, even though "
            "the ingredients are (this step does not yet explain itself)"
        )

    @property
    def qualified(self) -> str:
        return f"{self.home}.{self.name}"


class BoundStep:
    """One step with its extra arguments already supplied — an ``Expr -> Expr``.

    What a derivation wants in a list: a plain callable, so the list is data
    you can slice, reorder, reuse on another expression, or generate.
    """

    __slots__ = ("step", "context", "__name__")

    def __init__(self, step, context):
        self.step = step
        self.context = dict(context)
        self.__name__ = step.name

    def __call__(self, expr):
        return self.step(expr, **self.context)

    def run(self, expr):
        """Apply and say what happened — :meth:`Step.run` with the context."""
        return self.step.run(expr, **self.context)

    def __repr__(self):
        given = ", ".join(
            f"{k}={v!r}"
            for k, v in sorted(self.context.items())
            if k in self.step.needs + self.step.options
        )
        return f"<{self.step.qualified}({given})>"


class Bound:
    """The catalogue with a context already supplied (:func:`using`).

    Attribute access binds a step; calling binds one with extra arguments::

        b = ts.using(basis=frame)
        steps = [b.expand_in_basis, b.reduce_frame, b("reassemble", target="u")]

    Each step takes only what it declares, so *one* context serves a whole
    derivation — the same reason :func:`applicable` can try everything from a
    single frame.
    """

    def __init__(self, **context):
        self.context = context

    def __getattr__(self, name):
        try:
            step = info(name)
        except ValueError:
            raise AttributeError(
                f"no step named {name!r}; see tender.steps.names()"
            ) from None
        return BoundStep(step, self.context)

    def __call__(self, name, **extra):
        """Bind *name* with arguments beyond the shared context."""
        return BoundStep(info(name), {**self.context, **extra})

    def __dir__(self):
        return sorted(names())

    def __repr__(self):
        given = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"<bound steps: {given}>"


def using(**context):
    """Bind a context to every step, so a derivation can be a plain list.

    The alternative is a lambda per step (``lambda x: reduce_frame(x, frame)``)
    or a :func:`functools.partial` repeating the frame on every line.  Neither
    is wrong; both make the shared thing local, and the list stops reading as
    the derivation it is::

        b = ts.using(basis=frame)
        td.derive(a @ b, [b.expand_in_basis, b.reduce_frame, b.reassemble])
    """
    return Bound(**context)


_STEPS: dict = {}


def register(
    name,
    fn,
    *,
    category,
    summary,
    home="",
    needs=(),
    options=(),
    primary=False,
    wants=None,
    reported=None,
):
    """Add a step to the catalogue — yours sit alongside the shipped ones.

    ``category`` must be one of :data:`CATEGORIES`; ``needs`` and ``options``
    name the extra arguments by kind (``"basis"``, ``"coord"``, ``"rules"``,
    ``"level"``, ``"op"``, ``"target"``, ``"variance"``).
    """
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; expected one of {CATEGORIES}")
    step = Step(
        name=name,
        category=category,
        summary=summary,
        fn=fn,
        home=home,
        needs=tuple(needs),
        options=tuple(options),
        primary=primary,
        wants=dict(wants or {}),
        reported=reported,
    )
    _STEPS[name] = step
    return step


def names(category=None):
    """Every catalogued step name, optionally narrowed to one category."""
    return sorted(n for n, s in _STEPS.items() if category in (None, s.category))


def info(name):
    """The :class:`Step` called *name*; raises ``ValueError`` if unknown."""
    try:
        return _STEPS[name]
    except KeyError:
        raise ValueError(
            f"unknown step {name!r}; see tender.steps.names()"
        ) from None


def in_category(category):
    """Every :class:`Step` in *category*, primaries first."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; expected one of {CATEGORIES}")
    got = [s for s in _STEPS.values() if s.category == category]
    return sorted(got, key=lambda s: (not s.primary, s.name))


def primaries():
    """The steps a derivation reaches for by name."""
    return sorted(
        (s for s in _STEPS.values() if s.primary),
        key=lambda s: (CATEGORIES.index(s.category), s.name),
    )


def describe(category=None):
    """A readable table of the catalogue — what to reach for, and for what."""
    out = []
    for cat in CATEGORIES if category is None else (category,):
        rows = in_category(cat)
        if not rows:
            continue
        out.append(f"{cat} — {_CATEGORY_BLURB[cat]}")
        for s in rows:
            mark = "*" if s.primary else " "
            extra = ""
            if s.needs:
                extra = "  (needs " + ", ".join(s.needs) + ")"
            out.append(f"  {mark} {s.qualified:34s} {s.summary}{extra}")
        out.append("")
    out.append("* = a step to reach for by name; the rest are what they are built from.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The shipped catalogue.
# ---------------------------------------------------------------------------

_D = "tender.derivation"
_B = "tender.basis"
_C = "tender.chart"


def _r(name, mod, home, **kw):
    register(name, getattr(mod, name), home=home, **kw)


# ---- bridge: invariant ⇄ components ---------------------------------------
_r("expand_in_basis", _b, _B, category="bridge", primary=True,
   needs=("basis",), options=("variance",),
   summary="write an invariant in components on a frame")
_r("reduce_frame", _b, _B, category="bridge", primary=True, needs=("basis",),
   reported=_core.basis._reduce_frame_reported,
   wants={"basis_vectors": 1},
   summary="everything the frame licenses, to a fixed point")
_r("to_concrete", _b, _B, category="bridge", primary=True, needs=("basis",),
   wants={"index_slots": 1},
   summary="evaluate over the frame's concrete directions")
_r("reassemble", _b, _B, category="bridge", primary=True,
   reported=_core.basis._reassemble_reported,
   needs=("basis",), options=("target",),
   wants={"coordinates": 1},
   summary="fold components back into direct notation")
_r("simplify_basis_dot", _b, _B, category="bridge", needs=("basis",),
   wants={"basis_vectors": 2},
   summary="one pass of eᵢ·eⱼ → δ")
_r("simplify_basis_cross", _b, _B, category="bridge", needs=("basis",),
   wants={"basis_vectors": 2},
   summary="one pass of eᵢ×eⱼ → ε")
_r("reassemble_completeness", _b, _B, category="bridge", needs=("basis",),
   wants={"basis_vectors": 1},
   summary="Σᵢ (X·eᵢ) eᵢ → X alone")
_r("fold_resolution_of_identity", _b, _B, category="bridge", needs=("basis",),
   wants={"basis_vectors": 1},
   summary="a completed Σ eₖ⊗eₖ → I")
_r("expand_identity", _b, _B, category="bridge", needs=("basis",),
   wants={"identities": 1},
   summary="I → Σ eₖ⊗eₖ on a frame")
_r("unroll_sums", _d, _D, category="bridge",
   wants={"index_slots": 1},
   summary="expand a Σ over the index space's directions")
_r("eval_delta_concrete", _d, _D, category="bridge",
   wants={"deltas": 1},
   summary="evaluate δ on concrete indices")
_r("eval_eps_concrete", _d, _D, category="bridge",
   wants={"epsilons": 1},
   summary="evaluate ε on concrete indices")

# ---- bridge: the chart ----------------------------------------------------
# A third of the moves in real derivations happen on a chart, and none of them
# were catalogued until vibe 000108 §14 — the catalogue assumed a step was a
# module-level function, and these were methods on a parameter.
_r("expand", _c, _C, category="bridge", primary=True, needs=("chart",),
   summary="expand abstract fields into components on the physical frame")
_r("express", _c, _C, category="bridge", primary=True, needs=("chart",),
   wants={"basis_vectors": 1},
   summary="re-express in this chart's frame — the general change of basis")
_r("to_reference", _c, _C, category="bridge", needs=("chart",),
   wants={"basis_vectors": 1},
   summary="re-express in the reference (WCS) frame")

# ---- index algebra --------------------------------------------------------
_r("contract_delta", _d, _D, category="index", primary=True,
   reported=_core.derivation._contract_delta_reported,
   wants={"deltas": 1},
   summary="contract a δ against whatever carries its index")
_r("contract_eps_pair", _d, _D, category="index", primary=True,
   wants={"epsilons": 2},
   summary="the ε-δ identity: ε ε → δδ − δδ")
_r("contract_metric", _d, _D, category="index", primary=True, options=("target",),
   reported=_core.derivation._contract_metric_reported,
   wants={"metrics": 1},
   summary="raise, lower, or contract the inverse metric pair")
_r("insert_metric", _d, _D, category="index", primary=True,
   reported=_core.derivation._insert_metric_reported,
   needs=("level",), options=("target",),
   wants={"coordinates": 1},
   summary="move an index the other way, paying a metric")
_r("contract_identity", _d, _D, category="index", primary=True,
   wants={"identities": 1},
   summary="I·X → X")
_r("expand_eps", _d, _D, category="index",
   wants={"epsilons": 1},
   summary="expand ε into its δ-determinant")

# ---- operators ------------------------------------------------------------
_r("apply_operators", _d, _D, category="operators", primary=True,
   reported=_core.derivation._apply_operators_reported,
   wants={"derivs": 1},
   summary="carry out the first-class ∂ operators by Leibniz")
_r("partial", _d, _D, category="operators", primary=True, needs=("coord",),
   summary="differentiate with respect to a coordinate")
_r("fold_operator", _d, _D, category="operators", primary=True, needs=("op",),
   reported=_core.derivation._fold_operator_reported,
   wants={"deriv_marks": 1},
   summary="fold an operator's expansion back into the operator")

_r("evaluate", _c, _C, category="operators", primary=True, needs=("chart",),
   wants={"nablas": 1},
   summary="lower an invariant ∇ expression onto this chart's operators")
_r("expand_nabla", _c, _C, category="operators", primary=True, needs=("chart",),
   wants={"nablas": 1},
   summary="expand a chart-free ∇ into the free-index frame form eᵢ∂ᵢ")
_r("reassemble_nabla", _c, _C, category="operators", primary=True,
   needs=("chart",), wants={"deriv_marks": 1},
   reported=_core.chart._reassemble_nabla_reported,
   summary="fold a reduced free-index expression back into ∇ operators")
_r("to_contraction", _c, _C, category="bridge", needs=("chart",),
   reported=_core.chart._to_contraction_reported,
   wants={"coordinates": 1},
   summary="write a component as its defining contraction, aᵢ → a·eᵢ")
_r("componentize_nabla", _c, _C, category="operators", needs=("chart",),
   wants={"deriv_marks": 1},
   summary="lower an expand_nabla result to concrete components")

# ---- normalise ------------------------------------------------------------
_r("simplify", _d, _D, category="normalise", primary=True,
   summary="the general tidy-up, finishing in implicit form")
_r("simplify_scalars", _d, _D, category="normalise", primary=True,
   summary="scalar-field identities: cos²+sin²→1, √(x²)→x, powers")
_r("collect_terms", _d, _D, category="normalise", primary=True,
   summary="group addends by their tensor part")
_r("factor_common", _d, _D, category="normalise", primary=True,
   summary="factor a common tensor out of a sum")
_r("canonicalize", _d, _D, category="normalise",
   summary="the normal form; most steps now self-prepare instead")
_r("implicitize", _d, _D, category="normalise",
   summary="strip explicit Σ back to Einstein form")
_r("expand_products", _d, _D, category="normalise",
   summary="distribute products over sums")
_r("distribute_contraction", _d, _D, category="normalise",
   summary="push a contraction onto the near leg of a ⊗")
_r("expand_dyad_ops", _d, _D, category="normalise",
   summary="expand tr / vec / transpose on dyads")
_r("expand_double_dot", _d, _D, category="normalise",
   summary="expand a double contraction of dyads")
_r("fold_arithmetic", _d, _D, category="normalise",
   summary="evaluate numeric arithmetic")
_r("fold_equal_addends", _d, _D, category="normalise",
   summary="X + X → 2X, across dummy renaming")
_r("fold_equal_addends_structural", _d, _D, category="normalise",
   summary="the bare structural version of the above")
_r("fold_sums", _d, _D, category="normalise",
   summary="fold a materialised Σ back where it came from")
_r("sym", _d, _D, category="normalise",
   summary="the symmetric part ½(A + Aᵀ)")
_r("skew", _d, _D, category="normalise",
   summary="the skew part ½(A − Aᵀ)")

# ---- engine ---------------------------------------------------------------
_r("apply_identity", _d, _D, category="engine", primary=True,
   needs=("identity",),
   summary="apply one named identity from the rule library")
_r("engine_simplify", _d, _D, category="engine", primary=True, needs=("rules",),
   summary="equality saturation, extracting the cheapest form")
_r("saturate", _d, _D, category="engine", needs=("rules",),
   summary="the raw saturation pass behind engine_simplify")

# ---------------------------------------------------------------------------
# What applies here?
# ---------------------------------------------------------------------------


def shape(expr):
    """The structural fingerprint of *expr* — what a step's effect is measured by."""
    return _core.derivation._expression_shape(expr)


def rule_steps(identities, home="tender.identities"):
    """One :class:`Step` per identity — the rule library as a step set.

    Identities are context-bound, so they cannot live in the catalogue the way
    the shipped steps do; but they answer the same question, and this makes
    them answerable by the same call::

        print(applicable(expr, steps=rule_steps(td.rules("cross", ctx=ctx))))

    Which rules apply here, with what each would change — the counterpart of
    handing the whole set to :func:`~tender.derivation.engine_simplify` and
    accepting its choice.
    """
    from functools import partial

    def reporter(identity):
        # A rule that does not fire has exactly one reason, and it is worth
        # saying which pattern went looking: "no match" plus the lhs is the
        # whole story, where the synthesized fallback would guess.
        def run(expr):
            got = identity(expr)
            if _d.structural_eq(got, expr):
                return got, False, (
                    f"{identity.name} did not match; its left-hand side is "
                    f"{identity.lhs.latex()}"
                )
            return got, True, ""

        return run

    return [
        Step(
            name=r.name,
            category="engine",
            summary=f"the identity {r.name}",
            fn=partial(_d.apply_identity, identity=r),
            reported=reporter(r),
            home=home,
        )
        for r in identities
    ]


@dataclass(frozen=True)
class Hit:
    """One step that does something to an expression, and what it does."""

    step: Step
    result: Any
    change: dict  # fingerprint deltas; empty means "reshaped only"

    @property
    def reshapes_only(self) -> bool:
        return not self.change

    def __repr__(self):
        what = (
            ", ".join(f"{k}{v:+d}" for k, v in sorted(self.change.items()))
            or "reshaped only"
        )
        return f"<{self.step.qualified}  {what}>"


class Report(list):
    """The applicable steps, content-changing first; prints as a table.

    ``missing`` maps a step's qualified name to *every* need the context did
    not cover, because a step blocked for want of an argument is a third
    category — not tried, rather than tried and quiet — and supplying what it
    lacks is an obvious next move (vibe 000108 §3).
    """

    def __init__(self, hits, missing=()):
        super().__init__(hits)
        self.missing = {k: tuple(v) for k, v in dict(missing).items()}

    def blocked_on(self, kind):
        """The steps not tried because *kind* was absent."""
        return sorted(n for n, lack in self.missing.items() if kind in lack)

    @property
    def changing(self):
        return [h for h in self if not h.reshapes_only]

    def __str__(self):
        out = []
        if not self.changing:
            out.append("nothing changes the content of this expression.")
        for h in self.changing:
            out.append(
                f"  {h.step.qualified:36s} {', '.join(f'{k}{v:+d}' for k, v in sorted(h.change.items())):32s}"
                f" {h.result.latex()[:46]}"
            )
        quiet = len(self) - len(self.changing)
        if quiet:
            out.append(f"  (+{quiet} more that only reshape: canonical reordering)")
        if self.missing:
            by_kind = {}
            for qualified, lack in self.missing.items():
                by_kind.setdefault(", ".join(lack), []).append(qualified)
            for kinds, who in sorted(by_kind.items()):
                out.append(
                    f"  not tried — no {kinds} in context: {', '.join(sorted(who))}"
                )
        return "\n".join(out)


def applicable(expr, steps=None, **context):
    """Which steps actually do something to *expr*?

    The answer to "which step do I need?", asked of the expression rather than
    of memory.  Every catalogued step is tried; the ones that bite are reported
    with **what they changed** — because "not a no-op" is too weak a filter.  On
    a two-vector dot thirteen steps fire and seven merely commute factors into
    canonical order; those are collapsed to a count so the real options stand
    out.

    Extra arguments are supplied by kind, once, for every step that wants them::

        print(ts.applicable(expr, basis=frame))

    A step whose ``needs`` the context does not cover is not tried, and is
    listed at the end with what it was missing — so the report also says what
    you could hand it to see more.  That listing is ``report.missing``, keyed by
    the step and carrying every unmet need, not just the first.

    *steps* narrows or replaces what is tried — names or :class:`Step` objects,
    defaulting to the whole catalogue.  The catalogue is not the only source of
    steps: a rule library builds one per identity, and those belong to a
    session rather than to the module (vibe 000108 §11).
    """
    hits, missing = [], {}
    before = shape(expr)
    for entry in names() if steps is None else steps:
        st = info(entry) if isinstance(entry, str) else entry
        lack = tuple(k for k in st.needs if k not in context)
        if lack:
            missing[st.qualified] = lack
            continue
        try:
            got = st(expr, **context)
        except Exception:
            continue  # a domain error means "not applicable here"
        if _d.structural_eq(got, expr):
            continue
        after = shape(got)
        change = {k: after[k] - before[k] for k in before if after[k] != before[k]}
        hits.append(Hit(st, got, change))
    hits.sort(
        key=lambda h: (
            h.reshapes_only,
            CATEGORIES.index(h.step.category),
            h.step.name,
        )
    )
    return Report(hits, missing)


def why_not(expr, step, **context):
    """Why did *step* do nothing to *expr*?

    The inverse of :func:`applicable`, and the more useful one when you already
    have an expectation.  "Nothing happened" is the least informative thing a
    library can say; this says which of the four possible reasons it was:

    * the context is missing something the step needs (``basis``, ``coord``, …);
    * the expression has nothing for it to act on — no δ to contract, no basis
      vector to reduce — reported against the step's recorded precondition;
    * it raised, and on what;
    * it ran and genuinely changed nothing.

    *step* may be a name or a :class:`Step`.
    """
    st = info(step) if isinstance(step, str) else step
    lack = [k for k in st.needs if k not in context]
    if lack:
        return (
            f"{st.qualified} was not tried: it needs {', '.join(lack)}, which "
            f"the context does not have — pass {lack[0]}=… ."
        )
    sh = shape(expr)
    try:
        res = st.run(expr, **context)
    except Exception as ex:  # noqa: BLE001 - the reason is the answer
        return f"{st.qualified} raised {type(ex).__name__}: {ex}"
    if not res.fired:
        return f"{st.qualified}: {res.reason}"
    got = res.expr
    after = shape(got)
    change = {k: after[k] - sh[k] for k in sh if after[k] != sh[k]}
    if not change:
        return f"{st.qualified} only reshaped the expression (canonical reordering)."
    return (
        f"{st.qualified} did apply: "
        + ", ".join(f"{k}{v:+d}" for k, v in sorted(change.items()))
    )


def explain(before, after):
    """What changed between two expressions — the fingerprint delta, and both forms.

    A step's effect read off the structure rather than by squinting at two
    LaTeX strings.  `tender.render.labeled` gives the positional view when you
    need to know *where*.
    """
    b, a = shape(before), shape(after)
    change = {k: a[k] - b[k] for k in b if a[k] != b[k]}
    lines = [f"  before  {before.latex()}", f"  after   {after.latex()}"]
    if _d.structural_eq(before, after):
        lines.append("  change  none — the expressions are identical")
    elif not change:
        lines.append("  change  reshaped only (canonical reordering)")
    else:
        lines.append(
            "  change  " + ", ".join(f"{k}{v:+d}" for k, v in sorted(change.items()))
        )
    return "\n".join(lines)


__all__ = [
    "CATEGORIES",
    "Step",
    "Hit",
    "Report",
    "applicable",
    "Bound",
    "BoundStep",
    "using",
    "rule_steps",
    "StepResult",
    "why_not",
    "explain",
    "shape",
    "register",
    "names",
    "info",
    "in_category",
    "primaries",
    "describe",
    # the demoted moves, importable as before
    "distribute_contraction",
    "fold_equal_addends_structural",
    "implicitize",
    "saturate",
]
