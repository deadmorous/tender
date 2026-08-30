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
closed list — ``basis``, ``coord``, ``rules``, ``level``, ``op``.  ``options``
lists what it will *accept*: ``target`` names a single object to act on,
``variance`` picks co/contravariant.  Both are here so a tool can supply them
from context rather than each caller remembering.

Registration is open: :func:`register` adds your own step, so it appears
alongside the shipped ones — the same choice the identity library makes.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from . import _core
from . import basis as _b
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
    "engine": "goal-directed: hand the work to the rule engine",
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
    """The applicable steps, content-changing first; prints as a table."""

    def __init__(self, hits, missing=()):
        super().__init__(hits)
        self.missing = dict(missing)

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
            for kind, names in sorted(self.missing.items()):
                out.append(
                    f"  not tried — no {kind} in context: {', '.join(sorted(names))}"
                )
        return "\n".join(out)


def applicable(expr, **context):
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
    you could hand it to see more.
    """
    hits, missing = [], {}
    before = shape(expr)
    for name in names():
        st = info(name)
        lack = [k for k in st.needs if k not in context]
        if lack:
            missing.setdefault(lack[0], set()).add(st.qualified)
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
