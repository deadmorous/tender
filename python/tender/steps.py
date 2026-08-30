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
   summary="everything the frame licenses, to a fixed point")
_r("to_concrete", _b, _B, category="bridge", primary=True, needs=("basis",),
   summary="evaluate over the frame's concrete directions")
_r("reassemble", _b, _B, category="bridge", primary=True,
   needs=("basis",), options=("target",),
   summary="fold components back into direct notation")
_r("simplify_basis_dot", _b, _B, category="bridge", needs=("basis",),
   summary="one pass of eᵢ·eⱼ → δ")
_r("simplify_basis_cross", _b, _B, category="bridge", needs=("basis",),
   summary="one pass of eᵢ×eⱼ → ε")
_r("reassemble_completeness", _b, _B, category="bridge", needs=("basis",),
   summary="Σᵢ (X·eᵢ) eᵢ → X alone")
_r("fold_resolution_of_identity", _b, _B, category="bridge", needs=("basis",),
   summary="a completed Σ eₖ⊗eₖ → I")
_r("expand_identity", _b, _B, category="bridge", needs=("basis",),
   summary="I → Σ eₖ⊗eₖ on a frame")
_r("unroll_sums", _d, _D, category="bridge",
   summary="expand a Σ over the index space's directions")
_r("eval_delta_concrete", _d, _D, category="bridge",
   summary="evaluate δ on concrete indices")
_r("eval_eps_concrete", _d, _D, category="bridge",
   summary="evaluate ε on concrete indices")

# ---- index algebra --------------------------------------------------------
_r("contract_delta", _d, _D, category="index", primary=True,
   summary="contract a δ against whatever carries its index")
_r("contract_eps_pair", _d, _D, category="index", primary=True,
   summary="the ε-δ identity: ε ε → δδ − δδ")
_r("contract_metric", _d, _D, category="index", primary=True, options=("target",),
   summary="raise, lower, or contract the inverse metric pair")
_r("insert_metric", _d, _D, category="index", primary=True,
   needs=("level",), options=("target",),
   summary="move an index the other way, paying a metric")
_r("contract_identity", _d, _D, category="index", primary=True,
   summary="I·X → X")
_r("expand_eps", _d, _D, category="index",
   summary="expand ε into its δ-determinant")

# ---- operators ------------------------------------------------------------
_r("apply_operators", _d, _D, category="operators", primary=True,
   summary="carry out the first-class ∂ operators by Leibniz")
_r("partial", _d, _D, category="operators", primary=True, needs=("coord",),
   summary="differentiate with respect to a coordinate")
_r("fold_operator", _d, _D, category="operators", primary=True, needs=("op",),
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

__all__ = [
    "CATEGORIES",
    "Step",
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
