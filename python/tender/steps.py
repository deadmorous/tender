"""tender.steps — the internal rewriting steps.

These are the machinery the verbs and the chart layer are built from.  They
are importable and supported, but they are not the vocabulary: reaching for
one usually means the goal-directed surface is missing something, and that is
worth saying out loud rather than working around quietly.

The public vocabulary lives in :mod:`tender.derivation`:

============================  ==================================================
instead of                    prefer
============================  ==================================================
``saturate``                  ``prove_equal`` / ``engine_simplify``
``implicitize``               ``simplify`` (which finishes in implicit form)
``distribute_contraction``    ``canonicalize``, which self-prepares
``fold_equal_addends_st…``    ``fold_equal_addends`` (self-preparing)
============================  ==================================================

They were demoted by measurement, not taste (vibe 000098): across every
example and challenge in the repository, none of them is called.
"""

from .derivation import (  # noqa: F401
    distribute_contraction,
    fold_equal_addends_structural,
    implicitize,
    saturate,
)

__all__ = [
    "distribute_contraction",
    "fold_equal_addends_structural",
    "implicitize",
    "saturate",
]
