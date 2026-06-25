# 000059 — Nf migration follow-ups

Two non-blocking follow-ups carried out of the parallel-IR expression-model
migration (vibe 000058, Stage 5 / C14 complete).  Both are deferred niceties,
not correctness blockers for the work that closed C14; recorded here so they are
not lost.

## (1) Subtree-variable rank-checking in `nf_match`

**What.** A slot-less, non-well-known `Atom` in an identity LHS is a *subtree
variable* that binds any whole target factor.  The deleted Expr matcher
(`identity.cpp`, pruned at C14e) rank-checked this binding: a rank-1 pattern
variable would *not* bind a rank-2 subtree when both ranks were known
(`match_node`'s `TensorObject` arm: `if (p.rank) if (auto tr = infer_rank(tgt);
tr && *tr != *p.rank) return false;`).  The Nf matcher's equivalent
(`nf_match.cpp`, `match_factor`'s `Atom` arm) does **not** rank-check — it binds
the whole factor unconditionally via `try_bind_subtree`.

**Why it might matter.** Without the check a rank-mismatched identity could fire
where the Expr matcher would have declined, producing a rank-inconsistent
rewrite.  In practice the identities in `tender/identities` are rank-consistent,
so nothing observably regressed when the Expr check was dropped (the C14e prune
was suite-green) — hence "nicety", not "bug".  The old direct-test coverage
(`SubtreeVars.RankCheckRejectsMismatch`) was deleted with the Expr matcher.

**Sketch.** In `match_factor`'s subtree-variable branch, infer the target
factor's rank (an `Nf`-level `infer_rank(Factor const*)`, or reuse the Expr
`infer_rank` after a cheap raise of the factor) and reject the bind when the
pattern `Atom`'s declared rank is known and differs.  Add a focused matcher unit
test mirroring the deleted one.  Open question: is a robust `Factor` rank
available cheaply, or does this want a small `nf::infer_rank`?

**Status — TODO.**  User flagged this one as "not sure" whether it is needed;
decide when an actual rank-ambiguous identity shows up, or fold it in
opportunistically.

## (2) Make `canonicalize_nf` self-contained

**What.** `nf::canonicalize_nf(ctx, e)` currently assumes its input `Expr` has
already been through the Expr-side prep that the flip wired in around C10/C13 —
`materialize` (explicit binders) then `float_sums` (binders floated to the term
head).  Every caller therefore spells `canonicalize_nf(ctx, steps::canonicalize(
ctx, e))` (and `apply_identity` / `NfEGraph::add` / the differential harness all
repeat that prep).  The goal: fold the materialize/float prep *into*
`canonicalize_nf` so it lowers an arbitrary surface `Expr` directly, and callers
drop the now-redundant `steps::canonicalize` wrapper.

**Why.** DRY (the prep incantation is duplicated at every call site) and a
cleaner public contract ("lower any `Expr` to `Nf`", no pre-conditions).  See
`nf_lower.hpp`'s `canonicalize_nf` doc, which still lists the un-prepped cases it
throws on.

**Care.** The flip itself (`steps::canonicalize = raise ∘ canonicalize_nf`) must
not recurse: the prep folded into `canonicalize_nf` has to be the *minimal*
materialize/float step, not a re-entrant call to `steps::canonicalize`.  Keep
the round-trip property `canonicalize_nf(raise(nf)) == nf` green.

**Status — TODO.**  User flagged this one as safe to do independently of (1).
