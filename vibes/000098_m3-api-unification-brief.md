# 000098 M3 brief — API unification, increment breakdown

The per-milestone brief for M3 (vibe 000093), grounded in an audit of who
actually calls what across `examples/`, `challenges/`, and `python/tests/`.

## Two audit findings that reshape M3

**1. "Four differentiation routes" is really "one redundant route".**
Vibe 000092 counted four and vibe 000093 concluded `chart.grad/div/rot/
laplacian` should "become internal".  The measurement says otherwise:

| route | callers | verdict |
|---|---|---|
| `chart.grad/div/rot/laplacian` | **10 files, ~50 sites, 6 challenges** | keep public |
| core `t.nabla` + `chart.evaluate` | the invariant route (vibe 000084) | keep public |
| `td.partial` / `td.deriv` + `apply_operators` | the scalar layer, mostly internal | demote |
| `tender.operators` `DifferentialExpr` | **3 files** (1 example, 2 tests) | the redundant one |

`chart.grad(u)` is not a wart — it is direct, reads like the mathematics, and
is what six challenges use.  `nabla @ T` then `chart.evaluate` is the
*coordinate-free* route: you write the physics without choosing coordinates,
then evaluate.  Those are complementary, not duplicated, and the M3 job is to
**say which is for what**, not to delete one.  Making `chart.grad` internal
would remove a good API and break six certified challenges to no benefit —
so this brief declines that part of the M2/M3 sketch.

The genuinely redundant route is `tender.operators`: a *Python-side shadow
AST* that does not compose with `Expr` (no canonicalize, no mixing into sums,
no engine).  Per the vibe-000093 decision ledger it becomes sugar emitting
core `Expr` rather than being atticed.

**2. Chart-only coordinate systems are nearly done; the real gap is
boilerplate.**  Only three files still touch the old curvilinear factories
(`tb.cylindrical` and friends), and two are their own tests.  Meanwhile the
suite hand-writes **16 chart constructions and 9 cylindrical embeddings** —
`ws.chart(ws.wcs(), [r, th, z], [r*cos(th), r*sin(th), z])`, over and over,
with the `nonneg=("r",)` detail easy to forget and silently load-bearing
(it licenses √(r²) → r).  So the valuable move is not retiring the old
factories — that is a small cleanup — but giving the common charts a name.

## Increments

Each keeps all suites green.  Scoreboard moves are not expected; M3 is
surface work, and any promotion is a signal to stop and look.

1. **Named charts, and the coordinate-system route closed.**
   `Workspace.cylindrical_chart()` / `spherical_chart()` / `polar_chart()` /
   `cartesian_chart()`, each minting its coordinates (with `nonneg` set
   correctly) and returning a chart plus its coordinates.  Migrate examples,
   challenges and tests; retire the curvilinear well-known-basis factories and
   `coord_system.cpp` to the attic (`wcs()` stays — charts need the reference
   frame).  *Done when:* no example or challenge hand-writes a standard
   embedding, and `attic/` holds the old factories.

2. **One differentiation story, told properly.**
   `tender.operators` becomes sugar that emits core `Expr` (so
   `nabla * f` composes with everything) or, where that is not faithful, is
   atticed with its three callers migrated.  `chart.grad/div/rot/laplacian`
   and the `t.nabla` + `chart.evaluate` route both stay public, with
   docstrings that say when each is the right tool.  `td.deriv` /
   `apply_operators` / `td.partial` are documented as the low-level scalar
   layer.  *Done when:* one obvious way to write "the gradient of u in this
   chart", one for "coordinate-free, evaluated later", and no third thing.

3. **The verb surface becomes the public surface.**
   Verbs (`prove_equal`, `simplify`, `expand`, `factor`, `rules`) exported at
   top level and, where natural, as `Expr` methods.  The genuinely internal
   steps — measured: `distribute_contraction`,
   `fold_equal_addends_structural`, `implicitize`, `saturate` (superseded by
   `engine_simplify`), plus the operator-layer pair — move to
   `tender.derivation.steps` with shims.  Public docstrings rewritten in user
   vocabulary: **no vibe numbers in the public API docs** (they belong in the
   vibes and in comments, not in what a mechanician reads).
   *Done when:* `dir(td)` reads as a vocabulary, not a catalogue; every
   remaining public name has a docstring aimed at a user.

4. **Errors and defaults reach the user in their own terms.**
   - A verb must never surface a canon-internal message.  Today
     `prove_equal` on challenge 000010 raises
     `ValueError: encapsulate: unsupported factor node (a nested ⊗ inside an
     operand awaits fence distribution)`.  Wrap it: verbs report *what they
     could not do* and return a result where that is meaningful
     (`Unsupported`, beside `Proved` / `Refuted` / `Exhausted` / `Budget`).
   - Context-scoped budget defaults (vibe 000097 left this open): the verbs
     reach a `Context` through their expressions, so once they are
     Context-aware the session default becomes a per-Context one.
   *Done when:* challenge 000010's L2 xfail reports a tender-level reason, and
   a budget can be set per Context.

5. **Notebook experience.**
   `Derivation` gains `_repr_html_`: the step table with per-step fired flags
   (M1 increment 3) and rendered expressions — the derivation as a user reads
   it, not as a Python repr.  Integrate the labeled-path view
   (`tender.render.labeled`, vibe 000054), and give `ProofResult` a rich
   display showing which identities fired.  *Done when:* a notebook shows a
   derivation as a readable table, and the examples' notebooks use it.

## Deliberately out of scope

- Challenge promotions (M4) — including 000010, whose blocker is a canon
  capability, not the API.
- New identities or engine rules (M2 is closed).
- The physics arcs (M5).

## Risks

- **Increment 1 touches every example and challenge.**  Mechanical, but the
  suite is the gate; migrate in one commit per file group, not one big sweep.
- **Increment 3 is where the "single way to do a thing" principle can be
  overdone.**  Demote only what the audit shows is internal; a step used by a
  challenge is public by definition.
- The operators-as-sugar decision (increment 2) may prove unfaithful — the
  shadow AST defers evaluation, and core `Expr` may not express that without
  a chart.  If so, attic it and migrate the three callers, which the ledger
  already permits.
