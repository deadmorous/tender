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

## Progress record

- **Increment 1 DONE** (2c09cce): named charts (`cartesian_chart` /
  `cylindrical_chart` / `spherical_chart` / `polar_chart`, each minting its
  coordinates with `nonneg` correct and returning `(chart, coords)`); all 16
  hand-written chart constructions migrated, zero remain.  Because that
  removed every instance of arbitrary construction, `examples/custom_chart.py`
  was added — parabolic cylindrical coordinates from nothing but the
  embedding.  **Finding recorded there:** on a chart whose scale factors are
  *surds*, `∇R = I` is not recognised, because `simplify_scalars` cannot
  cancel `(σ²+τ²)/√(σ²+τ²)`.  The named charts hide this (their scale factors
  are `r`, `r sinθ`, `1`); a user's own chart likely will not.  The geometry
  is correct regardless — a simplification gap, now documented where a user
  meets it, and a backlog item.

- **Increment 2 PARTIAL**: `Workspace.nabla()` gives the composable
  coordinate-free operator as a real `Expr`, and `tender.operators` is marked
  superseded in its module docstring with the core route spelled out.
  **Deferred:** atticing it and migrating callers.  The brief said "three
  callers", which understated the work — one of them is a 982-line test suite
  written against the shadow AST.  Mechanical but large; better as its own
  commit than squeezed into this milestone.

- **Increment 3 DONE**: `tender.steps` holds the six internals the audit
  measured as uncalled (`saturate`, `implicitize`, `distribute_contraction`,
  `fold_equal_addends_structural`, `deriv`, `apply_operators`); the advertised
  surface drops 42 → 36 names.  They still import, so nothing breaks.

- **Increment 4 DONE (errors)**: `ProofStatus::Unsupported` + `detail`, and
  `SimplifyResult::unsupported`.  A canonicalization failure is now reported
  as a statement about *tender* — "tender cannot yet put this expression in
  canonical form: …" — instead of escaping as `ValueError: encapsulate:
  unsupported factor node`.  Challenge 000010 asserts the new behaviour; its
  xfail stands, since being reported is not being proved.  **Deferred:**
  Context-scoped budget defaults, which need the verbs to take a Context.

- **Increment 5 DONE**: `Derivation._repr_html_` renders the derivation as a
  table with per-step fired marks, and `ProofResult._repr_html_` shows the
  verdict, the identities that produced it, and — where relevant — that the
  rule set rather than the claim is what fell short.

- **`doc/cheatsheet.md` updated** through M3, including a new section on the
  verbs, which the headline change of M2/M3 had left undocumented.  Every API
  named in the file was checked to exist.

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

## Postscript: `tender.operators` atticed (2026-08-29)

The last M3 leftover, and it turned out the module was the smaller half of the
job.  `python/tests/test_operators.py` was 43 tests, but only **7** actually
touched the DSL — the other 36 were about the ∇ *node* all along (expand_nabla,
reassemble_nabla, chart.evaluate, the Navier–Lamé and strain-compatibility
endpoints).  The file was misnamed, which is why it looked like 1001 lines of
migration debt.

Determined by deleting the import and reading the failures rather than by
grepping: a first pass by regex misclassified `t.nabla(...)` as DSL use and got
the count wrong.

- 36 stayed, as `python/tests/test_nabla.py`.
- 3 of the 7 covered node behaviour the DSL merely wrapped and were **ported**
  to the core route: parenthesised operands, the directional derivative
  `(v·∇)R = v`, and `chart.nabla()` reproducing the gradient.
- 4 asserted the DSL's own surface and went to `attic/operators_dsl/` with the
  module and a README mapping every DSL form to its core equivalent.

`examples/field_operators.py` was ported, not deleted (CLAUDE.md principle 5),
and the README's opening snippet with it — it now runs verbatim and still prints
exactly the divergence it documents.  One nice consequence: the cheatsheet's
"two ∇/Δ surfaces — don't confuse them" warning is gone, because there is one.
