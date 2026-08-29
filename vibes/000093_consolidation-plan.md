# 000093 Consolidation plan — harness, IR, engine, API, then new physics

The implementation plan for the vibe-000092 decisions. The pieces are
entangled — the verb API wants the e-graph engine, the e-graph keys its e-node
dedup on the canonical form, the canonical form is what the IR consolidation
changes, and "done" for all of them is defined by the challenge suite. The plan
untangles this into one dependency spine:

```
M0 harness  →  M1 IR  →  M2 engine  →  M3 API  →  M4 certification (L2)
                                                        ↓
                          M5A applied-mechanics arc   M5B continuum arc
```

The spine is sequential because each stage is the foundation the next one
keys on; M5's two arcs are independent of each other and can interleave.
Throughout, the CLAUDE.md rule holds: the system stays alive at every
increment — no big-bang rewrite; old forms keep working behind shims until
their replacements are proven by the suite.

## Why this order

- **Harness first** (M0): it converts "new challenges fail most of the time"
  into a measurable, named list, and it is the safety net under every later
  change. It needs no product-code changes, so it cannot be blocked.
- **IR before engine** (M1 < M2): the e-graph's hash-consing keys on the
  canonical form. Reviving the e-graph on today's ANF and then flattening the
  IR means porting the engine twice. Flatten first, port once.
- **Engine before API** (M2 < M3): the goal-directed verbs (`prove_equal`,
  `simplify`) are thin wrappers *if* the engine exists; without it they would
  have to be re-implemented as step pipelines and rewritten later.
- **API before the L2 push** (M3 < M4): L2 certification is defined as "the
  derivation runs on the public surface with no internals", so the public
  surface must exist before challenges can be graded against it.
- **Physics last** (M5): every new capability (rotation tensors, formal small
  parameter, time/variation, complex scalars) multiplies on top of a stable
  engine. Adding them before consolidation would multiply the current
  step-ordering pain instead.

## M0 — Certification harness

- `challenges/` directory, one **directory per challenge** (refined
  2026-08-03 — a challenge is constituted by multiple files, including
  scanned hand-made derivations):

  ```
  challenges/
    NNNNNN_descriptive-name/
      test.py     # the runnable assertion of the claim
      meta/       # free-form human material: scans, notes, references
  ```

  `NNNNNN` is a six-digit global sequence in creation order, exactly the
  vibes convention; no tier letter in the name — the tier (vibe 000092 §4)
  is recorded in challenge metadata, since tiers are a planning taxonomy
  that may be reorganized and a challenge can span tiers. `test.py` is a
  fixed name (the directory carries the identity); pytest must run with
  `--import-mode=importlib` so same-named test modules don't collide.
  Each `test.py` declares machine-readable metadata — tier, title, source
  (book + section where applicable) — exact shape settled in the M0 brief;
  `meta/` inner structure is refined only when needed.
- A pytest-based runner plus a CI job; a generated scoreboard table
  (challenge → L0 failing / L1 verified / L2 performed, grouped by tier
  from metadata) committed or rendered in CI output.
- L1 = endpoint confirmed by component check (`algebraic_eq` after chart
  expansion). L2 = direct-notation derivation using only the documented
  public surface, no trial-and-error steps.
- Seed immediately with what already passes (tier D regressions: ∇R = I,
  Δr² in cyl/sph, rot(r e_θ); tier E: Navier–Lamé, strain compatibility,
  the elastic-energy reduction; tier A: the eps-delta and cross-identity
  example content). These start at L1; almost none start at L2 — that gap *is*
  the product metric.

Exit: scoreboard exists, CI-run, with the full A–E tier list from vibe 000092
enumerated (unimplemented challenges marked as expected-fail, not skipped
silently — the red is the roadmap).

## M1 — IR consolidation

Four increments, each keeping all tests and the M0 suite green.

1. **Flattened additive/multiplicative forms.** One n-ary additive node whose
   terms each carry a rational coefficient; one n-ary product node with a
   single scalar-rational prefactor. `Difference`, `Negate`, `ScalarDiv`
   become construction-time sugar and render-time forms only. The
   canonicalizer's output *is* this form; steps consume only it. This deletes
   the "six additive shapes every step must peel" problem (vibe 000091's
   `expand_double_dot` shape list is the regression test).
2. **Traversal combinators.** `map_addends`, `map_factors`,
   `bottomup_fixpoint` in one module; port the three worst offenders
   (`expand_double_dot`, `distribute_contraction`, `fold_equal_addends`) onto
   them as the proof of concept. Expected effect: `derivation.cpp` shrinks
   substantially; step coverage becomes uniform by construction.
3. **Step-boundary invariant + reporting.** A single wrapper gives every step
   the canonical-implicit contract (vibe 000062 promoted from convention to
   mechanism) and a fired/no-op report. `Derivation.step` surfaces no-ops
   (warning by default, `optional=True` to silence). `apply_identity` stops
   returning a canonicalized tree when it did not match.
4. **∇ fence as node data.** Operator applicativity becomes an explicit
   attribute the canonicalizer respects; the `Paren`-as-fence encoding
   (vibe 000085) is deleted. Render output unchanged.

Exit: suite green; the only observable rendering change is the vibe-000056 #3
class of minus-sign artifacts disappearing.

## M2 — Engine revival (equality saturation as the reasoning core)

1. **Port the e-graph to the flattened ANF.** Union-find/hash-cons/rebuild/
   extract and the e-matcher re-keyed on the M1 canonical form; benchmarks
   kept and re-baselined.
2. **Identity library as a first-class asset.** Organize `tender/identities`
   into named rule groups: `eps_delta`, `dyadic` (tr/vec/transpose/dyad
   expansion), `double_dot`, `cross` (bac-cab, Lagrange), `basis`
   (completeness, frame dots), later `leibniz`. Each rule marked directed or
   bidirectional; each verb selects groups, never "all rules".
3. **Verbs on the engine.**
   - `prove_equal(lhs, rhs)` — saturate until the two roots share an e-class
     (or budget exhausted); returns a result object, not a bare bool.
   - `simplify(expr)` — bounded saturation + ε-weighted cost extraction
     (vibe 000046's cost model, revalidated).
   - `expand(expr, what=...)` / `factor(expr)` — *directed* rewriting; these
     deliberately bypass saturation where a confluent directed pass is
     cheaper and predictable.
4. **Explanation.** Minimum viable: a rule-firing trace (which identities,
   where) attached to verb results, so an L2 derivation can show its work.
   Full e-graph proof extraction is a stretch goal, not a gate.

Blowup risk and mitigations: AC is handled by the ANF (never as rules);
distribution rules are the known explosive ones — budgeted iterations/node
counts, rule scheduling (cheap groups to fixpoint before explosive ones), and
a documented fallback to the directed pipeline when the budget trips.

Exit: tier-A challenges pass at L2 *via the verbs* (`prove_equal` one-liners);
saturation benchmarks within agreed budgets.

## M3 — API unification

1. **One differentiation route.** Core invariant `t.nabla` + `chart.evaluate`
   is the blessed path (vibe 000084). `tender.operators`' shadow AST either
   becomes sugar that emits core `Expr` or moves to the attic;
   `chart.grad/div/rot/laplacian` become internal (used by `evaluate`).
2. **Chart-only coordinate systems.** `Workspace` gains
   `cylindrical_chart()` / `spherical_chart()` / `polar_chart()` conveniences
   (chart + coords minted in one call); tests migrate; `coord_system.cpp` and
   the curvilinear well-known-basis factories retire to the attic. `wcs()`
   stays — charts still need the orthonormal reference frame.
3. **Public verb surface.** Verbs exported at top level and as `Expr` methods
   (`expr.simplify()`, `expr.expand(frame)`, …); the step catalog moves to
   `tender.derivation.steps` with deprecation shims; docstrings rewritten in
   user vocabulary (no vibe numbers in the public docs).
4. **Notebook experience.** `Derivation` rich display: per-step rendering with
   the rule-firing trace; labeled-path view (vibe 000054) integrated.

Exit: examples rewritten on the public surface only; no example or challenge
imports a demoted step.

## M4 — Certification to L2

Iterate over the tier A–E scoreboard: for each challenge not at L2, either the
verbs handle it (promote) or the failure names a concrete gap (file it, fix
it, promote). No new features except gap fixes. This milestone is where the
vibe-000056 usability thesis — "a correct derivation the user cannot discover
is not a usable system" — is declared resolved or not, with the scoreboard as
evidence.

Exit: tiers A–D fully L2; tier E at L2 with any principled exceptions
documented per challenge.

## M5 — Capability tracks (the new physics)

Both arcs sit on the consolidated core; they are independent of each other.
Every new node kind added here follows a checklist from day one: ANF rules +
identity-group entries + render + challenge, in the same increment.

### M5A — Applied-mechanics arc (Gantmacher; Zhilin/Eliseev; Neimark–Fufaev)

1. **Time & variation**: q(t) generalized coordinates, total d/dt, the
   variation δ with integration by parts in time. This is shared
   infrastructure — the same δ machinery serves the continuum arc's Ritz
   route (deliberate entanglement exploited once, built once).
2. **Rotation tensors** (angle-free first, per Zhilin/Eliseev): proper
   orthogonal tensors, rotation about an axis, angular-velocity tensor and
   vector, Poisson kinematic equations; Euler and other angle sets as derived
   parameterizations, not the foundation.
3. **Challenges**: virtual-work equilibrium of a constrained linkage; Lagrange
   equations of pendulum/double pendulum; inertia dyadic → Euler's dynamic
   equations; then Appell's equations with quasi-velocities; Neimark–Fufaev
   nonholonomic problems. Hamiltonian mechanics queued after these (design
   driver TBD, per vibe 000092 §6b).

### M5B — Continuum arc (Eliseev)

1. **Cross-section/domain resultants**: definite integrals over an
   unspecified domain as first-class named quantities (area, static moments,
   inertia moments) — the semi-inverse method's bookkeeping. Gates
   Saint-Venant.
2. **Saint-Venant generalized problem** (extension, bending, torsion) — first
   big challenge of the arc; needs 1 but not the small parameter.
3. **Formal small parameter & series**: formal ε, truncated expansions,
   `collect_orders` — gates rods, shells/plates (Kirchhoff–Love), and crack
   asymptotics (SIF).
4. **Rods → shells/plates → cracks**, in that order (each reuses the
   asymptotic machinery of the previous).
5. **Complex scalars + z/z̄ calculus** for Muskhelishvili 2D elasticity —
   explicitly last; inelasticity/nonlinear kinematics remain out of plan.

Challenge sourcing for both arcs: derivations lifted directly from the user's
Russian PDFs of the section-5 books; each challenge file cites book + section.

## Status — the spine is complete (2026-08-25)

**M0–M4 are done.**  Briefs and outcomes: M0 vibe 000094, M1 000095, M2 000096,
M3 000098, M4 000099.  Scoreboard **16 L2 / 6 L1 / 1 L0** of 23 challenges,
from a starting point where the suite did not exist.

Three places the audits **declined** what this plan sketched, each recorded
where it happened:

- **M1**: the flattened IR already existed (`nf::Nf`); the defect was that
  steps re-peeled the *raised* tree.  M1 became "close the gap", not "build
  the IR".
- **M2**: the e-graph was already Nf-native with subtree pattern variables.
  M2 became "put the engine to work", not "port it".
- **M3**: "make `chart.grad` internal" would have deleted a good API used by
  six challenges.  Declined; the redundant route was `tender.operators`.

Carried forward, in rough order of value:

1. **An invariant Leibniz rule group** (∇ over a product).  Blocks challenges
   000012, 000013, 000019 — the gateway to tier C, and the continuum arc will
   need it immediately.  The most valuable single next thing.
2. **The context-blocking problem** — vibe 000100.  Not a feature but a design
   question, and the one most likely to keep costing us: seven instances so
   far.  Its resolution touches ε-reassembly (000017), the metric fold
   (000018) and probably more.
3. **Fence distribution inside a contraction operand** — blocks 000010.
4. **Inverse chart embeddings** (vibe 000090 approach B) — blocks 000021, the
   last L0.
5. **Deferred from M3**: ~~atticing `tender.operators`~~ **done** (2026-08-29,
   vibe 000098 postscript — the test file was mostly core ∇ tests, not DSL
   ones); Context-scoped budget defaults (needs Context-aware verbs) remains.
6. **Open from M1**: `fold_equal_addends` may be canon + implicitize.
7. **`ddot-identity` has no derivation** — the one open proof obligation in
   the identity DAG.

**M5 has not started.**  Both arcs stand as written below.  Worth noting
before it does: M5B's first item (cross-section resultants → Saint-Venant)
needs no new engine capability, while everything in M5A's item 1
(time & variation) and much of tier C needs the Leibniz group — so (1) above
is on the critical path for the applied-mechanics arc but not for the
continuum one.

## Execution notes (for whichever model implements this)

This plan is strategy, not a work order. The protocol for executing it:

**Per-milestone briefs.** At the start of each milestone, write a new vibe
(the project's established practice, cf. vibe 000078) breaking it into
increments, each with a one-line testable "done when". Do not implement from
this vibe directly, and do not detail a future milestone early — M2+ details
would go stale under M1's changes.

**Decision ledger — pre-resolved so no mid-flight judgment is needed:**

- `tender.operators` shadow AST → becomes **sugar emitting core `Expr`**
  (keeps notebooks working); attic only if the sugar turns out to fight the
  core. Cheap to revisit.
- Saturation budgets (initial, revisit with benchmarks): **30 iterations /
  10k e-nodes** per verb call; on budget trip, fall back to the directed
  pipeline and say so in the result object — never fail silently.
- E-graph proof extraction: **out of scope for M2.** The rule-firing trace is
  the M2 deliverable; full proofs are a later, separate discussion.
- Anything not covered here that changes a public surface or deletes code:
  **ask Stepan** — do not resolve by picking the smaller diff.

**Guardrails (repeating the ones that get violated under pressure):**

- Never encode AC/commutativity as e-graph rules — the ANF owns that.
- Never encode semantics in presentation nodes (the vibe-000085 lesson).
- Attic, never delete; shims live one milestone.
- A red challenge is a roadmap entry, not a failure to hide: expected-fail,
  never skip.
- Keep every increment alive (build + tests green) — no long-lived broken
  intermediate states, per CLAUDE.md.

## Working agreements

- The scoreboard is the single definition of progress; a feature PR that
  doesn't move it (or unblock a mover) should justify itself.
- Attic, don't delete: `coord_system`, the operators shadow AST, and any
  superseded steps go to `attic/` like `attempt_01` did.
- New node kinds ship with ANF + identities + render + a challenge in the
  same increment — no representation debt of the vibe-000085 kind.
- Deprecation shims live for one milestone, then go to the attic.
