# 000096 M2 brief — engine revival, increment breakdown

The per-milestone brief for M2 (vibe 000093), grounded in an audit of
`nf_egraph.{hpp,cpp}`, `nf_match.{hpp,cpp}`, `identities.{hpp,cpp}`, the
`_saturate` binding, and the saturation benchmark.

## The audit finding that reshapes M2

Vibe 000093 sketched M2.1 as "port the e-graph to the flattened ANF".
**That port already happened**: the vibe-000034 Expr-structural `EGraph` is
gone from the tree, and what stands is `NfEGraph` (vibe 000058 / C14d) —
Nf-native e-nodes (`Factor`/`Term`/`Sum`), union-find, hash-consing,
congruence `rebuild`, ε-weighted `extract` (the vibe-000046 Levi-Civita
weight), and a `saturate` that fires identities through the `nf_match`
matcher.  Better still, **subtree pattern variables exist** (vibe 000051): a
slot-less, non-well-known named tensor in a rule's LHS binds a whole target
factor — so invariant-level rules (bac-cab, dyadic identities) are
expressible today, not just index rules.  `td.saturate` already drives all
of this from Python.  M1's ParenKind is already keyed into e-node congruence.

So M2 is not "build the engine" — it is **put the engine to work**: verbs
with budgets and a trace, a real rule library, the two engine gaps that
block it, and the tier-A scoreboard promotions that prove it.

Known limits found in the audit, addressed below:

- `saturate` skips multi-term-LHS rules (no Nf sub-sum matcher) — rules can
  only fire in the "expand" direction when their other side is a sum.
- Pattern variables bind `Factor`s, not arbitrary sub-`Nf`s.
- `saturate` caps passes but has **no node budget** and reports only a pass
  count — no reason, no per-rule trace.
- The M1 observation: `place_factors`' `has_operator` does not see an
  operator inside a fence factor (possible term-placement scope bug).

## Progress record

- **Increment 1 DONE** (ac90848): `SaturateBudget` / `SaturateOutcome` /
  `SaturateReport` (per-rule fired counts + *skipped* uncompilable rules),
  a `stop()` goal check, and the `tender::engine` verbs `prove_equal`
  (three-valued `Proved` / `Exhausted` / `Budget`) and `simplify`
  (best-form-so-far on a budget trip).  Python: `td.prove_equal`,
  `td.engine_simplify`, both warning `BudgetExceeded`.

- **Increment 2 DONE**: rule library as groups — `eps_delta` (4, existing),
  `cross` (bac-cab, cross-identity, cross-removal, Lagrange), `dyadic`
  (trace-cyclic, identity-dot); `identities::group/group_names/all_rules`,
  Python `td.rules("cross")` / `td.rule_groups()`.  Every rule fire-tested
  at birth.  **Findings, all from those tests:**

  1. **Canon already does much of the dyadic work.**  `tr(a⊗b) = a·b`,
     `vec(a⊗b) = a×b`, `(a⊗b)ᵀ = b⊗a`, `(a⊗b)··(c⊗d)`, `(Aᵀ)ᵀ = A`,
     `a·b = b·a` and `tr(I) = n` are all *canon-equal already* —
     `prove_equal` proves them in **0 passes with no rules**.  They were
     dropped from the library rather than shipped as inert decoration.
     Consequence: challenge 000004 (`a·b = b·a`) needs no rules at all.

  2. **Pattern-variable NAMES are load-bearing — a real engine defect.**
     Canon sorts a symmetric contraction chain (`:`/`··`, rank-1 `·`) by
     tensor name and the matcher compares chain factors *positionally*, so
     a rule's variable name decides which targets it matches.  Measured:
     the rule `X··I = tr X` fires when its variable is named B/C/H and
     silently fails when it is named X/Z — because the target `I··A`
     canonicalizes to `A··I` while the pattern `I··X` stays `I··X`.  **No
     single spelling works for all targets**, so the double-dot-with-`I`
     rules are deliberately NOT in the library.  The fix is AC matching
     (*associative-commutative*: matching modulo the operand orders the
     operator permits, rather than positionally — see
     `vibes/glossary.md`) for symmetric chains — promoted to increment 3 as the *third* engine gap,
     and the most important one.  Every shipped rule is guarded by a
     name-robustness sweep test.

  3. **Soundness holds where it matters.** A subtree variable binds any
     factor *regardless of rank*, so bac-cab could in principle fire on
     `a×(B×c)` with `B` rank-2 — where the identity is false.  It does not:
     canon's rank-2 fence reassociation (vibe 000055) puts that expression
     in a different shape.  Now pinned by an explicit soundness test rather
     than left to luck.

  4. `A··(b⊗c) = c·A·b` cannot even be *stated* today: canonicalizing it
     throws `encapsulate: unsupported factor node (a nested ⊗ inside an
     operand awaits fence distribution)`.  Filed for increment 3.

- **Library moved to Python** (7058525), on user direction: rules are data,
  and authoring them is empirical, so they belong in the language users
  iterate in — and extending the set needs no rebuild.
  `python/tender/identities.py` holds all rules + the group registry;
  `src/identities.{hpp,cpp}`, the `_rule_group` binding and
  `identities_test.cpp` are gone (the C++ library was a leaf — no engine
  code called it).  Coverage moved to `python/tests/test_identities.py`.
  Accepted tradeoff: a C++-only consumer gets the engine but no shipped
  rules; tender is Python-first.

- **Increment 3 DONE**: all three engine gaps resolved or bounded.

  1. **AC matching for symmetric chains — FIXED.**  `match_factor`'s
     Contraction arm now retries with the operands swapped, gated by
     `contraction_commutes_factors`, a Factor-level mirror of canon's own
     `contraction_commutes` (so `·` swaps only between two rank-1 vectors,
     `:`/`··` only between two rank-2 tensors).  The measured defect is
     gone: `X··I = tr X` now fires for every variable/target naming, where
     before *no* spelling worked for the whole alphabet.  Soundness pinned
     by a test that a **directional** contraction (`A·b` vs `b·A`) is still
     matched strictly.  This unblocked the `double_dot` group, now shipped.

  2. **`place_factors` fence scope — a real bug, FIXED, and worse than the
     M1 note guessed.**  The fence was not mis-*placed*, it was never
     *formed*: `multiplicative_flatten` flattened `λ μ (∇⊗u)` to the loose
     factors `[λ, μ, ∇, u]`, so ∇ never reached `encapsulate` as a
     TensorProduct and no `OperatorFence` Paren was built.  Two consequences,
     both measured: the rendered form read `∇ λ μ u` — the operator's scope
     silently widened — and, since a term holding a top-level operator keeps
     *every* factor positional, `λμ(∇⊗u)` and `μλ(∇⊗u)` canonicalized to
     **different** normal forms, i.e. canon was not canonical.  Fix:
     `flatten_operand` keeps a ∇-bearing ⊗ operand whole.  Deliberately
     ∇-only — a concrete `Deriv` product is the frame-expanded route canon
     means to distribute, and fencing those lands on encapsulate's throw
     (caught immediately by `test_express_div_of_symmetric_gradient_stress`).

  3. **Multi-term LHS — bounded by evidence, deferred.**  The boundary test
     settles it: `prove_equal` does **not** need a sub-sum matcher, because
     both sides saturate in one graph and a forward rule closes the gap from
     either end (proved, `fired={bac-cab: 1}`, starting from the expanded
     side).  It bites only `simplify`'s *factoring* direction, where no
     compilable rule introduces the compact form — recorded as a strict
     xfail (`test_simplify_can_factor_an_expanded_form_back`), with skipped
     rules already visible in the trace since increment 1.

  Benchmarks after the changes: `delta-contraction` 2 passes / 7 nodes,
  `eps-delta-2` 2 passes / 7 nodes — both far inside the default budget
  (30 passes / 10k nodes).

- **Increment 4 DONE — M2 COMPLETE.**  Scoreboard **9 → 14 at L2**
  (3 L1, 6 L0 unchanged).  Five promotions, each its own commit:

  | # | Challenge | How it proves | Honest? |
  |---|---|---|---|
  | 000004 | `a·b = b·a` | `prove_equal` with an **empty** rule set — canon decides it in 0 passes | nothing cited |
  | 000014 | Lagrange | 9-step ε-δ derivation, every step asserted to fire | **derived** |
  | 000005 | `a×I = I×a` | proved *and* rewritten by `engine_simplify` from the left side alone | derived-ish |
  | 000015 | `a×(b×I)` | one call, `cross` group | cites `cross-removal` |
  | 000016 | `tr(A·B)=tr(B·A)` | one call, `dyadic` group | cites `trace-cyclic` |

  **On circularity.**  Proving a stated identity by citing the library rule
  *for that identity* certifies little, so each promotion says in its
  docstring which it is.  Two are cite-based (000015, 000016) and both are
  defensible: the cited identity is independently verified in components by
  the *same challenge's* L1 test and carries its own fire-test, and citing a
  standard identity from a toolbox is what a human does.  000014 was
  deliberately upgraded from a citation to a real derivation once the ε-pair
  route turned out to work (the δ from the frame dot has to be contracted
  *first*, to bring the two ε's onto a shared index).

  **Discovery vs verification.**  `engine_simplify` reaches the answer from
  the problem alone only when the target is *cheaper*: it rewrites `a×I` to
  `I×a` and `I·a` to `a`, but keeps `a×(b×c)` rather than expanding it —
  correctly, since the compact cross form has fewer nodes.  Expansion-
  direction identities are therefore `prove_equal` material, not `simplify`
  material.  This is the cost function behaving, not a gap.

  **000010 (elastic energy) not promoted — no forced win.**  Its xfail now
  names the true blocker: the rule it needs (`A··I = tr A`) exists and
  fires; what fails is *stating* the problem, since `T = λ tr(ε) I + 2με`
  puts a ⊗-product inside the double-dot operand and `encapsulate` rejects
  it ("awaits fence distribution").  A second defect surfaced here and is
  filed for M3: `prove_equal` *raises* that canon-internal error instead of
  returning a result — a goal-directed verb should never leak an internals
  message.

  **Benchmarks** (invariant rules added to `egraph_saturate_bench`, built
  locally there since the library is Python now):

  | case | passes | nodes | ns/op |
  |---|---|---|---|
  | delta-contraction | 2 | 7 | 185k |
  | eps-delta-2 | 2 | 7 | 360k |
  | bac-cab | 2 | 12 | 371k |
  | cross-removal | 2 | 11 | 325k |

  No blow-up: the cross group — the one with right-hand sides larger than
  its left, i.e. the explosive shape — saturates in 2 passes and ~12 nodes,
  three orders of magnitude inside the 10k-node default budget.

## Increments

Each keeps all suites green.  Scoreboard moves are expected ONLY in
increment 4, each as an explicit strict-xfail-removal commit.

1. **Verbs on the existing engine: budgets, outcome, trace.**
   - `Budget{max_passes = 30, max_nodes = 10'000}` (the vibe-000093 decision
     ledger).  `NfEGraph::saturate` learns the node budget and returns an
     outcome — `Saturated` (fixed point) / `PassBudget` / `NodeBudget` —
     plus a per-rule fired count (the rule-firing trace; full proof
     extraction stays out of scope per the ledger).
   - `prove_equal(ctx, lhs, rhs, rules, budget)`: add both sides, saturate,
     proved ⇔ same e-class.  Returns a result object — proved / not-proved-
     within-budget are *different* answers (a budget trip must never read as
     "not equal"), with passes, node count, and the trace.
   - `simplify(ctx, e, rules, budget)`: saturate + ε-weighted extract;
     result carries the same outcome + trace.  On a budget trip the caller
     still gets the best extraction, loudly marked.
   - Thin provisional Python bindings (`td._prove_equal`-style; the blessed
     public names are M3's), with the loud-fallback warning on budget trips.
   - *Done when:* `prove_equal` proves the four existing library identities'
     targets; a deliberately tiny budget yields the budget outcome (test);
     the trace lists which rules fired how often.

2. **The rule library becomes a first-class asset.**
   Reorganize `tender/identities` into named groups with a registry
   (`identities::group("eps_delta")`, `all_groups()`), each rule carrying a
   name and a direction policy (directed, or bidirectional where the
   reversed LHS is single-term — emitted as two directed rules):
   - `eps_delta` — the four existing rules, regrouped.
   - `cross` — bac-cab `a×(b×c)`, `a×I = I×a`, the Lagrange contraction
     `(a×b)·(c×d)`.
   - `dyadic` — `tr(a⊗b) = a·b`, `vec(a⊗b) = a×b`, `(a⊗b)ᵀ = b⊗a`,
     `(Aᵀ)ᵀ = A`, `I·x = x`, `tr(I) = n` (dimensioned).
   - `double_dot` — `(a⊗b):(c⊗d)` both pairings, `I··A = tr A`,
     `A··(b⊗c) = c·A·b`.
   - `basis` (minimal) — completeness `Σ_i e_i⊗e_i = I` and frame-dot rules,
     factory-parameterized by a `Basis`; grown only as far as increment 4
     needs.  Differential/Leibniz rules are explicitly NOT an M2 group (see
     out-of-scope).
   Every rule lands with a test that it *fires* under `saturate` on a
   minimal target (the vibe-000040 lesson: canon α-renaming and symmetry
   normalization can silently kill a match — per-rule tests catch that at
   birth).
   - *Done when:* every tier-A challenge identity has its rules present and
     firing in isolation; groups are selectable per verb call.

3. **The two engine gaps.**
   - **Multi-term LHS:** decide by evidence, not upfront.  For
     `prove_equal`, both sides are saturated, so single-term forward rules
     can meet in the middle and a sub-sum matcher may be unnecessary; for
     `simplify`, the factoring direction needs it.  Increment 3 writes the
     boundary test (a proof that needs the factoring direction), and either
     implements the term-subset matcher or records the limit in this vibe
     with the failing test as a strict xfail.  No silent skipping: `saturate`
     reports skipped multi-term rules in the trace.
   - **`place_factors` fence scope** (M1 observation): determine whether a
     term like `s · Paren-fence(∇⊗u)` mis-places `s` into the commutative
     scalar region; fix (has_operator looks through `OperatorFence` parens —
     now trivial, the kind is data) or document why it cannot matter.  With
     a regression test either way.
   - *Done when:* both items closed with tests; any deferral is a strict
     xfail plus a paragraph here.

4. **Tier-A promotions + benchmark baseline.**
   The six strict-xfailed L2s of tier A/B become the acceptance suite:
   000004 (a·b = b·a), 000005 (a×I = I×a), 000010 (energy T··ε
   invariantly), 000014 (Lagrange), 000015 (a×(b×I) — THE vibe-000056
   case), 000016 (tr(A·B) = tr(B·A)).  Each becomes a verb one-liner in its
   L2 test; each promotion is its own commit removing the strict xfail.
   A challenge the engine genuinely cannot reach gets its xfail reason
   updated to name the missing piece instead — no forced wins.
   Benchmarks: extend `egraph_saturate_bench` with the cross/dyadic groups
   on the challenge shapes; record passes/nodes/wall-clock here and confirm
   the default `Budget` covers them with headroom.
   - *Done when:* the scoreboard shows the promotions (target: all six; the
     honest number is whatever survives), and this vibe records the bench
     numbers.

## Deliberately out of scope

- Public naming, `Expr` methods, step demotion — M3.
- Full e-graph proof production — decision ledger; the trace is the M2
  deliverable.
- Differential/Leibniz rule groups (∇ inside the e-graph): the fence
  machinery makes this a correctness minefield (vibes 000085/000088) and
  tier-C L2s are M4 material; M2 keeps ∇-bearing terms inert in the graph
  (fence parens are opaque congruence nodes — already the case).
- Chart/coordinate lowering through the engine.

## Risks

- **Blowup**: cross + dyadic rules are the explosive kind.  Mitigations:
  node budget (increment 1 lands before the rules), per-verb groups (never
  "all rules"), AC handled by canon not rules, and the benchmark gate in
  increment 4.
- **Canon eats matches**: symmetry/α normalization can make a textbook rule
  unmatchable in canonical form (vibe 000040).  Mitigation: per-rule
  fire-tests at birth (increment 2's rule).
- **Meet-in-the-middle optimism**: if increment 3's boundary test shows
  `prove_equal` needs the factoring direction after all, the sub-sum matcher
  becomes real work — it is the one item in M2 with genuine design risk,
  which is why it is isolated in its own increment with an honest-deferral
  exit.
