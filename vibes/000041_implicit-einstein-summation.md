# 000041 Implicit Einstein summation in the canonical form

Vibe 000028 is explicit: **"Every index occurrence in a term is either
contracted or free — there is no unspecified state."** The realm decides
(Oblique: one upper + one lower; Orthonormal: any two; Collection/Label:
never), and `ExplicitSum`/`NoSum` are *overrides* for the cases the implicit
rule can't reach.

That rule was specified but **never implemented**. Every consumer — the
canonicalizer, the matcher, the steps — keyed off explicit `ExplicitSum` nodes;
an implicitly-repeated index with no `ExplicitSum` was silently treated as
*free*, in direct violation of "no unspecified state". This surfaced while
discussing the identity matcher (vibe 000039): it looked like `ExplicitSum` was
required for the soundness of `delta-contraction`, when in fact the Einstein
rule already determines that a repeated index is summed (Orthonormal) or that
the expression is ill-formed (Oblique, same level). So `δ^r_m δ^r_n` and
`Σ_r δ^r_m δ^r_n` *must* mean the same thing.

## Decision

Materialize implicit contractions into `ExplicitSum` as part of
`canonicalize`, so the canonical form is uniform and every downstream consumer
(matcher, future e-graph) need only understand explicit binders. Two forks,
decided with the user:

- **Site = lazy, in `canonicalize`.** A `materialize_implicit_sums` sub-pass
  runs *before* `canon` (`steps::canonicalize`). Construction stays dumb; raw
  trees may carry implicit contractions until canonicalized. (The eager
  construction-time alternative was rejected to avoid touching every `make_*`.)
- **Errors throw immediately — but only with no override.** An ill-formed term
  (Oblique same-level pair, or ≥3 occurrences) throws `std::invalid_argument`.
  Because the pass only counts *free* ids, any id covered by an enclosing
  `ExplicitSum`/`NoSum` is excluded — so an explicit annotation both suppresses
  contraction and silences the error, exactly matching vibe 000028's
  "error (no override)" column.

## The pass

`materialize(ctx, e, bound)` recurses, carrying the set of ids bound by
enclosing `ExplicitSum`/`NoSum`. At each **component-valued** product (vibe
000036's `is_component_valued` line) — and at a lone `TensorObject` (a trace) —
it counts free `CountableIndex` occurrences across the direct tensor factors,
applies the realm rule (`contracted_ids`, which throws on the ill-formed
cases), and wraps the term in an `ExplicitSum` per contracted id. Invariant
factors and operands (`Dot`/`Cross`/…) recurse as independent scopes.

Then `canon` runs as before; its α-normalization relabels the freshly
materialized binders, so the implicit and explicit spellings converge to the
same tree. `algebraic_eq` (= `structural_eq` of canonical forms) therefore now
equates them.

## Scope / limitation

Contraction is detected among the **direct `TensorObject` factors** of a
product (the overwhelmingly common case: `δ^r_m δ^r_n`, `a^i b_i`, the trace
`δ^i_i`). An index that is free only *inside* a composite factor — e.g. an
un-distributed sum, `(a^i + b^i) c_i` — is **not** counted at the enclosing
term. That interacts with distribution and is deferred to the e-graph, the same
boundary the ANF already draws by not distributing. The pass never throws on
these; it only under-detects, leaving the index free.

## Why before the e-graph

The e-graph (vibe 000034) reasons entirely over canonical forms and rewrite
rules. If the canonical form didn't reflect implicit summation, every rule
would be built on an incomplete representation. Fixing it now keeps the
algebraic layer honest before saturation is layered on. It also retires the
false "ExplicitSum needed for soundness" reading from vibe 000039: both
spellings now canonicalize identically, so the matcher's binder logic is
correct and uniform with no change.

## Tests

`tests/derivation_test.cpp` — `ImplicitSum.*` (8): orthonormal pair, oblique
mixed-level, oblique trace, same-level throw, ≥3 throw, ExplicitSum override
silences, NoSum keeps free, Collection never auto-contracts. Python:
`test_implicit_summation_equals_explicit` (equivalence, identity-fires-on-
implicit, trace, throw-and-override). All 348 C++ / 95 Python pass.

## Next

Unchanged: the e-graph (vibe 000034). The parametric-slot + computed-RHS
capability (vibe 000040 — dimension-polymorphic `delta-trace`, general
`delta-substitution`) is designed into the e-graph rule type when it lands.
