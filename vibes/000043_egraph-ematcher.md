# 000043 E-class matcher (Stage 1, part 2 — slice 2)

Second slice of the e-graph (vibe 000034, item #3): **e-matching** — search the
e-graph for every place an `Identity` LHS matches, with its index binding.
`EGraph::ematch(Expr const* pattern) → vector<pair<EClassId, MatchBinding>>`.

## Reuse, don't fork the matcher

The Expr-tree matcher (vibe 000039) already encodes the index-binding semantics:
free LHS indices are pattern variables, ExplicitSum/NoSum binders are α-locals,
slots match on level/realm/space. E-matching changes only *where recursion goes*
— into child e-classes instead of child Exprs — so the index logic must not be
duplicated (CLAUDE.md #6). Two thin primitives were lifted out of the matcher:

- `match_into(pattern, target, bnd)` — the binding-threading core of `match()`,
  used to match a **leaf** pattern (`TensorObject`/`ScalarLiteral`) against a
  leaf e-node's stored `Expr` wholesale (slots, indices, levels — all of it);
- `bind_pattern_index(bnd, id, target)` — bind a pattern index to a target,
  used to bind an `ExplicitSum`/`NoSum` pattern binder to an e-node's binder id.

So the index semantics live in one place; the e-matcher adds only the e-class
descent and AC handling.

## The algorithm

`ematch` canonicalizes the pattern, then for every e-class tries to match the
pattern's root against each of the class's e-nodes (`match_class` →
`match_node`):

- **leaves** — delegate to `match_into` on the e-node's `Expr`;
- **binders** — `bind_pattern_index` the binder, recurse into the body (and the
  symbolic-bound child) e-classes;
- **positional binary** (Negate, ScalarDiv, Dot, DDot, DDotAlt, Cross, and
  invariant TensorProduct) — recurse into both child classes, taking the
  cartesian product of the resulting bindings;
- **commutative** (Sum, and component TensorProduct via `is_component_valued`) —
  flatten the pattern and the target and AC-backtrack, matching each pattern
  operand against a distinct target operand class.

Crucially, **all** e-nodes of a class are searched, not just the extracted
representative. So after a rule proves `Σ_p δ^p_m δ^p_n = δ_{mn}` and extraction
prefers the smaller `δ_{mn}`, the contraction pattern still matches via the
now-non-representative contraction e-node — the property that makes saturation
order-independent. This is tested directly (`SearchesNonRepresentativeNodes`).

## Scope / limitations

- **Single target flattening.** Commutative target operands are collected by
  following one same-kind e-node per class. Enumerating *alternative* flattenings
  of a class that holds several distinct commutative forms is a follow-up; the
  current scheme is complete for canonical, mostly-singleton classes and the
  size-reducing rule set saturation will start with.
- A canonical pattern gives every binder a distinct (depth-based) id, so the
  binder-consistency check never fails and the `Difference` arm is never reached
  (canonical forms carry signs as coefficients) — both coverage-excluded.

## Tests

`tests/egraph_test.cpp` EMatch.* (9): binds free indices on delta-contraction;
matches modulo factor order; finds the pattern in a sub-expression; empty on no
match; matches a non-representative e-node after a merge; the two-index eps-δ
through the generic e-matcher; every reachable node kind; NoSum and
symbolic-bound ExplicitSum; sum flattened across e-nodes. `egraph.cpp` 100% line
coverage; 380 C++ / 95 Python pass.

## Next (vibe 000034)

#4 instantiate-into-egraph (build RHS e-nodes from a binding and add them) and
#5 the `saturate` fixed-point loop, then #8 the rule set (where the
parametric-slot + computed-RHS rule type of vibe 000040 is designed in), #9
Python `td.saturate`, #10 benchmarks. With `ematch` returning `(class, binding)`,
saturate is: for each rule, for each match, `instantiate` the RHS, `add` it, and
`merge` it into the matched class; `rebuild`; repeat to a fixed point.
