# Rewrite search — automated identity sequencing

## Problem

`apply_identity_auto` applies a single identity when its LHS pattern matches
the *root* of the current expression.  Real derivations often require a
preparatory sequence of rewrites before the desired identity becomes
applicable.  Example:

```python
# (v×w)×u — does not directly match BAC-CAB (a×(b×c)).
# Must first recognise it as double-cross ((a×b)×c) and then the user
# sees the result.  Or, more generally: find what to apply and in what order.
expr = cross(cross(v, w), u)
```

The question is: given a *target* identity and an expression, find a sequence
of rewrites from a known library that transforms the expression into a form
where the target matches.

---

## Clarification: root-level vs sub-expression rewriting

`apply_identity` and `apply_identity_auto` both operate at the **root** of the
current expression — the entire expression is replaced by the result of the
RHS substitution.

Sub-expression rewriting (apply an identity inside a larger context, rebuilding
the parent chain) is implemented via:

- `find_and_rewrite_all(id, rl, root)` (C++) — walks `root` in pre-order;
  at every node where `id.lhs()` matches, computes the new root by substituting
  `id.rhs()` at that node and rebuilding the parent chain via `replace_in_tree`.
- `_find_and_rewrite_all(id, root)` (Python binding) — returns a list of
  `(new_root: Expr, step_name: str)` pairs.
- `_capture_step(name, result)` — a `DerivationStep` that ignores its input
  and returns the pre-computed `result`.  Used to encode sub-expression rewrites
  as standard derivation steps.

These lower-level helpers are used internally by `search_apply` and are not
part of the public API.

---

## Three tiers of automation

### Tier 1 — Oriented rewriting / normalization (simplest)

Choose a *canonical direction* for every identity and apply it always
left-to-right.  Example: always expand `a×(b×c)` and `(a×b)×c` via BAC-CAB /
double-cross, eliminating nested cross products.  The result is a *normal form*;
identity matching reduces to structural equality of normal forms.

**Good for**: well-structured domains with a finite set of oriented rules and a
known normal form (polynomial arithmetic, Lie algebras).  
**Bad for**: bidirectional rules (anti-commutativity cycles), large rule sets,
expressions that should stay compact.

### Tier 2 — BFS over sub-expression rewrites (implemented here)

Breadth-first search over the space of expressions reachable by one-step
applications of any rule **at any sub-expression site** in the tree.

- Queue: `(current_expr, steps_so_far)`.
- Expansion: for every rule, call `_find_and_rewrite_all` to enumerate all
  match sites in `current_expr`; for each site produce a `_capture_step`.
  Discard expressions already seen (keyed by `latex()` string).
- Termination: `target` can be applied anywhere in the new expression (checked
  via `_find_and_rewrite_all(target, ...)`); or timeout / exhaustion.
- The final step is always the application of `target`; `search_apply` returns
  a **complete** derivation sequence including that last step.
- Default rule set: all identities in `tender.lib`.

**Example**:

```python
# (v×w)×u needs anti-commutativity first, then BAC-CAB inside Scale(-1,…).
steps = search_apply(bac_cab, cross(cross(v, w), u))
# steps: [apply(cross-anticomm), apply(BAC-CAB)]
# Applying all steps yields: -v(u·w) + w(u·v)
```

**Good for**: preparatory sequences of 1–4 rewrites at any depth; interactive
derivation assistance.  
**Timeout**: floating-point seconds, checked once per dequeue.

### Tier 3 — E-graphs / equality saturation (future)

Instead of choosing *which* rewrite to apply, add **all** possible rewrites
simultaneously to a compact shared data structure called an equality graph
(e-graph).  Expressions equal under any combination of rules share equivalence
classes.  After saturation, extract the desired form cheaply.

**Key properties**: handles bidirectional rules without looping (anti-commutativity
is fine), complete up to the rule set, sub-linear extraction via cost functions.  
**Reference implementation**: `egg` (Rust), `egglog` (Datalog-style extension).  
**For tender**: would require either linking to `egg` via C FFI or re-implementing
a minimal e-graph in C++.  The right choice if tender grows into a general CAS
or if BFS proves too slow for practical derivations.

---

## Implementation decisions (Tier 2)

### API

```python
# Returns a complete derivation sequence including the final application of
# target.  Applies the sequence to expr to get the result directly.
# Raises TimeoutError if no sequence found in time.
# Raises RuntimeError if search space is exhausted.
steps = search_apply(target, expr, rules=None, timeout=5.0)
history = Derivation(steps).apply(State(expr))
result = history[-1].expr   # final expression with target applied
```

`rules=None` defaults to `tender.lib.ALL` — all standard-library identities.

### double_cross removed from epsilon.py

The `double_cross` identity `(a×b)×c = b(a·c) − a(b·c)` is derivable from
`bac_cab + anti_commutativity` in two steps via `search_apply`.  Keeping it in
`ALL` would add it to the default BFS rule set without shortening any search
(the BFS finds the two-step path anyway).  It was therefore removed from the
library to keep the rule set minimal.

### anti_commutativity added to epsilon.py

```python
anti_commutativity = Identity("cross-anticomm",
    lhs=cross(a, b),
    rhs=-cross(b, a),
)
```

This is bidirectional: `find_matches` + BFS will apply it in only one direction
per step, but it can cycle.  The visited-set in BFS prevents infinite loops.

### apply_identity_auto semantics tightened

Previously `apply_identity_auto` called `find_matches` (full tree search) and
then tried to apply the first binding at the root.  This was silently wrong for
non-root matches (the binding referred to a sub-expression, not the root).

Changed to use `match_pattern(id.lhs(), expr, {})` — root-only match — so the
function's promise ("automatically find and apply") is consistent with what it
actually does.  The `max_nodes` budget parameter is removed; root-level matching
is always O(|pattern|).

---

## Future work

- **E-graphs**: full equality saturation once the rule library is rich enough
  and BFS depth becomes a bottleneck.
- **Oriented normalization**: canonical normal forms for specific sub-domains
  (e.g. cross-product-free vector expressions).
- **Cost-guided search**: replace BFS with A* using a cost heuristic (e.g.
  expression size) to find shorter derivation sequences faster.
