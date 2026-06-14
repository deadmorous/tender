# 000034 E-graph saturation

## Why not BFS rule application?

Attempt-1 used a BFS strategy: at each step, try every identity at every
sub-expression node, collect all one-step rewrites, and explore the resulting
state space.  The problems:

1. **Exponential branching** — each identity applied at any node of a large tree
   produces a sibling in the BFS frontier.  Trees grow quickly; frontiers explode.
2. **Rewrite ordering matters** — applying rule A before B can block rule C.
   BFS mitigates this but does not eliminate the problem.
3. **No sharing** — if two subtrees are equal under different normal forms, BFS
   re-derives each independently.
4. **Termination is hard to guarantee** — a rule like `x + y = y + x` loops.
   BFS needs explicit cycle detection.

E-graph saturation solves all four problems.

---

## What is an e-graph?

An **e-graph** (equality graph) compactly represents a *set of equivalent
expressions* using two kinds of nodes:

- **E-nodes** — one node per expression constructor (`Sum`, `TensorProduct`,
  `Delta`, …) with children pointing to **e-classes**, not e-nodes.
- **E-classes** — equivalence classes of e-nodes.  When two expressions are proved
  equal, their e-classes are *merged* (union-find).

Saturation means: keep applying rewrite rules — each rule can add new e-nodes and
trigger merges — until no rule fires (fixed point).  After saturation, the e-graph
contains *all expressions derivable from the input by the rule set*, in compact
shared form.

**Extraction** then picks the cheapest (shortest, simplest, …) representative from
the root e-class.

---

## Why e-graphs fit tender well

tender already has most of the primitives:

| tender concept | e-graph concept |
|----------------|-----------------|
| `Expr const*` pointer into `Context` | e-node |
| `structural_eq` | equality test for e-node deduplication |
| `Context` (arena) | e-graph node storage |
| `Theorem { lhs, rhs }` | rewrite rule |
| `match()` (pattern matcher) | left-hand matcher |
| `apply_theorem()` | right-hand substitution |

The missing piece is the **union-find** data structure over e-classes and the
**worklist** that drives saturation.

---

## Sketch of the data structures

```cpp
// Each e-class is identified by a canonical id.
using EClassId = int;

struct ENode {
    NodeVariant node;              // Sum, Delta, TensorProduct, …
    std::vector<EClassId> children;// one per child slot
};

struct EGraph {
    // Union-find
    std::vector<EClassId> parent;  // parent[i] = representative of class i
    EClassId find(EClassId i);
    void merge(EClassId a, EClassId b);

    // E-nodes, keyed by (node-type, children) for deduplication
    std::unordered_map<ENode, EClassId, ...> nodes;

    EClassId add(ENode);           // add or return existing
    void rebuild();                // re-canonicalise after merges
};
```

`rebuild()` is the Egg-style batch rebuild: after a round of merges, update all
e-node child pointers to their canonical class id, then re-insert into the hashmap
to detect newly equal nodes.

---

## Saturation loop and termination

```cpp
auto saturate(EGraph& eg, EClassId root,
              std::vector<Theorem> const& rules) -> EClassId
{
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto const& rule : rules) {
            for (auto const& [enode, cls] : eg.nodes) {
                Binding sigma;
                if (match(rule.lhs, eg, cls, sigma)) {
                    EClassId rhs_cls = instantiate(rule.rhs, eg, sigma);
                    if (eg.find(cls) != eg.find(rhs_cls)) {
                        eg.merge(cls, rhs_cls);
                        changed = true;
                    }
                }
            }
        }
        eg.rebuild();
    }
    return eg.find(root);
}
```

After saturation, extract the cheapest expression from the root e-class:

```cpp
Expr const* extract(EGraph& eg, EClassId cls, Context& ctx);
```

### Saturation criterion and graph growth

**When does the loop terminate?**

The loop exits when `changed == false` after a full pass — i.e., no rule merges
any two previously distinct e-classes.  At that point the e-graph has reached a
*fixed point*: the rule set applied to the current set of expressions produces no
new equivalences.

**Can the graph grow without bound?**

Yes, in the presence of rules that *increase* expression size:

- `x → x + 0` (additive zero) is size-increasing and creates infinitely many new
  e-nodes.
- `x + y → y + x` (commutativity) is size-neutral but applied repeatedly to every
  sub-tree it generates exponentially many equivalent orderings.

Practical e-graph libraries (Egg, egglog) deal with this in two ways:

1. **Canonical ordering** — handle commutativity and associativity not as rewrite
   rules but as an invariant enforced during e-node insertion (sort children of
   symmetric nodes by a stable key).  This is sufficient for tender's current rule
   set.

2. **Size / iteration limit** — set a cap on the number of e-nodes or iterations.
   After the cap, extract the best available expression even if the graph is not
   fully saturated.  This is the standard production approach.

For the tender rule set (delta contraction, eps-delta, etc.) all productive rules
are *size-reducing* (they replace a larger pattern with a smaller expression).
Size-reducing rules always terminate: each application strictly decreases the
expression size, and the e-graph cannot merge an infinite number of non-equivalent
expressions whose sizes are bounded by the initial size.

The only size-neutral rules tender might need are commutativity and associativity,
which are handled by canonical ordering (§ "Commutativity and associativity" below),
not as explicit rewrite rules.  With this design, the tender rule set is guaranteed
to saturate in finite iterations.

---

## How this replaces the existing derivation pipeline

Today the pipeline is a *linear sequence of rewriting steps*:

```
expand_eps → unroll_sums → expand_products → eval_delta_concrete → fold_arithmetic
  → fold_sums → fold_sums → contract_delta
```

Each step is manually ordered and must be called the right number of times.

With saturation, the user provides the *rule set* and saturation finds the normal
form automatically:

```python
rules = theorems.standard_set()   # all the rules above
result = td.saturate(expr, rules)
```

The manual step ordering becomes unnecessary.  The user may still write linear
derivations for *pedagogical* output (showing each step), but *computing* the result
does not require choosing the order.

---

## Commutativity and associativity

These are the classic pitfall: `x + y = y + x` creates two e-nodes for the same
`Sum`, which merge, which triggers re-analysis of everything touching them, causing
the graph to grow without bound.

Standard mitigations:

1. **Canonical form** — sort `Sum` / `TensorProduct` children by a stable key
   before inserting into the e-graph.  Commutativity rules are then never needed.
2. **Restricted rule application** — apply commutativity/associativity only on
   demand during pattern matching (match modulo AC), not as rewrite rules in the
   graph.

For tender, option 1 is simplest: define a `canonicalise()` pass that sorts
children of symmetric nodes before saturation.

---

## Interaction with index quantification

ExplicitSum introduces a *bound variable* (the summation index).  Two e-nodes
`ExplicitSum{p, body(p)}` and `ExplicitSum{q, body(q)}` are equal iff `body(p)`
and `body(q)` are equal up to α-renaming.

The e-graph must identify expressions modulo α-equivalence.  The cleanest approach
is **de Bruijn levels**: replace each bound index by its nesting depth before
hashing.  Then `structural_eq` already handles α-equivalence for free.

tender's `CountableIndex` already has an opaque `id` that can serve as a de Bruijn
level if we normalise bound ids by depth during e-node construction.

---

## Implementation plan

1. **Canonical child ordering** (prerequisite — also benefits pattern matching).
2. **`EGraph` data structure** — union-find + e-node hash map + `rebuild()`.
3. **E-graph pattern matcher** — `match()` operating on e-class ids, not raw `Expr*`.
4. **`instantiate()`** — build right-hand side e-nodes from a binding.
5. **`saturate()`** — the fixed-point loop above.
6. **`extract()`** — greedy bottom-up cost extraction.
7. **α-normalisation** — normalise bound indices before e-node insertion.
8. **Rule set** — encode standard index identities as `Theorem` objects.
9. **Integration** — expose `td.saturate(expr, rules)` in Python bindings.
10. **Benchmarks** — saturation on the full eps-delta expression, measure node count
    and iteration count.

The first six items can be built and tested independently of the tensor algebra;
a simple arithmetic e-graph (`1 + 1 = 2`, `x * 0 = 0`, …) is a good prototype.

---

## Relationship to existing code

- `steps::fold_sums` and `steps::contract_delta` become **theorem applications**
  in the e-graph framework.  The hard-coded steps remain available for linear
  derivations but are no longer the primary mechanism.
- `structural_eq` (already in `expr.hpp`) is the e-node equality test.
- `Context` remains the arena; e-nodes point into it.
- The `Derivation` class stays for human-readable step-by-step output; saturation
  is a separate code path for automated simplification.
