# 000033 Theorems and pattern matching

## Context

The eps-delta derivation exposed a gap: after `fold_sums` and `contract_delta`, the
one-index case ($\varepsilon^{ijk}\varepsilon_{iml}$) ends in 12 concrete
Kronecker-delta products.  Folding them into
$\delta^j_m\delta^k_l - \delta^j_l\delta^k_m$ requires recognising the identity
$\sum_p \delta^p_A\delta^p_B = \delta_{AB}$, but that identity is itself a theorem.
`contract_delta` is a hard-coded special case of it; the general machinery does not
yet exist.

This vibe maps out what needs to be built.  Updated after a round of design review.

---

## 1. What is a theorem?

A **theorem** is a concrete derivation:

```
LHS  =  RHS
```

Both `LHS` and `RHS` are ordinary `Expr` trees, built with the same API used for
expressions.  Any **named tensor object** that appears in `LHS` acts implicitly as
a *pattern variable*: it matches any sub-expression during matching, and the same
name must match the same sub-expression throughout a single match attempt.

**Example** — delta contraction:

```python
A = tender.tensor("A")          # pattern variable (any sub-expression)
B = tender.tensor("B")
p, space = ctx.alloc_index(), space_3d

lhs = explicit_sum(p, delta(Upper, Lower, p, A) * delta(Upper, Lower, p, B))
rhs = delta(Upper, Lower, A, B)   # A, B slots inherit from lhs binding

theorem_delta_contraction = Theorem(name="delta-contraction", lhs=lhs, rhs=rhs)
```

This is more concrete than an abstract "pattern variable" node — the LHS is itself
a valid expression and the role of each named tensor is self-documenting.

---

## 2. Identities vs. theorems

**Identity** = an *axiom*, not derivable from other rules.  Examples: the symmetry
of the metric, the permutation sign of ε, `δ^i_i = n` (dimension).  These are
most naturally encoded in `TensorTraits` at the C++ level — they are built-in
properties of specific tensor types, not deduced from other expressions.

**Theorem** = *derivable* from axioms plus arithmetic rules.  Examples: the
eps-delta identity, delta contraction, the Binet–Cauchy identity.  Theorems live
in a theorem library (see §3) and are applied as named derivation steps.

The hard-coded step `contract_delta` is currently filling the role of a theorem.
Once the theorem machinery exists, `contract_delta` should become an application of
the `delta-contraction` theorem — with the hard-coded step kept as a fast-path
fallback if performance requires it.

---

## 3. Theorem library

A small, explicit set of named theorems covering index algebra:

| Name | LHS | RHS |
|------|-----|-----|
| `delta-contraction` | `Σ_p δ^p_A δ^p_B` | `δ_{AB}` |
| `delta-substitution` | `Σ_p δ^p_A f(p, …)` | `f(A, …)` |
| `delta-trace` | `Σ_p δ^p_p` | `ScalarLiteral(n)` (dimension) |
| `eps-delta-2` | `Σ_{ij} ε^{ijk} ε_{ijl}` | `2 δ^k_l` |
| `eps-delta-1` | `Σ_i ε^{ijk} ε_{iml}` | `δ^j_m δ^k_l - δ^j_l δ^k_m` |

Note: `delta-trace` (`δ_ii = 3` in 3D) needs both slots of one delta to share the
same bound index.  With the "named tensor as variable" approach sketched here there
is no way to force that, so it *looks* like an open limitation.  **Superseded — see
vibe 000040:** the shipped matcher (vibe 000039) binds the *binder itself* as a
pattern variable and reuses its id in both slots, so consistency checking forces the
two slots to agree for free.  `delta-trace` works through the generic engine.  The
real residual limit is being *dimension-polymorphic* (a parametric slot + computed
RHS), not the same-index constraint.

---

## 4. Pattern matching

### 4.1 Core match interface

```cpp
using Binding = std::unordered_map<std::string, Expr const*>;
// key = TensorName string of the pattern variable

auto match(Expr const* pattern, Expr const* target, Binding& out) -> bool;
```

`match` is recursive:

- If `pattern` is a `TensorObject` with no slots (a "free" named tensor), look up its
  name in `out`.  If already bound, require `structural_eq` with the bound value.
  Otherwise bind it.
- If `pattern` is a `TensorObject` with slots, require the same name and recurse into
  slots.
- If `pattern` is a composite (`Sum`, `TensorProduct`, `ExplicitSum`, …), require same
  node type and recurse into children.
- If `pattern` is a `ScalarLiteral` / `ConcreteIndex`, require exact equality.

### 4.2 Trait-based constraints

The user raised a valid concern: if we allow matching modulo tensor properties
(e.g., "any symmetric tensor"), every possible sub-expression must be tested for
the given property.  For a 36-term sum of delta products this could mean thousands
of symmetry checks per match attempt — potentially unacceptable.

**Resolution**: for the first version, pattern variables match *any* sub-expression,
with no trait filtering.  Structural matching alone is sufficient for the index
identities listed in §3.  Trait-constrained patterns are deferred to a later version
where we can benchmark the cost and decide whether to invest in pruning heuristics.

### 4.3 Sub-tree search

`apply_theorem(ctx, expr, theorem)` walks `expr` bottom-up and tries `match` at
each node.  On the first successful match it substitutes `theorem.rhs` under the
binding and returns the result.  Applying all matches in one pass is a future
extension.

---

## 5. Theorem application as a derivation step

```python
drv.step(td.apply_theorem(theorems.delta_contraction))
```

`apply_theorem(t)` returns a `(Expr) -> Expr` callable.  The derivation history
records the theorem name at each step, making proofs human-readable.

---

## 6. Open questions

- **`delta-trace` (`δ_ii = n`)**: ~~requires a pattern language extended with
  "same-index" constraints~~ — resolved by binder-as-variable matching; see vibe
  000040.  Residual: a single dimension-polymorphic form needs a parametric slot
  + computed RHS, deferred to the e-graph rule type.
- **Commutativity / associativity**: naïve structural matching misses `A + B` when
  the theorem is written as `B + A`.  Short-term: canonical ordering before matching.
  Long-term: match modulo AC (or e-graph saturation; see vibe 000034).
- **Matching depth**: apply_theorem currently finds the first (deepest-first) match.
  Applying all matches simultaneously requires a sweep or multiple passes.

---

## 7. Priority implementation order

1. `match()` — recursive, handles `Binding`, named-tensor pattern variables.
2. `apply_theorem()` — bottom-up search + substitution.
3. Encode `delta-contraction` as a theorem; make `contract_delta` call it.
4. Close the one-index eps-delta example.
5. Add `delta-substitution` and `eps-delta-1` to the library.
6. Canonical child ordering as a preprocessing pass (commutativity/AC readiness).
