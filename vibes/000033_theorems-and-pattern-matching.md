# 000033 Theorems and pattern matching

## Context

The eps-delta derivation exposed a gap: after `fold_sums` and `contract_delta`, the
one-index case ($\varepsilon^{ijk}\varepsilon_{iml}$) still ends in 12 concrete
Kronecker-delta products.  Folding them into
$\delta^j_m\delta^k_l - \delta^j_l\delta^k_m$ requires recognising the identity
$\sum_p \delta^p_A\delta^p_B = \delta_{AB}$, but that identity is itself a theorem.
`contract_delta` is a hard-coded special case of it; the general machinery does not
yet exist.

This vibe maps out what needs to be built.

---

## 1. What is a theorem?

A **theorem** is a universally-quantified rewrite rule of the form

```
LHS(x₁, …, xₙ) = RHS(x₁, …, xₙ)
```

where `LHS` and `RHS` are expression templates with free *pattern variables*
`x₁, …, xₙ` matching sub-expressions.  Applying the theorem means:

1. find a sub-expression that matches `LHS` under some variable binding σ,
2. replace it with `RHS[σ]`.

Theorems in tender are distinct from *derivation steps* (algorithmic
transformations like `fold_sums`) because they:

- are stated in the object language (as `Expr` trees with unbound slots), and
- have a human-readable name / citation that can be tracked in a derivation history.

---

## 2. Do we need an identity library?

Yes — but a small, typed one, not a flat list.

Attempt-1 had a bag of identities applied via BFS.  The problem is the search
space explodes: every identity is tried at every sub-tree node at every step,
with no guidance on termination.

The right structure is a **tiered library**:

| Tier | Content | How applied |
|------|---------|-------------|
| **Arithmetic** | `0+x=x`, `1*x=x`, `-(-x)=x`, … | `fold_arithmetic` (already done) |
| **Index identities** | `δ^i_j δ^j_k = δ^i_k`, `Σ_p δ^p_A δ^p_B = δ_AB`, … | pattern-match + theorem step |
| **Structural** | eps-delta, curl-of-gradient, … | dedicated steps or e-graph rules |

The index-identity tier is where the gap is.  It is small (maybe a dozen entries)
and the identities are well-known, so it is feasible to enumerate them explicitly
rather than discover them.

---

## 3. Pattern matching

Pattern matching maps a concrete `Expr` tree against a template tree containing
*pattern variables*.  Requirements:

- **Pattern variables** match any sub-expression (initially: match any `Expr`).
- **Index slots** may be concrete (`ConcreteIndex`) or abstract (`CountableIndex`);
  the matcher must handle both.
- **Commutativity / associativity** of `Sum` and `TensorProduct` makes naïve
  structural matching insufficient.  For the short term, canonical ordering (sort
  children by a stable key) is enough.  Long term, e-graph saturation handles this
  properly (see vibe 000034).
- **Binding occurs**: once a pattern variable is bound to a sub-expression, every
  subsequent occurrence of the same variable must match the *same* sub-expression
  (modulo `structural_eq`).

### 3.1 Sketch of the matcher interface

```cpp
struct PatternVar { int id; };

// A pattern is just an Expr tree where some leaves are PatternVar nodes.
// PatternVar is added to the node variant.

using Binding = std::unordered_map<int, Expr const*>;

auto match(Expr const* pattern, Expr const* target, Binding& out)
    -> bool;
```

`match` is recursive:

- If `pattern` is a `PatternVar`, try to unify with `out[id]`.
- If `pattern` is a `ScalarLiteral` / `ConcreteIndex` / etc., require exact equality.
- If `pattern` is a composite (`Sum`, `TensorProduct`, …), require same node type
  and recursively match children.

### 3.2 Sub-tree search

`apply_theorem(ctx, expr, theorem)` walks `expr` bottom-up and tries to match
`theorem.lhs` at each node.  On the first match, it substitutes `theorem.rhs[σ]`
and returns the new tree.  (Multiple matches in one pass is a future extension.)

---

## 4. Theorem application as a derivation step

A theorem is itself a first-class object:

```cpp
struct Theorem {
    std::string name;   // e.g. "delta-contraction"
    Expr const* lhs;
    Expr const* rhs;
    // optional: proof context (future)
};
```

Applying it produces a derivation step:

```python
drv.step(td.apply_theorem(delta_contraction))
```

where `apply_theorem(t)` returns a callable `(Expr) -> Expr`.

The derivation history then contains the theorem name at each step, making
proofs human-readable.

---

## 5. The identity needed to close the one-index eps-delta

```
Σ_p δ^p_A δ^p_B = δ_AB
```

This is exactly what `contract_delta` hard-codes.  With a general theorem layer,
`contract_delta` becomes an application of this named theorem rather than bespoke
code.  The index-identity library entry is:

```
name   : "delta-contraction"
LHS    : ExplicitSum{ p, TensorProduct{ δ(p, A), δ(p, B) } }
RHS    : δ(A, B)
```

where `A` and `B` are pattern variables ranging over index slots.

Once this theorem is in the library, the one-index eps-delta derivation closes:

```python
drv.step(td.apply_theorem(theorems.delta_contraction))  # applied twice
```

---

## 6. Immediate next steps (priority order)

1. **Add `PatternVar` to the node variant** — minimal change, no existing code
   affected.
2. **Implement `match()`** — recursive, handles binding.
3. **Implement `apply_theorem()`** — bottom-up sub-tree search + substitution.
4. **Encode the delta-contraction theorem** — replace the hard-coded `contract_delta`
   step with `apply_theorem(theorems::delta_contraction)`.
5. **Close the one-index eps-delta example** using the theorem step.
6. **Expand the index-identity library** — at minimum: `δ^i_j δ^j_k = δ^i_k`
   (delta product / index substitution), which is also needed for chained
   contractions.
7. **Commutativity/associativity** — canonical child ordering as a preprocessing
   normalisation pass, so pattern matching does not need to enumerate permutations.
