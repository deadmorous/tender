# Phase 13 — Identity library (full)

## Scope

Three deliverables, in priority order:

1. **Automatic targeting** for `apply_identity` — combinatorial pattern search
   so the caller no longer needs to supply the variable mapping manually.
2. **Standard identity library** — `tender.lib` Python package with the most
   commonly needed identities pre-defined.
3. **`doc()` function** — renders identity/theorem metadata as LaTeX or plain
   text.
4. **`Theorem` and `NamedObject`** — deferred to a follow-up; the architecture
   below keeps room for them but they are not implemented in Phase 13.

---

## 1. Automatic targeting

### The problem

`apply_identity(bac_cab, {a: u, b: v, c: w})` requires the caller to hand-code
the binding.  Phase 13 adds `find_matches(identity, expr)` → list of bindings,
and a convenience `apply_identity_auto(identity, expr)` that picks the first
match (or raises if none is found).

### Pattern-matching algorithm

```
match(pattern, expr, bindings) → list[bindings]
```

Base cases:
- `pattern` is a `PatternVar`:
  - If already bound to a different expression → no match (`[]`).
  - If already bound to the same expression → match (`[bindings]`).
  - If `expr` satisfies `pattern`'s rank and trait constraints → bind and
    return `[bindings ∪ {pattern: expr}]`.
  - Otherwise → `[]`.
- `pattern` is a concrete non-structural leaf (`NamedTensor`, `IdentityTensor`,
  `LeviCivitaTensor`, `RationalConst`, `NamedConst`, `Parameter`) → match only
  if `pattern == expr` (pointer equality for singletons, value equality for
  `RationalConst`).

Structural cases — `pattern` and `expr` must have the same dynamic type;
match fields recursively and intersect the resulting binding lists:
- `Sum(terms)`: match only if `|terms|` is the same; match `terms[i]`
  pairwise (order-sensitive for now).
- `Scale(coeff, inner)`: `coeff` must be equal; match `inner`.
- `TensorProduct(lhs, rhs)`, `Contract(lhs, rhs)`, `DoubleContract(lhs, rhs)`,
  `CrossProduct(lhs, rhs)`: match `lhs`, then `rhs` with the bindings from
  `lhs`.
- `Gradient(arg)`, `Divergence(arg)`, `Rotor(arg)`: match `arg`.
- `FunctionApply(kind, arg)`: `kind` must be the same enum value; match `arg`.
- `Pow(base, exp)`: both `base` and `exp` must match.
- `Integral(domain, integrand)`: `domain` pointer must be equal; match
  `integrand`.
- Everything else → no match.

### Tree search

```
find_matches(identity, expr) → list[bindings]
```

Walk `expr` in pre-order.  At each node, attempt `match(identity.lhs, node,
{})`.  Collect all successful bindings (a node can match at most one way per
position, but different positions in the tree may each match).

`apply_identity_auto(identity, expr, all_matches=False)`:
- Calls `find_matches`.
- If `all_matches=False` (default): apply to the first match, return a
  `DerivationStep`.
- If `all_matches=True`: apply to every match in sequence, returning a list
  of `DerivationStep`s.
- Raises `ValueError` if no match is found.

**Budget**: `find_matches` accepts an optional `max_nodes=10_000` parameter
that caps the number of nodes visited.  Exceeding the budget raises
`RuntimeError("find_matches: budget exceeded")`.

### Implementation location

Pattern matching lives in a new C++ module:
- `src/include/tender/match.hpp`
- `src/match.cpp`

Exposed to Python via `_tender.cpp`:
- `find_matches(identity, expr, max_nodes=10000)` → `list[dict]`
- `apply_identity_auto(identity, expr, all_matches=False, max_nodes=10000)`
  → `DerivationStep | list[DerivationStep]`

### Constraint checking in pattern variables

`PatternVar` already has `constrain_rank(n)` from Phase 8.  For Phase 13 we
add:
- `constrain_symmetric()` — match only tensors flagged symmetric
- `constrain_skew()` — match only tensors flagged skew-symmetric
- `constrain_scalar()` — shorthand for `constrain_rank(0)`
- `constrain_vector()` — shorthand for `constrain_rank(1)`

Trait flags on `NamedTensor` (opt-in, set once):
- `tensor("A", 2).declare_symmetric()` — marks the tensor as symmetric in its
  two indices
- `tensor("A", 2).declare_skew()` — marks it as skew

The `PatternVar` constraint check calls `Expr::is_symmetric()` /
`is_skew_symmetric()` — virtual methods returning `false` by default,
overridden in `NamedTensor`.

---

## 2. Standard identity library

Implemented as a Python package `tender/lib/` (no C++ required; builds
`Identity` objects using the public Python API).

### Directory layout

```
python/tender/lib/__init__.py
python/tender/lib/identities/__init__.py
python/tender/lib/identities/epsilon.py       # Levi-Civita / cross product
python/tender/lib/identities/identity_tensor.py
python/tender/lib/identities/functions.py     # sin²+cos²=1 etc.
```

### Identities to include in Phase 13

**epsilon.py**
- `bac_cab` — `a×(b×c) = b(a·c) − c(a·b)` (already in `bac_cab.py`; promote
  to library entry)
- `double_cross` — `(a×b)×c = b(a·c) − a(b·c)` (the companion rule)
- `eps_contract_eps` — `εᵢⱼₖ εᵢₗₘ = δⱼₗδₖₘ − δⱼₘδₖₗ` (for direct-notation:
  `ε:ε = 6`, `(ε·a):(ε·b) = 2(a·b)` — add as needed)

**identity_tensor.py**
- `identity_dot_vec` — `I·a = a`
- `identity_double_contract` — `I:A = tr(A)`
- `trace_of_identity` — `tr(I) = 3` (or `n` in n-d)

**functions.py**
- `sin_sq_plus_cos_sq` — `sin²θ + cos²θ = 1`
- `exp_product` — `exp(a)·exp(b) = exp(a+b)` (scalar product here means
  `TensorProduct` of two rank-0 scalars)

### Identity object structure

Each library module exposes its identities as module-level `Identity` objects.
The module also exposes an `ALL` list for bulk import:

```python
# tender/lib/identities/epsilon.py
from tender import make_pattern_var, constrain_rank, cross, dot, tp, Identity

a = make_pattern_var("a"); constrain_rank(a, 1)
b = make_pattern_var("b"); constrain_rank(b, 1)
c = make_pattern_var("c"); constrain_rank(c, 1)

bac_cab = Identity(
    "BAC-CAB",
    lhs=cross(a, cross(b, c)),
    rhs=b * dot(a, c) - c * dot(a, b),
)

ALL = [bac_cab, double_cross]
```

Import path:
```python
from tender.lib.identities.epsilon import bac_cab
from tender.lib.identities import epsilon  # then epsilon.bac_cab
```

---

## 3. `doc()` function

```python
def doc(entry, format="latex"):
    ...
```

- `entry` may be an `Identity` (or later a `Theorem`).
- `format="latex"` returns a compilable LaTeX snippet (suitable for pasting
  into a document).
- `format="plain"` returns a plain-text representation.
- `format="jupyter"` (default in a Jupyter environment) triggers
  `IPython.display.Math` directly.

For an `Identity`:
```latex
\textbf{BAC-CAB identity:}
\[
  \mathbf{a} \times (\mathbf{b} \times \mathbf{c})
  = \mathbf{b}(\mathbf{a} \cdot \mathbf{c})
  - \mathbf{c}(\mathbf{a} \cdot \mathbf{b})
\]
```

`doc` lives in `python/tender/__init__.py` (or a small `tender/doc.py`).

---

## 4. Implementation plan

### C++ additions (`src/match.{hpp,cpp}`)

```cpp
// Single-node match attempt.  Returns {} on failure, list of bindings on success.
using PatternBinding = std::map<PatternVar const*, Expr*>;
auto match(Expr* pattern, Expr* expr, PatternBinding const& bindings)
    -> std::vector<PatternBinding>;

// Tree search.
auto find_matches(Identity const& id, Expr* expr, int max_nodes = 10'000)
    -> std::vector<PatternBinding>;
```

### Python additions

- Bind `find_matches` and `apply_identity_auto` in `_tender.cpp`.
- Export from `__init__.py`.
- `declare_symmetric()`, `declare_skew()` on `Expr` (or just `NamedTensor`)
  in C++ + binding.
- `constrain_symmetric()`, `constrain_skew()` on `PatternVar`.
- New directory `python/tender/lib/` with the standard library modules.
- `doc()` function.

### Tests

- `tests/match_test.cpp` — unit tests for `match()` and `find_matches()` in
  isolation (no Python required).
- `python/test_tender.py` — Python-level tests for `find_matches`,
  `apply_identity_auto`, `doc()`, standard library imports.
- Updated `examples/bac_cab.py` — demonstrate auto-targeting.

### CMakeLists

- Add `src/match.cpp` to the `tender` library in `src/CMakeLists.txt`.
- Add `python/tender/lib/__init__.py` (and submodules) to the list of synced
  Python files in `python/CMakeLists.txt`.

---

## Questions resolved

**Q: Order-sensitive Sum matching?**  
A: Yes, for now.  Sum terms are ordered (as stored), so the pattern `a + b`
matches `u + v` with `{a→u, b→v}` but not `v + u` unless the pattern is
also written `b + a`.  This is the simplest correct first step; commutativity-
aware matching is a future refinement.

**Q: Should `find_matches` return all matches at a given node or stop after one?**  
A: All matches at each node.  Multiple bindings per node are possible if the
same pattern can match the same expression in more than one way (e.g., when
PatternVars have overlapping constraints).

**Q: What if the identity LHS is itself a PatternVar (trivially matches anything)?**  
A: Disallow — raise `ValueError` at `Identity` construction time if `lhs` is a
bare `PatternVar` with no further structure.

**Q: How are `Theorem` and `NamedObject` deferred?**  
A: Not implemented in Phase 13.  The `Theorem` and `NamedObject` classes are
left as stubs or entirely absent; the standard library uses only `Identity`.

**Q: Where does `tender.lib` live on disk?**  
A: Source at `python/tender/lib/`, synced to the build tree alongside
`__init__.py`.  No C++ involvement — pure Python.
