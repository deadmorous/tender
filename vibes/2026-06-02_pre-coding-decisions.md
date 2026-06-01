# Pre-coding decisions

_First discussed: 2026-06-02_

Gaps to resolve before writing code, organized by area.
Open questions are labeled `Q_subject` for easy grepping.
Resolved questions are marked with their answer inline.

---

## 1. Internal core

### Expression tree

Node taxonomy (settled): scalars (rational constants, symbolic variables), named special
tensors (**I**, **ε**, basis vectors **g**ᵢ), operations (Add, Scale, TensorProduct,
Contract, Gradient, Integral, ...), function application for scalar fields.

**Q_memory_model**: What memory model for tree nodes?
- `shared_ptr` with structural sharing — nodes are reused across rewrites; natural for
  an immutable tree, good for DAG representation, some overhead
- Arena allocation — faster, deterministic lifetime, harder to implement correctly
- Value semantics with copy-on-write — simple to reason about, potentially expensive
  for large trees

*Downstream consequence: affects the entire codebase and how rewriting rules are written.*

**Q_rank_tracking**: Static (compile-time template parameter) vs. dynamic (runtime integer) rank?
- Static: type safety — contracting two rank-1 tensors to a scalar is a compile-time
  error if ranks mismatch; but makes generic rewriting rules harder to write uniformly
- Dynamic: simpler to implement generic rules; rank errors caught at runtime only

*Downstream consequence: determines whether the rewriting engine can be written generically
or needs specialization per rank.*

**Q_index_slots**: Track covariant vs. contravariant slot types explicitly in the tree?
- Needed for full curvilinear correctness (raising/lowering indices involves the metric)
- Adds significant complexity to every operation and rewriting rule
- Alternative: work in a fixed frame (Cartesian internally) and handle
  co/contravariant distinction only at the CS evaluation layer

### Simplification / rewriting engine

**Q_normal_form**: What does "simplified" mean — where do we stop?
- Full canonical form for arbitrary tensor expressions is an open research problem
- Practical partial normal form: sums flattened and sorted, contractions with **I**
  resolved, ε-identities applied, scalar factors collected
- Need an explicit list of what the simplifier guarantees and what it leaves to the user

**Q_rule_repr**: Are rewriting rules data or code?
- Data (pattern trees → replacement trees, matched at runtime): user-extensible,
  declarative, can be inspected and printed
- Code (hand-written visitor cases per rule): faster, easier to debug, not extensible
  without recompiling
- Hybrid: built-in rules as code, user rules as data

**Q_termination**: How to guarantee the rewriting engine terminates?
- Term ordering (e.g. lexicographic on node type + size) that strictly decreases
  with every rule application
- Rewriting budget (max steps) with a fallback that returns the partially simplified form
- Detect cycles explicitly

### Coordinate system model

**Q_cs_construction**: How is a coordinate system constructed?
- User supplies the symbolic embedding map **x**(q¹, q², q³) → WCS; metric and
  Christoffel symbols derived automatically
- User supplies metric components gᵢⱼ(q) directly; Christoffels derived
- User supplies everything (metric + Christoffels) explicitly

**Q_cs_dispatch**: How does ∇ know which CS a field lives in?
- Field carries a CS reference (field and CS are coupled objects)
- CS is an explicit context passed at the call site: `grad(field, cs)`
- Both: field has a default CS, overridable at call site

### Integrals

**Q_divergence_theorem**: Is the divergence theorem a rewriting rule in the engine,
or a named macro the user invokes explicitly?
- Rule in engine: applied automatically during simplification (risk: surprising behavior)
- Named explicit step: user writes `apply_divergence_theorem(expr, domain)`;
  cleaner for a derivation assistant where every step should be visible
  (mirrors the "localization is a named inference step" decision already made)

---

## 2. User interface

**Settled: nanobind over pybind11** for Python bindings.
nanobind is the modern successor from the same author, faster compile times,
smaller binaries, C++17-native design. Python-in-Jupyter is the target interactive
experience. C++ API remains primary; Python is the interactive shell.

**Q_ui_cppapi**: Should the C++ API be designed for direct use by the user
(ergonomic DSL-like syntax via operator overloading), or is it primarily
an implementation layer with Python as the intended user-facing interface?

---

## 3. Technology

**Settled choices:**

| Concern | Decision |
|---|---|
| Build system | CMake ≥ 3.25 |
| Package management | CMake FetchContent primarily; Conan only if a heavy dep demands it |
| Testing | GoogleTest via FetchContent |
| Python bindings | nanobind |
| C++ standard | C++20 (concepts, ranges; solid compiler support GCC 11+, Clang 13+) |
| Code formatting | clang-format (committed early) |
| Static analysis | clang-tidy |

**Q_ci**: Continuous integration — GitHub Actions or something else?
What is the target platform (Linux only, or also macOS/Windows)?

---

## 4. License

**Settled: Apache 2.0**
Permissive (the user's inclination), with an explicit patent non-aggression clause
as low-cost insurance. No principled reason for copyleft given the build-from-scratch
approach and optional-bridge third-party strategy.
