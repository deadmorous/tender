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

**Q_memory_model**: RESOLVED — `mpk_mix::ResourceList`

Nodes are allocated via `ResourceList` from the user's `mpk_mix` library.
`ResourceList::create<T>(...)` allocates an object of any type, stores a
`std::shared_ptr<void>` to it internally, and returns a raw pointer.
All objects live until the `ResourceList` is destroyed — no manual management.

Properties:
- Raw pointers used for tree navigation — zero reference-counting overhead
- Structural sharing works naturally: multiple nodes hold raw pointers to the same child,
  child stays alive as long as the `ResourceList` lives
- Rewriting produces new nodes on the same `ResourceList`; old nodes are not eagerly freed
- `ResourceList` lifetime = derivation session lifetime — create one, do the algebra, destroy it

Similar in spirit to an arena, but without pre-allocated memory blocks.
`mpk_mix` will be added as a third-party dependency when coding begins.

**Q_rank_tracking**: RESOLVED — dynamic (runtime integer), with old-school polymorphism throughout

Rank is a runtime property. Users specify expressions freely (primarily through the Python
interface), so compile-time rank tracking is not applicable.

Broader architectural decision settled here:
- **Abstract interfaces + virtual functions** are the implementation vehicle for the entire
  core — not templates, not std::variant + visitors. Chosen explicitly for simplicity:
  the codebase will be much easier to understand and extend, and there is no latency
  requirement that would justify the complexity of heavy template metaprogramming.
- This decision may be revisited if performance bottlenecks appear in numerical evaluation,
  but the expected path there is **code generation** — generating simple, flat numerical
  code for a specific completed derivation — rather than making the symbolic core generic
  and heavily templated.
- Virtual dispatch overhead is acceptable for symbolic manipulation, which is not
  latency-critical.

**Q_index_slots**: RESOLVED — track covariant/contravariant slot types explicitly, with two refinements

Co/contravariant distinction is carried in the expression tree because it is necessary for
curvilinear correctness (raising/lowering indices involves the metric; a contraction is only
valid when one index is up and one is down).

**Refinement 1 — orthonormal basis relaxation:**
In an orthonormal basis (gᵢⱼ = δᵢⱼ), covariant and contravariant components coincide and
the up/down distinction is physically meaningless. The CS object carries an `is_orthonormal`
flag; when set, the system relaxes index slot checking and allows same-level index pairing.
This lets users write Cartesian expressions without decorating every index with a level.

**Refinement 2 — Einstein summation overrides:**
The default Einstein rule (one up + one down repeated index → contraction) must be
overridable in both directions:

- **Explicit sum over an index appearing 3+ times** (or at the same level in orthonormal mode):
  e.g. the spectral decomposition **A** = Σᵢ Aᵢ **v**ᵢ⊗**v**ᵢ where i labels eigenvalues and
  eigenvectors — i appears three times, which is invalid under Einstein's rule but the
  summation is well-defined and needed. Represented as an `ExplicitSum(body, index, range)`
  node that marks the summation regardless of occurrence count.

- **Explicit no-sum over a repeated index**: suppress automatic contraction when an index
  appears twice but summation is not intended. Represented as a `NoSum(body, index)`
  annotation on the relevant sub-expression.

These two annotation node types (`ExplicitSum`, `NoSum`) extend the expression tree
and must be respected by every rewriting rule that touches indexed expressions.

**Refinement 3 — index spaces as first-class objects:**

An index is not just a name — it is an object carrying a reference to an `IndexSpace`.
An `IndexSpace` defines:
- `range`: the cardinality of the index (integer constant, or a symbolic value like N
  for FEM shape function count); can be arbitrary, not limited to spatial dimensions
- `auto_sum`: whether Einstein summation applies by default when this index appears
  in a valid contraction position (one up, one down)

Examples:
- `IndexSpace("spatial_3d", range=3, auto_sum=true)` — standard 3D spatial index
- `IndexSpace("spatial_2d", range=2, auto_sum=true)` — surface/plate/shell index
- `IndexSpace("fem", range=N, auto_sum=false)` — FEM shape function index;
  u = uᵢφᵢ requires explicit summation because i is not a spatial Einstein index
- `IndexSpace("layers", range=M, auto_sum=false)` — e.g. laminate layers

An `Index` is a named symbol (a letter, for display) bound to an `IndexSpace`.
The space determines summation behavior; the name is for human readability only.

**On index space deduction:**

Index spaces are inferred from the geometric objects being indexed, not declared
separately by the user in common cases:
- A 3D vector or tensor → its indices are automatically `spatial_3d`
- A tensor living on a 2D CS (surface, plate, shell) → indices are `spatial_2d`
- **ε**, **I**, and other built-in spatial tensors → infer from their own definition

The user only needs to declare an index space explicitly when the index does not
correspond to any tensor's natural space — e.g. FEM shape function index,
laminate layer index, eigenmode label.

The Latin=3D, Greek=2D naming convention is not adopted as a default rule.
Letter names are display-only; space is always deduced from geometric context or
declared explicitly. Special-casing `spatial_3d` / `spatial_2d` in the simplifier
(e.g. for ε-contraction identities that depend on knowing dimension = 3) is acceptable
if it genuinely simplifies those rules — but we do not multiply abstractions beyond that.

**Summation rules, summarized:**

| Situation | Behavior |
|---|---|
| Index appears once up + once down, `auto_sum=true` | Automatic contraction (Einstein) |
| Index appears once up + once down, `auto_sum=false` | Free index — no contraction |
| Index appears 3+ times, any `auto_sum` | Error unless wrapped in `ExplicitSum` |
| Index appears twice same level, orthonormal CS | Allowed; `auto_sum` flag governs contraction |
| `ExplicitSum(body, index, space)` | Sum over `space.range` regardless of occurrence count |
| `NoSum(body, index)` | Suppress contraction regardless of `auto_sum` |

### Simplification / rewriting engine

**Q_normal_form**: RESOLVED — user-guided simplification via named operations

No single canonical normal form is attempted. Instead, the system provides a collection
of named simplification operations that the user invokes explicitly and composes in
whatever order the derivation requires. The primary interface is a Jupyter notebook,
where the user can inspect the expression at each step, apply an operation, and continue.

This is consistent with the "derivation assistant, not automaton" principle: the system
handles the algebra of each step; the user supplies the strategy.

**Automatic (always-on) simplification** — applied eagerly, expected to always reduce:
- Arithmetic on known constants (rational arithmetic)
- Flattening of nested sums and products
- Elimination of zero terms and unit factors
- Scalar factor collection within a sum

**Named simplification operations** — user-invoked, each a deliberate derivation step:

| Operation | Effect |
|---|---|
| `contract()` | Perform all valid Einstein contractions |
| `simplify_identity()` | Apply identity tensor rules: **I**·**a** = **a**, **I**:**A** = tr(**A**), etc. |
| `simplify_epsilon()` | Apply Levi-Civita identities (e.g. ε-contraction producing Kronecker deltas) |
| `expand_grad()` | Apply product rule to gradient of a product |
| `symmetrize()` / `skew()` | Decompose into symmetric and skew-symmetric parts |
| `collect(subexpr)` | Group terms by a given sub-expression |
| `substitute(what, with_what)` | Symbolic substitution |
| `expand()` | Expand products over sums |
| `apply_divergence_theorem(domain)` | Rewrite volume integral as surface integral |
| `localize()` | Logical step: integral = 0 for all virtual fields → integrand = 0 |
| `assume_kinematic(hypothesis)` | Substitute a kinematic hypothesis (user-defined) |

The list grows as real derivations reveal missing operations. No attempt is made to
define "fully simplified" globally — an expression is done when the user says it is.

**Q_rule_repr**: RESOLVED — rules are code; composition is the user-facing data

Built-in simplification operations are implemented as code (C++ algorithms with virtual
dispatch). This is the right choice because the operations we need are inherently
procedural and cannot be expressed as simple pattern→replacement pairs:
- `apply_divergence_theorem` must understand domain types and tensor ranks
- `simplify_epsilon` must know the ambient dimension is exactly 3
- `collect(subexpr)` requires structural tree traversal and matching

A data-driven rule engine powerful enough to express these would be reimplementing
code in a more fragile notation, adding complexity without adding power.

The user-facing "data" lives at a higher level: the **name and parameters of each
operation**, and their **composition into a derivation sequence** in Python/Jupyter.
A notebook cell sequence is already a clean, human-readable, inspectable record of
every derivation step — no lower-level rule editing is exposed or needed.

User extensibility, if ever needed, comes through Python: the user writes a function
that composes existing named operations, not by defining low-level rewrite rules.
Simple algebraic substitution rules (`substitute(what, with_what)`) cover the common
case of user-supplied identities without opening up the rule engine itself.

**Q_termination**: RESOLVED — termination by construction per operation, with pragmatic fallbacks

Since rules are code rather than a general term-rewriting system, there is no need for
a global term ordering or a system-wide termination proof. Each built-in operation is
responsible for its own termination. Three mechanisms, applied as appropriate:

1. **Finite by construction** — the primary approach. Each operation is designed to
   terminate structurally: it works over a finite expression tree, collects terms at
   a well-defined granularity (e.g. dyad level), and produces a result no larger in
   the relevant measure. This is the implementer's responsibility per operation.

2. **Visited-node tracking** — for any operation that traverses the expression graph
   (which may be a DAG due to structural sharing), track already-visited nodes to
   prevent re-processing cycles.

3. **Iteration cap as safety net** — for operations that iterate to a fixpoint,
   impose a maximum step count. If the cap is hit, return the partially processed
   expression and emit a warning. Not expected to trigger in correct usage.

**User control flow**: if the user constructs a loop in Python that does not terminate,
the system does not add its own safeguards — the user can interrupt execution (Ctrl+C
in Jupyter). Injecting hidden step counters into user-level control flow would add
noise and obscure what is happening. The boundary is clear: built-in operations are
safe by design; user-composed sequences are the user's responsibility.

### Coordinate system model

**Q_cs_construction**: RESOLVED — embedding map is primary; five construction paths

Option 3 (manual Christoffels) is rejected — computing them by hand is error-prone
in the general case and defeats the purpose of the system.

**Five construction paths, in order of abstraction:**

1. **Built-in named CS** — predefined embedding maps for well-known systems.
   System ships with at minimum: `CylindricalCS`, `SphericalCS`, `PolarCS2D`.
   These are the embedding-map approach with the map already filled in.

2. **From embedding map** *(primary general path)* — user supplies **x**(q¹, q², q³) → WCS
   as a symbolic expression. System derives covariant basis **g**ᵢ = ∂**x**/∂qⁱ,
   metric gᵢⱼ = **g**ᵢ·**g**ⱼ, and Christoffel symbols automatically.

3. **From metric components** — user supplies gᵢⱼ(q) directly; Christoffels derived.
   Kept as an alternative path for generality (e.g. general relativity use cases where
   the embedding in flat space may not exist or is not the natural starting point).

4. **From curve** — user supplies position vector **r**(s) of a curve (s = arc length
   or parameter). System derives the natural trihedron (Frenet-Serret frame):
   - Tangent: **t** = **r**'(s) / |**r**'(s)|
   - Principal normal: **n** = **t**'(s) / |**t**'(s)|
   - Binormal: **b** = **t** × **n**
   - Curvature κ and torsion τ follow from the Frenet-Serret equations
   This gives a 1D CS on the curve with the trihedron as its natural basis,
   used for elastic rod theory.

5. **From surface** — user supplies position vector **r**(u, v) of a surface.
   System derives:
   - Covariant basis of tangent plane: **g**₁ = ∂**r**/∂u, **g**₂ = ∂**r**/∂v
   - Surface normal: **g**₃ = **g**₁ × **g**₂ (unnormalized, or normalized as needed)
   This defines a 2D CS (u, v) on the surface. The basis is automatically completed
   to a 3D CS (u, v, w) by adding **g**₃ as the third direction (w measured along the
   normal — the thickness coordinate in shell/plate theory).
   Used for elastic shell, plate, and membrane theories.

**Q_cs_dispatch**: RESOLVED — field carries CS reference; invariant fields handled separately

**Primary rule**: a field carries a reference to its CS. When the user defines a scalar
field as f(q¹, q², q³), the CS those coordinates belong to is part of the field object.
`grad(field)` automatically uses the cobasis **g**ⁱ of that CS — no extra argument needed.

**Invariant fields**: some fields have no CS dependency and can be expressed in
coordinate-free form. The position vector **r** is the canonical example — it is just **r**,
not a function of any particular coordinates. For such fields:
- ∇ is computed symbolically, using WCS as an intermediate if helpful, and the result
  is expressed in invariant form where possible
- Known results are registered as built-in simplification rules:
  e.g. `simplify_identity()` knows ∇**r** = **I** (true in any CS:
  **g**ⁱ ⊗ ∂**r**/∂qⁱ = **g**ⁱ ⊗ **g**ᵢ = **I**)
- Other invariant results (∇|**r**| = **r**/|**r**|, etc.) are added as needed

**CS override**: `grad(field, cs=some_cs)` allows the user to specify which CS to use
for differentiation, overriding the field's own CS. Useful when expressing a field
in one CS but differentiating with respect to another.

**Result representation**: a method `.in_cs(cs)` on expressions allows the user to
request that the result be expressed in a specific CS basis — separate from which CS
was used to compute the derivative.

**Summary of dispatch logic for ∇**:
1. Field has a CS → use its cobasis automatically
2. Field is invariant → compute symbolically, return invariant form; apply known rules
3. Explicit `cs=` override → use that CS regardless of field's own CS

### Integrals

**Q_divergence_theorem**: RESOLVED — named macro, user-invoked; revisit automation later

The divergence theorem is an explicit derivation step the user invokes by name,
consistent with the "derivation assistant, not automaton" principle and with the
treatment of localization as a named inference step.

A family of related theorems falls under the same pattern — all are named macros:

| Macro | Transforms |
|---|---|
| `apply_divergence_theorem(expr, domain)` | ∫_V ∇·**A** dV → ∮_∂V **A**·**n** dS (and tensor generalizations) |
| `apply_stokes_theorem(expr, domain)` | ∫_S (∇×**F**)·d**S** → ∮_∂S **F**·d**l** |
| `apply_greens_identity(expr, domain)` | Green's first/second identity for scalar fields |
| `apply_integration_by_parts(expr, domain)` | Tensor integration-by-parts derived from divergence theorem |

Automatic application of these theorems would require the system to recognize the
appropriate structural pattern in an integral expression and decide when to apply —
a level of reasoning that is currently out of scope. This decision is flagged for
revisiting once the core system is working and real derivation experience reveals
whether automation would be valuable and tractable.

---

## 2. User interface

**Settled: nanobind over pybind11** for Python bindings.
nanobind is the modern successor from the same author, faster compile times,
smaller binaries, C++17-native design. Python-in-Jupyter is the target interactive
experience. C++ API remains primary; Python is the interactive shell.

**Q_ui_cppapi**: RESOLVED — Python is the sole user-facing interface; C++ is for contributors

The C++ API is an implementation layer. Ergonomics, operator overloading, and DSL-like
syntax in C++ are not goals — correctness, clarity, and extensibility for core contributors
are. The Python interface (via nanobind) is what end users see and interact with.

Consequences:
- C++ API design is not constrained by user-facing ergonomics; it can prioritize
  internal clarity and contributor experience
- All user-facing naming, convenience, and composition live in the Python layer
- The nanobind binding layer is a first-class part of the system, not an afterthought

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

**Q_ci**: RESOLVED — GitHub Actions, Linux only

CI: GitHub Actions (project is on GitHub; user has more GitLab CI experience but
GitHub Actions is the pragmatic choice for the host platform).

Target platform: Linux only. macOS and Windows support are not planned — a future
contributor may add them. No effort spent on cross-platform compatibility beyond
what modern C++20 and CMake provide for free.

---

## 4. License

**Settled: GPL-3.0**
Copyleft. Chosen to enable direct linking with GPL-licensed computational software
without restrictions — Gmsh (mesh generation), Code_Aster and CalculiX (structural FEM
solvers directly relevant to mechanics), GSL (numerical methods), and others.

LGPL libraries (deal.II, FEniCS, SLEPc) are compatible with GPL and can be linked directly.
Permissive libraries (PETSc/BSD, Eigen/MPL-2.0, SUNDIALS/BSD, nanobind/BSD) are compatible.
Permissiveness for proprietary use was not a priority.
