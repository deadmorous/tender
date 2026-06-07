# Implementation Plan

_Created: 2026-06-06_

This document is the master reference for building tender. It records the
coding principles and organises the design decisions scattered across vibes
000001–000011 into a concrete, incremental build sequence.

---

## Coding principles

1. **Incremental growth.** The system must be alive (buildable, testable) at
   every step. No dead branches; no stubs that don't compile.
2. **Test everything.** Unit tests for every module. Functional and integration
   tests when a full slice is needed. Mock components to isolate units.
3. **Coverage.** Track test coverage; gaps are bugs-in-waiting.
4. **Benchmark performance-sensitive operations.** Any operation that could be
   called millions of times (expression tree traversal, rational arithmetic,
   pattern matching) gets a benchmark from the start.
5. **Maintain feasibility examples.** A small set of end-to-end examples (PVW,
   Rodrigues formula, BAC-CAB identity) prove the system works as a whole.
   These evolve with the system; they are never deleted.

---

## Cross-reference legend

Each phase below cites the vibes where its design decisions live.

| Vibe | Topic |
|---|---|
| 000001 | Tensor conventions, CS construction paths |
| 000002 | System boundaries, PVW, integral rewriting |
| 000003 | Architecture tiers, third-party strategy |
| 000004 | Expression tree, memory model, simplification, CS model |
| 000005 | Python operators, free functions, State/Derivation, named tensors |
| 000006 | Tensor coordinates, slots, three-layer model |
| 000007 | Identity library, pattern variables, constraint verification |
| 000008 | Parameters, time, material derivative |
| 000009 | Standard functions, Polynomial, differentiation rules |
| 000010 | Libraries, rotation tensors, bootstrap philosophy |
| 000011 | Numeric types, rational engine, named constants |
| 000015 | Phase 13 identity library — scope and implementation |
| 000016 | Rewrite search — BFS over sub-expression rewrites |
| 000017 | Phase 13.5 — identity derivation and library growth |
| 000018 | Phase 13.2 — index/basis bridge, expand-in-basis tooling |
| 000019 | Phase 13.6 — indexed-sum notation (collect repeated component sums) |

---

## Design note — Expression dispatch and downcasting

_Recorded 2026-06-06 after Phase 7._

### Current situation

`dynamic_cast` is already pervasive in production code: `make_scale`,
`make_sum`, `make_product`, `deriv`, `depends_on_impl`, `expand_poly_impl`
all use it to dispatch on concrete type.  This is fine for now.  In tests,
`dynamic_cast` is permanently acceptable for structural assertions.

### Planned replacement: discriminator + static_cast

After Phase 9, when the set of expression node types stabilises, we will
introduce:

- `enum class ExprKind { RationalConst, NamedConst, SymbolicVar, Parameter,
  NamedTensor, IdentityTensor, LeviCivita, Scale, Sum, TensorProduct,
  Contract, DoubleContract, DoubleContractReversed, CrossProduct, Trace,
  FunctionApply, Pow, Product, MaterialDeriv, PolynomialExpr, PatternVar }`;
- `kind()` accessor on `Expr` (non-virtual; set in a protected constructor);
- switch-based dispatch + `static_cast<Concrete*>(e)` — analogous to LLVM's
  `isa<>` / `cast<>` / `dyn_cast<>`.

This is a straight mechanical refactor: all semantics stay the same, only the
cast mechanism changes.

### Expr interface growth

The `Expr` interface will stay lean.  Heavy algorithms (deriv, substitute,
expand, simplify) stay as external free functions that dispatch over the full
node set.  A small number of _canonical participation_ operations — where
every node has a uniform answer — may be added as virtual methods (e.g.
`collect_params()`).  These are added only when a concrete need arises.

---

## Phase 0 — Project scaffold

**Goal**: a compilable, testable skeleton. Nothing mathematical yet.

- CMake ≥ 3.25 project; C++20
- FetchContent for: GoogleTest, nanobind, mpk_mix
- `clang-format` config committed; enforced in CI
- GitHub Actions CI: build + test on Linux
- Single passing smoke test: `EXPECT_TRUE(true)`

**Sources**: vibe 000004 (technology table), vibe 000003 (third-party strategy)

**Exit criterion**: `cmake --build . && ctest` succeeds in CI.

---

## Phase 1 — Rational arithmetic engine

**Goal**: exact rational scalars with a stable, backend-agnostic API.

- `Rational` concrete type: numerator/denominator as `int64_t`; always
  normalised (GCD reduction, positive denominator)
- Arithmetic: `+`, `-`, `*`, `/`, unary `-`
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Conversion: `to_double()`, `from_int()`
- **Fatal error on overflow**: detect before the operation, call
  `std::abort()` with a diagnostic message
- API is declared behind an interface header so the backend can be swapped to
  arbitrary-precision later without changing any call site
- Unit tests: all operations, normalisation, GCD edge cases, overflow trigger
- Benchmark: rational addition and multiplication in a tight loop

**Sources**: vibe 000011 (Q_rational_engine)

**Exit criterion**: all tests pass; benchmark numbers recorded.

---

## Phase 2 — Expression tree core

**Goal**: an allocatable, traversable expression tree for scalar expressions.

### Node hierarchy

Abstract base `Expr`:
- `rank()` → `int` (runtime)
- `latex()` → `std::string`
- `python()` → `std::string`
- `name` attribute (empty by default; set at most once)

Concrete scalar nodes:
- `RationalConst(Rational)` — rank 0
- `NamedConst(std::string)` — rank 0; π, e, etc.
- `SymbolicVar(std::string, traits)` — rank 0 scalar variable

Structural nodes (rank inherits from operands):
- `Sum(vector<Expr*>)` — addition; at least two operands
- `Scale(Rational, Expr*)` — scalar multiple; collapses nested Scales
- `TensorProduct(Expr*, Expr*)` — ⊗; rank = sum of operand ranks

### Memory model

All nodes allocated via `mpk_mix::ResourceList`. Callers receive raw `Expr*`
pointers. The ResourceList lives for the session.

### Always-on simplification (eager, applied at construction)

Applied by each node's constructor; never produces a larger expression:
- Rational arithmetic (e.g., `Scale(2) * Scale(3)` → `Scale(6)`)
- Flatten nested `Sum` into one flat `Sum`
- Drop zero `Scale` terms from `Sum`
- Collapse `Scale(1, e)` → `e`
- Collapse `Scale(0, e)` → `RationalConst(0)`
- Collect equal sub-expressions in a `Sum` (merge coefficients)

### Tests

- Construction, rank queries, LaTeX output for each node type
- Always-on simplification rules
- ResourceList lifetime: expressions survive the session that made them

**Sources**: vibe 000004 (Q_memory_model, Q_rank_tracking, Q_normal_form),
vibe 000005 (named tensor syntax, `named()`)

**Exit criterion**: scalar expressions construct, simplify eagerly, and render.

---

## Phase 3 — Index and slot system

**Goal**: co/contravariant indices with explicit summation control.

- `IndexSpace(name, range, auto_sum)` — range may be integer or symbolic `N`
- `Index(letter, IndexSpace*)` — display name + space; created by user
- Shortcut constructors: `AutoSumIndex3d(letter)`, `AutoSumIndex2d(letter)`
- Slot annotation on `Expr` nodes: each node carries a `SlotList` (upper/lower
  positions with optional `Index*` or display-name-only for anonymous slots)
- Annotation nodes:
  - `ExplicitSum(Expr*, Index*, IndexSpace*)` — force sum regardless of count
  - `NoSum(Expr*, Index*)` — suppress automatic contraction

Convolution:
- `convolve(Expr*, slot_a, Expr*, slot_b)` — pair one upper + one lower slot,
  produce a new expression with those slots consumed

Error conditions:
- Index appears 3+ times without `ExplicitSum` wrapper → error at construction
- `convolve()` called with two upper or two lower slots (non-orthonormal CS) → error

**Sources**: vibe 000004 (Q_index_slots), vibe 000006 (slot model, Layer 3)

**Exit criterion**: index tracking and convolution unit tests pass.

---

## Phase 4 — Tensor nodes and geometric operations

**Goal**: invariant tensors and the Layer 1 geometric operations.

Tensor nodes:
- `NamedTensor(name, rank)` — generic named tensor; rank tracked at runtime
- `IdentityTensor()` — **I**; rank 2; singleton
- `LeviCivitaTensor()` — **ε**; rank 3; singleton

Operations (produce new `Expr*` nodes):
- `Contract(Expr*, Expr*)` — single contraction `@`; rightmost lower ↔ leftmost upper
- `DoubleContract(Expr*, Expr*)` — `:` Frobenius `//`
- `DoubleContractReversed(Expr*, Expr*)` — `··` reversed `**`
- `CrossProduct(Expr*, Expr*)` — `×`; rank adds as for tensor product

Python operator wiring (applied in the nanobind layer, Phase 12):
- `a % b % c` → error at construction (non-associativity of cross product)

Always-on simplifications now extended:
- `I · a` → `a`
- `I : A` → `tr(A)`
- `0 ⊗ a` → `0` (zero propagation)

**Sources**: vibe 000001 (conventions), vibe 000004 (Q_rank_tracking),
vibe 000005 (operator table)

**Exit criterion**: `I @ v` reduces to `v`; `eps` can be constructed and
contracted; cross product raises error on chaining.

---

## Phase 5 — Standard scalar functions

**Goal**: exp, log, trig, hyperbolic, pow, Polynomial, with derivative rules.

Node type `FunctionApply(FunctionKind, Expr* arg)`:
- `FunctionKind` enum: `Exp`, `Log`, `Sin`, `Cos`, `Tan`, `ASin`, `ACos`,
  `ATan`, `ATan2`, `Sinh`, `Cosh`, `Tanh`, `Sqrt`, `Pow`

`Polynomial(vector<pair<Rational, int>>)` — coefficients × integer exponents;
arithmetic, evaluation, differentiation as methods.

Each `FunctionKind` carries:
- LaTeX template
- Derivative rule: `d/dx f(g) = f'(g) · g'` — chain rule fires during
  symbolic differentiation in Phase 7

Error: applying a scalar function to a non-rank-0 expression.

**Sources**: vibe 000009

**Exit criterion**: `sin(x).latex()` renders correctly; derivative rules
verified by unit tests against known results.

---

## Phase 6 — Parameters, time, and dependency tracking

**Goal**: parametric differentiation with automatic independence detection.

- `Parameter(name)` — a named scalar parameter; subtype of `SymbolicVar`
- Default `t` (time) exportable from the `tender` namespace
- `deriv(param, expr)` — symbolic partial differentiation
  - If `expr` does not depend on `param`: return zero tensor of matching rank
  - If `expr` does: apply differentiation rules recursively
- `dt(expr)` = `deriv(t, expr)`; `ddt(expr)` = `deriv(t, deriv(t, expr))`
- `material_deriv(v, f)` = `dt(f) + dot(v, grad(f))` (named node; expands on demand)

Dependency tracking:
- Each `Expr` node carries a set of `Parameter*` it depends on
- Set is computed lazily and cached on the immutable node

**Sources**: vibe 000008

**Exit criterion**: `deriv(t, A)` returns zero tensor when `A` has no `t`
dependency; chain rule fires through function applications.

---

## Phase 7 — State and Derivation model

**Goal**: auditable, composable derivation sequences.

- `State` — an immutable wrapper around an `Expr*`; no history inside it
- `DerivationStep` — named transformation: `State → State`; stores step name
  and both before/after expressions
- `Derivation(vector<DerivationStep>)` — ordered recipe
  - `apply(State) → vector<State>` — one state per step
  - `operator+(Derivation)` — concatenation
- `show(vector<State>)` — renders the full history
- `show(vector<State>, final_only=true)` — renders only the last state

Built-in named steps (initially just wrappers, implementations grow over time):
- `expand_step`, `contract_step`, `substitute_step(what, with_what)`
- `simplify_identity_step`, `simplify_epsilon_step`
- `collect_step(subexpr)`, `symmetrize_step`, `skew_step`

**Sources**: vibe 000005 (Q_derivation_display, Q_python_output)

**Exit criterion**: a two-step derivation applies, returns correct State list,
and renders step-by-step.

---

## Phase 8 — Identity library (basic)

**Goal**: named algebraic identities; manual-targeted application.

- `PatternVar(name)` — distinct from `SymbolicVar`; fluent constraint API:
  `.rank(n)`, `.symmetric()`, `.skew_symmetric()`, `.rank(n).symmetric()`, …
- `pattern_vars(*names)` — convenience batch constructor
- `Identity(name, lhs, rhs)` — rewrite rule; directed LHS→RHS
- `apply_identity(identity, mapping={a: expr_a, …})` — manual targeting;
  produces a `DerivationStep`
- `Identity.from_derivation(name, history, vars=[…])` — promote a completed
  derivation to a reusable identity

Constraint verification (structural first, then algebraic):
- Structural: detect symmetry/skew-symmetry directly from expression tree shape
- Algebraic: compute `A − Aᵀ` and test for zero
- User declaration: `tensor('A').declare(symmetric=True)` writes to constraint cache

Constraint cache: one attribute per constraint on the immutable `Expr` node;
`absent` / `True` / `False`.

**Sources**: vibe 000007 (pattern matching, constraint verification, caching)

**Exit criterion**: BAC-CAB identity (`a % (b % c) = b*(a@c) - c*(a@b)`)
can be stated, applied manually, and confirmed by a unit test.

---

## Phase 9 — Named simplification operations

**Goal**: the full set of user-invokable simplification steps.

All implemented as `DerivationStep` factories:

| Function | Effect |
|---|---|
| `contract()` | Perform all valid Einstein contractions |
| `simplify_identity()` | Apply identity tensor rules |
| `simplify_epsilon()` | Apply Levi-Civita identities |
| `expand_grad()` | Product rule on gradient |
| `symmetrize()` / `skew()` | Decompose into symmetric/skew parts |
| `collect(subexpr)` | Group terms by sub-expression |
| `substitute(what, with_what)` | Symbolic substitution |
| `expand()` | Expand products over sums |

Termination guarantee: each operation is finite by construction (tree is
finite; visited-node tracking guards DAG traversal; iteration cap as safety net).

**Sources**: vibe 000004 (Q_normal_form, Q_rule_repr, Q_termination)

**Exit criterion**: `simplify_identity()` applied to `I @ v` returns `v`
as a `DerivationStep` with correct before/after states.

---

## Phase 10 — Coordinate systems

**Goal**: WCS, transformations, built-in CSs, embedding-map construction.

- `WCS` singleton — root reference frame; Cartesian orthonormal
- `CS(reference, rotation=None, translation=None)` — general CS
- CS carries: basis vectors, cobasis vectors, metric, Christoffel symbols
  (computed lazily from the construction path)

Construction paths (in priority order for implementation):
1. `WCS` — hardcoded; already available as basis **i**, **j**, **k**
2. `DirectBasisCS(e1, e2, e3)` — basis specified as vector expressions;
   metric, cobasis, Christoffels derived automatically (vibe 000001, path 6)
3. `EmbeddingMapCS(r_func)` — position vector **r**(q¹,q²,q³); covariant basis
   from partial derivatives (vibe 000004, Q_cs_construction)
4. `CylindricalCS`, `SphericalCS`, `PolarCS2D` — pre-defined built-ins
5. `CurveCS(r_func)` — Frenet-Serret frame for rods
6. `SurfaceCS(r_func_uv)` — surface tangent basis + normal for shells

Gradient dispatch (vibe 000004, Q_cs_dispatch):
1. Field carries CS reference → use its cobasis
2. Invariant field → compute symbolically; apply built-in rules
3. `grad(f, cs=...)` → override

`is_orthonormal` flag on CS: relaxes slot level checking in contractions.

**Sources**: vibe 000001 (CS paths), vibe 000004 (Q_cs_construction, Q_cs_dispatch)

**Exit criterion**: `DirectBasisCS` derives its cobasis correctly for a known
non-orthogonal example; `CylindricalCS` gradient of a scalar field produces
the known formula.

---

## Phase 11 — Integral expressions and PVW

**Goal**: domains, integrals, divergence theorem, localization.

Domain types: `Volume(name)`, `Surface(name, normal)`, `Curve(name, tangent)`,
`Point`. Each carries its boundary (`boundary(domain)` returns ∂domain).

`Integral(domain, integrand)` node — rank matches integrand rank.

Named derivation steps:
- `apply_divergence_theorem(domain)` — ∫_V ∇·**A** dV → ∮_∂V **A**·**n** dS
- `apply_stokes_theorem(domain)`, `apply_greens_identity(domain)`
- `apply_integration_by_parts(domain)`
- `localize()` — logical inference: ∫ = 0 ∀ δ**u** → integrand = 0

Feasibility example: write the PVW integral (vibe 000002) as a tender
expression and apply `apply_integration_by_parts` + `localize` to derive the
strong form balance equation.

**Sources**: vibe 000002, vibe 000004 (Q_divergence_theorem)

**Exit criterion**: PVW example runs end-to-end and produces the correct
strong-form equation as a derivation history.

---

## Phase 12 — Python bindings (nanobind)

**Goal**: the full Python API; Jupyter rendering.

Bind all types from Phases 1–11. Python-layer additions:
- `__add__`, `__mul__`, `__matmul__`, `__pow__`, `__floordiv__`, `__mod__`
  wired to the C++ node constructors; `%` operator raises error on chaining
- `_repr_latex_()` for Jupyter auto-rendering
- `expr.python()` output; `named(name, expr)` convenience function
- `wcs`, `I`, `eps`, `nabla` importable singletons
- `AutoSumIndex3d`, `AutoSumIndex2d` shortcut constructors
- `tensor(name)`, `scalar(name)`, `field(name, depends_on=[…])`
- `Derivation`, `State`, `show(history)`

Feasibility test: run the BAC-CAB derivation interactively in a Jupyter notebook.

**Sources**: vibe 000004 (Q_ui_cppapi), vibe 000005 (operators, named syntax,
State/Derivation), vibe 000006 (coord extraction)

**Exit criterion**: `from tender import I, eps, wcs` works; Jupyter cell renders
tensor expressions as LaTeX automatically.

---

## Phase 13 — Identity library (full)

**Goal**: automatic targeting, Theorem type, standard library.

- Automatic targeting: combinatorial search with configurable budget;
  `all_matches=True` returns every valid mapping
- `Theorem(name, statement, introduced_objects, reference)` — existence
  assertion; `apply_theorem()` introduces named objects with constraints
- `NamedObject` (prototype + singleton distinction, vibe 000010)
- `doc(entry)` / `doc(entry, format='plain')` — renders identity/theorem spec

Standard identity library submodules:
- `tender.lib.identities.epsilon` — ε-contraction identities, BAC-CAB rule
- `tender.lib.identities.identity_tensor` — **I**·**a** = **a**, **I**:**A** = tr(**A**)
- `tender.lib.identities.functions` — sin²+cos²=1, exp rules, log rules
- `tender.lib.theorems.spectral` — spectral decomposition of symmetric tensors

**Sources**: vibe 000007 (full), vibe 000010 (library architecture)

**Exit criterion**: `apply_identity(bac_cab)` with auto-targeting finds the
correct binding on a concrete expression; spectral theorem can be invoked and
introduces the correct named objects.

---

## Phase 13.2 — Index/basis bridge

**Goal**: build the tooling that allows derivations to proceed via explicit
component expansion in a coordinate system, enabling proofs like `a·b = b·a`
from first principles rather than by assertion.

### Deliverables

1. **Python bindings** for `Index`, `IndexSpace`, `CoordSystem`, `wcs()`,
   `direct_basis_cs`, `explicit_sum`, `no_sum`

2. **`expand_in_basis_step(tensor, cs, covariant=True)`** — replaces a named
   tensor with its explicit 3-term component expansion in `cs`; component
   scalars are fresh `NamedTensor` nodes

3. **`simplify_basis_dot_step(cs)`** — replaces `Contract(e_i, e^j)` pairs
   with their scalar value (`1` or `0` for WCS; `metric(i,j)` in general)

4. **`collect_zero_terms_step()`** — removes `Scale(0, ...)` terms from sums
   and unwraps single-element sums

5. **`reassemble_from_components_step(cs)`** — inverse of expansion; recognises
   a sum of `(scalar × basis_vector)` terms as a tensor, emitting
   `IdentityTensor` when the coefficients match, or a fresh `NamedTensor`
   otherwise; needed because tensor-valued results (e.g. `δ_i^j δ_j^k = δ_i^k`)
   leave multiple non-zero terms that cannot be collected into a scalar

6. **Scalar commutativity** — either a step or an identity rule allowing
   `Product(s, t)` for rank-0 operands to be reordered

7. **Example derivation** of `a·b = b·a` in WCS demonstrating the full
   pipeline: expand both vectors, distribute, evaluate basis dots, collect

### Exit criterion

A Python script (and notebook) proves `a·b = b·a` step-by-step using only
the tools above (no `dot_comm` axiom); all intermediate states are valid
`State` objects in a `Derivation` history.

**Sources**: vibe 000018

---

## Phase 13.5 — Identity derivation and library growth

**Goal**: build the identity library from the ground up — starting from
definitional axioms and deriving all other results mechanically.

### Deliverables

1. **Definitional layer** — a new `tender.lib.identities.definitions` module
   containing the smallest set of axiomatic rules:
   - `dot_bilinear` — `(a+b)·c = a·c + b·c` (and symmetric counterpart)
   - `tp_contract_right` — `(u⊗v)·w = u*(v·w)` (dyad contracts from right)
   - `tp_contract_left` — `w·(u⊗v) = (w·u)*v` (dyad contracts from left)
   - `trace_dyad` — `tr(a ⊗ b) = a·b` (defines trace; linearity extends it)
   - `identity_dot_vec` — `I·a = a` (defines **I** on vectors; already exists)
   - `cross_def` — `a×b = ε:(a⊗b)` (defines cross product)
   - `eps_antisym` — anti-symmetry of **ε** under any index swap
   - `eps_norm` — `ε₁₂₃ = 1` (normalization; with antisymmetry determines all components)
   - `scalar_comm` — `s·t = t·s` for rank-0 operands (axiom; see vibe 000018)
   - Note: `a·b = b·a`, `eps_contract`, `eps_sq`, `ε·a = -(I×a)` are all *derived*

2. **Derived identity tier 1** (immediate from definitions):
   - `cross_self_zero` — `a×a = 0`
   - BAC-CAB re-derived (currently asserted; now produced by `search_apply` +
     `Identity.from_derivation`)
   - scalar triple product cyclic shifts
   - `(a×b)·(c×d) = (a·c)(b·d) − (a·d)(b·c)` (Lagrange identity)

3. **`derive_identity` convenience function** — wraps `search_apply` +
   `Identity.from_derivation` in one call:
   ```python
   id = derive_identity("double-cross", cross(cross(a, b), c),
                        target=bac_cab)
   ```

4. **Pattern-variable promotion in `Identity.from_derivation`** — accept a
   `substitute` mapping from concrete expressions to `PatternVar`s, replacing
   both endpoints of the derivation history.

5. **User extension workflow** — a documented, tested template for creating
   custom identity modules that plug into `search_apply`'s default rule set.

6. **Updated examples** — `bac_cab.py` and `bac_cab_search.py` updated to
   import from the new library layout; a new `triple_product.py` example
   demonstrating derivation of cyclic-shift identities.

### Identity structure

All derived identities live in `tender.lib.identities.derived` (or in their
natural sub-module, e.g. `epsilon.py` for cross-product results).  The
defining axioms live in `tender.lib.identities.definitions`.  The `ALL` list
exported from `tender.lib` covers both.

### Exit criterion

`search_apply(bac_cab, cross(cross(a, b), c))` produces a two-step derivation
(anti-comm + BAC-CAB) and `Identity.from_derivation` converts it into a new
`double_cross` identity; a notebook demonstrates the full workflow including
user-defined extensions.

**Sources**: vibe 000017

---

## Phase 13.6 — Indexed-sum notation

**Goal**: add a `collect_repeated_sum_step(cs)` that collapses an explicit
component sum such as `a^1 b_1 + a^2 b_2 + a^3 b_3` into the compact indexed
form `a^i b_i`, completing the WCS component-expansion pipeline with readable
output.

### Deliverables

1. **`IndexedSum` AST node** — display-only rank-0 node; stores pre-rendered
   body LaTeX with the running suffix replaced by an index letter.
2. **`collect_repeated_sum_step(cs)`** — recognises a `Sum` of `dim` terms each
   of the form `Product(NamedTensor, NamedTensor)` where symbols differ only by
   a 1-based integer suffix; emits an `IndexedSum` node.
3. **Python binding and `__init__.py`** — expose `IndexedSum` type and
   `collect_repeated_sum_step`.
4. **Tests** — C++ unit tests; Python tests in `test_tender.py`.
5. **Updated `dot_commutativity.py`** — final step shows `a^{i} b_{i}`.

### Exit criterion

`show(history)` for the `dot_commutativity` derivation ends with

    [collect_repeated_sum]  a^{i} b_{i}

and the compiled PDF renders it correctly.

**Sources**: vibe 000019

---

## Phase 14 — Named constants and symbolic scalar traits

**Goal**: π, e as named objects; symbolic scalars with traits.

- `NamedConst` nodes registered in the named object library
- Simplification rules: `sin(pi) = 0`, `exp(1) = e`, `log(e) = 1`, etc.
  as entries in `tender.lib.identities.functions`
- `scalar(name).non_negative()`, `.positive()`, `.integer_valued()`, `.unit()`
  — trait annotations usable in pattern variable constraints
- Identity matching can condition on scalar traits (e.g., √(λ²) = λ if
  λ `.non_negative()`)

**Sources**: vibe 000011 (Q_named_constants)

**Exit criterion**: `sin(pi)` simplifies to 0 via identity application;
`sqrt(lam**2)` simplifies to `lam` when `lam` is declared non-negative.

---

## Phase 15 — Rotation tensor library (bootstrap)

**Goal**: demonstrate the bootstrap philosophy; populate the rotation library.

Steps:
1. Using the tender system as built, derive:
   - **R**^T · **R** = **I** from the definition of orthogonality
   - det(**R**) = +1
   - **Ω** = **Ṙ** · **R**^T is skew-symmetric
   - Rodrigues formula: **R** = **I** + sin(θ)**N** + (1−cosθ)**N**²,
     **N** = **n** × **I**
2. Register each result as a library entry (Identity or Theorem)
3. `R_proto = NamedObject('R', constraints=[orthogonal, det_plus_1])` as prototype
4. `from tender.lib.objects.rotation import R_proto, Omega_proto`

Feasibility examples updated to include a rigid-body kinematics derivation.

**Sources**: vibe 000010 (rotation tensor library, bootstrap philosophy)

**Exit criterion**: `R1 = R_proto.instance('R_1')` creates a named rotation
tensor; `doc(R_proto)` renders all known identities that apply to it.

---

## Deferred tiers (not in this plan)

- **Tier 2a**: Taylor series, symbolic indefinite integration via bridge
- **Tier 2b**: ODE/PDE characteristic polynomials, stability analysis
- **Tier 3**: Complex analysis, arbitrary-precision numerics, Kolosov-Muskhelishvili
- **Bridges**: SymPy/Mathematica integration bridge, FEniCS/UFL emission bridge

---

## Persistent feasibility examples

Maintained throughout all phases (added to as each phase completes):

| Example | Tests |
|---|---|
| `examples/scalar_arithmetic.py` | Rational arithmetic, named constants |
| `examples/bac_cab.py` | Identity library, tensor product, cross product |
| `examples/rodrigues.py` | Rotation library, standard functions, trig identities |
| `examples/pvw_continuum.py` | Integrals, divergence theorem, localization |
