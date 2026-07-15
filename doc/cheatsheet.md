# tender cheatsheet

A quick reference to the Python surface of **tender**, a tensor-algebra library
for computational mechanics in direct (coordinate-free) notation. This is a
method reference, not a tutorial — for worked derivations see `examples/`.

Conventions used in the tables:

- **Returns `Expr`** unless stated otherwise. An `Expr` is an immutable,
  hash-consed expression tree; you build new ones with operators and factories,
  you never mutate.
- ⊗ = tensor product, · = dot, × = cross, : / ·· = the two double contractions,
  ∇ = nabla, Δ = Laplacian.
- "does **not**" columns flag the common expectation a method deliberately does
  not meet.

---

## The pieces and how they fit

tender is layered. From the top down:

- **`Workspace`** (`tender.Workspace`) — the recommended entry point. It owns a
  single `Context` and forwards the common factories with `ctx=` bound, and
  mints chart coordinates with their `chart_id`/`slot` filled in. Everything
  below can be driven with an explicit `ctx=` instead, but the workspace removes
  the bookkeeping.
- **`Context`** (`tender._core.Context`) — the arena that owns every allocated
  `Expr` and hands out globally-unique index ids. One per derivation session.
  Rarely touched directly once you use a `Workspace`.
- **`Expr`** — the expression object. Overloaded Python operators (`*`, `@`,
  `%`, `/`, `**`, `//`, `+`, `-`) build the algebra; a handful of methods
  (`.tr()`, `.vec()`, `.transpose()`, `.latex()`, `.rank`) round it out.
  Produced by the **factories** (`tensor`, `scalar`, `identity`, `field`,
  `coordinate`, `delta`, `levi_civita`, …) in the top-level `tender` module.
- **`Derivation`** (`tender.derivation`, usually imported as `td`) — records an
  expression through a sequence of rewrite **steps**. Each step is a plain
  `Expr -> Expr` callable, so the built-ins and your own compose the same way.
  Also home to `Identity` / `saturate` (rule-based rewriting via an e-graph)
  and the equality predicates.
- **`Basis`** (`tender.basis`, usually `tb`) — vector frames and the
  invariant⇄coordinate bridge (`expand_in_basis`, `reassemble`, …). Built by
  hand or obtained from a well-known system (`wcs`, `cylindrical`, …).
- **`CoordinateChart`** (`tender.chart`) — a coordinate mapping from which the
  whole orthogonal-curvilinear geometry is *derived*: radius vector, tangent
  basis, metric, scale factors, physical frame, connection, and the
  differential operators `grad`/`div`/`rot`/`laplacian`, plus `evaluate`
  (lower a coordinate-free ∇ expression onto this chart) and
  `expand`/`components` (surface an invariant into frame components).
- **`tender.operators`** — a small DSL (`nabla`, `d`, `laplacian`) for writing
  differential expressions symbolically and `.evaluate(chart)`-ing them later.

A typical session flows: build a `Workspace` → declare fields/coordinates and a
chart → write an invariant expression (optionally with ∇ operators) → `evaluate`
/ `expand` / `components` onto the chart → `simplify` and read off the LaTeX.

---

## `Workspace` — the facade (`tender.Workspace`)

`ws = t.Workspace()`. Every method binds `ctx=ws.ctx` for you.

| Method | Returns | Does | Does **not** |
|---|---|---|---|
| `tensor(name, rank=None)` | `Expr` | Named tensor object; `rank=None` is dimension-agnostic (doubles as an identity pattern var) | Attach coordinate dependence — use `field` |
| `field(name, rank, deps=None, symmetric=False)` | `Expr` | Tensor field; `deps=None` ⇒ depends on all coords (∂ never silently 0); `symmetric=True` marks rank-2 `T_ij=T_ji` | Bind to a chart frame; it's still invariant |
| `scalar(value)` | `Expr` | Scalar literal (int or `Rational`) | Accept floats |
| `identity(space=None)` | `Expr` | Identity tensor `I`, carries its dimension (`tr(I)=n`); defaults 3-D | Provide a dimension-agnostic `I` |
| `coordinate(name, chart_id=0, slot=0, nonneg=False)` | `Expr` | One coordinate atom; prefer `coords` for a set | — |
| `coords(*names, chart_id=None, nonneg=())` | `list[Expr]` | Mint a coordinate set, slots by position, one fresh `chart_id`; `nonneg` names license `√(x²)→x` | — |
| `wcs()` | `Basis` | World Cartesian frame i, j, k — **memoised** (same basis every call, so sibling charts share a reference) | Return a fresh frame per call |
| `cylindrical()` / `spherical()` / `polar_2d()` | `Basis` | Well-known physical frames | — |
| `chart(reference, coords, embedding)` | `CoordinateChart` | Build a chart from a reference basis, coord atoms, and the Cartesian embedding `xᵃ=fᵃ(q)` | — |
| `.ctx` | `Context` | The owned context (pass as `ctx=` to any core factory) | — |

---

## `Expr` — building the algebra

Operators (all return `Expr`; a Python int/`Rational` is accepted wherever a
scalar is expected):

| Op | Meaning | Example |
|---|---|---|
| `a + b`, `a - b`, `-a` | sum / difference / negate | `u + v` |
| `a * b` | tensor product ⊗ (scalar on either side ⇒ scaling) | `2 * u`, `u * v` |
| `a @ b` | dot · | `u @ v`, `nabla @ v` |
| `a % b` | cross × | `u % v` |
| `a / b` | scalar division | `x / r` |
| `a ** n` | power (scalar fields) | `r ** 2` |
| `a // b` | alternate double contraction ·· | `A // B` |

Methods:

| Method | Returns | Does | Notes |
|---|---|---|---|
| `.ddot(other)` | `Expr` | Double contraction `:` (`(a⊗b):(c⊗d)=(a·c)(b·d)`) | `:` isn't a Python op, so method-only |
| `.ddot_alt(other)` | `Expr` | Alternate `··` (`=(a·d)(b·c)`) | also the `//` operator |
| `.tr()` | `Expr` | Trace `tr(A)` (scalar; `tr(a⊗b)=a·b`) | |
| `.vec()` | `Expr` | Vector invariant `vec(A)` (`vec(a⊗b)=a×b`) | |
| `.transpose()` | `Expr` | Transpose `Aᵀ` (`(a⊗b)ᵀ=b⊗a`) | |
| `.latex(map=None)` | `str` | LaTeX math string (no `$`); pass an `IndexNameMap` to keep index names stable across renders | |
| `.rank` | `int` or `None` | Inferred invariant rank (⊗ adds, · removes 2, `:` 4, × 1). `None` if a leaf rank is undeclared or a contraction is ill-formed | property |

### Selecting and rewriting parts (vibe 54)

Address one occurrence by its **path** — a `list[int]` of child selectors from
the root (binary `left`=0/`right`=1, unary operand=0, …) — to apply a step to,
or extract, just that part instead of the whole tree.

| Method | Returns | Does | Notes |
|---|---|---|---|
| `.find(kind=None, name=None)` | `list[list[int]]` | Paths (pre-order) to matching tensor objects: `kind=` a well-known name (`"Identity"`, `"Delta"`, `"LeviCivita"`, `"Metric"`), `name=` a tensor's name; both narrow (AND) | `find()[k]` is the k-th occurrence |
| `.addends()` | `list[list[int]]` | Paths to the top-level terms (walking the `+`/`−` spine) | so `.at(e.addends()[k])` extracts a term |
| `.paths(which="all")` | `list[list[int]]` | Paths (pre-order) to nodes by policy: `"all"`, `"atoms"` (leaves), `"tensors"`, `"wellknown"` | feeds the labeled view below |
| `.at(path)` | `Expr` | The subexpression at `path` — **extract** a part as a first-class expression | `IndexError` if out of range |
| `.replace_at(path, sub)` | `Expr` | Copy with the part at `path` replaced by `sub` — **splice** back | off-path nodes are shared (cheap) |
| `.rewrite_at(path, fn)` | `Expr` | Apply `fn` (an `Expr→Expr` step) at `path` only — **selective application** | see `td.at` below |

`td.at(expr, path, step)` is the free-function form of `.rewrite_at` — it
retargets *any* step to one occurrence: e.g. expand only one `I` in `a × I × b`,
leaving `a`, `b` symbolic:

```python
p = expr.find(kind="Identity")[0]
out = td.at(expr, p, lambda s: tb.expand_in_basis(s, frame, tb.Variance.Covariant))
```

> **Don't** canonicalize between selecting a path and rewriting it — a path
> addresses one specific tree, and canonicalize reshapes it. The order is
> *canonicalize → `find`/address → `at`*.

**Reading a path off the expression** — `tender.render.labeled(expr,
which="all")` returns a display object (Jupyter table / terminal text): the
whole expression above a **path → part** legend, so you can see which path to
pass to `.at` / `td.at`. Each part is rendered on its own, so the legend is
exact even where the pretty renderer folds structure (a `Δ` hides the
`∇·(∇⊗X)` whose parts you still need to address).

```python
print(t.render.labeled(expr, which="wellknown"))
# \Delta \mathbf{X}
# [1, 1]  \mathbf{X}          ← e.at([1,1]) is the field X inside the Δ
```

Building blocks come from the top-level `tender` module (or the `Workspace`
methods above). These take an explicit `ctx=` when used directly:

| Factory | Returns | Does |
|---|---|---|
| `tensor(name, rank=None, ctx=, space=None)` | `Expr` | Named tensor object; `space=t.space_3d` makes it dimension-aware |
| `scalar(value, ctx=)` | `Expr` | Scalar literal (int / `Rational`) |
| `identity(ctx=, space=None)` | `Expr` | Identity tensor (defaults 3-D) |
| `field(name, rank, deps=None, symmetric=False, ctx=)` | `Expr` | Tensor field (see `Workspace.field`) |
| `coordinate(name, chart_id=0, slot=0, nonneg=False, ctx=)` | `Expr` | Coordinate variable (rank-0 scalar field) |
| `sin/cos/tan/exp/log/sqrt(x)` | `Expr` | Elementary scalar functions of a scalar field |
| `tr(A)` / `vec(A)` / `transpose(A)` | `Expr` | Free-function forms of the `Expr` methods |
| `delta(realm, space, level0, level1, idx0, idx1, ctx=)` | `Expr` | Kronecker delta (indexed) |
| `levi_civita(realm, space, levels, indices, ctx=)` | `Expr` | Levi-Civita symbol (indexed) |
| `explicit_sum(index, body, ctx=)` | `Expr` | Annotate `body` with an explicit Σ over `index` |
| `no_sum(index, body, ctx=)` | `Expr` | Suppress implicit summation over `index` |
| `alloc_index(ctx=)` | `CountableIndex` | Fresh dummy index id |
| `nabla(ctx=)` | `Expr` | The **chart-free ∇** as a core `Expr` (combine with `*`/`@`/`%`) |
| `laplacian(operand)` | `Expr` | Invariant `Δ(operand) = ∇·(∇⊗operand)`; any rank, renders `Δ operand` |
| `render_latex(expr, map=None)` | `str` | Free-function LaTeX render |
| `space_2d` / `space_3d` / `space_4d` | `IndexSpace` | Predefined index spaces (values {1,2}, {1,2,3}, {1,2,3,4}) |

> **Two ∇/Δ surfaces — don't confuse them.** `t.nabla(ctx)` / `t.laplacian(e)`
> build **core `Expr`** ∇-nodes that a chart later lowers (via `chart.evaluate`
> or the `expand_nabla` pipeline). `tender.operators.nabla` / `.laplacian` build
> a **deferred `DifferentialExpr`** you `.evaluate(chart)` (see the operators
> section). `t.laplacian` is always `∇·(∇⊗·)`; bare `nabla @ nabla` with no
> operand is **not** a Laplacian.

---

## `Derivation` and steps (`tender.derivation`, `td`)

```python
import tender.derivation as td
drv = td.Derivation(expr)
drv.step(td.canonicalize).step(td.unroll_sums).step(td.fold_arithmetic)
drv.current.latex()
```

### `Derivation` class

| Member | Returns | Does |
|---|---|---|
| `Derivation(initial, index_map=None)` | — | Start a history; pass an `IndexNameMap` to keep index names consistent across all renders |
| `.step(step_fn)` | `Derivation` | Apply `step_fn: Expr->Expr` to the current expr, append to history; chainable |
| `.history` | `list[Expr]` | Every expression, initial through last |
| `.current` / `.initial` | `Expr` | Latest / first expression |
| `.latex(k, index_map=None)` | `str` | Render history step `k` |

### Steps — `Expr -> Expr` callables

Any of these can be passed to `.step()` or called directly.

| Step | Does | Does **not** |
|---|---|---|
| `canonicalize` | Algebraic normal form: sort commutative operands, carry one rational coeff/term, combine like terms, α-normalise dummies, materialise implicit Einstein sums. Equal-under-T0 ⇒ structurally identical | **Distribute** products over sums; preserves a ∇-fence (`∇·(∇⊗X)` stays put) |
| `simplify` | `canonicalize` then `implicitize` — the clean, canonical, implicit-sum *finish* of a derivation | — |
| `implicitize` | Drop each `explicit_sum` binder whose index repeats in one product term (Einstein form) | Touch an index straddling a `+` |
| `unroll_sums(expr, *indices)` | Expand `ExplicitSum` into concrete `Sum` trees; with `*indices`, only those (raises if none found) | — |
| `fold_sums` | Detect concrete N-addend `Sum` cycles → fold into one `ExplicitSum` over a fresh index | — |
| `expand_products` | Distribute product nodes (⊗, ·, `:`, ··, ×) over `Sum`/`Difference` | — |
| `fold_arithmetic` | Constant-fold literals over sum/diff/⊗//÷/negate; normalise `X+(−Y)→X−Y` | — |
| `eval_delta_concrete` | `δ(a,b)` with concrete indices → 1 / 0 | Touch symbolic-index deltas |
| `eval_eps_concrete` | Levi-Civita with concrete indices → its permutation sign / 0 | Touch symbolic-index ε |
| `expand_eps` | Every rank-3 ε → its 6-term δ cofactor expansion | — |
| `contract_delta` | `Σ_m δᵐ_a·δᵐ_b → δ_{ab}` | — |
| `contract_identity` | `I·x→x`, `x·I→x` | — |
| `contract_eps_pair` | `Σ ε·ε` sharing summed indices → generalized δ (3D, exactly two rank-3 ε's) | Any other shape |
| `distribute_contraction` | `op(L, A⊗B)→op(L,A)⊗B` (·/× over the adjacent ⊗ leg), one pass | Cross a ∇-fence (barrier) |
| `expand_double_dot` | `(a⊗b):(c⊗d)→(a·c)(b·d)` etc.; distributes over sums/binders | Fire when a side isn't a dyad; cross a ∇-fence |
| `expand_dyad_ops` | `tr(a⊗b)→a·b`, `vec(a⊗b)→a×b`, `transpose(a⊗b)→b⊗a`; linear | Fire when operand isn't a dyad |
| `fold_equal_addends` | **Self-preparing**: canonicalize → group identical addends (`X+X→2X`, `X−X→0`, even across dummy renaming) → implicitize | — |
| `fold_equal_addends_structural` | Bare structural version — merges only addends written identically | Rename dummies / normalise order |
| `collect_terms` | Group addends by their tensor (dyad) part, sum an **arbitrary scalar** coeff | Handle numeric-only (that's `fold_equal_addends`) |
| `factor_common` | Factor a common rank-0 non-literal scalar out of a group: `λ(∇·u)+μ(∇·u)→(λ+μ)(∇·u)`; runs bottom-up | Pull a numeric or tensor factor |
| `simplify_scalars` | Targeted scalar simplifier to a fixed point: `cos²·C+sin²·C→C`, `x⁰→1`/`x¹→x`, `√(x²ᵏ)→xᵏ` (nonneg), `r(a+b)→ra+rb` | General CAS simplification |
| `partial(expr, coord)` | ∂expr/∂coord: linearity + Leibniz + quotient + chain rules; only the matching coord →1, all else constant; result canonical | Differentiate w.r.t. a non-coordinate |
| `deriv(coord)` | The unapplied ∂/∂coord operator (a `Deriv` node); build `deriv(x)*f` then `apply_operators` | Differentiate immediately (use `partial`) |
| `apply_operators` | Carry out the first-class ∂ operators by Leibniz, rightmost-first | — |
| `sym(A)` / `skew(A)` | `(A±Aᵀ)/2` builders (`A=sym+skew`) | Recognise the *result* as (anti)symmetric |
| `structural_eq(a, b)` | `bool` — deep structural tree equality | Algebraic equality |
| `algebraic_eq(a, b)` | `bool` — T0-canonical equality, falling back to `simplify_scalars(a−b)==0` (so `x/r+y/r == (x+y)/r`) | — |

### Rule-based rewriting

| API | Returns | Does |
|---|---|---|
| `Identity(name, lhs, rhs)` | — | A directed rule `lhs=rhs`; free indices of `lhs` are pattern vars. Callable as a step; matches deepest-first, result canonical |
| `apply_identity(identity)` | step fn | Wrap an `Identity` as an `Expr->Expr` step |
| `saturate(expr, rules, max_iterations=30)` | `Expr` | Equality-saturate under `Identity` rules in an e-graph; return the cheapest extraction. No manual step ordering; all share one `Context` |

---

## `Basis` and the coordinate bridge (`tender.basis`, `tb`)

A `Basis` is a vector frame; the basis-parameterized functions expand an
invariant into components and fold it back. Steps take `(expr, basis, …)` —
wrap in a `lambda` to drive a `Derivation`.

Constructors / well-known frames:

| Function | Returns | Does |
|---|---|---|
| `make_orthonormal_basis(...)` / `make_oblique_basis(...)` | `Basis` | Build a frame from vectors (symbols, handedness, value names) |
| `wcs(ctx)` | `Basis` | World Cartesian i, j, k (prefer `Workspace.wcs()` — memoised) |
| `cylindrical(ctx)` / `spherical(ctx)` / `polar_2d(ctx)` | `Basis` | Well-known physical frames |

`Basis` properties/methods: `.space`, `.basis_id`, `.is_orthonormal`,
`.vector_symbol`, `.basis(i)`, `.cobasis(i)`, `.direction(i)`,
`.covariant_vector(i)`, `.contravariant_vector(i)`.

Bridge steps:

| Function | Does | Does **not** |
|---|---|---|
| `expand_in_basis(expr, basis, variance)` | Expand invariants into `Σ` over components on `basis` | — |
| `simplify_basis_dot(expr, basis)` | Reduce `eᵢ·eⱼ` (concrete / frame-vector directions) | — |
| `simplify_basis_cross(expr, basis)` | Reduce `eᵢ×eⱼ` | — |
| `reassemble(expr, basis)` | Fold component sums back into invariants (incl. completeness) | — |
| `reassemble_completeness(expr, basis)` | Fold `Σ_k e_k⊗e_k → I` (resolution of identity) | — |
| `fold_resolution_of_identity(expr, basis)` | Collapse a completed `i⊗i+j⊗j+k⊗k → I` | — |
| `expand_identity(expr, basis)` | Expand `I` into `Σ_k e_k⊗e_k` on `basis` | — |

`Variance` = `Covariant` / `Contravariant`; `Handedness` = `Right` / `Left`.

---

## `CoordinateChart` (`tender.chart`)

```python
cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
cyl  = ws.chart(ws.wcs(), [r, th, z], [r*t.cos(th), r*t.sin(th), z])
```

Built from a reference `Basis`, the coordinate atoms `qⁱ`, and the Cartesian
embedding `xᵃ=fᵃ(q)`. Everything else is derived.

### Geometry (derived from the embedding)

| Method | Returns | Does |
|---|---|---|
| `.coords` | `list[Expr]` | The coordinate variables `qⁱ`, in order (property) |
| `radius_vector()` | `Expr` | Position `R=Σ fᵃ(q) uₐ` in the reference frame |
| `position()` | `Expr` | Position in this chart's **own** physical frame (`r e_r + z e_z`); `grad(position())→I` |
| `field(name, rank)` | `Expr` | A field depending on all this chart's coords |
| `tangent_vector(i)` | `Expr` | Holonomic `gᵢ=∂R/∂qⁱ` |
| `metric_component(i, j)` | `Expr` | `g_ij=gᵢ·gⱼ`, simplified |
| `scale_factor(i)` | `Expr` | `hᵢ=√(g_ii)` (positive root) |
| `physical_basis()` | `Basis` | Physical orthonormal frame `eᵢ=gᵢ/hᵢ` |
| `physical_frame()` | `Basis` | Same, **and** registers the connection so operators differentiate `∂ⱼeᵢ` via the Christoffel table; idempotent per chart |
| `basis_derivative(i, j)` | `Expr` | `∂_{qʲ}eᵢ` as a reference-frame vector |
| `connection_coefficients(i, j)` | `list[Expr]` | `γᵏ_{ij}` with `∂_{qʲ}eᵢ=Σ γᵏ_{ij} eₖ` |

### Differential operators

| Method | Returns | Does | Note |
|---|---|---|---|
| `grad(f, fold_identity=True)` | `Expr` | `∇f=Σ(1/hᵢ)eᵢ⊗∂ᵢf` (rank +1; `∇R=I`) | `fold_identity=False` keeps the raw `Σ eₖ⊗eₖ` |
| `div(v, fold_identity=True)` | `Expr` | `∇·v` (rank −1) | |
| `rot(v, fold_identity=True)` | `Expr` | `∇×v` (3D vector) | |
| `laplacian(f, fold_identity=True)` | `Expr` | `Δf=div(grad f)` | |
| `dot(u, v)` / `cross(u, v)` | `Expr` | Invariant `u·v` / `u×v` reduced in the reference frame (for custom operators like `v·∇`) | |
| `nabla()` | `Expr` | The first-class ∇ operator `Σ(1/hᵢ)eᵢ∂ᵢ` — composable/inspectable; apply with a product + `apply_operators` | |

### The abstract-∇ pipeline (vibe 78/84)

| Method | Returns | Does |
|---|---|---|
| `evaluate(e)` | `Expr` | **Lower a coordinate-free core-∇ expression onto this chart**: `Dot(∇,X)→div`, `⊗→grad`, `Cross(∇,X)→rot`, `∇·(∇⊗X)→Δ`, inner-first; sums/coeffs/transpose/`I` pass through. Reprojects a sibling chart's WCS coords (`cyl.evaluate(∇⊗cart.position())=I`). Returns an invariant in this chart's frame |
| `expand_nabla(e)` | `Expr` | Expand every `t.nabla()` into the free-index frame form `eᵢ∂ᵢ` and apply (constant unit-scale frames only) |
| `componentize_nabla(e)` | `Expr` | Lower a free-index expansion to concrete components (`eᵢ→e_d`, `∂ᵢ→∂_{qᵈ}`) |
| `reassemble_nabla(e)` | `Expr` | Fold a reduced free-index expression back into chart-free ∇ operators |

### Surfacing invariants into components

| Method | Returns | Does |
|---|---|---|
| `express(v)` | `Expr` | Re-express `v` in **this** chart's physical frame (general change of basis) |
| `to_reference(v)` | `Expr` | Re-express `v` in the **reference (WCS)** frame (`e_r→cosθ i+sinθ j`); folds a completed resolution of identity |
| `expand(v)` | `Expr` | Expand every abstract field into components `Σ T_ij eᵢeⱼ` and reduce; `∂_q T` expanded by Leibniz with the connection (moving-frame terms kept) |
| `components(v)` | `list` / `list[list]` | Physical components on this frame: rank-1 → `[c₀,c₁,…]` with `cᵢ=v·eᵢ`; rank-2 → matrix `m[i][j]=eᵢ·v·eⱼ` (field expanded first, symmetry folded) |

---

## Differential-operator DSL (`tender.operators`)

Write differential expressions **symbolically**, then `.evaluate(chart)`. This
lowers to the chart's `grad`/`div`/`rot`/`laplacian` and the `partial`
differentiator.

```python
from tender.operators import nabla, d, laplacian
expr = nabla @ (nabla * f)      # symbolic  ∇·∇f
expr.latex()                    # "\\nabla \\cdot \\nabla f"
lap = expr.evaluate(cart)       # == cart.laplacian(f)
```

| API | Builds | Evaluates to |
|---|---|---|
| `nabla * T` | gradient (⊗) | `chart.grad(T)` |
| `nabla @ T` | divergence (·) | `chart.div(T)` |
| `nabla % T` | rotor (×) | `chart.rot(T)` |
| `nabla @ (nabla * T)` | `∇·∇T` | `chart.laplacian(T)` (nests div over grad) |
| `laplacian(T)` | `Δ T` (citable atom) | `chart.laplacian(T)` |
| `d(q) * T` or `d(q)(T)` | `∂_q T` | `td.partial(T, q)` |
| `nabla.along(v) * T` | directional `(v·∇)T` | `chart.dot(v, chart.grad(T, fold_identity=False))` |
| `nabla.at(chart)` | the first-class ∇ **Expr** for `chart` | apply with a product + `td.apply_operators` |
| `evaluate(expr, chart)` | — | free-function form of `expr.evaluate(chart)` |

Every operator expression carries `.latex()` and stays inspectable **before**
evaluation — nothing computes until `.evaluate(chart)`.

---

## Rendering (`tender.render`)

| Function | Returns | Does |
|---|---|---|
| `display(expr, map=None)` | `IPython.display.Math` | Rich Jupyter render; pass an `IndexNameMap` to share index names across cells |
| `labeled(expr, which="all", map=None)` | `LabeledExpr` | Path-labeled view (Jupyter table / terminal text): the expression + a path→part legend, for reading off selection paths (see "Selecting and rewriting parts") |
| `to_latex_document(exprs, title=None)` | `str` | Standalone LaTeX doc from `(label, Expr)` pairs, one display equation each |
| `expr.latex(map=None)` / `render_latex(expr, map=None)` | `str` | Bare LaTeX math string |

---

*See `examples/` for end-to-end derivations (`navier_lame`,
`curvilinear_operators`, `strain_compatibility`, `cyl_equilibrium`, …). Design
rationale for any feature lives in `vibes/NNNNNN_*.md`.*
