# 000070 — Usability of the differential functionality

The vibe-000069 layer (scalar fields, ∂_q, coordinate charts, curvilinear
geometry, ∇/grad/div/rot/Laplacian) works, but driving it from Python is still
clunky. This vibe collects concrete usability problems found by exercising the
API on realistic scripts, then proposes fixes. Rethinking some primitives is
on the table.

## Method

A fixed preamble plus interchangeable `# STEPS` blocks. We run each variant,
record what is awkward, surprising, verbose, or broken, and grow the problem
list below.

### Preamble (the agreed template)

```python
import tender as t
import tender.basis as tb
import tender.chart as tc
import tender.derivation as td
from IPython.display import Math, display

ctx = t.Context()


def disp(*exprs):
    for e in exprs:
        display(Math(e.latex()))


I = t.identity(ctx)

WCS = tb.wcs(ctx)
x = t.coordinate("x", chart_id=1, slot=0, ctx=ctx)
y = t.coordinate("y", chart_id=1, slot=1, ctx=ctx)
z = t.coordinate("z", chart_id=1, slot=2, ctx=ctx)
cart = tc.CoordinateChart(
    WCS,
    [x, y, z],
    [x, y, z],
)

R = cart.radius_vector()
disp(R)

# STEPS
```

Status: preamble runs, `R = x i + y j + z k`.

## Problems

(collected per STEPS variant below)

### From the preamble itself

- **P1 — coordinate declaration is boilerplate.** Each coordinate needs an
  explicit `chart_id`, `slot`, and `ctx`, repeated by hand and kept in sync with
  its position in the chart's `coords` list. The chart already knows the slots;
  the user should not restate them. A `chart.coords` / factory that mints the
  coordinate atoms (and returns them) would remove the duplication and the
  off-by-one risk.
- **P2 — `ctx` threaded through every call.** Almost every constructor takes
  `ctx=ctx`. Candidate for a context-bound facade so the context is implicit.

### STEPS variant 1 — `∇R` should collapse to `I`

```python
grad_R = cart.gradient(R)            # i⊗i + j⊗j + k⊗k
grad_R = tb.reassemble_completeness(grad_R, WCS)
grad_R = tb.reassemble(grad_R, WCS)
grad_R = td.fold_sums(grad_R)
disp(grad_R)                          # still i⊗i + j⊗j + k⊗k — no fold
```

- **P3 — the resolution of identity does not fold in its concrete, expanded
  form.** `∇R = Σ_k u_k⊗u_k = I`, but none of the reassembly passes collapse
  `i⊗i + j⊗j + k⊗k` to `I`. Root cause: `fold_completeness`
  (`src/basis.cpp:729`) only recognises the **symbolic bound** form
  `Σ_i e_i⊗e_i` (an `ExplicitSum` binder over a countable index); it bails
  immediately when there is no binder (`basis.cpp:760–761`). The differential
  operators emit the **fully expanded concrete** form — separate addends over
  the concrete reference vectors — so every reassembly pass is a no-op. This is
  the single biggest usability gap found so far: the headline result `∇R = I`
  cannot actually be obtained from the public API.

  **Key generalising fact (verified, cylindrical chart):** every operator
  returns its result in the chart's **constant reference frame** (the
  orthonormal Cartesian WCS `i, j, k`) — *for any CS*, not just Cartesian.
  `cyl.gradient(cyl.radius_vector())` is also `i⊗i + j⊗j + k⊗k`. So the
  resolution of identity to collapse is always `Σ_k u_k⊗u_k` over the
  **reference** basis's concrete vectors (the user's instinct to reassemble
  against `WCS` was right). What is missing is a fold that recognises this
  concrete expanded shape; the basis it must be given is the chart's reference
  basis, not the physical one.

- **P4 — `reassemble` / `reassemble_completeness` are the wrong public surface
  for this.** The user reasonably reached for both plus `fold_sums` and chained
  them; the correct answer turned out to be "none of these work, a different
  fold is needed." A user cannot be expected to know that. The operators
  themselves should collapse the resolution of identity in their own output
  (they already hold the reference basis), so `cart.gradient(R)` returns `I`
  with zero post-processing.

- **P5 (minor) — coordinate names restricted to one letter / LaTeX command.**
  `t.coordinate("phi", ...)` raises `ValueError: TensorName must be a single
  ASCII letter or a LaTeX command`; `"\\phi"` is required. Surprising for a
  plain variable name; at least the error could suggest the fix.

### STEPS variant 2 — `rot(R × I)` (`%` is cross) should give `−2I`, but errors

```python
cart.rot(R % I)
# ValueError: encapsulate: unsupported factor node
#   (a nested ⊗ inside an operand awaits fence distribution)
```

Correct result: `R × I` is the rank-2 skew tensor with `(R×I)·a = R×a`, and
`rot(R×I) = ∇×(R×I) = ε_{ikl}ε_{ilm} e_m e_k = (δ_{kl}δ_{lm} − 3δ_{km}) e_m e_k
= I − 3I = −2I`.

- **P6 — the cross product does not reduce against the identity tensor `I`
  (nor against rank-2 operands generally).** Verified minimal facts:
  - `i × (i⊗i)` *does* fence-distribute → `(i×i)⊗i` (works when `I` is spelled
    out as an explicit `⊗`).
  - `i × I` is left **atomic** — the cross never expands `I = Σ_k e_k⊗e_k`, so
    there is nothing to distribute or reduce.
  - the actual `rot` leg `e_i × ∂_i(R×I) = i × (i × I)` is returned completely
    **unreduced** by both `canonicalize` and `expand_products`.
  - inside the operator's compound output the same inability surfaces as the
    hard `encapsulate` error above: Nf lowering refuses a `Cross` whose operand
    is a `⊗` (here `I`) because it expects prior fence distribution, and there
    is no rule to fence-distribute over the atomic `I`.

  Net: the operators cannot accept a rank-2 (or higher) field that involves `I`
  or a nested cross. `reduce_cross` (`src/chart.cpp`, `split_frame`/`eps3`)
  only knows `e_a × e_b` (vector × vector via the frame table); it has no
  vector × rank-2 case and does not fence-distribute the operand into legs.
  The failure is also **ungraceful** — a crash, not a no-op-returning-the-input
  as the reassembly passes do.

  This pairs naturally with P3: to reduce, expand `I → Σ_k e_k⊗e_k`
  (completeness, the *forward* direction); after `reduce_cross`/`reduce_dot`
  collapse the legs, fold the resulting `Σ_k e_k⊗e_k` back to `I` (P3, the
  *reverse* direction). One completeness primitive serves both.

### STEPS variant 3 — opaque tensors are not fields (`div T = 0`)

```python
T = t.tensor("T", 2, ctx)
div_T = cart.divergence(T)        # 0  — WRONG (should be a symbolic vector)
```

- **P7 — there is no concept of a tensor field; differential operators see
  every opaque tensor as a constant.** The differentiator
  (`src/derivation.cpp:2997-3004`) returns `∂_q X = 1` only when `X` is the
  *matching coordinate atom* and `0` for everything else — opaque tensors,
  reference vectors, parameters. So `∂_q T = 0` and every operator gives `0` on
  a generic tensor. The whole differential layer currently only works on
  *concrete* fields built explicitly from coordinate atoms (e.g. `r e_r`); you
  cannot write `div T`, `∇·σ`, `∇v`, etc. for an abstract field. This is a
  missing fundamental, not a polish item.

  Requirements the user stated:
  1. Declare a **tensor field of any rank** (rank 0..n), i.e. an opaque tensor
     whose value varies in space.
  2. Specify which **coordinates it depends on** (e.g. `f = f(r)` only).
  3. Allow dependence on coordinates from **different charts at once** (e.g. a
     field that depends on spherical `r` *and* Cartesian `x`).

## Field design plan (for P7)

### Model

A **field** is a `TensorObject` (any rank) carrying a new trait beside the
existing `coordinate` trait: a **dependency descriptor** — the set of
coordinate atoms it depends on, each identified by `(chart_id, slot, name)`
(the same identity `is_same_coord` already uses). A scalar field is just a
rank-0 field (subsumes M1 nicely).

**Differentiation rule** (extends the `TensorObject` arm of `diff`):
`∂_{q} T` where `T` is a field with dependency set `D`:
- if `q ∈ D` → an **opaque derivative field** `∂_q T` (see below), *not* `0`;
- if `q ∉ D` and no chain-rule coupling is known → `0` (correct when `T`'s
  declared dependence is in the *same* chart you differentiate by);
- if `q ∉ D` but some `p ∈ D` depends on `q` through a known chart relation →
  chain rule `∂_q T = Σ_{p∈D} (∂_p T)·(∂p/∂q)` (Stage F3).

**The opaque derivative `∂_q T`** is a fresh `TensorObject`:
- rank = rank(T) (the derivative of a rank-r field along a scalar is rank-r);
- carries the field trait with the *same* dependency set `D` (so it can be
  differentiated again);
- stores a `FieldDeriv{base, multiindex}` trait = the base field id plus a
  multi-index of `(chart_id, slot)` derivative directions, kept **sorted** so
  mixed partials are automatically symmetric (`T_{,ij} = T_{,ji}`) and
  hash-cons to one node;
- renders as a comma-derivative `T_{,i}` / `T_{,ij}` (or `∂_i T`).

Because an opaque field carries no `e_j(q)` inside it, `∂_q T` produces **no
connection terms** — correct: the partial of an invariant w.r.t. a scalar is
itself an invariant. A field written *in the physical frame* (e.g.
`f(r) e_r⊗e_r`) still differentiates its `e_r(q)` automatically via the
existing M6 machinery, so opaque fields and frame-expressed fields compose.

### What the operators then yield (coordinate-free, symbolic)

- `∇f  = Σ_i (1/h_i) e_i  f_{,i}`            (gradient of a scalar field)
- `∇·T = Σ_i (1/h_i) e_i · T_{,i}`          (divergence of a tensor field)
- `∇v  = Σ_i (1/h_i) e_i ⊗ v_{,i}`          (gradient raises rank)
- `Δf  = ∇·∇f`, `rot v = Σ_i (1/h_i) e_i × v_{,i}`
These are the textbook direct-notation results and follow with **no new
operator logic** — only the differentiation rule changes.

### Staging

- **F1 — opaque fields + same-chart differentiation.** Field trait + factory
  (`t.field(name, rank, deps=[...], ctx=...)`, and a chart shorthand
  `cart.field(name, rank)` meaning "depends on all of this chart's coords");
  the `∂_q T` derivative symbol; the `q∈D / q∉D → 0` rule. Fully correct when a
  field's declared dependence lives in the same chart the operator uses. Covers
  `div T`, `∇f(r)`, `∇v`, `Δ`, `rot`.
- **F2 — derivative algebra & display.** Canonical sorted multi-index, mixed-
  partial symmetry, higher derivatives for the Laplacian, comma/∂ LaTeX.
- **F3 — cross-chart dependence (the hard, explicitly-scoped stage).** The
  user's "depends on spherical `r` *and* Cartesian `x`" case lands here:
  differentiating a field by a coordinate not in `D` but related to a `D`
  member through the charts. All charts embed into the common reference
  Cartesian space (`chart.embedding` gives `x^a = f^a(q)`, so `∂x^a/∂q^i` is
  free); the reverse `∂q^i/∂x^a` needs an **inverse Jacobian** (or user-supplied
  inverse relations). Plan: register chart relations through the shared
  reference coords, compute the needed Jacobian entries, and apply the chain
  rule above. Flagged as the expensive part; F1/F2 deliver most of the value
  first.

### Decisions settled

1. **Default dependency** of `t.field(...)` with no `deps`: **depends on all
   coordinates** (general position) — `∂` is never silently zero; narrow with
   `deps=[...]`.
2. **Notation**: the derivative renders as the **operator form `∂_i T`**.
3. **Scope**: P7 stays a recorded plan for now (F1/F2/F3 staging above);
   continue collecting other usability problems before implementing.

### Design topic — make `∇` and `∂_q` first-class citizens (P8)

Today `∇`/grad/div/rot/Δ are Python functions (`cart.gradient(T)`); `∂_q` only
exists transiently inside them. The user wants `∇` and `∂_q` to be **expression
nodes**, so operators compose and users can build their own (`e·∂_q`, the
directional derivative `v·∇`, the material derivative `∂_t + v·∇`, …).

Key observations (user):
- `∇` is an **invariant vector**; in a basis it expands to a concrete
  `Σ_i (1/h_i) e_i ∂_{q^i}` — basis vectors paired with coordinate-partials.
- `∂_q` is **not an ordinary scalar**: it acts on everything to its right up to
  the end of its **scope** (limited by parens). It is non-commuting and
  position-bearing.
- A custom operator like `e·∂_q` must know **which coordinate set (chart) `q`
  belongs to** — `q` carries `(chart_id, slot)`, so `∂_q` inherits the chart.

**New IR nodes**
- `DiffOp{coord}` — unapplied, scalar-rank differential operator *factor* (the
  thing the user writes and combines, e.g. `e @ d(q)`); carries the coordinate
  atom, hence its chart. **A new factor category that is rank-0 yet
  order-fixed** — canonicalize currently floats rank-0 scalars to the front as
  coefficients, which would destroy `∂_q`'s scope. This is the main IR change.
- `Diff{coord, operand}` — the *applied* form after scope binding; evaluation is
  just `steps::partial(operand, coord)` (the existing M2 engine).
- `Nabla` — unapplied invariant vector operator atom (rank 1), chart-free.

**Surface operators (Python), per user mapping** — with a nabla object `nabla`
(reusing `*`=⊗, `@`=·, `%`=×):
- `nabla * T` → `∇⊗T` (gradient, rank +1)
- `nabla @ T` → `∇·T` (divergence, rank −1)
- `nabla % T` → `∇×T` (rotor)
- `d(q)` builds `∂_q`; `v @ nabla` builds the scalar operator `v·∇`.

**Pipeline (kept explicit so symbolic `∇` identities stay provable)**
1. Build expressions freely with `Nabla` / `DiffOp`.
2. `expand_nabla(expr, chart)`: rewrite `BinOp(Nabla, X)` for
   `BinOp ∈ {⊗,·,×}` → `Σ_i (1/h_i) ( e_i BinOp DiffOp{q_i}·X )`. Needs the
   chart for `h_i`, `e_i`, `q_i`.
3. **Scope binding**: a `DiffOp` factor binds its scope = the product of all
   factors to its right within the enclosing sub-expression (parens =
   sub-tree boundary limit it) → `Diff{coord, scope}`. In a contraction `∇⊙T`
   the `∂_i` scopes the **entire right operand** `T` and the `e_i` contracts
   with the differentiated result — the classic `e_i ⊙ ∂_i(T)`.
4. **Evaluate**: `Diff{q, S} → partial(S, q)`, then canonicalize/simplify.

**Payoffs**
- M6 operators collapse to thin wrappers: `gradient = evaluate∘expand_nabla(∇*T)`,
  etc.; `Δf = ∇@(∇*f)` (Laplacian as a composite, no new atom).
- Custom operators become first-class: directional derivative `(v@nabla)*T`,
  convective term, material derivative, `∇∇`, etc.
- Builds directly on P7: `∂_q` applied to an opaque field yields P7's `∂_i T`.

**Decisions settled**
1. **Chart on `∇`: chart-free.** `∇` is a pure symbol (`t.nabla()`); the chart
   is supplied at `expand_nabla(expr, chart)`. Enables chart-free identity
   proving (`∇·(∇×v)=0` without choosing a CS). A convenience `chart.nabla()`
   may return a pre-bound one, but the canonical form is chart-free.
2. **Evaluation timing: explicit.** Building stays purely symbolic; nothing
   fires implicitly. `expand_nabla` and `evaluate` are explicit steps, so the
   symbolic `∇` form is inspectable and identities are provable before
   expansion.
3. **Laplacian: first-class atom *with* a defining expansion.** `Δ` is its own
   expression node (so it renders as `Δ` and can be pattern-matched directly),
   **but** it is not an independent code path — it carries the definition
   `Δ ≡ ∇·∇`, which `expand_nabla` rewrites before evaluation. So `Δ` is both a
   citable atom and derivable from `nabla @ nabla`; the two must agree by
   construction (a single expand rule, no duplicated logic). Same treatment is
   the template for any future named composite operator.

## Proposals

- **Concrete-completeness fold.** Add a fold keyed on an orthonormal `Basis b`:
  in a `Sum`, find addends of shape `c · u_k⊗u_k` (same concrete basis vector
  `k = b.basis(k)` on both adjacent legs, common scalar `c`); if all `n =
  b.dim` vectors are present with that `c`, replace those `n` addends with
  `c·I`. Add the partial-contraction analog `Σ_k (X·u_k) u_k → X` for the
  concrete case. Guard on `b.is_orthonormal()` (completeness only holds there).
- **Operators self-fold.** Have `del_apply` run the concrete-completeness fold
  against `chart.reference` before returning, so `∇R = I`, `div R = 3`, etc.,
  come out directly. Keep an opt-out if a user wants the raw expansion.
- **Fence-distribute cross/dot over `⊗`, expanding `I`.** Teach the
  cross/dot reduction `a ⊙ (b⊗c) = (a⊙b)⊗c` and, crucially, expand the atomic
  `I → Σ_k e_k⊗e_k` (over `chart.reference`, which is orthonormal) when it is an
  operand of a contraction. Then `i×I = Σ_k (i×e_k)e_k` reduces through the
  existing frame table, and the whole `rot(R×I)` collapses — via the same
  completeness fold — to `−2I`. At minimum, make the unreducible case a no-op
  instead of a crash (P6's ungraceful failure).

## Implementation plan

Effort is in relative points (rough story-points, not hours); the **P2 share**
is computed at the end. Phases are ordered by dependency, then by value.

### Phase 0 — shared completeness primitive (P3) · 8 pts

The keystone: one primitive used in **both** directions by P4 and P6.

- `src/basis.cpp`: add a **concrete-completeness fold** keyed on an orthonormal
  `Basis b`. In a `Sum`, detect addends of shape `c · u_k⊗u_k` (same concrete
  `u_k = b.basis(k)` on both legs, common scalar `c`); when all `n = b.dim`
  vectors are present with that `c`, replace the `n` addends with `c·I`. Add the
  partial-contraction analog `Σ_k (X·u_k) u_k → X`. Guard on
  `b.is_orthonormal()`.
- Add the **forward** direction `expand_identity(I, b) → Σ_k u_k⊗u_k` (concrete),
  used by Phase 2.
- Public entry: extend (or add beside) `fold_completeness` so the concrete shape
  is recognised, not only the symbolic bound `ExplicitSum` form
  (`basis.cpp:760–761` currently bails when there is no binder).
- Tests: `i⊗i+j⊗j+k⊗k → I`; `2(i⊗i+…) → 2I`; partial sum (missing `k`) stays
  put; non-orthonormal basis is left alone; partial-contraction `Σ_k(X·u_k)u_k`.

### Phase 1 — operators self-fold (P4) · 2 pts (needs Phase 0)

- `src/chart.cpp` `del_apply`: run the Phase-0 concrete fold against
  `chart.reference` before returning, with a `simplify=True` default and an
  opt-out for the raw expansion.
- Tests: `cart.gradient(R) == I` and `cyl.gradient(cyl.radius_vector()) == I`
  with zero post-processing; `div R == 3`.

### Phase 2 — cross/dot reduce against `I` and rank-2 (P6) · 8 pts (needs Phase 0)

- `src/chart.cpp` `reduce_cross`/`reduce_dot`: fence-distribute a contraction
  over a `⊗` operand, `a ⊙ (b⊗c) = (a⊙b)⊗c`, and **expand atomic `I`** via
  Phase-0's `expand_identity` when `I` is an operand. `i×I = Σ_k(i×u_k)u_k` then
  reduces through the existing `eps3` frame table; the result folds back with
  Phase 0.
- Make the previously-crashing Nf-lowering path (`encapsulate` of a `Cross` over
  a `⊗`) a **graceful no-op returning the input** when no rule applies (P6's
  ungraceful failure).
- Tests: `i×I` reduces; `rot(R % I) == -2*I`; rank-2 field × vector; unreducible
  case returns input rather than raising.

### Phase 3 — Python ergonomics (P1, P2, P5) · 7 pts (independent)

- **P1 · 2 pts** — `CoordinateChart` mints its coordinate atoms. Add
  `chart.coords` (and/or a factory) returning the atoms with `chart_id`/`slot`
  filled from their position; preamble drops the three hand-written
  `t.coordinate(...)` lines.
- **P2 · 4 pts** — context-bound facade so `ctx` is implicit. Wrap `Context`
  with a thin object exposing `identity()`, `tensor()`, `coordinate()`,
  `field()`, `CoordinateChart(...)`, etc., each forwarding with `ctx` bound.
  Keep the explicit `ctx=` form working (facade is additive).
- **P5 · 1 pt** — relax/clarify coordinate names: accept common multi-letter
  names (or at minimum, make the error message suggest the `\\phi` LaTeX form).

### Phase 4 — tensor fields (P7) · 32 pts

- **F1 · 13 pts (needs nothing above)** — opaque fields + same-chart diff.
  - `src/include/tender/expr.hpp`: dependency-descriptor trait beside the
    `coordinate` trait — set of `(chart_id, slot, name)`. Default = all coords
    (decision 1).
  - `src/derivation.cpp` `diff` TensorObject arm (`2997-3004`): if `q ∈ D` emit
    an **opaque derivative field** `∂_q T` (fresh `TensorObject`, rank = rank(T),
    same `D`, `FieldDeriv{base, multiindex}` trait); if `q ∉ D` → `0`.
  - Factories: `t.field(name, rank, deps=[...], ctx)` and `cart.field(name,
    rank)` (= depends on all the chart's coords).
  - Tests: `div T` is a symbolic vector (not 0); `∇f(r)`; `∇v`; `Δ`, `rot`.
- **F2 · 6 pts (needs F1)** — derivative algebra & display: canonical **sorted**
  multi-index ⇒ mixed-partial symmetry `T_{,ij}=T_{,ji}` and hash-cons; higher
  derivatives for the Laplacian; `∂_i T` LaTeX (decision 2). Tests: symmetry,
  second derivatives, render.
- **F3 · 13 pts (needs F1; expensive, DEFERRED)** — cross-chart dependence.
  *Status: not implemented* — F1/F2 deliver the value; F3 (a field on spherical
  `r` AND Cartesian `x`, differentiated by a coordinate related through the
  charts) needs the inverse-Jacobian chain-rule machinery and is left for later.
  Chain rule `∂_q T = Σ_{p∈D}(∂_p T)(∂p/∂q)`; charts embed into the shared
  reference Cartesian (`∂x^a/∂q^i` free), reverse `∂q^i/∂x^a` via inverse
  Jacobian or user-supplied inverse relations; register chart relations through
  the shared reference coords. Tests: a field on spherical `r` and Cartesian `x`
  differentiated by each.

### Phase 5 — first-class `∇`/`∂_q` (P8) · 21 pts (best after F1)

> **Implemented as a deferred operator layer, not core IR nodes.** Adding
> `Nabla`/`DiffOp`/`Diff`/`Laplacian` to the `Expr` variant would force a new arm
> across ~6 exhaustive `Expr` visitors (canonicalize, nf lowering, nf_match,
> nf_egraph, render, structural_eq/tensor_order) plus `rewrite_tree` — high risk
> of destabilising the whole library for a layer whose nodes are eliminated
> before canonicalize ever runs (explicit evaluation, decision 2).  Instead
> Phase 5 ships `tender.operators`: a small **Python** algebra (`Nabla`, `d(q)`,
> `laplacian`, `DifferentialExpr`) that builds a symbolic, inspectable tree and
> lowers to the existing chart operators on `.evaluate(chart)`.  All three
> decisions hold: chart-free `nabla`, explicit `.evaluate(chart)`, and `Δ` a
> citable atom whose evaluation routes through `div(grad)` so it equals
> `nabla @ (nabla * f)` by construction.  Custom operators work: the directional
> derivative `nabla.along(v) * T = v·∇T` evaluates via the new public
> `chart.dot`/`chart.cross` (frame reductions).  *Limitation vs the IR-node
> design:* `∇` identities cannot be proven chart-free (evaluation needs a chart);
> that is the one thing the full IR nodes would add, and remains a future option.

- **IR · main change** — `src/include/tender/expr.hpp`: add `DiffOp{coord}`
  (rank-0 **order-fixed** factor category — canonicalize must NOT float it to
  the front like a scalar coefficient), `Diff{coord, operand}`, `Nabla`
  (chart-free rank-1 atom, decision 1), and `Laplacian{operand}` (atom that
  renders as `Δ`, decision 3). Touches `canonicalize` factor-ordering.
- **Surface** (Python): chart-free `t.nabla()` with `nabla*T`→grad, `nabla@T`→
  div, `nabla%T`→rot; `d(q)`→`∂_q`; `v@nabla`→`v·∇`; `t.laplacian(T)`→`Δ`.
- **Pipeline (explicit, decision 2)**: `expand_nabla(expr, chart)` rewrites
  `Laplacian{X} → Nabla·(Nabla⊗X)` (single rule — `Δ` and `nabla@nabla` agree by
  construction), then `BinOp(Nabla,X) → Σ_i(1/h_i)(e_i ⊙ DiffOp{q_i}·X)`;
  **scope binding** `DiffOp` ⇒ `Diff{coord, scope}` (scope = factors to the
  right within the enclosing sub-tree; parens bound it); **evaluate**
  `Diff{q,S} → partial(S,q)` (reuses M2), then simplify. Nothing fires until
  `expand_nabla`/`evaluate` are called.
- M6 operators become thin wrappers over this; chart at `expand_nabla` with an
  optional `chart.nabla()` convenience (decision 1).
- Tests: `expand_nabla` matches M6 outputs; `(v@nabla)*T` directional
  derivative; `Δf` via composite; scope/paren handling.

### Effort summary

| Phase | Problems | Points |
|------|----------|-------:|
| 0 completeness primitive | P3 | 8 |
| 1 operators self-fold | P4 | 2 |
| 2 cross/dot vs `I` | P6 | 8 |
| 3 ergonomics | P1, **P2**, P5 | 7 (P1 2 / **P2 4** / P5 1) |
| 4 fields | P7 (F1 13 / F2 6 / F3 13) | 32 |
| 5 first-class ∇/∂_q | P8 | 21 |
| **Total** | | **78** |

**P2 share ≈ 4 / 78 ≈ 5%** of the whole vibe (≈ 6% if F3 is deferred:
4 / 65). P2 is one of the smallest items — a mechanical, low-risk Python facade —
dwarfed by fields (P7, ~41%) and first-class ∇/∂_q (P8, ~27%).

### Suggested order

Phase 0 → 1 → 2 (completeness chain, unblocks the headline `∇R=I` and
`rot(R×I)=−2I`); Phase 3 anytime (independent ergonomics); Phase 4 F1→F2 (the
missing fundamental) with F3 deferrable; Phase 5 after F1.
