# 000077 — first-class composable differential operators

A foundational detour out of the strain-compatibility work (vibe 000076 / gap
D).  Deriving `inc ε = ∇×(∇×ε)ᵀ` "as performed" kept running into the same wall:
tender has no honest notion of a *differential operator*.  ∇ was either a Python
`DifferentialExpr` shell or (the abandoned first attempt) a hard-wired
`Del{grad/div/rot}` node.  Both cut corners.  The user's requirement is broader
and non-negotiable: **div/grad/rot must be particular cases of an operator
algebra**, because we will need `∂_t`, the material derivative `∂_t + v·∇`,
products of derivatives `∂_x²`, the thin-rod `∇ = ∇_⊥ + λ⁻¹ ∂_s t`, commutators
`[X,Y]`, and (eventually) derivatives with respect to tensors (`T = dΠ/dε`).  So
we stop and build the operator layer first.

## The abandoned first attempt (Del), and why we pivoted

Vibe 000076's `Del{Grad/Div/Curl, operand}` node (committed, to be dropped) fused
∇ with a product into three primitives.  It is real, tested, and *too narrow*:
you cannot write a bare ∇, so you cannot build operators *from* ∇ — no `v·∇`, no
`∇_⊥`, no `[X,Y]`.  It also drifted from the agreed principle that differential
operators are first-class citizens.  Retired in favour of the algebra below.

## The model

### Direction = "differentiate with respect to a tensor object"

Coordinates *are* rank-0 `TensorObject`s (a `CoordinateRef` trait).  So "which
way it differentiates" generalizes to: **a derivative is taken with respect to a
tensor object**, identified structurally (a `CoordinateRef` today; any object
tomorrow).  Consequences:

- **Operator rank = rank of the wrt-object.**  `∂/∂(scalar)` is rank 0 (hence it
  can sit among the scalars); `∂/∂ε` is rank 2; `dΠ/dε` = rank-0 numerator over
  rank-2 denominator = rank-2 result — no special case.  In product-rank
  arithmetic a `∂/∂q` contributes 0 and leaves its operand's rank unchanged
  (`∂_q` of a rank-2 field is rank 2), so grad/div/rot rank falls out of the
  vector part.
- Time derivatives, general-scalar derivatives, tensor derivatives are all one
  thing.  We build only the rank-0 (coordinate) case now; the door stays open.

### The dependency layer (conservative now, DAG-ready)

`∂x/∂q` is meaningless without knowing how `x` depends on `q`.  Guiding
principle: **independence must be positively established; otherwise the result
stays formal** — collapse to 0 only with proof, never silently.  That makes
"leave it unevaluated" the default.

Two derivative notions, kept distinct:

- **Partial** `∂/∂q^i`: vary `q^i`, hold the *same chart's other coordinates*
  fixed.  `∂q^j/∂q^i = δ_ij` is then the defining axiom of a coordinate system,
  not a computation.  "Hold siblings fixed" is part of what the direction means.
- **Chain / total**: when the object depends on `q` through other variables
  (another chart, a defined function `q'(q)`), `∂_q x = (∂x/∂q'_k)(∂q'_k/∂q)`.

Natural home: a **dependency DAG** — coordinates independent within their chart
(no sibling edges), transition maps / definitions are edges; differentiation
traverses it (no path ⇒ 0, a path ⇒ chain, no information ⇒ formal).  This is the
elementary-derivative layer every `∂_q → atom` bottoms out in.  **Start
conservative** (independent within a chart via δ; a field's `FieldDeps`;
everything else formal — enough for the single-chart Cartesian strain
derivation) and grow the DAG when the first real cross-chart derivative demands
it.

### Operators are first-class; application is Leibniz is commutation

The Jacobi identity is the decisive argument: to write `[X,Y] = XY − YX` you need
`XY` and `YX` as distinct composite operators with *no operand yet* — the order
of composition is the whole content.  Marks on value-factors cannot represent
that.  So **operators live in "operator-land" as first-class, composable objects**
(closed under `+`, composition, and the `⊗/·/×` products), independent of any
operand.

Applying an operator is Leibniz, and Leibniz *is* the operator commutation rule:

    ∂ ▷ (x · E)  =  (∂ ▷ x)·E + x·(∂ ▷ E)          ( = [∂,x]·E + x·(∂▷E) )

so `∂·x = (∂x) + x·∂`, i.e. `[∂,x] = ∂x`.  "Apply a derivation to a factor" and
"commute a derivation rightward past a factor" are the *same rewrite*.  The
tensor product is **not** abandoned: application's *result* is written with
ordinary products; application is just "a product that Leibniz-distributes and
bottoms out in an elementary derivative."  Non-commutativity (`∂_x x ≠ x ∂_x`,
`∂_1 e_1 ≠ e_1 ∂_1`) is exactly operator-product order, resolved by this rewrite.
"Don't expand" = leave the operator·operand product formal; expansion = run
Leibniz.  Both are first-class resting states.

**grad / div / rot are one operation** — "apply ∇ to T" — differing only in the
product the vector part uses: `⊗`→grad, `·`→div, `×`→rot.  No hard-coded nodes.

### Unapplied flag + scope marks (the user's rule)

Each derivation carries an **unapplied flag** and a **scope** (a set of marks
naming the factors it acts on; *mark count* = its size).  Rule:

- **flag required when mark count = 0**, optional otherwise.

Semantics:

- **Unapplied** (flag set, no marks): acts **greedily on everything to its right**
  in the enclosing product.  This is the freshly-built / composable form, and
  it is order-sensitive (canon must not reorder across it).
- **Applied / marked** (marks present): acts on **exactly** its marked factors,
  order-independent (canon may reorder freely — the action is in the marks, not
  the position).  This is where the earlier "store the operator among the
  scalars, mark its factors, detach the basis vectors" idea lives.

Worked example (`(d_x x)·f(x)`):

- `d_x` **unapplied**: `d_x` grabs everything right → `∂_x(x·f) = (1 + x∂_x)f
  = f + x ∂_x f`.
- `d_x` **marked to {x}** only: `(∂_x x)·f = 1·f = f`.

The two forms are the same operator at different lifecycle stages; the Leibniz
rewrite converts held-unapplied → marked atoms.

### Non-commutative, operator-aware canon

Because operators are carried explicitly, canon has a clean signal: it must (a)
**not reorder** an operator past a value or another operator, and (b) **may** fire
the Leibniz/commutation rewrite on demand.  The cost I flagged two rounds ago
("canon must know about ∇") is now principled rather than a hack — it keys off
operator-ness, not off recognizing `∇`.

## Representation decisions

1. **`Deriv` operator node** (new leaf): `{ wrt: <object-ref>, unapplied: bool,
   scope: [LinkId] }`.  Rank = rank(wrt) (0 for a coordinate).  Its very node
   kind is the "this is an operator" signal canon keys off.  `wrt` is a
   structural reference (a `CoordinateRef` today), consulted by the dependency
   layer.
2. **Marks are `LinkId`s** — canon-α-renamable ids like `CountableIndex`, shared
   between a `Deriv` and the factors in its scope (so they survive canon and
   hash-consing exactly as summation indices do; a globally-unique object id or a
   raw pointer would break hash-cons dedup).  A `Deriv` also carries a **second,
   independent** relation — its *direction* (the `wrt` reference) — distinct from
   its *scope* (the LinkIds).
3. **Marks attach to atoms only.**  A key simplification: composite
   differentiation (`∂` over a whole product/contraction) is represented in its
   **held, unapplied** form (order-based); only when Leibniz pushes it down to
   `TensorObject` atoms do marks land.  So the per-factor mark set lives *only on
   `TensorObject`* (where `FieldDerivDir` lives today), not on every node.
4. **`FieldDerivDir` is the degenerate case** — "operator already applied, to one
   field, concrete direction, cannot be unapplied/free-indexed/composed."  It is
   subsumed and retired; its slot on `TensorObject` becomes the scope-LinkId set.

## Implementation plan (incremental — buildable/testable at every step)

**Step 0 — remove Del.**  `git reset --hard HEAD~1` drops the vibe-000076 commit
in one shot (cleanest way to unthread `Del` from ~15 files); this vibe recaps its
still-valid context so nothing is lost.

**A — the `Deriv` node & operator-land basics.**  Node + factory `deriv(wrt)`
(unapplied); rank rule; structural identity (wrt + flag + scope); render
(`∂_{q}`, marks/flag visible); thread through `rewrite_tree`/`map_children`/the
exhaustive visitors; nf lowering as an order-fixed operator factor; canon keeps
an unapplied `Deriv` in place (no reordering, no Leibniz yet).  *Alive:* build,
render, compose, and canon operators — no differentiation.

**B — application = Leibniz = commutation.**  The `▷` rewrite: an unapplied
`Deriv` over its rightward product → sum of marked-atom terms; elementary
derivative at atoms via the conservative dependency check (within-chart δ; field
`FieldDeps`; else formal).  Marks on `TensorObject` atoms.  Tests: `∂_x x = 1`,
`∂_x f` formal, `∂_x(fg)` Leibniz, the `(d_x x)f` example.  *Alive:*
differentiation works, formal by default, expansion on request.

**C — vector operators; ∇ = e_i ∂_i; grad/div/rot as apply+product.**  Assemble
∇ from the chart's `e_i` and `∂_{q^i}`; apply with `⊗`/`·`/`×`.  Parity tests
against the current M6 `chart.grad/div/rot` results.  *Alive:* the operator
reproduces the curvilinear results.

**D — migrate & retire `FieldDerivDir`.**  Re-express `partial()` and
`chart.grad/div/rot/laplacian` on the operator algebra; delete `FieldDerivDir`;
keep every vibe 000069–075 curvilinear test green.  *Alive:* one derivative
representation.

**E — Python surface.**  First-class `nabla`, `d(q)`, composition, apply/expand;
`DifferentialExpr` becomes a thin façade over `Deriv` (keep `.evaluate(chart)`
working).

**Then resume strain compatibility (gap D)** on this foundation: ∇-only WCS
expansion with ε kept abstract (native now — "don't expand" is a resting state),
apply `a×B×c` once to `e_i × ε × e_j`, reassemble into named operators.

## Deferred / open

- Tensor-valued `wrt` (`dΠ/dε`) — model accommodates it (operator rank =
  wrt-rank); not built.
- The full dependency DAG (transition maps, chains) — start conservative.
- Chart coordinates addressable *by index* (one indexed object) vs *n* named
  scalars — the free-index `∇ = e_i ∂_{q^i}` wants the former; revisit in C.
- Partial-application edge case (flag set *and* marks present — differentiates
  marks and still acts rightward): allowed by the rule, detail when needed.
