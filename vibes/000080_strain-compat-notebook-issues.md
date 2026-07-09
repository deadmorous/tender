# 000080 — strain_compatibility.ipynb: notebook-play issues

**Status: IN PROGRESS — eight issues + a plan whose sprint endpoint is the
Navier–Lamé reduction (Increment 8).  DONE:** Increment 0 (Issue 7 crash fix,
commit e9d9fb8) and the `(Aᵀ)ᵀ→A` involution (Issue 8 b1 / Increment 7, commit
4b2b6e3).  **Remaining:** Increments 1–5, rest of 7, **6 (now a *correctness*
prerequisite for 8, not last)**, and **8 (Navier–Lamé, the endpoint)**.
**Deferred (needs special care):** vibe 000054 (selective application) and its
riders Issue 6 (equation→identity) + Issue 8(C) (symmetry-guarded identity).
**Key session lesson:** author operator derivations with the operand *abstract*
(strain-compat template) and expand the basis *last*; basis-expanding `u` first
is a correctness trap (μ-Laplacian vanishes).  **Notation: never `∇²` — use
`∇·∇` / Laplacian.** From notebook-driven derivations on the reassembled
operator forms: Issues 1–6 come from `examples/strain_compatibility.ipynb` (one
fixed helper cell + one changing experiment cell); **Issue 7** (a **hard crash**,
the priority item) and **Issue 8** (symmetric/antisymmetric-part constructors +
symmetry recognition + a symmetry-guarded-identity idea) come from a second
example — continuum **balance equations via linear displacement** (Hooke's law,
`∇·T`).  Mix of real bugs and API/usability friction.  The **Implementation
plan** at the bottom covers the Issue 7 crash-fix (Increment 0, lead), the five
reduction/display gaps (Increments 1–5), and Issue 8's constructors + symmetry
recognition (Increment 7).  **Deferred (conditional-identity capabilities):**
Issue 6 (turn the trace *equation* into an *identity*) and Issue 8(C) (a
*symmetry-guarded* identity), both built on the vibe-000054 selective-application
primitive — recorded with brainstorms but intentionally left out of the plan.

Fixed helper cell (unchanged across variants):

```python
from IPython.display import Math, display

def disp(x):
    display(Math(x.latex()))
```

Each issue below is one variant of the experiment cell.

---

## Issue 1 — `canonicalize` moves bare `∇` operators to the *right* of their operand (unrenderable)

**Cell:**

```python
disp(td.canonicalize(reass))
```

where `reass = cart.reassemble_nabla(phase1)` is the reassembled closed identity
(the same `reass` the example builds).

**Symptom.** `reass` itself renders correctly, ∇ on the left:

```
−∇·(∇·ε) I + ∇·∇ tr(ε) I − ∇·∇ ε − (∇∇ tr(ε))ᵀ + ∇(∇·ε) + (∇(∇·ε))ᵀ
```

but `td.canonicalize(reass)` flips ∇ to the **right** in three of the six terms:

```
−(∇·ε)·∇ I + ∇·∇ tr(ε) I − ∇·∇ ε + (∇·ε) ∇ + ∇(∇·ε) − tr(ε) ∇ ∇
```

- `−∇·(∇·ε) I`  → `−(∇·ε)·∇ I`   (the outer `Dot(∇, ∇·ε)` is commuted — `∇·X` sorted to `X·∇`)
- `−(∇∇ tr ε)ᵀ` → `−tr(ε) ∇ ∇`   (transpose-of-dyad materialized, legs swapped, both ∇ pushed right)
- `+(∇(∇·ε))ᵀ`  → `+(∇·ε) ∇`     (same transpose leg-swap)

**Diagnosis.** The `ε` deriv-marks are untouched — the *applied* content is
still correct — but the **bare `Nabla` nodes** that `reassemble_nabla` emits get
repositioned by two ordinary canon mechanisms:

1. **commutative `Dot`** — canon treats `Dot(∇, X)` as symmetric (`a·b = b·a`)
   and sorts ∇ after `X`, giving `X·∇`;
2. **transpose-of-dyad** — canon expands `(∇⊗X)ᵀ → X⊗∇`, moving ∇ to the right.

A ∇ (or any operator) on the right of its operand reads as acting on nothing —
mathematically misleading and not how we can render it.  Contrast vibe 000077,
which made canon **operator-position-aware for applied `Deriv`** (`∂_x x ≠ x ∂_x`);
that positional rule does **not** extend to a bare `Nabla` factor, so a
reassembled operator expression is not safe to `canonicalize` for display.

**Repro (headless).** Build `reass` as in `strain_compatibility.py`
(`cross_removal_identity` → `expand_nabla` → `apply_identity(id_inc)` →
`reassemble_nabla`), then compare `reass.latex()` vs
`td.canonicalize(reass).latex()`.

**Notes / open questions for triage.**
- The example never displays `canonicalize(reass)` — it only uses
  `algebraic_eq(reass, closed)` (which canonicalizes *both* sides, so the
  reorder is consistent and the equality still holds).  So this is a **display**
  hazard, not a correctness break in the existing example.
- Bug vs. usability: is the fix (a) make canon keep a bare `Nabla` positional
  like `Deriv`, (b) a render-time normalization that pulls operators left, or
  (c) simply document that reassembled operator forms are display-ready and must
  not be re-canonicalized?  Leaning (a)/(b) since a user will naturally reach for
  `canonicalize` before `disp`.

---

## Issue 2 — `expand_dyad_ops` resolves the trace only on dyad (`a⊗b`) terms, not on `c·I` or operator-applied fields

**Cell:**

```python
disp(reass.tr())
disp(td.expand_dyad_ops(reass.tr()))
```

**Symptom.** `reass.tr()` is the lazy `tr(Σ …)` (the whole six-term sum wrapped
in one `Trace` — expected, `tr` is a node).  `td.expand_dyad_ops(reass.tr())`
distributes `tr` over the sum but resolves it on only **three** of the six
terms; the user read it as "did not do much":

```
−tr(∇·(∇·ε) I) + tr(∇·∇ tr(ε) I) − tr(∇·∇ ε)      ← tr LEFT in place (3 terms)
  − tr(ε) ∇·∇  + ∇·(∇·ε)         + (∇·ε)·∇          ← tr resolved (3 terms)
```

**Diagnosis.** `expand_dyad_ops` (src/derivation.cpp:1462) reduces `tr` / `vec` /
`transpose` via `expand_unary`, which distributes over `Sum`/`Difference`/
`Negate` and then only fires when the operand **`split_dyad`s into an explicit
`scalar ⊗ leg0 ⊗ leg1` outer product** (`tr(a⊗b)=a·b`).  The three unresolved
terms are not dyads:

- `∇·(∇·ε) I` and `∇·∇ tr(ε) I` are **`scalar · I`** — the identity is a single
  rank-2 well-known tensor, not an `a⊗b` dyad, so there is no rule for
  `tr(c·I) = c·tr(I) = c·n` (n = dimension).  ⇒ terms 1, 2 unresolved.
- `∇·∇ ε` is a **differential-operator contraction on a single field** (`Δε`),
  not a dyad; `tr` should commute through the ∇ operators onto the field
  (`tr(Δε) = Δ tr ε`, `tr(∇⊗v)=∇·v`, …), which `expand_dyad_ops` does not do.
  ⇒ term 3 unresolved.

The dyad terms 4/5/6 (`∇⊗X` shapes and their transposes) *do* resolve because
`∇⊗X` splits as a dyad — but they then render with **∇ on the right** again
(`−tr(ε) ∇·∇`, `(∇·ε)·∇`), i.e. Issue 1 recurring in the reduced output.

**Missing capability.** Computing `tr(inc ε)` (a natural next step — the scalar
compatibility equation) needs, beyond the dyad rule: (i) `tr(c·I) = c·n`
(trace of a scalar multiple of the identity), and (ii) `tr` commuting through
differential operators to land on the field.  Without these the trace of a
closed operator identity can't be reduced to a scalar.

**Repro (headless).** Same `reass` as Issue 1; compare
`td.expand_dyad_ops(reass.tr()).latex()` — three `\operatorname{tr}(…)` wrappers
survive.

**Notes / triage.**
- The `c·I` case (i) is the cleaner, more clearly-missing rule — arguably a
  small gap in `expand_dyad_ops` (a well-known-tensor arm alongside the dyad
  arm).  The operator-commuting case (ii) is bigger (trace/operator interaction)
  and overlaps with the reassembly/operator machinery.
- Compounds with Issue 1 (∇-on-the-right in the resolved terms).
- See Issue 3 for the minimal isolation of case (i) — `tr(I)` — and the deeper
  δ_ii/unroll chain it exposes.

---

## Issue 3 — `tr(I)` does not reduce to the dimension (3); no `tr(well-known)` path, and `δ_ii` / `Σ_i δ_ii` stalls

**Cell:**

```python
td.expand_dyad_ops(t.identity().tr())     # user expects 3
```

**Symptom.** `tr(I)` stays symbolic through *every* natural surface:

```
t.identity().tr()                      →  tr(I)
td.expand_dyad_ops(t.identity().tr())  →  tr(I)          (unchanged)
td.canonicalize(t.identity().tr())     →  tr(I)          (unchanged)
```

The user expects `3` (trace of the identity in 3D).

**Diagnosis — three stacked gaps.**

1. **No `tr(well-known)` rule.** `expand_dyad_ops` only reduces `tr` on a
   `split_dyad` `a⊗b` (Issue 2); `I` is a single well-known tensor, so `tr(I)`
   is never even rewritten.  (`canonicalize` likewise leaves it.)

2. **The manual basis path stalls at `Σ_i δ_ii`.** Expanding first still doesn't
   get there:

   ```
   tb.expand_in_basis(I, wcs, Covariant)   →  e_i ⊗ e_i
   …tr, canonicalize                        →  Σ_i e_i · e_i
   tb.simplify_basis_dot                    →  Σ_i δ_ii
   td.contract_delta / canonicalize         →  Σ_i δ_ii     (stuck)
   ```

   `contract_delta` does **not** collapse a *self-contracted* delta `δ_ii`
   (= trace = dimension), and nothing evaluates `Σ_i δ_ii` to `n`.

3. **The sum never unrolls to a concrete dimension.** Contrast the working
   `examples/delta_trace.py`, which reduces `δ^i_i → 3` by
   **`unroll_sums`** over a *concrete 3D index space* (`space_3d`) then
   evaluating δ on concrete indices `δ^1_1+δ^2_2+δ^3_3 = 1+1+1`.  That δ carries
   a concrete-dimension space; the invariant `I` (and the `wcs` basis index in
   the path above) does not drive an unroll here, so `Σ_i` stays symbolic.

Also note `t.identity()` was called **with no `ctx`** and worked — the identity
is effectively dimension-agnostic, so strictly `tr(I) = n`, and getting the
literal `3` presupposes a 3D space is in scope (true in the strain example's WCS,
but not obviously attached to a bare `t.identity()`).

**Missing capability.** A direct `tr(I) → n` (and, in a concrete-dimension
context, `→ 3`) reduction — either a `tr(well-known)` arm in `expand_dyad_ops`,
or a `contract_delta`/scalar rule that evaluates `δ_ii` / `Σ_i δ_ii` to the
space's dimension.  This is the atom under Issue 2's unresolved `c·I` terms:
without `tr(I)=n` the whole `tr(inc ε)` scalar reduction is blocked.

**Repro (headless).** `td.expand_dyad_ops(t.identity(ctx).tr()).latex()` returns
`\operatorname{tr}(\mathbf{I})` unchanged.

**Notes / triage.** Strong candidate for a real, small fix: a `tr` (and the
`δ_ii`) → dimension rule.  Open question the fix must settle: where does the
dimension come from — the identity's/​index's space (`space_3d`), or must the
caller supply it?  A dimension-agnostic `I` may only reduce to a symbolic `n`.

---

## Issue 4 — no `∇·∇ → Δ` fold: reassembled output shows literal `∇·∇`, not the Laplacian

**Cell:**

```python
disp(reass)
```

**Observation / request.** `reass` renders two `∇·∇` occurrences —
`∇·∇ tr(ε) I` (= Δθ·I) and `∇·∇ ε` (= Δε).  The user asks: is there a way to
contract `∇·∇` into the Laplacian `Δ`?

**Answer today: no** (Expr level).

- `reassemble_nabla` emits a Laplacian as the literal `∇·(∇⊗X)` —
  `Dot(nabla, TensorProduct(nabla, cur))` (src/chart.cpp:1069–1071).
- A `Δ` symbol exists only in a **separate representation**: the high-level
  `tender.operators.laplacian` atom (`laplacian(f).latex() == "\\Delta f"`,
  which `evaluate`s to `chart.laplacian`).  It is a different object graph, not
  something a `∇·∇` `Expr` can fold into.
- There is **no `∇·∇ → Δ` fold step** — nothing in `td.*` (no `laplacian`, no
  `fold_laplacian`), no render-time recognition of `∇·(∇⊗X)`.

**The knowledge is already there.** `reassemble_nabla` *detects* the Laplacian
structure internally — it counts `e_ℓ·e_m` δ-pairs into a `laplacians` counter
(src/chart.cpp:1029–1071) and then re-emits each as `∇·(∇⊗·)`.  So the term is
*known* to be a Laplacian at reassembly time; it just has no `Δ` node to emit.

**Missing capability.** A first-class Laplacian in the `Expr`/nf model (a `Δ`
node, or a well-known operator), so that (a) `reassemble_nabla` can emit `Δ X`
directly where it already knows a δ-pair Laplacian, and/or (b) a fold step
`∇·(∇⊗X) → Δ X` collapses an existing `∇·∇`.  Bridges to the existing
`operators.laplacian` atom (which already renders `\Delta`).

**Notes / triage.** Feature gap, not a bug.  Relates to Issue 1 (a `Δ` node
would also sidestep the ∇-on-the-right reorder for those terms) and to the
operator/reassembly line (vibes 000077–000078).  Cleanest scope is probably an
optional Laplacian-folding mode on `reassemble_nabla` plus a standalone
`∇·(∇⊗·) → Δ` step for already-built expressions.

---

## Issue 5 — `∇∇(scalar)` (the Hessian) is not recognized as symmetric; a redundant `ᵀ` survives

**Cell:**

```python
disp(reass)
```

**Observation.** The term `−(∇∇ tr(ε))ᵀ` renders with a transpose.  But the
gradient-of-gradient of a **scalar** (here θ = tr ε) is the Hessian
`∂_i∂_j θ`, which is **always symmetric** (mixed partials commute), so the `ᵀ`
is redundant — it should read `−∇∇ tr(ε)`.

**Diagnosis.** tender does not know `∇∇(scalar)` is symmetric:

```
algebraic_eq(∇∇θ, (∇∇θ)ᵀ)   →  False
canonicalize(∇∇θ)            →  ∇∇θ
canonicalize((∇∇θ)ᵀ)         →  θ ∇ ∇      (transpose materialized as a leg-swap,
                                            ∇-on-the-right — Issue 1)
```

The two `∇` are **bare, unapplied `Nabla` operators**, so nothing carries the
`∂_i∂_j = ∂_j∂_i` symmetry — the existing symmetric-transpose fold (vibe 000078,
`encapsulate`) only fires for a *concrete* symmetric tensor (well-known `I`/`δ`/`g`
or a `symmetric=True` field), not for a symmetry that arises from operator
commutation.  So `(∇∇θ)ᵀ` is a genuine (unfoldable) leg-swap to tender.

`reassemble_nabla` even emits this `ᵀ` **knowingly**: its gradient-leg loop wraps
a right-side leg over a rank-≥1 `cur` in a transpose (src/chart.cpp:1061–1068),
with the comment *"∂'s commute, so ∇∇θ is symmetric regardless"* — the author was
aware the result is symmetric, but the transpose is still emitted and never
folded.  (The `strain_compatibility.py` `closed` identity likewise writes
`(nabla*(nabla*theta)).transpose()` with a "∇∇θ symmetric" comment, to *match*
reassemble's output so `algebraic_eq` holds.)

**Missing capability.** A symmetry rule for `∇⊗∇(scalar)`: `(∇∇θ)ᵀ = ∇∇θ`, so the
transpose folds away — either (a) `reassemble_nabla` skips the transpose when the
leg wraps a scalar-gradient operand (the Hessian case it already recognizes in
its comment), or (b) a general fold/`algebraic_eq` rule that a
gradient-of-gradient of a scalar is symmetric.  Note this differs from the
concrete-tensor symmetry fold — the symmetry is a property of the *operator
composition*, not of a stored tensor trait.

**Repro (headless).** `td.algebraic_eq(nabla*(nabla*t.tr(eps)),
(nabla*(nabla*t.tr(eps))).transpose())` returns `False`.

**Notes / triage.** Real gap (correctness-of-simplification + display).  Cleanest
narrow fix is (a) in `reassemble_nabla` (drop the transpose for the scalar-Hessian
leg it already identifies); the general rule (b) is more powerful but needs the
operator layer to track when a composition is symmetric.  Also compounds with
Issue 1 (the `ᵀ` that *does* survive re-canonicalizes to ∇-on-the-right).

---

## Issue 6 — no way to turn an *equation* into a rewrite *identity* (the final trace-closure step)

**Not a cell — the derivation's final step, and a design gap.** To *complete* the
strain-compatibility derivation one takes the **trace** of the (cross-free) inc ε
and uses it to close the form:

```
inc ε = 0                                    (compatibility)
  ⟹  tr(inc ε) = 0
  ⟺  ∇·(∇·ε) = ∇·∇(tr ε)     i.e.  ∇∇··ε = Δθ,  θ = tr ε   (the "trace identity")
```

then **applies that identity** back into the cross-free inc ε — substituting
`∇∇··ε → Δθ` — which cancels the two I-terms `−(∇∇··ε)I + Δθ·I → 0` and yields
the final compact result.

**Math (for the record).** `tr(inc ε) = (n−2)(Δθ − ∇∇··ε)` (n = dimension); in
3D the factor is 1, so `tr(inc ε)=0 ⟺ ∇·(∇·ε) = ∇·∇(tr ε)`.  (In 2D it vanishes
identically — the trace closure is a 3D statement.)

**What already works.** The *application* half is fine.  A hand-built identity
substitutes and collapses the terms:

```python
ddI  = nabla @ (nabla @ eps)     # ∇·(∇·ε)  = ∇∇··ε
lapI = nabla @ (nabla * theta)   # ∇·∇(tr ε) = Δθ
trace_id = td.Identity("trace_compat", ddI, lapI)
td.apply_identity(trace_id)(-(ddI)*I + lapI*I)   # ==> 0
```

So once we *have* the identity, `apply_identity` closes the derivation (verified).

**What's missing — turning an equation into an identity.** tender has:
`Identity(name, lhs, rhs)` (a directed rule, but you must **hand-assert** both
sides) and `Derivation` (a rewrite *chain* from an initial expr).  It has **no
`Equation` abstraction and no algebraic equation-manipulation** — no way to:

1. represent an equation `LHS = RHS` (or `expr = 0`) as a value;
2. **rearrange / isolate** — move additive terms across the `=` and solve for a
   chosen subterm, turning `Δθ − ∇∇··ε = 0` into `∇∇··ε = Δθ`;
3. emit the oriented result as an `Identity` for `apply_identity`.

The user: *"the tricky part is to have an equation and make the identity out of
it — we don't have enough to express it in tender."*  (And even *forming* the
equation `tr(inc ε)=0` is currently blocked by Issues 2/3 — computing `tr` of a
`c·I` term and of `∇·∇ ε` — and would read cleaner with a `Δ` node, Issue 4.)

**Dependencies.** Blocked upstream by **Issue 2/3** (compute `tr(inc ε)` at all)
and eased by **Issue 4** (`Δ` node for a clean RHS).  The *isolate→identity* step
here is the genuinely new capability.

**Brainstorm — possible solutions (to discuss).**

- **(A) A light `Equation` value + `isolate`/`solve_for`.** `td.Equation(lhs, rhs)`
  (and `Equation.from_zero(expr)` for `expr = 0`).  `eq.solve_for(subterm)` moves
  every other additive term to the far side and negates, giving `subterm = rest`
  — tractable for a **linear/additive** occurrence (exactly this case).  Then
  `eq.as_identity(name, direction=...)` picks the rewrite orientation and hands a
  normal `Identity` to `apply_identity`.  Non-linear "solving" is explicitly out
  of scope.
- **(B) `Identity.from_equation(eq, solve_for=subterm)`.** Skip a first-class
  Equation type; a single constructor takes `lhs, rhs` (or `expr==0`) plus the
  subterm to isolate, and builds the oriented `Identity` directly.  Smallest
  surface; covers the use case.
- **(C) A standalone `isolate(expr_eq_zero, subterm) -> (lhs, rhs)` primitive.**
  The core additive move-and-negate, reusable by (A)/(B); pairs the subterm
  against the negated remainder.  Requires only: `subterm` occurs additively and
  linearly (assert/za fail otherwise).
- **(D) Hypothesis/provenance on the identity.** `∇∇··ε = Δθ` holds **only under
  the compatibility hypothesis** `inc ε = 0` (it is *not* an unconditional
  algebraic identity — `tr(inc ε)` is generally nonzero).  A derived-from-equation
  identity should probably carry a label/provenance (`"under inc ε = 0"`) so its
  conditional nature is visible; soundness is the user's responsibility (as with
  any `apply_identity`), but the record should not masquerade as a theorem.
- **(E) Keep it manual, document the pattern.** Since the *application* already
  works, the minimal stop-gap is a documented recipe: compute the trace, read off
  the scalar equation by eye, and hand-write `td.Identity(lhs, rhs)`.  Loses the
  "derive it, don't assert it" spirit the rest of the example upholds.
- **(F) Build the isolate step on the *selective-application* primitive (vibe 000054).**
  Moving *one* LHS term to the RHS is a **targeted, positional** rewrite: pick a
  single additive term by its place in the tree and relocate it (with a sign
  flip) to the other side — leaving the rest untouched.  That is precisely what
  vibe 000054's **selective basis expansion** primitive was designed to enable —
  `rewrite_at(path, f)` + `find_occurrences(e, predicate)` (positional addressing
  forced by hash-consing: identical subterms are one pointer, so "this term, not
  that one" can only be a *route from the root*).  The feature generalizes beyond
  expansion — better named **selective *application*** — to "apply this operation
  at *this* occurrence," of which "move this addend across the `=`" is one case.
  So the `isolate`/`solve_for` of (A)/(B)/(C) should be **layered on the vibe-54
  `rewrite_at` primitive** rather than reinvented: `find_occurrences` locates the
  chosen additive term, `rewrite_at` removes it on the LHS and adds its negation
  on the RHS.  This ties Issue 6 to the still-unbuilt vibe 000054 work (status
  there: options captured, not committed) and argues for building the positional
  primitive first, since it is the shared enabler (targeted `apply_identity`,
  selective expansion, and now equation term-moving all want it).

**Leaning.** (B)/(C) as the smallest real capability (linear additive isolate →
oriented `Identity`), **built on vibe 000054's `rewrite_at` positional primitive
(F)**, with (D)'s provenance label so the compatibility hypothesis is recorded.
A full `Equation` algebra (A) is the more general home but more than this step
needs.  Whatever the shape, the produced `Identity` plugs straight into the
existing (verified) `apply_identity` path.  **This whole equation→identity step
(with its vibe-54 dependency) is deferred — it is *not* part of the
implementation plan below.**

---

## Issue 7 — `express(∇·T)` crashes: `encapsulate: unsupported factor node` on a ScalarDiv-wrapped transposed *basis-expanded* gradient (**hard bug**) — FIXED (Increment 0, commit e9d9fb8)

**Resolution.** The choking node was an **`ExplicitSum`**, not a `⊗`: after
`materialize` wrapped the implicit `u = u_i e_i` in an `ExplicitSum` binder,
materializing the transpose `(∇u)ᵀ → Σ_i(u_i e_i)⊗∇` stranded that binder as a
bare product factor inside `∇·(…)`, which `distribute_contraction` cannot split
(it distributes over `Sum`, not a binder).  Fixed by self-preparing in
`steps::canonicalize` *before* `materialize` — iterating
`expand_dyad_ops ∘ distribute_contraction` to a joint fixpoint while the
summation is still implicit (so the exposed sum stays a plain `⊗`/scalar the
distributor reduces).  `cart.express(∇·T)` now yields the Navier–Lamé expansion
(rank 1); 803 C++ + 245 Python green.  The result still carries Issues 1–5
(∇-on-the-right, unresolved `tr`), addressed by the later increments.


**Different example — balance equations via linear displacement.** Deriving the
continuum balance equation `∇·T + … = 0` for an isotropic (Hooke) material:

```python
u = ws.field(r"u", 1)
u = tb.expand_in_basis(u, cart.physical_basis(), tb.Variance.Contravariant)  # u_i e_i
eps = (nabla*u + (nabla*u).transpose())/2        # symmetric strain  (½(∇u+(∇u)ᵀ))
I   = t.identity()
lam = t.tensor(r"\lambda", 0); mu = t.tensor(r"\mu", 0)   # Lamé constants (not fields)
T   = lam*eps.tr()*I + 2*mu*eps                   # Hooke's law
cart.express(nabla@T)                             # ⟶ ValueError
```

```
ValueError: encapsulate: unsupported factor node
            (a nested ⊗ inside an operand awaits fence distribution)
```

Goal: `∇·T` expressed through the displacement `u` in *any* CS, so the balance
equations follow per coordinate system.

**Trigger — needs all three ingredients together** (each removed individually
works):

1. a **basis-expanded gradient** `∇u = ∇⊗(u_i e_i)` — a *multi-factor* nested `⊗`
   (∇, the component `u_i`, the frame vector `e_i`), **not** an atomic field;
2. **transposed inside a sum**: `∇u + (∇u)ᵀ`;
3. wrapped in a **`ScalarDiv`**: `(…)/2`; under an outer **`∇·`**.

Verified isolations (all `cart.express(…)`):

| expression | result |
|---|---|
| `∇·((∇u + (∇u)ᵀ)/2)`, **basis-expanded** u | **crash** |
| `∇·((∇u + (∇u)ᵀ)/2)`, **plain** (non-expanded) u | OK |
| `∇·((∇u)ᵀ/2)` (no sum) | OK |
| `∇·(∇u + (∇u)ᵀ)` (no `/2`) | OK |
| `∇·(∇u/2 + (∇u)ᵀ/2)` (`/2` distributed by hand) | OK |
| `∇·((A + Aᵀ)/2)`, A an atomic rank-2 field | OK |
| `∇·((a⊗b + (a⊗b)ᵀ)/2)`, plain dyad | OK |

**Diagnosis.** `(∇u)ᵀ` of a basis-expanded gradient is the transpose of a
*multi-factor* product `(∇ ⊗ u_i ⊗ e_i)ᵀ`.  The `ScalarDiv` fence (and the
enclosing `Sum`) **block the transpose-fence distribution** from reaching that
nested `⊗` before nf-lowering, so `encapsulate` (src/nf_lower.cpp) meets an
undistributed nested `⊗` inside a `Transpose` operand and throws.  `express` does
**not self-prepare** — it lowers without first canonicalizing / distributing the
`ScalarDiv`+`Transpose` fences.  Same family as vibe 000078 bug 3c (a transpose
fence hiding a nested/zero `⊗`), and a direct instance of the *"steps must
self-prepare"* principle (vibes 000060/000061): the step should materialize /
distribute internally rather than require the caller to prepare the input.

**Workarounds (today).**
- Distribute the `/2` by hand: `∇·(∇u/2 + (∇u)ᵀ/2)`.
- **Canonicalize the operand before `∇·`:** `cart.express(nabla @ td.canonicalize(T))`
  succeeds (rank 1) and yields the Navier–Lamé expansion — but the result still
  carries Issues 1 (∇-on-the-right: `u_i e_i ∇`, `∇·e_i ∇`) and unresolved `tr`,
  so a *clean* balance equation still needs Issues 1–5.
  (Note `cart.express(td.canonicalize(nabla@T))` — canon *after* wrapping — does
  **not** help; the `∇·` must sit outside the already-prepared operand.)

**Repro (headless).**
```python
u  = tb.expand_in_basis(ws.field("u",1), cart.physical_basis(), tb.Variance.Contravariant)
gu = nabla*u
cart.express(nabla @ ((gu + gu.transpose())/2))   # ValueError
```

**Notes / triage.** **Real, hard bug (a crash), not a display gap** — and it
blocks an entirely new workflow (elasticity balance equations / Navier–Lamé),
so it is the **highest-priority** item here and is **independent** of the
operator-display cluster (Issues 1–6).  Also a `.tr()` (trace) vs `.transpose()`
naming trap sits nearby: writing `(∇u).tr()` for the strain's symmetric part is a
natural mistake (`.tr()` is *trace*, rank 0), though it is orthogonal to this
crash (both spellings hit the same `encapsulate` error).

---

## Issue 8 — no `sym`/`skew` part constructors; a symmetric-*by-construction* tensor isn't recognized as symmetric; and a "symmetry-guarded identity" idea

**Context.** The strain is written `ε = (∇u + (∇u)ᵀ)/2` — symmetric *by
construction*.  Two asks:

1. **Convenience constructors** for the symmetric / antisymmetric part of a
   rank-2 tensor: `sym(A) = (A + Aᵀ)/2` and `skew(A) = (A − Aᵀ)/2`.
2. **Establish the symmetry of the result** — the point is not just sugar: the
   value should be *known* symmetric, so `sym(A)ᵀ = sym(A)` folds and downstream
   reductions can rely on it.

**Current state.**
- **No `sym`/`skew` methods** on the Python surface.
- **A symmetric-by-construction tensor is not recognized as symmetric.**
  `td.algebraic_eq((A+Aᵀ)/2, ((A+Aᵀ)/2)ᵀ)` is **False** (and even without the
  `/2`).  By contrast a **declared** `symmetric=True` field folds
  (`algebraic_eq(E, Eᵀ) = True`, via the vibe-000078 trait), and
  `algebraic_eq(½(E+Eᵀ), E) = True`.
- **Two supporting gaps that block recognition:**
  - **Double-transpose involution `(Aᵀ)ᵀ → A`** — was not folded for a generic
    field (`algebraic_eq((Aᵀ)ᵀ, A)` False; rendered as stacked superscripts
    `A^{\mathsf{T}}^{\mathsf{T}}`).  **FIXED (commit 4b2b6e3):** `encapsulate`'s
    Transpose arm now folds a transpose-of-transpose directly (the outer swap
    undoes the inner), collapsing chains too (`((Xᵀ)ᵀ)ᵀ → Xᵀ`).  Test
    `CanonicalizeNf.DoubleTransposeInvolutes`.
  - **Transpose is not distributed over a `Sum`/`ScalarDiv`**, so
    `((A+Aᵀ)/2)ᵀ` stays opaque instead of normalizing to `(Aᵀ + A)/2 = (A+Aᵀ)/2`
    (same fence-distribution family as **Issue 7**).

**The deeper idea — symmetry as a *guard* on identity application.** In a
derivation that involves `ε` *and assumes its symmetry*, one wants to treat the
definition `ε = (∇u + (∇u)ᵀ)/2` (i.e. `ε = sym(∇u)`) as an **identity applied to
the derivation result** — but firing **only where the constraint (the symmetry
of the matched subterm / the identity's LHS) holds**.  That is a **guarded /
conditional identity**: `apply_identity` with a *predicate* checked against the
match, not an unconditional rewrite.  It generalizes Issue 6's *provenance label*
(a passive note that an identity holds under a hypothesis) into an *active guard*
that is verified during matching.

**Brainstorm.**

- **(A) `sym`/`skew` as thin constructors.** `sym(A) → (A+Aᵀ)/2`,
  `skew(A) → (A−Aᵀ)/2` on the expr/chart surface.  Trivial on its own — the value
  is in (B).
- **(B) Establish symmetry — two routes (not exclusive):**
  - **(b1) Structural recognition.** Fix the small canon laws so a
    symmetric-part *normalizes* to a canonical symmetric form: fold
    `(Aᵀ)ᵀ → A` (transpose involution), distribute transpose over `Sum`/`ScalarDiv`,
    then sum-commutativity makes `((A+Aᵀ)/2)ᵀ ≡ (A+Aᵀ)/2` — no trait needed, it
    just folds (and `algebraic_eq` sees it).  Concrete and mostly small; the
    involution and transpose-over-sum are reusable wins (the latter overlaps
    Issue 7).
  - **(b2) A carried symmetry trait.** Have `sym(A)` produce a value tagged
    symmetric (extend the `symmetric=True` trait — today only on a *field* — to a
    compound expression, or a transparent `Symmetrized`/`Skew` wrapper node that
    canon/`encapsulate` read like the well-known symmetric fold).  Stronger
    (works even where structural normalization is hard, e.g. through operators —
    cf. Issue 5's operator-composition symmetry), but heavier (a trait/​node wired
    through the visitors).
- **(C) Guarded identities.** Add a *side-condition predicate* to `Identity` /
  `apply_identity` — the rewrite fires only where the match satisfies the guard
  (here: "the matched subterm is symmetric").  New capability; shares its home
  with Issue 6 (conditional/provenance identities).  Needs: a way to *ask* "is
  this subterm symmetric?" — which is exactly what (B) provides — so (C) depends
  on (B).

**Notes / triage.** (A)+(b1) are concrete and partly quick (the involution
`(Aᵀ)ᵀ→A` is a tiny canon rule; transpose-over-sum overlaps Increment 0/Issue 7).
(b2) and (C) are the deeper capabilities: a symmetry that is *carried* and a guard
that is *checked*.  (C) (symmetry-guarded application) is deferred **alongside
Issue 6** (both are conditional-identity capabilities); it should be designed
together with the equation→identity work and the vibe-000054 positional
primitive.  Relates to Issue 5 (operator-composition symmetry) and Issue 7
(transpose fence distribution).

---

# Implementation plan (Issues 1–5 + 8, the Issue 7 crash-fix, and the Navier–Lamé endpoint)

Scope: the Issue 7 crash-fix (done), the five reduction/display gaps, Issue 8's
constructors+recognition, **and — a sprint goal — reducing `∇·T` (Hooke stress)
to the clean Navier–Lamé displacement form** (Increment 8).  **Out of scope
(deferred, needs special care — user):** vibe 000054 (selective application) and
the two capabilities that ride on it — Issue 6 (equation→identity) and Issue
8(C) (symmetry-guarded identity).

**Notation (user, going forward): never write `∇²`.**  ∇ is not a
multiplicative-ring element, so a "power" is meaningless; the intended operation
is a dot.  Write `∇·∇` (or "the Laplacian"/`Δ`, once Increment 3 lands), never
`∇²`.

**Course correction — the Navier–Lamé endpoint reframes the plan (session
lessons).**  Driving `cart.express(∇·T)` with a **basis-expanded** displacement
`u = u_i e_i` produces a **wrong** result, not merely an ugly one: the components
`u_i` land to the *left* of the operators (`u_i ∇·∇ e_i`), so `∇·∇` differentiates
only the constant frame vector `e_i` and the μ-Laplacian term **vanishes in
Cartesian**.  Two findings reshape the plan:

- **`reassemble(u_i e_i) = u` already works** (basis-invariant reassembly folds
  the displacement) — the earlier "we can't recognize `u_i e_i = u`" worry was a
  false alarm.  The real problem is *when* the basis is expanded, not whether it
  can be folded.
- **Keep `u` abstract** (exactly as strain-compat kept ε abstract).  With an
  abstract `u`, `canonicalize(∇·T)` is *well-formed* — `μ (∇·∇ u + …) + λ …` with
  `u` correctly to the **right** of the operators.  The design template is the
  strain-compat pipeline (**abstract operand → expand ∇ → reduce → reassemble →
  expand per-CS at the very end**); the basis should be expanded *last*, never
  before the operator algebra.  This is a *usage/derivation-strategy* lesson, not
  a from-scratch rebuild — but it does mean the Navier example must be authored
  the abstract way, and a couple of new reductions are needed (below).

- **Operator positioning (Issue 1) is a *correctness* blocker, not just
  display.**  Even with abstract `u`, `∇` commutes past scalar factors to the
  wrong operand: the `λ tr(sym ∇u) · (∇·I)` term has **dropped its Leibniz
  `∇(∇·u)` piece** because `∇` skipped past the scalar `tr(…)` factor to act on
  `I` alone (and `∇·I = 0` for the constant identity).  So **Increment 6 is
  promoted from "display, do last" to a correctness prerequisite for the
  Navier–Lamé goal** — it must land before Increment 8 can produce the right
  answer.  (Design risk the user flagged: if a clean positional-operator model
  proves hard to retrofit onto canon, this is the increment where a small
  redesign, informed by these lessons, would live.)

**Ordering rationale (revised).** Increment 0 (Issue 7) is **done**.  Next the
additive reductions (1–5, 7) — each a new rule, low blast radius, independently
testable.  **Increment 6 (operator positioning) moves earlier** — it is now a
correctness prerequisite for Increment 8, not a cosmetic finish; still the
highest-risk edit, so Increments 3 (`Δ`) and 5 (symmetry) should precede it to
shrink its surface, but it must precede **Increment 8** (Navier–Lamé), which is
the endpoint that consumes everything.

## Increment 0 — `express`/`expand` self-prepares before nf-lowering (Issue 7, **priority lead**) — DONE (commit e9d9fb8)

Landed as a self-prep fixpoint (`expand_dyad_ops ∘ distribute_contraction`) at
the front of `steps::canonicalize`, *before* `materialize` (chosen over the
narrower encapsulate-hardening: it fixes a direct `canonicalize(∇·T)` too, and
both distributors are no-ops on ordinary inputs).  Full suite + examples green.
Original plan below.


**Goal.** `cart.express(∇·T)` (and the whole balance-equation pipeline) no longer
crashes on a `ScalarDiv`-wrapped transposed basis-expanded gradient; the
transpose fence over a multi-factor `⊗` is distributed before `encapsulate` runs.

**Approach.**
- Make the `express`/`expand` operator path **self-prepare** (per vibes
  000060/000061): before nf-lowering, distribute the `ScalarDiv` into its
  numerator sum and push the `Transpose` fence through the multi-factor `⊗`
  (`(a⊗b⊗c)ᵀ` distribution / `expand_dyad_ops`-style fence removal) — i.e. run
  the same normalization that canonicalizing the operand first already achieves
  (the verified workaround `express(∇· canonicalize(operand))`).
- Alternatively/additionally, harden `encapsulate` (src/nf_lower.cpp) so a
  `Transpose` operand carrying an undistributed nested `⊗` is distributed in
  place rather than throwing (mirrors the vibe 000078 bug-3c fence handling).

**Design question.** Is the right fix (a) a self-preparation pass at the front of
`express`/the operator lowering (broadest, fixes the class), or (b) a targeted
fence-distribution in `encapsulate` for the `ScalarDiv`/`Transpose` shape (narrow,
local)?  Prefer (a) if it does not perturb already-working paths; keep (b) as the
safety net.

**Files.** `src/chart.cpp` (`express`/`expand`), `src/nf_lower.cpp`
(`encapsulate` fence handling), possibly `src/derivation.cpp`.

**Verify.** The Issue-7 repro `cart.express(nabla@((gu+gu.T)/2))` (basis-expanded)
succeeds; the full Hooke `cart.express(nabla@T)` succeeds without a manual
`canonicalize`; the isolation table above all pass; regression-check that
plain-`u`, no-`/2`, and atomic-field variants still work; full suite green.
(Clean *rendering* of the result still depends on Increments 1–6.)

## Increment 1 — `tr(W) → dim` for a well-known symmetric tensor (Issue 3; atom of 2(i))

**Goal.** `tr(I) → n` (and `tr(δ)`, `tr(g)`), where n is the dimension; a
self-contracted `δ_ii` likewise collapses to n.

**Approach.**
- Add a reduction that recognizes `Trace(W)` for `W` a well-known symmetric
  tensor (`Identity`/`Delta`/`Metric`) and returns the space dimension.  Natural
  home: `expand_dyad_ops`'s `Trace` arm (a well-known branch beside the dyad
  branch) and/or `simplify_scalars`.
- Add the component-level companion: `δ_ii` (self-contracted Kronecker) and
  `Σ_i δ_ii` fold to the index space's cardinality (extend `contract_delta` /
  the δ-evaluation the `delta_trace` example already uses via `unroll_sums`).

**Design question (must settle first).** *Where does the dimension come from?*
The well-known tensor's index **space** (`space_3d` ⇒ 3) is the principled
source; a bare, dimension-agnostic `t.identity()` has none, so it can only yield
a **symbolic `n`** (a named scalar) — decide whether to (a) require a
space/dimension on the tensor to get a literal, (b) introduce a symbolic
dimension scalar `n`, or (c) both (symbolic by default, literal when the space is
concrete).

**Files.** `src/derivation.cpp` (`expand_dyad_ops`, maybe `simplify_scalars`),
possibly `src/nf_lower.cpp`/`contract_delta`; Python surface unchanged.

**Verify.** `tr(I)`, `tr(δ)`, `Σ_i δ_ii` reduce to the dimension (3 in a 3D
context, or symbolic n); `delta_trace` example still passes; unit tests for each.

## Increment 2 — `tr(scalar · W)` distributes; identity-scaled terms resolve (Issue 2(i))

**Goal.** `tr(c · I) → c · n` — the `(∇∇··ε)I` / `Δθ·I` terms of `reass` lose
their trace.

**Approach.** In `expand_dyad_ops`, extend the `Trace`/`vec` handling so that a
`scalar ⊗ W` operand (well-known W — *not* a `split_dyad` `a⊗b`) reduces via
`tr(c·W) = c·tr(W)`, reusing Increment 1 for `tr(W)`.  `expand_unary` already
distributes over `Sum`/`Difference`/`Negate`; this adds the well-known leaf case.

**Files.** `src/derivation.cpp`.

**Verify.** `td.expand_dyad_ops(reass.tr())` resolves the two I-terms;
`tr(c·I)` unit tests; the surviving unresolved term is only the operator-applied
one (Increment 4).

## Increment 3 — first-class Laplacian `Δ` + `∇·(∇⊗·) → Δ` fold (Issue 4)

**Goal.** A `Δ` representation the `Expr`/nf layer can carry, emit, and render as
`\Delta`; `reassemble_nabla` emits `Δ X` where it already detects a δ-pair
Laplacian; a standalone fold collapses an existing `∇·(∇⊗X)`.

**Approach (two options — pick in design).**
- **(a) A new `Laplacian`/`Δ` operator node** in the Expr model (and nf), wired
  through the visitors (rank = operand rank, renders `\Delta`), bridged to the
  existing `tender.operators.laplacian` atom.  Robust, but pays the
  "new-node-through-every-visitor" cost (as `Trace`/`Transpose` did).
- **(b) Pattern-recognition only** — no new node: a fold/renderer that recognizes
  `Dot(∇, TensorProduct(∇, X))` and prints/labels it `Δ X`, keeping the
  underlying `∇·∇` structure.  Cheaper, but the `Δ` is not a first-class operand
  (harder to match/consume downstream).

**Files.** `src/include/tender/expr.hpp`, `src/expr.cpp`, `src/render.cpp`,
`src/chart.cpp` (`reassemble_nabla`: emit `Δ` for the `laplacians` it counts),
`src/nf*.cpp` if a node; `python/_core.cpp` surface.

**Verify.** `reassemble_nabla` output shows `Δ tr(ε)` / `Δ ε` not `∇·∇ …`; a
`∇·(∇⊗X) → Δ X` step folds a hand-built expression; round-trips with
`chart.laplacian`; render test `\Delta`.

## Increment 4 — trace through differential operators (Issue 2(ii))

**Goal.** `tr(∇⊗v) = ∇·v`, `tr(Δε) = Δ(tr ε)`, `tr((∇⊗w)ᵀ) = ∇·w` — the
`tr(∇·∇ ε)` term resolves to `Δ(tr ε)`.

**Approach.** A trace/operator commutation pass: `tr` of an operator-applied
field pushes the trace onto the innermost field and rewrites the outer operator
pair to its scalar contraction (`∇⊗· → ∇·`, and with Increment 3, `∇·∇ → Δ`).
Builds on the reassembly/operator machinery (vibes 000077–000078); likely lives
beside `expand_dyad_ops` or as a dedicated `trace_through_operators` step.

**Files.** `src/derivation.cpp` and/or `src/chart.cpp`.

**Verify.** `tr(∇·∇ ε) → Δ(tr ε)`; all six terms of `expand_dyad_ops(reass.tr())`
resolve to trace-free scalars; matches the hand-computed
`tr(inc ε) = (n−2)(Δθ − ∇∇··ε)` component-wise (Cartesian + cylindrical).

## Increment 5 — Hessian symmetry: fold `(∇∇θ)ᵀ → ∇∇θ` (Issue 5)

**Goal.** The redundant transpose on a gradient-of-gradient of a scalar folds
away; `algebraic_eq(∇∇θ, (∇∇θ)ᵀ)` becomes True.

**Approach (two levels).**
- **Narrow:** in `reassemble_nabla` (src/chart.cpp:1061–1068), drop the transpose
  for the right-side gradient leg when `cur` is a scalar-gradient (the Hessian the
  comment already flags), so the `ᵀ` is never emitted.
- **General:** a symmetry rule that `∇⊗∇(scalar)` is symmetric, so
  `transpose(∇∇θ)` folds and `algebraic_eq` sees them equal — distinct from the
  concrete-tensor symmetric-transpose fold (vibe 000078), since the symmetry is a
  property of the *operator composition*, not a stored trait.

**Files.** `src/chart.cpp` (narrow); `src/derivation.cpp`/canon (general).

**Verify.** `reass` shows `−∇∇ tr(ε)` (no `ᵀ`); `algebraic_eq(∇∇θ, (∇∇θ)ᵀ)` True;
`strain_compatibility` example's `closed` term can drop its `.transpose()`.

## Increment 6 — operator-positional canon for bare `∇` (Issue 1) — **correctness prerequisite for Increment 8**

**Goal.** `canonicalize` keeps a bare `∇` (Nabla) **left of its operand and left
of any scalar factors it must act on**, so operator expressions stay both
renderable *and semantically correct* after canon.

**Approach.** Extend vibe 000077's operator-positional treatment (which pins an
*applied* `Deriv` in a product) to a **bare `Nabla`** in the commuting contexts
that currently move it right: (i) a `Dot(∇, X)` must **not** be treated as the
symmetric `a·b = b·a` (an operator contraction is directed); (ii) the
transpose-of-dyad materialization `(∇⊗X)ᵀ → X⊗∇` must not swap an operator leg to
the right; **(iii) a scalar factor must not commute *leftward past* a `∇` it is
the operand of** — this is the correctness leak behind the dropped Leibniz term
(`λ tr(sym∇u) (∇·I)` lost its `∇(∇·u)` because `tr(…)` slid left of `∇`).  All in
the canonicalization / factor-ordering / dyad-transpose logic.

**Not just display (revised).** Because of (iii), a **render-time normalization
fallback is *insufficient*** — the mis-positioning changes what `apply_operators`
differentiates, so the *value* is wrong (μ-Laplacian → 0 in Cartesian, dropped
`∇(∇·u)`).  The canon/positioning fix is mandatory, not cosmetic.

**Design note (revised).** Promoted from last-and-cosmetic to a **correctness
prerequisite for Increment 8** (Navier–Lamé).  Still the highest-risk edit, so
Increments 3 (`Δ`) and 5 (symmetry) should precede it (they remove `∇·∇` /
`(∇∇θ)ᵀ` shapes), but it must land **before** Increment 8.  If retrofitting a
clean positional-operator model onto canon proves too invasive, this is where a
small, lessons-informed redesign of the operator/positioning model belongs
(user's "better design fed by lessons learned" — kept *scoped* to operator
positioning, not a from-scratch rewrite).

**Files.** `src/derivation.cpp` (canon / tensor ordering), possibly
`src/tensor_order.cpp`; or `src/render.cpp` for the fallback.

**Verify.** `td.canonicalize(reass)` keeps every `∇` left; full suite green
(watch the operator/differentiation and strain tests for reorder regressions);
`strain_compatibility` example unchanged in result.

## Increment 7 — `sym`/`skew` constructors + recognize a symmetric-by-construction tensor (Issue 8, parts A + b1) — PARTIAL

**Done (commit 4b2b6e3):** the `(Aᵀ)ᵀ → A` transpose involution (one of the two
(b1) canon laws).  **Remaining:** the `sym`/`skew` constructors (A), transpose
distribution over `Sum`/`ScalarDiv` for symmetry recognition (the other (b1)
law — note `express`/canonicalize already distributes the transpose fence in the
contraction path via Increment 0, but plain `algebraic_eq((A+Aᵀ)/2, sym)` recog
still needs the sum-level rule), and the round-trip recognition assertions.

**Goal.** `sym(A)`/`skew(A)` build the (anti)symmetric part, and the result is
*recognized* (anti)symmetric — `algebraic_eq(sym(A), sym(A)ᵀ)` True,
`algebraic_eq(skew(A), −skew(A)ᵀ)` True — via structural normalization (no new
trait/node needed).

**Approach.**
- **(A)** Add `sym`/`skew` on the expr/chart surface as thin
  `(A ± Aᵀ)/2` builders (Python + binding).
- **(b1)** Add the small canon laws that make a symmetric-part fold:
  `(Aᵀ)ᵀ → A` (transpose involution) and transpose distribution over
  `Sum`/`ScalarDiv` (`(X+Y)ᵀ → Xᵀ+Yᵀ`, `(X/s)ᵀ → Xᵀ/s`) — with sum
  commutativity these normalize `((A+Aᵀ)/2)ᵀ` to `(A+Aᵀ)/2`.  The transpose-over-
  sum/ScalarDiv piece overlaps **Increment 0** (Issue 7 fence distribution) —
  share the implementation.

**Design question.** Is structural recognition (b1) sufficient for the workflow,
or is a *carried* symmetry trait (b2) needed where the symmetry survives through
operators (Issue 5) / can't be normalized structurally?  Start with (b1); revisit
(b2) if a real case needs it.

**Out of scope here.** Issue 8(C) — the *symmetry-guarded* identity application —
is **deferred with Issue 6** (conditional-identity capability, on the vibe-000054
primitive).  (b1) is a prerequisite for (C): the guard "is this symmetric?" reuses
the same recognition.

**Files.** `python/tender/*.py`, `python/_core.cpp` (`sym`/`skew`);
`src/derivation.cpp` / canon and `src/nf_lower.cpp` (involution + transpose
distribution — shared with Increment 0).

**Verify.** `sym`/`skew` round-trip; `(Aᵀ)ᵀ→A`; `((A±Aᵀ)/2)` recognized
(anti)symmetric; the malformed `A^{T}^{T}` render is gone; the strain
`ε = sym(∇u)` reads and folds as symmetric; full suite green.

## Increment 8 — reduce `∇·T` to the clean Navier–Lamé displacement form (**sprint endpoint**)

**Goal.** From the isotropic Hooke stress `T = λ(∇·u)I + 2μ sym(∇u)` (`u`
abstract), derive
`∇·T = (λ+μ) ∇(∇·u) + μ ∇·∇ u`
as an invariant operator identity, then expand it per coordinate system — the
continuum balance-equation term, in *any* CS.  (No `∇²`: write `∇·∇ u` or "the
Laplacian of `u`".)

**Approach — mirror the strain-compat pipeline, `u` abstract throughout.**
1. Build `T` with `u = ws.field("u", 1)` **abstract** (do *not* basis-expand);
   `ε = sym(∇u)` (Increment 7's `sym`).
2. Apply/expand the operators keeping `u` abstract, so `∇·T` reduces to a sum of
   `∇(∇·u)`, `∇·∇ u`, `∇·((∇u)ᵀ)`, `∇(∇·u)`-from-`∇·((∇·u)I)` terms — **with the
   operators correctly positioned** (needs **Increment 6**: today `∇` slides past
   the scalar `tr`/`λ` factors and drops the Leibniz `∇(∇·u)` piece, and `∇·I`
   wrongly collapses to 0).
3. Reduce the pieces with Increments 1–5: `∇·((∇·u)I) = ∇(∇·u)` (product rule +
   `∇·I` handling), `∇·((∇u)ᵀ) = ∇(∇·u)`, `∇·∇u` stays / folds to `Δu`
   (Increment 3), collect like terms → `(λ+μ)∇(∇·u) + μ ∇·∇u`.
4. **Reassemble / present** as the invariant identity, then **expand per-CS at
   the very end** (`chart.express` / `chart.components`), with `u` basis-expanded
   *last* — where `reassemble(u_i e_i)=u` etc. already work.

**Likely new sub-gaps to surface while building (record as found):**
- `∇·((∇·u) I) = ∇(∇·u)` — divergence of a scalar times the identity (the
  product-rule / `∇·I` reduction); the current output mis-handles it (Increment
  6's positioning + an `∇·(fI)` rule).
- Collecting `∇(∇·u)` from two different terms (the `λ` term and half of the `2μ`
  term) into one `(λ+μ)` coefficient — like-term collection over operator
  expressions.
- Whether a `reassemble`-style step is needed to name `∇·∇u`/`∇(∇·u)`, or the
  reduced form is already clean.

**Files.** Driven by Increments 1–6 landing; a new `examples/` example
(`elasticity_balance` / `navier_lame.{py,ipynb}`) as the witness; reductions in
`src/derivation.cpp` / `src/chart.cpp` as the sub-gaps demand.

**Verify.** `∇·T` reduces to `(λ+μ)∇(∇·u) + μ ∇·∇u` (abstract), checked
component-wise against a brute-force `chart.div(expand T)` in a Cartesian **and**
a cylindrical frame (the curvilinear payoff — the μ-Laplacian must **not** vanish,
the regression the basis-expanded route hit); the example runs end-to-end.

## Cross-cutting

- **Per CLAUDE.md:** every increment stays buildable/testable (incremental
  growth), ships unit tests (and, where a full slice moves, an example/notebook
  update), runs clang-format, keeps notebooks stripped, and commits naturally on
  the current branch.
- **Coverage:** keep line coverage ≥ 90% (the CI gate) as new branches land.
- **Examples — two end-to-end witnesses.** (i) `strain_compatibility.{py,ipynb}`:
  after Increments 1–5 it should reduce `tr(inc ε)` cleanly and display the closed
  form without stray `ᵀ` / `∇·∇` / ∇-on-the-right (the final equation-closure
  remains Issue 6, deferred).  (ii) A new **Navier–Lamé / elasticity-balance**
  example (Increment 8): `∇·T` reduced to `(λ+μ)∇(∇·u) + μ ∇·∇u` with `u`
  abstract, then expanded per-CS — authored the *abstract-operand* way (u
  basis-expanded last), and checked in Cartesian **and** cylindrical.  This is the
  sprint endpoint; it consumes Increments 1–6.
