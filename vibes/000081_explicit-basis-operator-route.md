# 000081 — explicit basis + operator route for ∇·u (cases-driven)

Status: **IN PROGRESS** — cases 1–3, 5–8 resolved (fixed or refused-with-guidance);
I9 fixed. Remaining: **I8** (merge-vs-distinguish coincident bases) deferred to the
**operations-revisit vibe** (next step); I7 is doc; I5 wontfix.

The user drives the next round of fixes with a Python example that derives `∇·u`
several ways (per coordinate system), each a numbered **case**. Every distinct
problem gets a short **issue label**. Discipline: **after each fix, re-run every
case and assess** whether it worked (a runnable harness lives in scratch;
`examples/` are success-stories only, so corrected cases stay in this vibe, not
in a committed example).

## Case & issue labels

**Cases** (from the driving example; `disp` shows each derive step):
| # | label | what it does | status |
|---|-------|--------------|--------|
| 1 | `baseline-div` | `∇·u` via `cs.grad → tr → apply_operators×2 → simplify_basis_dot → eval_delta → fold_arithmetic` | ✓ correct (cart & cyl) |
| 2 | `basis-first` | expand basis (`expand_in_basis`) while ∇ is still abstract, then `apply_operators … express … expand_nabla` | ✓ refused loudly (Fix B) |
| 3 | `nabla-first` | `expand_nabla` THEN `expand_in_basis`, then `apply_operators → unroll_sums → …` | ✓ refused loudly at unroll_sums → use `componentize_nabla` (Fix A) |
| 4 | `express-anomalies` | manual `express` / `expand_in_basis` interaction probes | I7 doc, I8 reproduced (case 9) |
| 5 | `scalar-complexify` | `grad(e_r)` then `express` — express should be a no-op (cyl) | ✓ FIXED+test (I9) |
| 6 | `nabla-appears` | `expand_in_basis(i) → reassemble_nabla → expand_nabla` (cart) | ✓ FIXED+test (I10) |
| 7 | `appends-one` | `reassemble_nabla(i)` — should be a no-op (cart) | ✓ FIXED+test (I11) |
| 8 | `mystery-1` | `expand_in_basis(i)` → `i_i e_i` (reference for I8) | ✓ ok+test |
| 9 | `mystery-2` | `express(i)` then `expand_in_basis(i)` → `i` (not `i_i e_i`) | I8 (render collision) |

**Issues:**
| label | case | one-liner | status |
|-------|------|-----------|--------|
| I1 `sum-surfaced` | 2,3 | `apply_operators` surfaced an explicit `Σ` on an implicit index | FIXED (implicitize) d9f0815 |
| I2 `basis-first-zero` | 2 | ∇ applied to constant basis vectors, gradient dropped | FIXED (refuse) e4ec23b |
| I3 `express-opaque` | 2 | `express` output mysterious after the broken state | SUBSUMED (case 2 refused) |
| I4 `final-zero` | 2 | final result 0 | FIXED (refuse) e4ec23b |
| I5 `cyl-expand-nabla` | 2 | cyl `expand_nabla` raises (free-index ∇ needs unit-scale frame) | WONTFIX (documented limit) |
| I6 `nabla-first-weird` | 3 | wrong vector `i+j+k`, then (worse) stuck `Σ_i δ_{i1} i` | FIXED (refuse→componentize) 9a67a18/5af9ae5 |
| I7 `express-no-reassemble` | 4 | `express` won't fold `u_i e_i → u` | USE `tb.reassemble` (doc) |
| I8 `express-render-collision` | 4,9 | reference `R` and physical frame `F` coincide for a Cartesian chart but are two `Basis`es with distinct `basis_id`, both rendering `𝐢`; `express` maps R→F invisibly | OPEN → operations-revisit vibe (merge vs distinguish elaborated) |
| I9 `express-complexify` | 5 | `express` (no-op) explodes coeff `1/r` into a 4-term trig sum that equals `1/r`; terms won't collect (tensor-in-div vs -out) | FIXED (canon `(s·T)/d→(s/d)·T`; express collects) 97bd283 |
| I10 `reassemble-fabricates-nabla` | 6 | `reassemble_nabla(i_i e_i)` invented `i_i ∇ 1` → `expand_nabla → 0` | FIXED (has_deriv_mark no-op) |
| I11 `reassemble-appends-one` | 7 | `reassemble_nabla(i)` appended `·1` | FIXED (has_deriv_mark no-op) |

**Note (basis(0) vs direction(0)):** for cart both render `𝐢`; for cyl
`direction(0)=e_r` (symbolic) but `basis(0)=cosθ 𝐢+sinθ 𝐣` (WCS). The new cases
use `basis.basis(0)`.

**Regression discipline (user):** once a case's result is accepted, make it a
Python test and mark it ✓test in the tables.

**NEXT STEP (new vibe, after I9):** the user wants to **revisit the set of
derivation operations** — rearrange which operations exist and what each does
(`express`, `expand_in_basis`, `reassemble`, `reassemble_nabla`, `componentize_
nabla`, `basis(i)` vs `direction(i)`, reference vs frame bases). This is where I8
(merge coincident bases) and the reference/frame model get resolved. To be
started as its own vibe, NOT inside 000081. I9 is now done → this is the next
thing to open.

## Side thread — `tr(inc ε)` of the strain-compat interim (`reass`)

User computing `tr(inc ε)` where `reass = −(∇∇··ε)I + Δθ I − Δε − ∇∇θ + ∇∇·ε +
(∇∇·ε)ᵀ` (θ=tr ε). Goal `Δ tr(ε) − ∇·(∇·ε)` — **identity verified via chart
operators** (`tr(inc ε) == chart.laplacian(tr ε) − chart.div(chart.div ε)`, True).

**RECIPE (works):** with a *dimensioned* identity,
`td.fold_equal_addends_structural(td.expand_dyad_ops(t.tr(reass)))` →
`Δ tr(ε) − ∇·(∇·ε)`, evaluable. (`cs.nabla()` is the CS-specific
∇=Σ_i(1/h_i)e_i∂_i; chart-free is `t.nabla(ctx)`.)

**Blockers:**
- **DESIGN FIX (dimension attribute) — 0c0888b.** Review caught a defect: the
  dimensioned identity carried its space in TWO FAKE unbound index slots. Index
  slots are for objects with real indices — the fake slots defeated
  `expand_in_basis`'s slotless-identity special case (broke `a%I%c`) and printed
  `I^{•·}_{•·}` (needing a render hack). Dimension-awareness is now a dedicated
  **orthogonal `TensorObject::dim` attribute** (`IndexSpace*`, kept last for
  aggregate-init compat). `make_identity(ctx, space)` sets `dim` and stays
  SLOTLESS → behaves exactly like the bare I in basis/contraction/render (clean
  `I`, hack removed); only `tr(I)=n` reads it. `dim` is part of identity (3-D I ≠
  agnostic I): in `tensor_object_cmp`, `structural_eq`, `hash_tensor_object`.
  Per the review, `expand_in_basis` now REFUSES a dimension mismatch (2-D I on a
  3-D frame). A sized I in the derivation works too now (slotless).
- **B1 (dimensioned identity) — FIXED** (7b40bcb; superseded by 0c0888b design).
  Threading a *dimensioned* I
  through the derivation BREAKS the cross-removal (`expand_in_basis` /
  `simplify_basis_cross` only handle the slotless bare I — `a%I%c` stops
  reassembling). So keep bare I in the derivation and have **`reassemble_nabla`
  stamp the chart's dimension onto the I in its invariant output**
  (`dimension_identities`, chart-aware). reass's I becomes `identity(space)`, so
  `tr(c·I)→c·n` folds. The dimensioned I's space-carrying slots are not indices,
  so render shows a clean `I` (not `I^{•·}_{•·}`) via `all_unbound_well_known`
  (both render sites). END-TO-END now works from the actual bare-I strain
  derivation: `td.fold_equal_addends_structural(td.expand_dyad_ops(t.tr(reass)))`
  → `Δ tr(ε) − ∇·(∇·ε)`.
- **B2 (scalar-Hessian trace) — FIXED** (8adad47, generalized in 0920b2e).
- **B3 (combine without float) — FIXED** (0920b2e). NOT the full canon operator
  normal form — sidestepped for this workflow: (1) the `expand_dyad_ops` Trace
  dyad rule now puts an operator leg on the LEFT with the other leg + scalars as
  its operand, so `tr(∇⊗v)` and the transposed `tr(v⊗∇)` both give `∇·v` (not
  `v·∇`) and `tr(∇⊗∇⊗s)=Δs` (B2 subsumed) — transposed and direct terms become
  structurally identical; (2) `extract_coeff` recurses so a `−`/factor buried in a
  coefficient (`(−X)·3`) lifts out, matching a bare-`X` sibling. Then
  `fold_equal_addends_structural` combines every term WITHOUT `collect_terms`/canon
  (which floats the scalar off `Δθ`). The deep canon operator-normal-form (the
  real I2 wall) is still deferred — `collect_terms` still floats; use the
  structural fold for bare-∇ operator sums.

## Driving example — preamble

```python
import tender as t
import tender.basis as tb
import tender.derivation as td
from IPython.display import Math, display

def disp(x):
    display(Math(x.latex()))

def derive(initial, steps, *, cb = None):
    if cb is None:
        cb = lambda a: None
    d = td.Derivation(initial)
    cb(initial)
    for s in steps:
        d.step(s)
        cb(d.current)
    return d.current

ws = t.Workspace()
nabla = t.nabla(ctx=ws.ctx)

x, y, z = ws.coords("x", "y", "z")
cart = ws.chart(ws.wcs(), [x, y, z], [x, y, z])
r, th, zc = ws.coords("r", r"\theta", "z", nonneg=("r",))
cyl = ws.chart(ws.wcs(), [r, th, zc], [r*t.cos(th), r*t.sin(th), zc])

co = tb.Variance.Covariant
contra = tb.Variance.Contravariant

u = ws.field(r"u", 1)

all_cs = {"cart": cart, "cyl": cyl}
```

## Cases (verbatim from the driving example)

### Case 1 `baseline-div` — `∇·u` per CS (correct)

Status: **ok** (reference derivation, both `cart` and `cyl`).

```python
# Derivation of nabla@u in a coordinate system
# Status: ok
for cs_name, cs in all_cs.items():
    print(f"--------\nCoordinate system: {cs_name}")
    basis = cs.physical_basis()
    derive(
        ws.field(r"u", 1),
        [
            cs.grad,
            t.tr,
            td.apply_operators,
            td.apply_operators,
            lambda a: tb.simplify_basis_dot(a, basis),
            td.eval_delta_concrete,
            td.fold_arithmetic
        ],
        cb=disp)
```

Establishes the step vocabulary the broken cases will reuse:
`grad → tr → apply_operators×2 → simplify_basis_dot → eval_delta_concrete → fold_arithmetic`.

### Case 2 `basis-first` — expand basis while ∇ abstract (was: zero)

Status: **incorrect result (zero)** — plus rendering/usability problems.

```python
# Derivation of nabla@u later expanded in a coordinate system
# Status: incorrect result (zero)
for cs_name, cs in all_cs.items():
    print(f"--------\nCoordinate system: {cs_name}")
    basis = cs.physical_basis()
    derive(
        ws.field(r"u", 1),
        [
            lambda a: nabla*a,
            t.tr,
            lambda a: tb.expand_in_basis(a, basis, contra),
            td.apply_operators,
            td.unroll_sums,
            cs.express,
            cs.expand_nabla,
        ],
        cb=disp)
```

Issues exposed by Case 2:

1. **[rendering]** after step 4 (`apply_operators`), an *explicit sum* is rendered
   (should stay in implicit/Einstein form, or at least not surface a raw
   `ExplicitSum`).
2. **[CORRECTNESS]** after step 4 (`apply_operators`), `∇` is applied to the
   **constant basis vectors** rather than to the coordinate-dependent
   components — so the derivative hits `∂ e_i = 0` and the whole thing later
   collapses to **zero**. The correct behavior: derivatives of the *components*
   times basis-vector dot-products, i.e. `(∇ u_i)·e_i` = `∂_j u_i (e_j·e_i)`.
   This is the *basis-expand-first correctness trap* already flagged in vibe
   000080 (components land on the wrong side of the operator) — but here it bites
   through `expand_in_basis` + `apply_operators` rather than the chart path.
3. **[usability/rendering]** after step 6 (`cart.express`) the output is
   "weird" — it is unclear how that state could follow from the step input
   (opaque transformation; provenance is lost).
4. **[CORRECTNESS]** final result is **zero** (consequence of #2).
5. **[usability/error]** for **cylindrical**, step 7 (`cs.expand_nabla`) raises
   `ValueError: expand_nabla: the free-index ∇ expansion currently supports only
   a constant unit-scale (Cartesian) frame; use the chart operators
   (grad/div/rot) for curvilinear charts`.

### Case 3 `nabla-first` — expand_nabla before basis (was: `i+j+k`)

Status: **incorrect result (vector `i+j+k`)** — Cartesian only (`cyl` commented out).

```python
# Derivation of nabla@u later expanded in a coordinate system
# Status: incorrect result (vector i+j+k)
for cs_name, cs in (("cart", cart),):#all_cs.items():
    print(f"--------\nCoordinate system: {cs_name}")
    basis = cs.physical_basis()
    derive(
        ws.field(r"u", 1),
        [
            lambda a: nabla*a,
            t.tr,
            cs.expand_nabla,
            lambda a: tb.expand_in_basis(a, basis, contra),
            td.apply_operators,
            td.unroll_sums,
            lambda a: tb.simplify_basis_dot(a, basis),
            td.eval_delta_concrete,
            td.fold_arithmetic,
            cs.express,
        ],
        cb=disp)
```

Issues exposed by Case 3:

6. **[CORRECTNESS]** ordering `expand_nabla` *before* `expand_in_basis` yields the
   **wrong vector `i+j+k`** instead of the scalar `∇·u`. `∇·u` is a scalar
   (rank-0); a vector result signals the sum over components was mis-contracted
   (the trace/divergence contraction lost, leaving a dangling free basis index
   that sums the unit vectors). Distinct failure mode from Case 2's zero, but the
   same root theme: basis expansion interleaved with operator expansion corrupts
   the contraction.

### Case 4 `express-anomalies` — express / expand_in_basis interaction

Manual exploration (not a full script); observed on both CS.

7. **[usability/correctness]** Cartesian: after the first step the output is
   `u_i e_i` (basis-expanded), and it **does not fold back** to `u` after
   `cs.express`. `express` is expected to recover the invariant `u` from
   `u_i e_i` but doesn't.
8. **[correctness/state]** Hidden-state effect of `express`: making `cs.express`
   the *very first* step outputs (visibly) still `u` — a seeming no-op — but then
   `expand_in_basis` yields `u` **instead of** `u_i e_i`, i.e. on apparently the
   *same* input the downstream step behaves differently. So `express`
   mutates/normalizes some internal representation *invisibly* (display
   unchanged, semantics changed). Cylindrical: with `cs.express` uncommented it
   *does* visibly change the input. → `express` has an order-dependent,
   display-invisible side effect; its interaction with `expand_in_basis` is not
   idempotent/commuting.

### Cases 5–9 (second batch; `basis = cs.physical_frame()`, `initial = basis.basis(0)` unless noted)

```python
# Case 5 scalar-complexify (cyl): grad(e_r) then express; express must be a no-op
derive(basis.direction(0), [cs.grad, cs.express], cb=disp)
# Case 6 nabla-appears (cart): reassemble_nabla must not invent a ∇
derive(basis.basis(0), [lambda a: tb.expand_in_basis(a, basis, contra),
                        cs.reassemble_nabla, cs.expand_nabla], cb=disp)
# Case 7 appends-one (cart): reassemble_nabla(i) must be a no-op
derive(basis.basis(0), [cs.reassemble_nabla], cb=disp)
# Case 8 mystery-1 (cart): expand_in_basis(i) → i_i e_i  (reference, ok)
derive(basis.basis(0), [lambda a: tb.expand_in_basis(a, basis, contra)], cb=disp)
# Case 9 mystery-2 (cart): express(i) then expand_in_basis(i) → i  (I8)
derive(basis.basis(0), [cs.express, lambda a: tb.expand_in_basis(a, basis, contra)], cb=disp)
```

**Diagnoses (grounded by runtime traces):**
- **I10/I11 (cases 6, 7) — FIXED (chart.cpp `has_deriv_mark` guard in
  `reassemble_term`).** `reassemble_term` treated any frame vector as a gradient
  leg; with no operand it fabricated a base `1` and wrapped ∇ around it — so `i`
  → `i·1` (I11) and `i_i e_i` → `i_i ∇1` → `expand_nabla → 0` (I10).  A term with
  NO ∂-mark is a plain expanded tensor, not a ∇-expression; reassembly is now a
  no-op on it.  Cases 6/7 give clean no-ops (`i_i e_i` / `i`).  Guards:
  py `test_reassemble_nabla_is_noop_without_derivative`,
  `test_expand_in_basis_of_reference_vector` (case 8).
- **I8 (case 9) — NOT a hidden-state bug; a RENDER COLLISION of two coincident
  bases.** Each step is individually correct; the confusion is that two
  *structurally distinct* vectors render identically in Cartesian.

  **The two bases (verified):**
  1. **Reference (WCS) basis** `R = chart.reference` (`ws.wcs()`) — the ambient
     constant Cartesian axes `i, j, k`, its own `basis_id` (call it `idR`).
  2. **Physical frame basis** `F = physical_frame(chart)` — the chart's
     orthonormal (moving) frame, a *separate* tender `Basis` with its own
     `basis_id` (`idF ≠ idR`). Its symbolic atoms are `F.direction(i) = e_i^F`
     (carry `idF`).

  Key API distinction (verified): on any frame `F`,
  - `F.basis(i)` returns the STORED CONSTRUCTION of `e_i^F` **in R** (its WCS
    components): cart → `𝐢`; cyl → `cosθ 𝐢 + sinθ 𝐣`. In fact for cart
    `F.basis(i)` is *structurally equal to* `R.basis(i)` (`structural_eq` True).
  - `F.direction(i)` mints the SYMBOLIC atom `e_i^F` carrying `idF`: cart → `𝐢`;
    cyl → `e_r`.

  **Why the collision is Cartesian-only:** for a Cartesian chart aligned with
  WCS the map is the identity, so `e_i^F` equals `R.basis(i)` *mathematically*,
  and F's value-names spell it `𝐢` too. So `F.direction(0)` (idF, `𝐢`) and the
  reference `𝐢` (idR) are structurally distinct (`structural_eq` False) yet
  render the same. In cyl they render differently (`e_r` vs `cosθ𝐢+sinθ𝐣`), so
  no collision. In case 9 `express(𝐢_R)` maps R→F giving `e_x^F` (renders `𝐢`),
  and `expand_in_basis(·, F)` then leaves the already-F vector (`→ 𝐢`), whereas
  on the reference `𝐢_R` it expands (`→ i_i e_i`) — both correct, invisibly
  different inputs.

  **Merge vs distinguish (for the operations-revisit vibe, not here):**
  - **(A) Merge coincident bases.** When `physical_frame` would reproduce the
    reference exactly (unit scale factors AND `F.basis(i) ≡ R.basis(i)` ∀i),
    register F with `idR` (one basis, not two). Then `F.direction(i) ≡
    R.basis(i)`; `express` is a true no-op on reference vectors and case 9 →
    `i_i e_i` (matches case 8). PRO: kills the collision at the source, `𝐢` and
    `e_x` become one object. CON: (1) `expand_in_basis(𝐢, F)` on a merged basis
    is a no-op (a basis vector expanded in its own basis is itself) so **case 8's
    `i_i e_i` would become `𝐢`** (arguably also correct); (2) only applies to a
    Cartesian chart *aligned* with WCS — a rotated Cartesian frame does not
    coincide; (3) couples F's connection/∂ machinery to R (trivial for cart:
    ∂e=0).
  - **(B) Keep distinct, make visible.** Always render a physical-frame vector
    distinctly (`e_x, e_y, e_z`), never `i,j,k`. PRO: no ambiguity. CON: cart
    loses the familiar `i,j,k`; a rendering regression for the common case; the
    two-basis modeling stays, just surfaced.
  - **(C) Document only.** Accept F≡R for cart renders alike; `express` maps
    R→F; no code change.
  RECOMMENDATION LEAN: (A) merge, since the two bases are genuinely the same
  object for a WCS-aligned Cartesian chart — but the right call depends on
  redefining `express` / `expand_in_basis` / `basis(i)` vs `direction(i)`
  semantics, which belongs to the operations-revisit vibe (see NEXT STEP below).
- **I9 (case 5) — OPEN, simplification.** `express(grad(e_r))` gives four
  `e_θe_θ` terms whose coeffs sum to 1 (`= 1/r`), but they DON'T collect because
  the terms are structurally inconsistent: term 1 is `(sin⁴θ·e_θe_θ)/r` (tensor
  INSIDE the `/r`) while the rest are `(scalar/r)·e_θe_θ` (tensor OUTSIDE).
  `collect_terms` can't unify tensor-in-div with tensor-out-of-div.  FIX
  DIRECTION: a canon normalization `(s·T)/d → (s/d)·T` (factor non-scalar legs
  out of a ScalarDiv numerator) so all terms share one shape and the trig folds
  to 1.  Scope: a canon/simplify change of moderate risk — hold for user OK.

### API note — obtaining `e_r`

`cs.physical_frame()` / `cs.physical_basis()` return a **`Basis`**, not a vector.
The abstract symbolic frame vector is `basis.direction(i)`:
`cs.physical_frame().direction(0).latex()` → `\mathbf{e}_{r}` (verified). The WCS
form (`cosθ i + sinθ j`) is the Basis's internal construction, not what
`.direction(i)` returns. Prefer `physical_frame()` (registers the connection
table → `∂ e_r` resolves symbolically) over `physical_basis()` when
differentiating frame vectors.

## Root causes (grounded by runtime tracing)

### Case 2 `basis-first` zero (I2, I4) — abstract ∇ never applied; canon commutes the coefficient across it
Trace of `nabla@(f·e_x)` (a *scalar field* f, not just a component):
`apply_operators(∇·(f e_x)) = f (∇·e_x)`. The `(∇f)·e_x` term is **dropped**.
Mechanism: `apply_operators_impl` only *applies* concrete `Deriv` operators
(the frame ∂'s that `expand_nabla` emits). An **abstract, unexpanded `Nabla`**
falls through to `cv->rebuild(∇, go(f e_x))`, i.e. ∇ is left in place — and then
`canon_tolerant`'s `canonicalize` treats `Nabla` as an ordinary rank-1 vector
factor and **commutes the scalar `f` to the front** of the `∇·` contraction,
producing `f (∇·e_x)`. That is exactly the "canon treats bare ∇ as a plain
factor" hazard from vibe 000080 — here it silently corrupts the *value* (drops a
Leibniz term), not just display. In Cartesian `∇·e_x = 0`, so the whole thing → 0.
So: **Case 2's order (basis-expand while ∇ is still abstract) feeds
`apply_operators` an operator it cannot apply, and canon then drops terms.**

### Case 3 `nabla-first` weird (I6) — ∂-direction index orphaned from its frame vector
Trace: `expand_nabla(tr(∇u)) = Σ_i e_i·(∂_i u)` (∂ direction index *is* the frame
index i). `expand_in_basis` then expands `u→u_k e_k`, renaming to display
`Σ_j e_j·(∂_j u_i) e_i` — BUT the ∂-mark's free direction index is **not renamed
in lockstep with its frame vector** `e_j`. After `apply_operators`
(`Σ_j Σ_i (∂_j u_i) e_j·e_i`), `unroll_sums` concretises the two *summation*
indices (e_j→{i,j,k}, e_i→component u_x/u_y/u_z) but the ∂-direction stays the
orphaned symbolic `i`: output rows read `(∂_i u_x) i·i + (∂_i u_y) i·j + …`.
δ-contraction keeps the diagonal `∂_i u_x + ∂_i u_y + ∂_i u_z` — a leftover free
index `i` where it should be `∂_x u_x + ∂_y u_y + ∂_z u_z`. `express` then
renders the dangling free index as the sum of unit vectors → `i+j+k`.
Same *class* as vibe 000080 sym-form (b) (inconsistent α-renaming orphans ∂-mark
direction indices) — here triggered by `expand_in_basis`, not canonicalize.

### Issue 1 (rendering) — `apply_operators` surfaces an `ExplicitSum` `Σ_i`
`apply_operators` materialises the Einstein sum as an explicit `Σ_i`
(`u_i ∇·e_i` → `Σ_i u_i ∇·e_i`). Cosmetic; `td.implicitize` re-folds it.

### Issue 3 (express weirdness) — express is fragile on the already-broken Part-2 term
`express(u_x ∇·i + …)` returns `∇·i i + ∇·j j + ∇·k k` — it mis-reads the loose
component symbols `u_x,u_y,u_z` as a vector's reference components and reshuffles
roles. This is *downstream of* the Part-2 corruption; it largely evaporates once
Case 2 is refused/corrected. Kept as a dependent item.

### Issue 5 (cylindrical error) — deliberate limitation, not a bug
`expand_nabla` asserts a constant unit-scale (Cartesian/WCS) frame: a moving
frame's `1/h_i` and `∂_i e_j ≠ 0` cannot ride one summed index uniformly. Message
already says "use grad/div/rot for curvilinear." Usability, not correctness.

### Issue 7 (express ≠ reassembly) — the tool exists, express is just the wrong one
`express(u_i e_i)` returns `u_i e_i` (express = frame change, via `to_reference`).
The invariant-folding inverse of `expand_in_basis` is **`tb.reassemble(expr,
basis)`**, which already returns `u` (verified). So issue 7 = discoverability /
express not opportunistically reassembling.

### Issue 8 (hidden state in express) — not reproduced with a field; needs the exact expr
`express(u); expand_in_basis(u)` still gives `u_i e_i` here (no order-dependence
observed). The user's `i_i e_i` case likely involves a *reference basis vector*
and `physical_frame`'s Context connection-table registration side effect. Needs
the exact expression to repro before diagnosing.

## Plan

The two orders of "expand everything then compute" are the crux:
- **∇-first (Case 3 order):** the sound route in principle — make it correct
  (Fix A). This becomes the recommended explicit workflow.
- **basis-first (Case 2 order):** feeds `apply_operators` an operator it can't
  apply; canon then silently drops terms. Stop the silent corruption (Fix B).

**Fix A — Case 3 (CORRECTNESS, primary). DONE.** ROOT CAUSE refined: a *free*
∂-mark stores an empty `CoordinateRef{}` (no chart id, no coord name), so the
chart-free `substitute`/`unroll_sums` genuinely *cannot* turn `∂_{link=i}` into
`∂_{q^v}` — only the chart-aware `componentize_nabla` can (it concretizes the
frame vector `e_i→e_v` AND the ∂-mark in lockstep). The bug was `unroll_sums`
*half*-concretizing: unrolling `e_i` while orphaning the linked ∂_i → the bogus
`i+j+k`. FIX: `unroll_sums` now skips (leaves intact) any ExplicitSum whose index
is the `link` of a free ∂-mark in its body (`body_has_free_deriv_link`), so those
indices are left for `componentize_nabla`. The correct explicit route is
`… apply_operators → cs.componentize_nabla → unroll_sums → simplify_basis_dot →
eval_delta_concrete → fold_arithmetic` → `∂_x u_x+∂_y u_y+∂_z u_z` (verified).
Guards: py `test_divergence_via_explicit_basis_and_componentize`,
`test_unroll_sums_leaves_free_deriv_linked_index`; C++
`UnrollSums.LeavesFreeDerivLinkedIndex`. Part-3 example step list to switch
`td.unroll_sums`→`cs.componentize_nabla` (then unroll) when writing the witness.

**Fix B — Case 2 (CORRECTNESS). DECISION was B2; SCOPE FINDING blocks the cheap
version.** Investigation nailed the corruption locus and, more importantly, an
architectural wall:
- Exact corruption: `distribute_contraction` (canon internal, derivation.cpp
  ~1261) floats a *legless scalar* out of a contraction fence
  (`∇·(f⊗v) → f (∇·v)`), treating the abstract ∇ as a plain vector and dropping
  the gradient `(∇f)·v`.  This is the normally-correct bilinearity `X·(f v) =
  f(X·v)` firing when `X` is the operator ∇.
- KEY: the user's Part-2 pipeline is **already correct** if the abstract ∇ is
  resolved by `expand_nabla` *before* any canonicalization — verified:
  `expand_nabla(tr(∇⊗(u_i e_i)))` → `∂_x u_x+∂_y u_y+∂_z u_z`.  The premature
  `apply_operators` (which calls `canon_tolerant`) is the sole cause; canon never
  should have seen an unexpanded ∇ over expanded components.
- WALL: making canon itself correct here needs an **operator normal form**.
  Canon has NO representation for a grad/div nested as a contraction operand
  (`∇⊗f`, `∇·v` inside a `Dot`): `encapsulate` throws "nested ⊗ inside an operand
  awaits fence distribution".  A probe that merely *stopped* the scalar-float
  (guarding `ends_in_operator`/`begins_with_operator`) left the node
  un-distributed → `encapsulate` crash.  And the mathematically-correct Leibniz
  output `(∇f)·v` is itself a `∇⊗f` inside a Dot → same crash.  The working
  chart/operator paths avoid this by expanding ∇ into concrete frame ∂'s FIRST,
  so a grad becomes `Σ_i e_i (∂_i f)` (∂ is a rank-0 field-deriv atom, no operator
  nested in a fence).  Teaching canon an operator normal form is a large, risky,
  multi-increment redesign — the exact vibe-000080 minefield.

So the practical choice narrows to: **B1-style guard** (make `apply_operators` /
canon refuse or no-op an abstract ∇ it can't apply, turning the silent-zero into
a loud pointer to expand_nabla — the correct answer is already reachable by the
∇-first order, Fix A) vs **B2-full** (operator normal form in canon; large).
Re-checked with the user before proceeding (scope was not known at decision time).

**DECISION (re-confirmed with user): guard + clear error. DONE.** New predicate
`steps::abstract_nabla_over_expanded_basis(ctx, e)` = an abstract `Nabla` node
coexists with a basis-frame vector (a slot `basis_id ≠ 0`).  `apply_operators`
and `chart::express` check it up-front and raise `std::invalid_argument`
("∇ is still abstract over an expanded basis — … Expand ∇ first (expand_nabla /
grad / div / rot), then expand the basis.") BEFORE any canonicalization, so both
failure modes (the silent scalar-float zero AND the cryptic encapsulate crash)
become one actionable message.  Scoped to `Nabla` only (NOT `Deriv`), so the
∇-first order — where `expand_nabla` has already replaced ∇ by frame ∂'s (Deriv)
+ e_i — passes untouched; abstract field/identity forms (no basis vector) also
pass (vibe-080 unaffected).  Guards: py
`test_apply_operators_refuses_abstract_nabla_over_expanded_basis`,
`test_abstract_nabla_over_basis_ok_when_nabla_expanded_first`; C++
`AbstractNabla.DetectsNablaOverExpandedBasis`.  Issue 3 (express weirdness) is
subsumed: express now refuses that input outright.  Full suite 814 C++ / 267 py.

**Fix C — Issue 1 (rendering).** The surfaced explicit `Σ_j Σ_i` after
`apply_operators` re-folds to Einstein form with **`td.implicitize`** (verified).
Tool already exists — steer users to it rather than change `apply_operators`'
output contract (many callers/tests read its explicit sums). Awaiting user
preference on whether `apply_operators` should auto-implicitize.

**Fix E — Issue 7 (usability). DECISION: keep `express` pure; document.**
`express` stays a pure frame-change; document `tb.reassemble(expr, basis)` as the
invariant-folding inverse of `expand_in_basis` and steer users to it (docstring
and/or example comment). No behavioural change to `express`.

**Issue 3 (G) — dependent:** re-check after Fix B; likely resolved. If not,
harden `express` against loose non-canonical component sums.

**Issue 5 (D) — out of scope for now:** keep the documented limitation; only
improve the message/guidance. Curvilinear free-index ∇ is a separate large item.

**Issue 8 (F) — blocked:** need the exact `i_i e_i` expression from the user to
reproduce; then investigate `physical_frame` registration side effects.

Ordering: **A** (correctness, unblocks the recommended route) → **B** (stop
silent zero) → **C**, **E** (polish) → revisit **G/F**. Two example witnesses to
grow: the Part-1 baseline (keep green) and a *corrected* Part-3 as the explicit
route. See [[vibe80-notebook-gaps-sprint]], [[route-b-curvilinear-derivations]],
[[differential-operators-and-strain-compat]].
