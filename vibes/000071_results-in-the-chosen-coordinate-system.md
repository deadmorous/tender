# 000071 — Work intrinsically in the chosen coordinate system

The differential operators (vibe 000069 M6, vibe 000070) currently **expand
everything into the chart's constant Cartesian reference frame** (WCS `i, j, k`),
differentiate the scalar coefficients there, and return the result still in WCS.
This vibe rejects that approach: the operators should compute **intrinsically in
the chosen curvilinear basis** using the *known derivatives of the basis
vectors* (the connection / Christoffel symbols we already compute), never
returning to WCS.  Results stay in the chosen system's physical basis
`e_r, e_θ, e_z`.

## Guiding idea

The whole purpose of curvilinear coordinates and their natural vector bases is to
serve as the **most convenient representation** of a tensor object.  Once a
coordinate system is chosen, that is where the tensors live — the components and
the basis vectors of *that* system — and we **stay there**.  Returning to WCS is
almost never wanted; it is a computational convenience of the current
implementation, not the mathematics.

Crucially, staying in the curvilinear basis is not only more convenient to read,
it is how the work is actually done by hand and the more **efficient** path:

- A basis vector is **not** treated as "a non-constant thing we must re-expand in
  WCS".  It is an atom whose derivatives are **already known**:
  `∂_r e_r = 0`, `∂_θ e_r = e_θ`, `∂_θ e_θ = −e_r`, … — the connection
  coefficients `γ^k_{ij}` (vibe 000069 M5, `connection_coefficients` /
  `basis_derivative`).  Differentiation *uses* this table directly.
- The reference frame (WCS) never appears, so trigonometric functions never
  appear.  Expanding `e_r = cos θ i + sin θ j`, differentiating the trig, and
  reassembling is exactly the wasteful round-trip we want to avoid.

So the design should compute *and* present in the chart's physical basis by
default, driving `∂` on basis vectors through the connection, and treat WCS (or
any other frame) as an explicit, opt-in destination only.

## Motivating exchange (verbatim)

> I created a cylindrical CS like this:
> ```python
> WCS = ws.wcs()
> r, th, z = ws.coords("r", r"\theta", "z")
> cyl = ws.chart(
>     WCS,
>     [r, th, z],
>     [r*t.cos(th), r*t.sin(th), z])
> ```
>
> **Q1.** How do I get basis vectors e_r, e_theta, e_z?
>
> Then I created a radial field:
> ```python
> f = t.field("f", 0, deps=[r], ctx=ws.ctx)
> ```
> and calculated its gradient in the cylindrical CS:
> ```python
> disp(cyl.gradient(f))
> ```
>
> **Q2.** Is it possible to see a vector expanded in the basis of the cylindrical
> CS, rather than the Cartesian CS (I'm seeing the latter)?
>
> **Q3.** If I used `nabla*f` instead, how could I expand the nabla in the
> cylindrical CS?

### Answer given (current state of the API)

**Q1 — basis vectors `e_r, e_θ, e_z`.** They come from the chart's *physical
basis*:

```python
pb = cyl.physical_basis()
e_r, e_th, e_z = pb.basis(0), pb.basis(1), pb.basis(2)
# e_r  = cos θ i + sin θ j
# e_th = −sin θ i + cos θ j
# e_z  = k
```

They render *in Cartesian* on purpose: the chart **derives** the whole geometry,
so `e_r` literally *is* `cos θ i + sin θ j`.  There is no opaque `e_r` symbol at
this layer — the frame is expressed in the constant reference vectors.

**Q2 — seeing a result in the cylindrical basis.** By design, every operator
returns its result in the **constant Cartesian reference frame**.  That is what
lets `∂_i` act on the scalar coefficients alone (the M6 formula
`∇⊙T = Σ_i (1/h_i) e_i ⊙ ∂_i T`); if the output moved with the frame, the
connection terms would not fall out cleanly.  `tb.reassemble(v, pb)` does **not**
fold it back — the result is in the reference frame, not a symbolic `Σ aᵢ eᵢ`
expansion.

To view a result in the physical basis you must **project** onto the
(orthonormal) `e_k` with `cyl.dot`:

```python
comps = [td.simplify_scalars(cyl.dot(v, pb.basis(k))) for k in range(3)]
# grad f(r)  ->  [∂_r f, 0, 0]
```

Those components (`∂_r f, 0, 0`) *are* the cylindrical-basis representation.  For
a symbolic vector printout, rebuild against the hand-written symbolic basis
`tb.cylindrical(ws.ctx)` (whose vectors render as bold `r, θ, z`):

```python
cb = tb.cylindrical(ws.ctx)
sum(comps[k] * cb.basis(k) for k in range(3))   # → ∂_r f · 𝐫
```

There is **no single built-in call** for this today — a genuine gap.

**Q3 — expanding `nabla` in the cylindrical CS.** `nabla` is chart-free; the
chart passed to `evaluate` is how the CS is chosen:

```python
from tender.operators import nabla
(nabla * f).evaluate(cyl)          # ∇ expanded in the cylindrical chart
```

This is identical to `cyl.gradient(f)` (`structural_eq == True`) and, like it,
comes out in the Cartesian reference frame.  Evaluation *performs* the expansion
`∇ = Σ_i (1/h_i) e_i ∂_i`; it does not print the symbolic operator form
`e_r ∂_r + (1/r) e_θ ∂_θ + e_z ∂_z`.  For the physical-basis view one applies the
same `cyl.dot` projection as Q2.

## Worked example — the intrinsic way (how it *should* work)

Second gradient of a radial field `f(r)` in cylindrical `(r, θ, z)`, done the way
it is done by hand — entirely in the curvilinear basis, using the known basis
derivatives, **no trigonometry, no WCS**:

```
∇ = e_r ∂_r + (1/r) e_θ ∂_θ + e_z ∂_z          (h_r = 1, h_θ = r, h_z = 1)

∇f      = e_r ∂_r f                              (f depends on r only)

∇∇f     = ∇(e_r ∂_r f)
        = e_r ∂_r(e_r ∂_r f) + (1/r) e_θ ∂_θ(e_r ∂_r f) + e_z ∂_z(e_r ∂_r f)
```

Now differentiate the basis vector *using the connection*, not by expanding it:

```
∂_r e_r = 0            →  e_r ∂_r(e_r ∂_r f) = e_r ⊗ e_r ∂_r∂_r f
∂_θ e_r = e_θ          →  (1/r) e_θ ∂_θ(e_r ∂_r f) = (1/r) e_θ ⊗ e_θ ∂_r f
                          (∂_θ(∂_r f) = 0, again because f = f(r))

∇∇f = e_r e_r ∂_r∂_r f + (1/r) e_θ e_θ ∂_r f
```

The result is expressed in `e_r, e_θ, e_z` with curvilinear components and never
touches `cos θ, sin θ, i, j, k`.  Contrast the current implementation, which
would expand `e_r → cos θ i + sin θ j`, differentiate the trig, and return a WCS
dyad — far more work and in the wrong frame.

The key rule the engine must apply: **`∂_j eᵢ = Σ_k γ^k_{ij} eₖ`**, taken from
the connection table we already compute — used *whenever* a derivative meets a
basis vector, instead of falling back to a WCS expansion.

## Problem statement

The three questions and the example expose one root issue: **the engine computes
in the wrong frame.**  Everything reduces to WCS to differentiate, which is both
the wrong home for the result *and* the inefficient way to compute it.

1. **Computation is extrinsic.** The operators (vibe 000069 M6) expand each field
   into the constant Cartesian frame so `∂` only ever meets scalar coefficients,
   then differentiate trig.  The intended way — differentiate basis vectors via
   `∂_j eᵢ = γ^k_{ij} eₖ` and stay curvilinear — is not used, even though the
   connection coefficients already exist (M5).
2. **Wrong output frame.** Results come back in WCS components even when the user
   works entirely in a curvilinear chart; the natural representation (symbolic
   `e_r, e_θ, e_z` with curvilinear components) needs manual projection.
3. **Two disconnected cylindrical representations.** The chart's
   `physical_basis()` vectors are concrete Cartesian expansions
   (`cos θ i + sin θ j`); the hand-written `tb.cylindrical` basis vectors are
   opaque symbols (bold `r, θ, z`).  Nothing bridges the two, so a basis vector
   cannot be both differentiable-via-connection *and* symbolic in display.

## Direction to explore (to be refined)

Recompute intrinsically.  The `∂` engine and the operators should treat the
physical basis vectors as **first-class symbolic atoms with a known derivative
rule**, and never expand to the reference frame.

Concrete threads to work out later:

- **A. Symbolic basis vectors with a connection-driven derivative.** Make
  `e_r, e_θ, e_z` opaque atoms (rendered as `e_r`, …) that the differentiator
  recognises, with the rule `∂_{q^j} eᵢ = Σ_k γ^k_{ij} eₖ` sourced from the
  chart's already-computed connection (M5).  This replaces "a basis vector is a
  WCS expansion, differentiate its trig".  A field written on these atoms (e.g.
  `f(r) e_r`) then differentiates fully within the curvilinear basis.
- **B. Operators compute and return in the chart basis.** Reformulate
  `gradient` / `divergence` / `rot` / `laplacian` (and the `nabla` layer) as
  `∇⊙T = Σ_i (1/h_i) eᵢ ⊙ ∂_i T` with `∂_i` acting by Leibniz on curvilinear
  components *and* basis vectors (via A), producing a result on `e_r, e_θ, e_z`.
  No WCS in, no WCS out.  Preserve the known results (`∇R`, `∇²`, `rot`, and the
  worked `∇∇f(r)` above) in the new intrinsic form.
- **C. One curvilinear basis object.** Reconcile `physical_basis()` and the
  symbolic `tb.cylindrical` view into a single basis that is simultaneously the
  differentiation domain (knows its `γ`) and the display (`e_r`), instead of
  pairing two.
- **D. Basis-to-basis expansion is the general primitive (subsumes "expand in
  WCS").** Expanding a result into WCS is nothing special — it is just
  *expressing the basis vectors of one CS in the basis of another*: `e_r` written
  in `i, j, k`.  So instead of a bespoke `to_reference`, provide a general
  `express eᵢ of CS-A in the basis of CS-B`, of which WCS is one target.  This is
  precisely the capability that lets M6 be dropped: WCS output stops being a
  privileged path and becomes an on-demand change of basis.  A symbolic basis
  vector is always expandable because it is linked (via the chart) to the data
  that relates it to any other frame.  (Longer term: a **tree of coordinate
  systems**, CAD-style, where each CS is placed relative to a parent, and
  expansion composes along the tree.)
- **E. Oblique bases come (almost) for free; orthonormal niceties are the
  special case.** Nothing about the intrinsic rule `∂_j eᵢ = γ^k_{ij} eₖ` needs
  orthonormality — a general oblique basis has a connection too, so the same
  machinery applies to any basis.  What is *orthonormal-only* is the convenient
  rigid-body form `∂_q eᵢ = Ω × eᵢ` (the moving frame rotates as a rigid body
  along a curve, with an angular-velocity vector `Ω`); that is a useful shortcut
  to add **later, for orthonormal bases only**.  So the first cut should already
  target general bases; only the `Ω×` convenience is deferred.
- **F. Symbolic operator form (stretch).** Optionally expose the expanded
  operator `∇ = e_r ∂_r + (1/r) e_θ ∂_θ + e_z ∂_z` as an inspectable object (Q3),
  distinct from its evaluated result.

## Decisions (settled this iteration)

1. **Physical basis vectors are opaque symbolic atoms linked to their connection
   table.** Because they carry (or reference, via the chart) that table, they
   are always expandable — in WCS, or in any other basis (thread D).  A basis
   vector no longer *is* a Cartesian expansion; it *has* one on demand.
2. **Source of `∂_j eᵢ` = the pre-computed connection (Christoffel) table.**
   Keep `γ^k_{ij}` pre-computed per chart (we already do — M5); the
   differentiator substitutes `∂_j eᵢ → Σ_k γ^k_{ij} eₖ` from that table whenever
   a derivative meets a basis vector.  No re-derivation, no WCS detour.
3. **Target general (oblique) bases from the start** (thread E).  The connection
   machinery is basis-agnostic; the plan is to analyse and remove whatever is
   orthonormal-specific rather than bake in orthonormality.  Defer only the
   `Ω×` rigid-rotation convenience (orthonormal-only).
4. **Output symbol configurable at CS creation** (already supported): physical
   frames render `e_r, e_θ, e_z` (via `BasisNaming.value_names`), WCS renders
   bold `i, j, k` (via `BasisNaming.vector_symbols`).  No new mechanism needed —
   just wire it through so intrinsic results display with the right symbols.
5. **Drop M6.** The reference-frame expansion approach is the wrong move and
   should be removed, not kept alongside.  Its one legitimate use — getting WCS
   components — is recovered as a special case of general basis-to-basis
   expansion (thread D): if a user wants a vector in `i, j, k`, they expand its
   basis vectors in that basis.

## Implementation plan

The good news: most of the *contraction* machinery already works on symbolic
basis vectors — `simplify_basis_dot` (`eᵢ·eⱼ → δ_ij` orthonormal, `g_ij`
oblique), `simplify_basis_cross` (`eᵢ×eⱼ → √g ε_{ijk} e^k`), `reassemble`
(`Σ aᵢ eᵢ → a`), and the symbolic completeness fold (`Σᵢ eᵢ⊗eᵢ → I`).  The
connection `γ^k_{ij}` is already computed (M5).  The **one missing primitive** is
intrinsic differentiation of a basis-vector atom: today `∂` of a symbolic `eᵢ`
is `0` (it is an opaque tensor).  Once `∂_j eᵢ = Σ_k γ^k_{ij} eₖ` fires, the
operators fall out on the symbolic basis with no reference frame.

Effort in relative points.

### Phase 1 — connection registry + symbolic physical basis · 8 pts

- Precompute, when the chart's physical basis is built, the per-chart data the
  differentiator needs: the connection table `γ^k_{ij}` (from
  `connection_coefficients`, M5), the scale factors `h_i`, and the Cartesian
  embedding `eᵢ → (…)` in the reference frame (for later expansion, Phase 4).
- Register it in the `Context` keyed by the physical basis's `basis_id` (a
  `ConnectionData` map), so a bare `eᵢ` atom — which carries `basis_id` — can be
  connected back to its chart and `γ`.  The coordinate `q^j` already carries
  `(chart_id, slot)`; store `chart_id` in `ConnectionData` to match them.
- Expose the physical basis's vectors as **symbolic atoms** `eᵢ` (name `e`,
  `ConcreteIndex{value}` tagged with `basis_id`, rendering `e_r` via
  `value_names`) — the `e_dir` shape that already exists — rather than only their
  Cartesian expansions.  Keep the expansion reachable (Phase 4), not as the
  vector's identity (decision 1).
- Tests: the registry round-trips; `eᵢ` renders `e_r`; `γ` matches M5.

### Phase 2 — intrinsic differentiation of basis vectors · 8 pts (needs 1)

- Extend `steps::partial` / `diff` (`src/derivation.cpp`): recognise a
  registered basis-vector atom `eᵢ`; differentiating by coordinate `q^j` of its
  own chart substitutes `∂_j eᵢ → Σ_k γ^k_{ij} eₖ` from `ConnectionData`.  A
  coordinate of a *different* chart → cross-chart (defer to a later stage; `0`
  when provably independent, chain rule otherwise, mirroring P7 F3).
- Leibniz (already present) then differentiates `a eᵢ` correctly:
  `∂_j(a eᵢ) = (∂_j a) eᵢ + a Σ_k γ^k_{ij} eₖ`, with the scalar `∂_j a` handled by
  the P7 field rule.
- Tests: `∂_θ e_r = e_θ`, `∂_θ e_θ = −e_r`, `∂_r e_r = 0` (cylindrical/polar);
  `∂` of `f(r) e_r`.

### Phase 3 — intrinsic operators, on the chart basis · 8 pts (needs 2)

- Reformulate `gradient`/`divergence`/`rot`/`laplacian` as
  `∇⊙T = Σ_i (1/h_i) eᵢ ⊙ ∂_i T` evaluated **on the symbolic basis**: `∂_i`
  differentiates fields (P7) and basis vectors (Phase 2); `⊙` uses the existing
  `simplify_basis_dot` / `simplify_basis_cross`; grad's `⊗` leaves a symbolic
  dyad `eᵢ⊗eⱼ`.  No reference frame in or out.
- Result simplification: the symbolic completeness fold already gives `∇R = I`
  (now `Σ_i eᵢ⊗eᵢ` in the physical basis); `reassemble` cleans partial
  contractions.
- Verify the known outputs intrinsically, incl. the worked example
  `∇∇f(r) = e_r e_r ∂²_r f + (1/r) e_θ e_θ ∂_r f` (no trig); `div`, `rot`,
  `laplacian`; cylindrical/spherical.
- **Retire M6** (decision 5): remove the reference-frame `del_apply` /
  `reduce_dot` / `reduce_cross` path and the vibe-70 concrete-completeness plumbing
  that only served the WCS-expanded form (keep whatever the intrinsic path still
  uses).  Port/replace the M6 tests to the intrinsic results.

### Phase 4 — basis-to-basis expansion (WCS as one target) · 8 pts (needs 1)

- `express(v, target_basis)`: rewrite every `eᵢ` atom of a source chart into its
  representation in `target_basis` (decision-D primitive).  WCS is the special
  case using the stored Cartesian embedding (`e_r → cos θ i + sin θ j`); the
  general case expands `eᵢ^A` in basis `B` (compose through the shared reference
  frame first, then generalise).
- This *replaces* M6's WCS output as an explicit, on-demand change of basis.
- Tests: `express(e_r, WCS) = cos θ i + sin θ j`; round-trip a gradient to WCS
  and back; `express` between two curvilinear charts (through WCS).

### Phase 5 — oblique-basis generality · 5 pts (needs 3)

- Audit the intrinsic path for orthonormal-only assumptions (dot via `δ` vs
  metric `g_ij`; `√g` and cobasis `e^i` in the cross; index level
  raising/lowering).  The connection rule itself is already general; make the
  operators and contractions honour an oblique physical basis.
- Tests: a worked oblique example; confirm orthonormal remains a special case.

### Deferred

- **Ω× rigid-rotation form** `∂_q eᵢ = Ω × eᵢ` (orthonormal-only convenience,
  thread E).
- **Tree of coordinate systems** (CAD-style relative placement; expansion
  composes along the tree, thread D).
- **Cross-chart differentiation** (a field/basis of chart A differentiated by a
  coordinate of chart B — the P7 F3 inverse-Jacobian machinery).
- **Symbolic operator form** `∇ = e_r ∂_r + (1/r) e_θ ∂_θ + e_z ∂_z` (thread F).

### Effort summary & order

| Phase | What | Points |
|------|------|-------:|
| 1 | connection registry + symbolic basis | 8 |
| 2 | intrinsic `∂ eᵢ = γ^k_{ij} eₖ` | 8 |
| 3 | intrinsic operators; retire M6 | 8 |
| 4 | basis-to-basis expansion (WCS opt-in) | 8 |
| 5 | oblique generality | 5 |
| **Total** | | **37** |

Order: **1 → 2 → 3** is the spine (delivers the intrinsic `∇∇f(r)` and lets M6
go); **4** can proceed after 1 (needed for the "I want it in WCS" path and to
justify dropping M6's output); **5** hardens generality after the orthonormal
spine works.  The system stays alive at every step: build the intrinsic operators
alongside M6, reach parity, then remove M6 in Phase 3.

### Progress

- **Phase 1 DONE** (`04803be`): `BasisConnection` + Context registries;
  `Basis::direction(i)` symbolic e_i atoms; `physical_frame` builds + registers
  the connection table.
- **Phase 2 DONE** (`2f8ee37`): `diff` resolves `∂_j e_i = Σ_k γ^k_{ij} e_k` from
  the registry; Leibniz differentiates `f(r) e_r`.
- **Phase 3 DONE**: intrinsic `gradient`/`divergence`/`rot`/`laplacian` on the
  physical frame (`del_apply` over symbolic e_i, `reduce_dot`/`reduce_cross`
  symbolic, `fold_resolution_of_identity` extended to the frame atoms); M6's
  reference-frame `del_apply` and the Cartesian cross helpers removed.  Delivers
  `∇∇f(r) = ∂²_r f e_r e_r + (1/r ∂_r f) e_θ e_θ` (no trig), `∇R = I`,
  `rot(r e_θ) = 2 e_z`.  Chart frames are idempotent per chart via a Context
  cache; the connection/chart-frame registries are per-context (not shared
  across `new_context`, so reused chart ids across contexts stay isolated).
  Tests ported to the intrinsic frame; the M6-only P6 `rot(R×I)` showcase was
  removed (a field carrying the atomic `I` in a cross is not an intrinsic input;
  robustness there is a later concern).  Python: `chart.physical_frame()`,
  `Basis.direction(i)`.
- **Phase 4 DONE**: basis-to-basis expansion.  `BasisConnection` now stores each
  frame vector's reference (WCS) expansion; `to_reference(v)` rewrites every
  registered frame atom to its Cartesian form (`e_r → cos θ i + sin θ j`) — the
  explicit WCS opt-in, answering the earlier Q2.  `express(target, v)` is the
  general change of basis: bring `v` to the shared reference frame, then project
  onto `target`'s orthonormal frame (`w = Σ_k (w·e_k) e_k`) — so a WCS `i`
  becomes `cos θ e_r − sin θ e_θ`, and the WCS→frame round-trip recovers e_r.
  Python: `chart.to_reference(v)`, `chart.express(v)`.
- **Phase 5 DONE** (audit + generality of the machinery): the intrinsic
  *machinery* is basis-agnostic, verified —
  - **differentiation** `∂_j e_i = Σ_k γ^k_{ij} e_k` resolves through the
    connection registry for *any* registered basis (test on an oblique basis
    with `is_orthonormal() == false`);
  - the **dot** is metric-aware (`e_i·e_j = g_ij` on an oblique basis, not
    `δ_ij` — `simplify_basis_dot`) and the **cross** carries `√g` and the
    contravariant `e^k` (`simplify_basis_cross`);
  - the one orthonormal-only step, `fold_resolution_of_identity`
    (`Σ_k e_k⊗e_k = I` holds only for an orthonormal frame), is correctly
    guarded (no-op on an oblique basis).

### Follow-up polish (post-P5, from usage feedback)

- **`rot(R×I)` regression fixed.** A field built with the identity tensor and a
  cross (`R×I`) tripped `encapsulate` after M6 removal.  The frame-aware field
  prep is restored: `del_apply` now `reduce_field`s the input — expand each `I`
  into the frame's `Σ_k e_k⊗e_k` (on the symbolic direction atoms) and reduce the
  frame crosses via their ε expansion — so `∇×(R×I) = −2I` again (with `R` on the
  frame, or WCS for a Cartesian chart).  `nabla % (R%I)` then evaluates too.
- **Operator-form parens.** `tender.operators` now parenthesises a compound
  operand: `∇×(x i + y j + z k)`, `∇×((…)×I)`, while `∇f` / `∇·∇f` stay bare.
- **`collect_terms` step.** A curvilinear second gradient `∇∇(f(r) sinθ)` comes
  out as six raw terms; `td.collect_terms` groups addends by their dyad and sums
  the scalar coefficients — even non-numeric ones (unlike `fold_equal_addends`),
  collapsing to one term per `e_i⊗e_j` (four here).
- **Known gap — the symmetric 3-term form.** `A e_r e_r + B(e_r e_θ + e_θ e_r) +
  C e_θ e_θ` needs the coefficients of the transposed dyads recognised as equal.
  They are *algebraically* equal but not *structurally*, because `simplify_scalars`
  does not put a sum of fractions over a common denominator (`A/r² + B/r`
  stays split), so `algebraic_eq` on them is false.  The missing pieces are
  (1) **common-denominator fraction combination** in the scalar simplifier and
  (2) a **symmetric-dyad factoring** step keyed on the combined coefficients.
  Deferred; the 4-term `collect_terms` form is available now.

  **Scope boundary (surfaced):** the chart *operators* (`gradient`/`divergence`/
  `rot`) build an **orthonormal** physical frame `e_i = g_i/h_i`, because the
  charts model *orthogonal* curvilinear systems (diagonal metric).  A genuinely
  **skew** chart — non-diagonal metric, so the physical frame is oblique and the
  operator needs the contravariant cobasis `e^i` and the covariant-derivative
  form (`∇T = e^i ∂_i T`, index raising/lowering, `Ω×` deferred) — is a larger
  follow-up than this cut and is left as scoped future work.  The differentiation
  and contraction pieces it would build on are already general (above).
