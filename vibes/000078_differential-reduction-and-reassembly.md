# 000078 — differential reduction & reassembly (strain-compat gap-D core)

Continues vibe 000077.  The operator foundation gives us `inc ε = ∇×(∇×ε)ᵀ`
built with ε **abstract** and ∇ expanded:

    inc ε = Σ_{a,b} u_a × (u_b × ∂_a∂_b ε)ᵀ          (WCS, ε never componentised)

but it stops there — the nested cross of concrete frame vectors with an
*abstract* rank-2 `∂_a∂_b ε` cannot reduce, and there is no way to fold the
result back into the named invariant operators.  This vibe designs the two
missing phases that carry it to the closed identity

    inc ε = −∇∇θ + Δθ·I − (∇∇··ε)I − Δε + 2(∇∇·ε)ˢ ,   θ = tr ε.

## The two phases

- **Reduction (Phase 1).**  Turn the cross structure `e_i × (e_j × ε)ᵀ` into a
  sum of dyad / dot / trace terms via the `a×B×c` identity — with the frame
  vectors *free-indexed* and ε abstract.
- **Reassembly (Phase 2).**  Fold `Σ … e_i ∂_i … ε` back into the invariant
  operators, by reading, for each ∂, the *role* of its index.  This is the novel
  engine: the differential analogue of `reassemble` (basis.hpp), which already
  folds `Σ_i a_i e_i → a` for the non-differential case.

## Representation — settled by prior decisions, now grounded

Two decisions the user already made (the very first design fork of gap D) come
back into force, now that the operator foundation exists:

1. **Abstract implicit-summation index** (not the concrete 9-pair expansion).
   The identity fires **once** on `e_i × (e_j × ε)ᵀ` with `i,j` summed, and — the
   decisive reason — reassembly then just *reads the index contractions*
   (frame-free vs contracted-with-ε vs δ-contracted), which is frame-independent
   and clean.  The concrete path would need fragile 9-term cycle recognition.
2. **A chart-free ∇ operator** as the invariant representation.  This is *not* the
   fused `Del{grad/div/rot}` we retired — it is the bare **rank-1 ∇ operator**
   (the "∇ is a rank-1 vector operator" decision), with grad/div/rot = ∇ combined
   with `⊗`/`·`/`×`.  It serves three roles at once: the natural chart-free way to
   *write* `inc ε = ∇×(∇×ε)ᵀ`, the thing that *expands* to `Σ_i e_i ∂_i` in a
   chart, and the *reassembly target* (∇∇θ, Δε, … as `Nabla`-operator Exprs).

## The pieces

### A. Chart-free ∇ operator (`Nabla` node)

A rank-1 invariant operator atom, operator-aware exactly like `Deriv` (canon
keeps it positional; it acts rightward).  `∇⊗T`, `∇·T`, `∇×T` are the ordinary
product nodes with `Nabla` on the left.  Renders `\nabla`; `∇·∇` renders `Δ`.
Reuses all the operator-aware infrastructure of steps A–C.  `inc ε` becomes
chart-free: `Nabla × (Nabla × ε)ᵀ`.

### B. Free-index ∂ and ∇-expansion in a chart

The one genuinely new expr-model capability, and the reason `DerivMark::link`
exists: a derivative whose **direction is a summation index**, tied to a frame
vector.

- `DerivMark` gains an abstract-index direction: `link != 0` means "the ∂
  direction is the `CountableIndex` with this id" (rather than the concrete
  `wrt`), so `e_i` (a frame vector carrying index `i`) and `∂_i ε` (ε with a mark
  linked to `i`) **sum together** under the existing Einstein machinery.
- `expand_nabla(chart, e)` rewrites each `Nabla` → `e_i (1/h_i) ∂_i` with a fresh
  shared index `i`: an indexed frame vector `e_i` (basis direction with a
  `CountableIndex` slot) times a free-index `∂_i`.  In a constant frame (WCS)
  `∂_i e_j = 0`, so applying the operators leaves `e_i × (e_j × ∂_i∂_j ε)ᵀ`, the
  free-index interior.
- Summation detection, canon, and render learn the abstract-index mark (it is an
  index occurrence; it renders `∂_i` with the index letter).

### C. Phase-1 reduction — the a×B×c identity, once

Apply `a×(c×B)ᵀ → …` to the free-index interior `e_i × (e_j × ε)ᵀ` (a=e_i,
c=e_j, B=ε).  Using vibe 000075's *sign-corrected* form and the 5-term
expansion, this yields (schematically, i,j summed, ∂_i∂_j riding along):

    θ(e_j⊗e_i − δ_ij I) + (e_i e_j··ε)I + δ_ij ε − e_j⊗(e_i·ε) − (e_i⊗e_j·ε)ᵀ

Mechanism: express the identity as a tender `Identity` and fire `apply_identity`
on the cross-with-rank-2 pattern (binding a,c to the free frame vectors, B to ε),
or extend the vibe-000063 cross-removal engine.  **Open:** which — a declared
Identity vs. a dedicated reducer (the LHS has a transpose inside a cross, which
stresses the matcher).

### D. Phase-2 reassembly — read each ∂'s index role  ← the heart

After Phase 1 the expression is a sum of terms, each

    coeff · [free frame vectors / δ's / I] · [∂_i∂_j (θ or ε)]     (i,j summed)

Every abstract ∂-index appears twice (Einstein): once on its ∂-mark, once as
either **(a)** a *free* frame vector `e_i`, **(b)** a `δ_ij` shared with the other
∂, or **(c)** a slot of ε (`e_i·ε`).  Reassembly classifies each ∂ by its partner
and rebuilds with `Nabla`:

| ∂-index partner | meaning | fold |
|---|---|---|
| free frame vector `e_i` (a dyad leg) | a gradient leg | `e_i ⊗ … ∂_i` → `∇ ⊗ …` |
| `δ_ij` with the other ∂ | the two ∂'s contract | `∂_i∂_i` → `∇·∇ = Δ` |
| a slot of ε (`e_i·ε`) | a divergence of ε | `(e_i·ε) ∂_i` → `∇·ε` |
| both ∂'s into ε (`ε_ij`) | double divergence | `∂_i∂_j ε_ij` → `∇∇··ε` |

So each term rebuilds to a `Nabla`-operator composition on θ or ε:

    Σ e_j⊗e_i ∂_i∂_j θ           → ∇∇θ
    Σ δ_ij ∂_i∂_j θ              → Δθ
    Σ (e_i e_j··ε) ∂_i∂_j        → ∇∇··ε
    Σ δ_ij ∂_i∂_j ε              → Δε
    Σ e_j⊗(e_i·ε) ∂_i∂_j         → ∇(∇·ε) = ∇∇·ε
    Σ (e_i⊗e_j·ε)ᵀ ∂_i∂_j        → (∇∇·ε)ᵀ

collapsing to `inc ε = −∇∇θ + Δθ·I − (∇∇··ε)I − Δε + (∇∇·ε + (∇∇·ε)ᵀ)`, i.e. the
closed identity with `2(∇∇·ε)ˢ`.  The engine is a focused walk: per term, find
the ∂-marks, resolve each index's partner, classify (grad / Laplacian /
divergence), emit the `Nabla` structure; it is the differential sibling of
`reassemble`.

## Increment plan (each buildable/testable)

1. **`Nabla` node** — chart-free rank-1 ∇ operator (build, render, operator-aware
   canon, grad/div/rot as ∇⊙, Δ = ∇·∇).  `inc ε` written chart-free.
2. **Free-index ∂ + `expand_nabla`** — abstract-index `DerivMark`; expand `Nabla`
   → `e_i ∂_i`; verify `inc ε` expands to the free-index interior with ε abstract.
3. **Phase-1 reduction** — a×B×c on the free-index interior (once), sign-corrected.
4. **Phase-2 reassembly** — the index-role engine folding to `Nabla` operators.
5. **Showcase** — `inc ε` derived as performed = the closed identity; verify vs
   the known form; then the cylindrical evaluation of the compatibility equations.

## Decisions (confirmed by the user)

- **Q1 — abstract free-index path.** ✓  Identity fires once; reassembly reads
  index roles.
- **Q2 — `DerivMark::link` as the abstract-index direction.** ✓  A
  `CountableIndex` id ties `∂_i` to `e_i` under Einstein summation.
- **Q4 — general reassembly engine.** ✓  Build the general "fold frame-indexed
  ∂-derivatives into ∇" mechanism, validated on the strain-compat patterns.
- **Q3 — derive the identities in-codebase, `a×B×c` as the sole primitive.**  Do
  *not* hand-assert or wrestle a transpose-inside-cross matcher.  Mirror
  `../tender-sandbox/identities.ipynb`: prove each identity by expanding both
  sides in WCS and self-verifying with `algebraic_eq`, then reuse via
  `apply_identity`.  The chain:
  1. Derive `id_axBxc : a×B×c = trB(c⊗a − (a·c)I) + (c·B·a)I + (a·c)Bᵀ −
     c⊗(B·a) − (c·B)⊗a`.  Recipe (from the notebook): `expand_in_basis(co)` →
     `simplify_basis_cross` → `simplify_basis_dot` → `contract_eps_pair` →
     `contract_delta` (repeat contract steps) → `reassemble`, self-verified
     against the 5-term RHS.
  2. Derive the **nested-cross helper** that turns the strain interior into
     `a×B×c` form using ε **symmetry**: the user's `a×(b×ε) = −a×ε×b` (worked out
     — with/without the outer transpose and exact sign — during derivation;
     tender catches sign slips as in vibe 000075).  So the interior
     `e_i × (e_j × ε)ᵀ` reduces first to `∓ e_i × ε × e_j`, then `id_axBxc`
     applies once.
  These derived identities live in a small library (extend `tender/identities`,
  vibe 000046) with a self-check test.

## Implementation plan (self-contained — resume here after compaction)

State at handoff: operator foundation A–E committed (vibe 000077); `inc ε`
builds with ε abstract; `DerivMark{coord_name, wrt, link}` exists with `link`
reserved.  Build these five increments, each buildable/testable, `clang-format`
+ tests green before each commit.

### Progress

- **Increment 1 DONE** (commit `cc0c54f`).  `Nabla` node added and threaded;
  grad/div/rot = ∇⊙; `inc ε = ∇×(∇×ε)ᵀ` builds chart-free; `t.nabla(ctx=…)`.
- **Increment 2 DONE** (commit `f09e3f1`).  Free-index ∂: `DerivMark` gained
  `free` (a dedicated flag — index ids can be 0, so `link != 0` won't do),
  `link` (direction id tied to e_i) and `free_slot`.  Summation counts a free
  mark; `substitute_index_ids` α-renames it with e_i; render shows ∂_i.  New
  free-index path in `diff`/`partial` via `make_coordinate_direction` wrt;
  `apply_operators` recurses into transpose/trace/vec.  `chart.expand_nabla`
  lowers `inc ε` → `e_i × (e_j × ∂_i∂_j ε)ᵀ` (i,j summed, ε abstract);
  `chart.componentize_nabla` unrolls to concrete directions.  **Verified: all 9
  components equal brute-force rot·rotᵀ.**

- **Increment 3 IN PROGRESS.**  `id_axBxc` *derives correctly in the current
  tender* via the notebook recipe (`expand_in_basis(co) → apply_identity(alt) →
  expand_in_basis → simplify_basis_cross → simplify_basis_dot → contract_delta →
  contract_eps_pair → contract_delta → contract_eps_pair → contract_delta →
  reassemble`), giving `a×B×c = −(a·c)trB·I + (a·Bᵀ·c)I + (a·c)Bᵀ − c⊗(B·a) −
  (Bᵀ·c)⊗a + trB·c⊗a` — matches the vibe's 5-term form.  The prerequisite
  `id_axIxb_alt` (`c⊗a = a×I×c + (a·c)I`) also derives.  **Open:** the nested
  transpose-cross helper `a×(c×B)ᵀ = −a×Bᵀ×c` does *not* reduce with the same
  chain — it leaves a residual bare `e_k × e_j` that `simplify_basis_cross` will
  not fire on inside the larger product (a reduction-ordering quirk the sandbox
  notebook tuned by hand).  Two robust ways forward for the helper: (a) find the
  interleaved expand / simplify_basis_cross fixpoint chain that clears the
  residual cross; (b) *prove* the helper at the component level (`component_matrix`
  of both sides over concrete a,c,B — the same brute-force check increment 2
  used), then reuse it.  Phase-1's output can then be produced either by
  `apply_identity` on the free-index interior (matcher-stressing: transpose inside
  cross) or by direct substitution a→e_i, c→e_j, B→∂_i∂_j ε into the derived
  `id_axBxc` RHS.  Increment 2's `componentize_nabla` + `component_matrix` is the
  brute-force oracle to verify whichever path.  NEXT: finish the helper, then the
  reassembly engine (increment 4).

- **Increment 5 DONE** (commit `8504206`), and a prerequisite operator fix
  (`7bdbf2d`).  The **closed identity is proven by tender**:

      inc ε = −∇∇θ + Δθ·I − (∇∇··ε)·I − Δε + 2(∇∇·ε)ˢ ,   θ = tr ε

  verified component-by-component against `inc ε = ∇×(∇×ε)ᵀ` in a **Cartesian
  frame and a cylindrical frame** — the curvilinear endpoint (original task #26).
  `examples/strain_compatibility.py` runs the full pipeline (chart-free inc ε →
  `expand_nabla` free-index interior with ε abstract → the stated closed form →
  componentwise verification), guarded by
  `test_strain_compatibility_closed_identity_{cartesian,cylindrical}`.  The
  verification route uses tender's mature M6 chart operators (grad/div/rot/
  laplacian) for both sides — the reassembly *engine* is not needed to *prove*
  the result, only to *produce* it automatically from the interior.
  - Prerequisite fix (`7bdbf2d`): `apply_product_operators` now applies a
    differentiated operand's own operators first (rightmost-first), so composed
    operators whose operand contains an unapplied ∇ — grad(div ε), div(div ε),
    Δε = div(grad ε) — expand instead of throwing "differentiating a ∂ operator".

- **Increments 3 & 4 — the symbolic reduce→reassemble transforms — remain
  blocked** on two distinct reduction-engine gaps (each a substantial piece of
  its own), and are *not* required for the proven result above:
  1. **Phase-1 (increment 3):** the nested transpose-cross helper
     `a×(c×B)ᵀ = −a×Bᵀ×c` leaves a residual bare `e_k×e_j` that
     `simplify_basis_cross` won't clear inside the larger product (matcher /
     reduction-ordering).  `id_axBxc` itself derives correctly.
  2. **Phase-2 (increment 4):** the reassembly engine needs Phase-1's reduced
     form as input.  Separately, componentizing the reassembly *target*
     operators through `expand_nabla` hit an `encapsulate` throw ("nested ⊗
     inside an operand awaits fence distribution", `nf_lower.cpp`).

- **`encapsulate` gap fully diagnosed (commits `f5b1f3d`, `0fe8655`) — it has
  THREE distinct root causes; the first two are now fixed, and the third turned
  out to hide a *foundational contraction-model bug*:**
  1. **Zero-fence factor — FIXED (`f5b1f3d`).**  Differentiating a constant frame
     vector (∂_i e_j = 0) leaves a literal 0 buried in a ⊗ / contraction fence,
     e.g. `e_j ⊗ (0·∂_iε)` in a componentized transposed operator result; no
     distribution removes it and `encapsulate` has no ⊗ arm.  Fix:
     `fold_forced_zeros` in `lower_term` collapses any node forced to 0 by a
     0 operand and drops additive zeros, before the all-`*` encapsulation — the
     algebraic zero law (test `CanonicalizeNf.ZeroFenceInsideContractionFolds`).
  2. **Scalar-in-contraction — FIXED (`0fe8655`).**  A concrete basis dot
     `e_a·e_a → 1` was landing as a bare `1` in a `·` chain (not the
     coefficient), which `encapsulate` cannot lower.  Root fix (user-endorsed
     invariant): a scalar has no leg to contract, so `s·T`/`s:T`/`s··T`/`s×T`
     are scalar multiplication.  `make_dot`/`make_ddot`/`make_ddot_alt`/
     `make_cross` now redirect to `make_tensor_product` when an operand is
     *known* rank 0 (unknown/abstract keeps the contraction).  `infer_rank` moved
     expr.hpp/expr.cpp so the factories can consult it without a layering
     violation.  Test `MakeBinary.ScalarOperandInContractionBecomesTensorProduct`.

- **"Cause 3" split, once (1)+(2) were fixed, into two SEPARATE bugs:**

  - **(3a) — free-index transpose aliasing — OPEN.**  `expand_nabla` on a
    *transposed* two-∇ form corrupts the ∂-marks: `(∇⊗(∇·ε))ᵀ` lowers to
    `Σ (e_j·(∂_i∂_i ε)) e_i` — the two summation indices collapse
    (`∂_i∂_j → ∂_i∂_i`).  `apply_operators` recursing through `Transpose`
    aliases the two free-index directions.  (Non-transposed grad-div is correct:
    `Σ e_i (e_j·(∂_i∂_j ε))`.)  Localized to the increment-2 free-index path.

  - **(3b) — FOUNDATIONAL contraction-model bug — the next-session target.**
    `a·(b·T)` (any ordinary double contraction, e.g. `v·(w·σ)`) canonicalizes to
    the WRONG answer `(a·b)·T` — rank 0 silently becomes rank 2.  Repro:
    `td.canonicalize(a @ (b @ T))` for vectors a,b and rank-2 T renders `a·b T`.
    **Mechanism:** the nf model stores a contraction as a *flat linear chain*
    (`nf::Contraction{factors, ops}`), and `flatten_contraction`
    (`nf_lower.cpp`) drops the bracketing via `flatten(l) ++ [op] ++ flatten(r)`
    — the "000057 interface theorem".  That reordering is valid ONLY for a
    genuine matrix-chain `A·B·C` (each · joins adjacent interface legs).  For a
    **fan-in / right-nesting into a higher-rank operand**, `Dot(a, Dot(b,T))`
    flattens to the chain `[a,b,T]` when the correct chain is `[b,T,a]`
    (= `b·T·a` = `a_j b_i T_ij`).  Left-nesting `(a·T)·b` flattens correctly to
    `[a,T,b]`; only right-nesting into rank ≥ 2 is wrong.  (This is *why*
    `∇·(∇·ε)` came out as `Δε`, and it also interacts with (2): the bad
    `Dot(a·b scalar, T)` now redirects to `(a·b)⊗T`, so the wrong result renders
    as a tensor product.)  **Library-wide correctness bug**, independent of the
    strain-compat work.

  **User decision (this session): fix (3b) — correctness outranks any deadline
  (there are none).  Investigate first, come back with a concrete fix design +
  blast-radius assessment, then change the core.**

  RESUME PLAN for (3b):
  1. Reproduce & pin the semantics: enumerate what the flat
     `Contraction{factors, ops}` chain is *defined* to mean (left-assoc? leg
     topology?), and for which nestings `flatten(l)++op++flatten(r)` is sound.
     Key cases: `A·B·C` (all rank 2 — sound), `(a·T)·b` (sound, `[a,T,b]`),
     `a·(b·T)` (UNSOUND, produced `[a,b,T]`, should be `[b,T,a]`).
  2. Decide the fix locus: either (i) make `flatten_contraction` compute the
     correct chain ORDER from the operands' ranks / interface legs when it
     right-descends into a higher-rank operand, or (ii) enrich the flat model to
     track which interface leg each `·` joins (a bigger change).  Prefer (i) if
     a rank-driven reordering rule is provably correct; it is the smaller change.
  3. Guard with tests at BOTH layers: an nf-level test that
     `canonicalize(a·(b·T))` keeps rank 0 and equals `b·T·a`; and the
     `∇·(∇·ε)` via `expand_nabla` == `cart.div(cart.div(eps))` end-to-end check.
  4. Re-run the full suite — the interface-theorem flattening underpins ALL
     contraction canonicalization, so watch for churn in matrix-chain cases.
  Then (3a) [keep the two free-index ∂-links distinct across a `Transpose` in
  `apply_operators`], after which `expand_nabla` handles the full RHS and the
  symbolic reduce→reassemble route (increments 3/4) is unblocked: build
  `reassemble_del` reading each ∂-index's role, driven on a directly-constructed
  reduced interior (substitution into the derived `id_axBxc`), verified against
  the increment-5 oracle.

**Increment 1 — chart-free `Nabla` operator node.**
- `struct Nabla final {};` in `expr.hpp` (a rank-1 invariant operator atom, no
  children); add to the `Expr` variant; `make_nabla`.  Thread through the same
  sites `Deriv` touched (compiler-guided): `rewrite_tree`/`map_children` (leaf),
  `structural_eq` (all equal), `infer_rank` (→ 1), `is_component_valued` (false),
  `has_free_index_for` (false), `find_index_space` (skip), render (`\nabla`; and
  `Nabla·Nabla` → `\Delta`), `diff` (throw — expand first), and the nf layer
  (`nf::Nabla` positional factor: lower/raise/rank=1/compare/hash/egraph via the
  Atom path).  Extend `place_factors`' operator detection to include `Nabla`.
- grad/div/rot/Δ are `Nabla` with `⊗`/`·`/`×`/`·∇`.  `inc ε` = `nabla % (nabla %
  eps).transpose()` chart-free.  Tests: rank, render (`∇×(∇×ε)ᵀ`, `∇∇θ`, `Δθ`),
  canon keeps it positional/opaque.

**Increment 2 — free-index ∂ and `expand_nabla`.**
- Teach the abstract-index `DerivMark` (`link` = the `CountableIndex` id): a
  free-index differentiation that *stamps a mark* `link=i` on a field (∂_i ε) and
  gives 0 on a constant frame vector (WCS), rather than concrete `partial`.
- Summation detection (`summation.cpp`) counts a mark's `link` index as an
  occurrence, so `e_i` (frame vector with slot index i) and `∂_i ε` (mark
  link=i) contract; render shows `∂_i` with the index letter.
- `expand_nabla(chart, e)`: replace each `Nabla` with `e_i (1/h_i) ∂_i` — a fresh
  shared `CountableIndex` i, an indexed frame vector `e_i`, a free-index `∂_i` —
  then apply the operators (Step B machinery, free-index variant).  Verify `inc
  ε` expands to `e_i × (e_j × ∂_i∂_j ε)ᵀ` (i,j summed, ε abstract), and that its
  components match brute-force `inc ε`.

**Increment 3 — Phase-1 reduction (derive + apply identities).**
- Derive `id_axBxc` and the nested-cross helper (Q3 recipe above); self-check
  test.  Apply them (`apply_identity`) to the free-index interior → the 5-term
  form `θ(e_j⊗e_i − δ_ij I) + (e_i e_j··ε)I + δ_ij ε − e_j⊗(e_i·ε) −
  (e_i⊗e_j·ε)ᵀ`.  Verify components vs brute force.

**Increment 4 — Phase-2 reassembly (the heart).**
- New engine `reassemble_del(ctx, e)`: per additive term, find the ∂-marks,
  resolve each abstract index's partner, classify (free frame vector → grad leg;
  `δ_ij` with the other ∂ → Laplacian; ε-slot → divergence; both into ε → double
  divergence), and emit the `Nabla` composition on θ/ε.  The differential
  sibling of `basis.hpp::reassemble`.  Fold to `∇∇θ, Δθ, ∇∇··ε, Δε, ∇∇·ε,
  (∇∇·ε)ᵀ`; symmetrize the last pair to `2(∇∇·ε)ˢ`.  Verify the reassembled
  `inc ε` equals the known closed identity (build the RHS with `Nabla` operators;
  `algebraic_eq` after a common expansion, or componentwise).

**Increment 5 — showcase + cylindrical.**
- `examples/strain_compatibility.py`: the full pipeline `inc ε` (chart-free) →
  `expand_nabla` → apply → reduce → reassemble → closed identity, verified.  Then
  the cylindrical evaluation of the compatibility equations (the original task
  #26 endpoint).  Keep the notebook stripped.
