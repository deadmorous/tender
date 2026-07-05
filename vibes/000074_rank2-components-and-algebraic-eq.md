# 000074 — rank-2 chart.components and the algebraic_eq fraction fallback

Follow-up items from vibe 000073 (user: "Let's do tomorrow, too. I already need
it for next example" — the next example being the strain compatibility
condition ∇×(∇×ε)ᵀ = 0).

## 1. `chart.components` for rank-2 tensors

`components(chart, v)` projected a *vector* onto the frame.  For the coming
rank-2 work we need the physical component **matrix**:

    m[i][j] = e_i · v · e_j

Implementation:

- C++ `component_matrix(ctx, chart, v)` in `src/chart.cpp`, declared in
  `chart.hpp`.  Same expand-first policy as `components`: `expand_fields`
  materializes an abstract field into Σ T_kl e_k ⊗ e_l before projection, so
  the entries come out as the minted physical components with symmetry folded
  (`m[1][0]` *is* `T_rθ`).  Each row computes `e_i·v` once, then dots with
  `e_j` per column.
- `components` now throws `std::invalid_argument` on a rank-≥2 input instead
  of silently returning vector-valued "components" (it used to hand back
  `e_i·T`, which nothing downstream expected).
- Python: `chart.components(v)` dispatches on `infer_rank` — rank 2 returns
  the nested 3×3 list, rank 1 the flat list as before.  One method, both
  shapes, matching how the user thinks about "the components of T".

### Bug found on the way: `0 · e_j` in reduce_dot

`component_matrix` of an explicit dyad (e.g. `e_θ ⊗ e_θ`) crashed with
"encapsulate: unsupported factor node".  Cause: a fully projected-out row
(`e_r·(e_θ⊗e_θ) → 0`) feeds the scalar `0` back in as a dot operand, and
`Dot(0, e_j)` is malformed for canonicalize's encapsulate — the exact class of
problem vibe 000073 hit with `T_ij·0` connection terms.  Fix mirrors
`reduce_cross`'s existing `0×b` guard: `reduce_dot` now folds a dot with a
zero operand to 0 right after `distribute_bilinear`, before any basis
reduction.

## 2. `algebraic_eq` sees through fraction shapes

Theory T0 (canonicalize + structural_eq) keeps `x/r + y/r` and `(x+y)/r`
apart — they are different canonical forms.  This forced the
`cyl_equilibrium` example into a `same(a, b)` workaround:
`simplify_scalars(a - b).latex() == "0"`.

Decision: fold that fallback into `algebraic_eq` itself.

- Fast path unchanged: `structural_eq(canonicalize(a), canonicalize(b))` —
  every previously-true comparison stays on the cheap path.
- Fallback (only when T0 says "different"): `simplify_scalars(canonicalize(a −
  b))` is checked against the literal scalar 0.

### Recursion trap

`simplify_scalars`' own helpers (trig-square pairing in the Pythagorean fold,
factor bagging in fraction cancellation / LCD collection) compare subterms
with `algebraic_eq`.  Feeding `simplify_scalars` into `algebraic_eq` would
make the two mutually recursive.  Resolution: those three internal call sites
now use a file-local `algebraic_eq_t0` (the old T0-only body); only the public
`algebraic_eq` carries the fallback.  So `simplify_scalars` never re-enters
itself, and internal comparisons keep their original (cheap, T0) semantics.

## Fallout

- `examples/cyl_equilibrium.{py,ipynb}`: the textbook cross-check lost all
  three helpers (`reduce_scalar`, `Tij`, `same`) — it is now
  `Tc = cyl.components(T)` + `td.algebraic_eq(div_i, textbook_rhs)` directly,
  with `d = td.partial` unwrapped (no simplify_scalars needed).  The notebook
  gained a cell rendering the component matrix and doing the θ-equation check.
- Tests: `Chart.ComponentMatrixOfSymmetricField`, `Chart.ComponentMatrixOfDyad`
  (the 0·e_j regression), `Derivation.AlgebraicEqFoldsFractionShapes`;
  Python `test_components_of_rank2_returns_matrix`,
  `test_component_matrix_of_explicit_dyad`,
  `test_textbook_check_needs_no_fraction_workaround`,
  `test_algebraic_eq_folds_fraction_shapes`.
