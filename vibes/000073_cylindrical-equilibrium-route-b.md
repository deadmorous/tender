# 000073 — Cylindrical equilibrium: making a real hand-derivation work (Route B)

## Context

The goal shifted from feature-building to **practical usefulness**: tender must
automate the derivations Stepan used to do by hand, and stop stumbling on
"trivial" steps.  To close the gap between our mental models, Stepan supplied a
scanned hand derivation — the balance equations of a continuous medium in
cylindrical coordinates (`vibes/images/cyl-equilib_0{1,2,3}.png`), deriving eq
(7): `∇·T` for a rank-2 stress `T`, expressed on the physical frame
`e_r, e_θ, e_z`, leading to the boiler (hoop-stress) formula.

## The mental model (how Stepan derives)

The load-bearing move is **(4) + (6) → (7)**:

> `∇·T` with `T` given by its frame components means: substitute the component
> expansion (4) into `∇·`, then apply the product rule to **both** the scalar
> components **and** the moving basis vectors, using the basis-derivative table
> (6): `∂_θ e_r = e_θ`, `∂_θ e_θ = −e_r`.  Then contract the leading dot
> (`e_i·e_j = δ_ij`) and collect by output vector.

The connection terms are what make the curvilinear divergence differ from the
flat one.  Everything after (7) is deleting terms.

## Two routes, both were broken

- **Route B** — expand `T` in the basis explicitly, *then* apply `div`.  Mirrors
  the hand derivation (4)→(7).  This is the route we implemented.
- **Route A** — keep `T` abstract, expand under the hood only when needed.
  Deferred (task): should reduce the typing Route B needs.

## Gaps found and fixed

**Gap 2 — `div` crashed on any dyad whose contracted (first) leg is `e_θ`**
(`div(a e_θ⊗e_θ)` etc.), with `encapsulate: unsupported factor node`.  Root
cause: the connection substitution `∂_θ e_θ = −e_r` produced a **`Negate`
wrapping a `⊗`** inside a dot operand, and `steps::distribute_contraction`'s
fence-distribution only matched a *bare* `TensorProduct`, not one behind a
`Negate`.  Fix: peel `Negate`s off both operands in the `distribute` helper,
distribute, then re-apply the sign (`src/derivation.cpp`).  Also wired
`distribute_contraction` into `reduce_dot` (`src/chart.cpp`) so the dot pushes
through the fence before `canonicalize`.

**Gap 1 (enabler) — `expand_in_basis` minted components as constants, not
fields.**  A component of a field is itself a field; the minted `T_ij` dropped
the source's `FieldDeps`, so `∂_r T_rr = 0` and `div` silently lost every
derivative term.  Fix: carry the source tensor's `field` trait and
`field_derivs` onto each component (`src/basis.cpp`).  This makes the faithful
Route B a near one-liner:
`T_cyl = canonicalize(unroll_sums(expand_in_basis(T, frame, Covariant)))`
— literally step (4) — and then `cyl.div(T_cyl)` is step (7).

**Gap 4 (cosmetic) — subscript spacing.**  `slots_str` concatenated concrete
index labels raw, so `\theta` + `r` rendered as the invalid control word
`\thetar`.  Fix: a shared `append_idx` helper inserts one space after a LaTeX
control word when the next label starts with a letter, leaving plain Latin
`rr`/`zr` unspaced (`src/render.cpp`).

**Gap 3 — `physical_frame()` vs `physical_basis()` minted different basis
identities** (`structural_eq` was `False` though both render `e_r`).  Cause:
`physical_frame` cached its basis (via `chart_frame`) but `physical_basis` built
a fresh `make_orthonormal_basis` — a new `basis_id`, baked into each `e_i`'s slot
tag — on every call.  Passing `physical_basis()` where the operators used
`physical_frame()` silently no-op'd `simplify_basis_dot`.  Fix (`src/chart.cpp`):
`physical_basis` is now the cached, idempotent frame builder (keyed by chart id +
geometry fingerprint); `physical_frame` delegates to it and only adds the
connection table (skipped if already registered for that basis).  Both now return
the identical Basis in either call order, and the user's *original* pipeline
(expand/reduce on `physical_basis()`) reduces correctly.

## Verification and a finding

Tender's `∇·T` was cross-checked term-by-term against the **standard textbook**
cylindrical equilibrium equations (general rank-2, contracting the first index)
and matches exactly, including the `(T_rθ + T_θr)/r` shear term in the
θ-component — i.e. `2 T_rθ/r` for a symmetric stress.

**Note on the hand derivation:** the hand (7)/(8) θ-equation carries only
`(1/r) T_rθ` (coefficient 1), whereas the correct/textbook value is `2 T_rθ/r`.
One of the two connection contributions (`∂_θ` acting on the *second* basis
vector of the off-diagonal dyad) appears to have been dropped by hand.  It does
not affect the final boiler formula (10), which uses only the r-equation.

## Deliverables

- `examples/cyl_equilibrium.py` + `.ipynb` (stripped) — Route B end-to-end,
  reproducing (4)→(7) and the boiler formula, asserting the textbook match.
  Added to `examples/Makefile`.
- Tests: C++ `Chart.CylindricalDivEThetaDyad`,
  `Chart.ExpandInBasisComponentsAreFields` (also covers the render spacing);
  Python `test_div_of_e_theta_dyad_does_not_crash`,
  `test_expand_in_basis_components_are_fields`,
  `test_cylindrical_divergence_matches_textbook`,
  `test_mixed_coordinate_subscript_spacing`.
- 747 C++ + 210 Python tests pass; coverage 90.4% lines.

## Still deferred

- **Route A** — abstract `T`, expanded under the hood; a one-call
  `cyl.components(expr, variance)` (or `.div` returning already-reduced frame
  components) to remove the expand → basis-dot → δ-eval → fold pipeline the
  Route B example still spells out by hand.
- Declaring a **symmetric** field (so `T_θr = T_rθ` folds automatically) is not
  yet exposed; the example uses a general `T` and notes the symmetric reduction.
