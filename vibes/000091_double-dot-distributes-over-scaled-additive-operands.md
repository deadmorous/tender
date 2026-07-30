# 000091 — expand_double_dot must distribute over scaled / signed sum operands

Status: **DONE**

## The goal (rot-cyl Lagrange functional)

Toward the rotating-cylinder energy method: in the cylindrical chart, with a
displacement field `u = u e_r + v e_θ + w e_z`, the elastic strain energy density
is a *scalar*

    Π = ½ T ·· ε,   T = λ(∇·u) I + μ(∇u + (∇u)ᵀ),   ε = (∇u + (∇u)ᵀ)/2.

The user wants Π reduced to a plain scalar in `u,v,w` and their partials — no
vectors, no dyads. Writing `Pi = (T // eps)/2` and evaluating in the chart left
the double contraction `··` **un-reduced**: `cyl.evaluate` lowers `∇` and expands
`T`, `ε` into frame dyads, but `expand_double_dot` then did nothing and
`canonicalize` raised *"nested ⊗ inside an operand awaits fence distribution"*.

## Root cause

`dd_expand` (behind `expand_double_dot`) distributed a contraction only over a
**bare `Sum`** operand (and pulled out summation binders). The strain-energy
operands are not bare sums:

- `ε = (…)/2` is a **`ScalarDiv`** wrapping a sum — `split_dyad` peels
  *multiplied* scalars off a single dyad but not a divisor, so `dyad/2` stalled.
- A curvilinear `∇u` expansion carries **signed addends** (`− v/r e_θ⊗e_r`),
  i.e. `Difference`/`Negate` nodes, which the Sum-only distribution skipped.
- `λ(∇·u)·I`, once `I` resolves to `Σ e_k⊗e_k`, is a **scalar × sum** —
  neither a bare `Sum` nor a single dyad, so `split_dyad` returned `nullopt`.

Any one of these left an intact `··` node, which then tripped `canonicalize`.

## The fix

`dd_expand` now normalises each operand over the **same** additive set that
`expand_unary` already handles for `tr`/`vec`/`transpose` — `Sum`, `Difference`,
`Negate`, `ScalarDiv` — plus a **scalar-weighted additive** peel `s·(A+B) →
s·(A··r + B··r)` (flatten factors; if exactly one non-scalar factor and it is
additive, pull the scalars out front and recurse). The bilinear contraction
commutes through all of these, so the moves are unconditionally valid.

With that, the natural `T // eps` route reduces in one pass:

    P = cyl.evaluate((T // eps)/2)
    P = td.expand_dyad_ops(P)          # (a⊗b)ᵀ → b⊗a
    P = td.expand_double_dot(P)        # (a⊗b)··(c⊗d) → (a·d)(b·c), now fully
    P = tb.simplify_basis_dot(P, fr)   # e_i·e_j → δ_ij
    P = td.canonicalize(P); P = td.unroll_sums(P); P = td.eval_delta_concrete(P)
    P = td.simplify_scalars(P)         # scalar Π, no vectors

Note: the invariant identity `I` must be resolved **in the frame**
(`Σ e_k⊗e_k`, the abstract physical directions), not via `tb.expand_identity`,
which drops `I` to the concrete WCS `(i,j,k)` and cannot then contract against
the abstract `e_r,e_θ,e_z`. Building `T` with `Ifr = e_r⊗e_r+e_θ⊗e_θ+e_z⊗e_z`
does the job. (`chart.cpp`'s `expand_identity_frame` already does the in-frame
resolution internally but is not on the Python surface.)

## The robust alternative — component matrices

`cyl.components(cyl.evaluate(ε))` returns the 3×3 physical strain matrix
directly (handling `I` on its own), reproducing the textbook cylindrical strain
tensor (`ε_rr = ∂_r u`, `ε_θθ = (u+∂_θ v)/r`, `ε_rθ = ½(∂_θ u/r − v/r + ∂_r v)`,
…). Then `Π = ½ Σ_ij T_ij ε_ij` is a one-liner, and the compact physical form

    Π = ½ λ (∇·u)² + μ Σ_ij ε_ij²

is `algebraic_eq`-equal to it (after `expand_products` clears the squared-sum
`Power` nodes that `simplify_scalars` leaves opaque). Both routes agree.

## Tests

`tests/derivation_test.cpp` — `ExpandDoubleDot.{CommutesThroughScalarDiv,
DistributesOverDifference,PeelsScalarOffScaledSum}`; full suite 848/848.

The showcase lives in `tender-sandbox/rot-cyl-lagrange.ipynb`.
