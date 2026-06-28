# 000067 — Basis-aware indices and coordinate-system naming

Design discussion before Stage 4/5 (roadmap 000038), and a prerequisite for
sub-expression / subspace splitting. Two gaps in the coordinate↔invariant layer
(vibes 000036, 000049) must close first.

## Gap 1 — a coordinate/basis-vector must know which basis it came from

Today `wcs()`, `cylindrical()`, `spherical()` all share the global `space_3d()`
singleton and emit the generic symbol `e` over it, so two orthonormal 3D bases
are indistinguishable at the index level. Needed for: (a) using more than one
basis in one expression without coordinate ambiguity, (b) basis-parameterized
steps acting only on *their* basis's coordinates.

### Rejected: enrich/per-basis `IndexSpace`

First idea was to give each basis its own `IndexSpace` (pointer = basis) and add
a "same-space required to contract" guard. **Killed by the rotation tensor**
`R = e_i ⊗ e'_i` (`e_i ∈ A`, `e'_i ∈ B`): `i` legitimately sums over the *shared*
range across two bases. So:

- **Index space = the summation range, and it is shared.** It is not the basis.
- **Einstein summation is purely formal** — repeated index ⇒ sum over range,
  *basis-blind*. `R` is fine; even `a^i_A b_i_B` is a legal formal sum (it just
  won't simplify to `a·b`). Therefore the basis label must **not** enter the
  summation key. What it governs is the basis-*aware rewrites*
  (`simplify_basis_dot`/`_cross`, `expand_in_basis`, `reassemble`), which must
  only fire within one basis. **Label ⟂ summation.**

### Decision: basis is a per-INDEX attribute (approach B)

Per-object tagging is insufficient: the **deformation gradient**
`F = F_{iJ} e_i ⊗ E_J` (`i` current basis, `J` reference basis) and the overlap
`Q_{iJ} = e_i^A·e_j^B` are single component objects with two indices from two
different bases. So basis identity lives on each **index slot**.

- `IndexSlot` gains `int basis_id` next to `level/realm/space`. `0` = none /
  basis-unaware (the default → δ, ε, and all existing hand-written tensors are
  untouched and stay neutral).
- **Summation explicitly ignores `basis_id`** (formal Einstein) ⇒ `R` sums.
- `structural_eq` / canonical ordering / hashing **include** `basis_id`, so
  `e_i^A ≠ e_i^B`. The matcher treats an unset (`0`) `basis_id` in a *pattern*
  as don't-care (basis-generic identities still match basis-tagged terms).
- Basis-aware steps filter by `basis_id == step's basis`: `e_i^A·e_j^A → δ_ij`
  (δ emitted with `basis_id = 0`, basis-unaware); `e_i^A·e_j^B` left as the
  overlap. "Basis-unaware = `basis_id 0`" — the clean form of the δ/ε caveat.

### Storage of the id → Basis mapping: the Context

The `Context` owns the registry (it already mints index ids and owns
`IndexSpace`):

- `Context` owns the `Basis` objects (`ctx.make<Basis>(…)`; lifetime-safe — a
  `Basis` only holds `Expr const*` already living in the ctx) and a
  `vector<Basis const*>` indexed by id; `ctx.basis(id)` is an O(1) lookup.
- `make_*_basis(ctx, …)` allocates an id, constructs the `Basis` carrying it,
  registers it, and returns **`Basis const*`** (it returns by value today — the
  one API ripple, propagating to `coord_system` and Python `PyBasis`).
- Each `Basis` stamps the slots it emits (`covariant_vector`,
  `contravariant_vector`, and the coordinate indices produced by
  `expand_in_basis`) with its own id. Render resolves
  `slot.basis_id → ctx.basis(id) → label / value_names`.

Pointer-speed lookup at render, id-stability on the slot.

## Gap 2 — familiar coordinate-system notation

`ConcreteIndex` renders as the bare integer; the space schema names only
dummies. Want cylindrical to read `r, θ, z`, WCS to read `i, j, k`, etc.,
while the underlying values stay `{1,2,3}` for summation. Target readability
(also exercises Stage-4 metric / Stage-5 operators):

    ∇ = e_r ∂_r + (1/r) e_θ ∂_θ + e_z ∂_z      (1/r is a Stage-4 Lamé factor;
                                                 ∂ is Stage-5 — not this gap)

### Two orthogonal naming tracks on the Basis

1. **Index-value naming** — a `value_names` list on the `Basis` (populated by
   the coordinate system, which knows the coordinate letters). Maps a *concrete*
   index value to a letter (`1→x` / `1→r`). Governs how concrete indices print
   on **both** coordinates (`aˣ`) and indexed vectors (`e_x`). Numeric is the
   always-available fallback. (= the user's modes 1 and 2.)
2. **Vector-symbol override** — an optional per-direction standalone symbol list
   on the `Basis`. When set, a concrete *basis vector* prints as that symbol
   (`i, j, k`) instead of `e`+index. Vector-only: WCS coordinates still read
   `aˣ`/`a¹`. User opt-in. (= mode 3.)

Render precedence for a concrete basis vector: distinct-symbol (if set) → else
`e` + value-name (if set) → else `e` + numeric. Dummies keep the space's
abstract schema (`i,j,k…`). A separate optional **display label** on the basis
disambiguates multiple bases in one term (the prime in `e_i ⊗ e'_i`). Render
reaches all of this via `slot.basis_id → ctx.basis(id)`.

## Implementation slicing (alive at every step)

1. `IndexSlot.basis_id` (default 0) threaded through slot construction,
   `structural_eq`, canonical ordering, hashing, matcher (0 = don't-care);
   summation ignores it. All existing tests pass unchanged (nothing sets it).
   **DONE** (commit ef18cb1).
2. `Context` basis registry; `make_*_basis` registers + stamps emitted vectors;
   `expand_in_basis` stamps coordinate indices. Behavior unchanged (steps not
   yet filtering). **DONE** (commit 94a9b70). Deviation from the plan below:
   `make_*_basis` kept its **by-value** return (`-> Basis`) rather than switching
   to `Basis const*`; instead `intern_basis()` stores a context-owned copy in the
   registry and stamps the same id on the returned value. This realizes the id
   handle with no API ripple (coord_system / Python `PyBasis` unchanged); the
   registry copy keeps a slot's `basis_id` resolvable.
3. Basis-aware steps filter by `basis_id`. Add multi-basis tests: rotation
   `e_i ⊗ e'_i` (must sum), overlap `e_i^A·e_j^B` (must NOT become δ),
   two-point `F_{iJ}`.
4. Gap-2 rendering: `value_names` + concrete-index rendering via
   `basis_id→value_name`; vector-symbol override; display label. Coordinate
   systems populate `value_names` (and optionally the WCS `i,j,k` symbols).

## Settled vs open

Settled: approach B (per-index `basis_id`); interned id with the registry in
`Context`; `basis_id 0` = basis-unaware; summation basis-blind, rewrites
basis-aware; Basis owns label + `value_names` + optional vector symbols; two
naming tracks; the slicing above.  Implementation refinement (increment 2):
`make_*_basis` returns `Basis` by value + a registered context-owned copy,
rather than `Basis const*` — same id-handle semantics, no API ripple.

Open / deferred: exact matcher don't-care policy for partially-basis-tagged
patterns; whether `make_*_basis` returning `Basis const*` warrants a thin
value-handle wrapper for Python ergonomics; metric / scale factors and `∂`
(Stage 4/5 proper). Related: [[fold-equal-addends-self-prepare]] (000065),
[[zero-literal]] (000066).
