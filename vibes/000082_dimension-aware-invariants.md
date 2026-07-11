# 000082 — dimension-aware invariant tensors

Status: **IN PROGRESS**.

Follows the vibe-000081 identity-dimension thread. Generalises: **every abstract
positive-rank invariant tensor carries the `IndexSpace` its implicit indices
range over**, inherited from the CS/basis. This closes the identity-dimension
design (bearing + non-optional) and extends it to `∇`, radius vectors, user
vectors, etc.

## Why (recap of the reasoning)

- The identity needed `dim` so `tr(I) → n` while its abstract form has no index
  slots (vibe 000081).
- Making `dim` *neutral* (a sized I ≡ a bare I) broke **congruence**: `I2 ≡ I3`
  yet `tr` gives 2 vs 3 — and collection could silently strip a dimension.
- Resolution: **identity-BEARING** dimension + **no unspecialized invariant**
  (bare construction defaults to a space). With a shared default there is no
  bare-vs-sized mismatch, so library-emitted and user invariants still cancel.
- Generalising to all positive-rank invariants: dimension is really the **index
  space** (not an integer), which the codebase already models (`IndexSpace`);
  the identity's `dim` is just that same info on a slotless object.

## Key subtlety (from `R2 + z·k`)

`R2` (2-D chart's radius vector) `+ z·k` (3rd axis) is a **valid 3-D vector**, not
a dimension clash: a radius vector's index lives in the **embedding** space, so
`dim(R2)=dim(k)=3`. The "2-D-ness" is the **chart's** (2 coords / 2-D tangent
space), not the vector's index space. So an embedded chart has **two** spaces —
tangent/chart (metric, intrinsic `tr`, `∇`'s summation) vs embedding/reference
(radius vector, frame vectors, `k`). Each tensor inherits the space **its own
indices** range over. Today `coords.size() == reference.dim()` (chart-dim =
embedding-dim), so they coincide; the design must let them diverge later.

## Design decisions

- **D1 — dimension = `IndexSpace const*`** (`TensorObject::dim`), on every
  abstract positive-rank invariant; inherited from the generating CS/basis.
- **D2 — BEARING**: `dim` is part of structural identity (`tensor_object_cmp`,
  `structural_eq`, `hash_tensor_object`). A 2-D I ≠ a 3-D I; congruence holds.
  (Reverts the vibe-000081 neutral choice.)
- **D3 — no unspecialized invariant**: bare construction defaults to
  `space_3d`. Positive-rank invariants always carry a space; rank-0
  scalars/coordinates keep `dim = null` (irrelevant). Shared default ⇒ a
  library I and a user I are both 3-D ⇒ they cancel.
- **D4 — internal producers stamp the OPERATIVE space**, not the global default:
  `fold_resolution_of_identity` / basis reassembly / `reassemble_nabla` use the
  basis/chart space. In flat 3-D these equal the default; for other-D they must
  follow context.
- **D5 — readers**: `tr` reads `dim` (identity → n; extend to δ/g if their
  abstract traces surface). `expand_in_basis` validates the invariant's `dim`
  against the basis it expands into (already done for I; generalise).
- **D6 — `+`/`·`/`⊗` stay PERMISSIVE** (no dimension-mismatch error): `R2 + z·k`
  must not be rejected. Space compatibility on sums = "same, or embeddable" —
  deferred (needs a subspace model).
- **DEFERRED**: tangent-vs-embedding two-space split; subspace/embedding
  compatibility & promotion; sum-compatibility enforcement; a Workspace-level
  ambient dimension so the default follows the workspace instead of hard-3-D.

## Increments

1. **Identity: BEARING + non-optional (default 3-D). DONE.** `make_identity(ctx)`
   → `space_3d`; `dim` back into `tensor_object_cmp`/`structural_eq`/hash.
   Internal identity folds (basis.cpp resolution-of-identity ×4, `reassemble_nabla`
   `dimension_identities` now *overwrites* to the chart space) stamp the operative
   space. `tr(I)` always folds. Migrated the `tr(I)`-stays-symbolic tests + the
   vibe-081 neutral test. **KEY BUG FOUND**: the identity matcher's `inst_factor`
   rebuilt an atom positionally as `TensorObject{name,rank,traits,slots}`,
   silently DROPPING `dim` (and `deriv_marks`) — so `apply_identity` produced a
   dimensionless I and the 4 cross-commute/reassociation matching tests broke once
   `dim` was bearing. Fixed: copy the whole obj, replace only slots. Kept the
   null-safe `space_cmp` and the `expand_in_basis` dimension check. Python
   `t.identity`/`ws.identity` default 3-D; docstrings updated.
2. **Generalise `dim` to abstract positive-rank invariants.** User vectors
   (`t.tensor(name, rank≥1)`), `∇`, radius/tangent vectors get a `dim`
   (default 3-D / inherited from chart). Bearing. Constructors stamp it.
3. **Space-readers validate; sums stay permissive.** `expand_in_basis` (and
   friends) check the invariant's space matches the basis; `+`/`·` untouched.
   `well_known_trace_dim` already generalises to any dimensioned symmetric
   well-known rank-2.

Constraints: buildable/tested per increment; ≥90% coverage; run clang-format;
strip notebooks. See [[vibe81-explicit-basis-operator-route]] (identity-dimension
origin), [[basis-aware-indices-plan]] (per-index basis_id), [[differential-foundations-plan]].
