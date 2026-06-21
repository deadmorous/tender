# 000050 Theorem catalog

A living catalog of **theorems** — invariant (direct-notation) identities the
system aims to *derive*, not assume.  Distinct from the axiom-level `Identity`
rules (vibe 000033): a theorem is proved by a derivation (expand into a basis,
manipulate at the coordinate/index level, fold back), and once proved it can be
reused.  We grow this list gradually; each entry tracks how far the machinery
can currently take it.

## Status legend

| mark | meaning |
|------|---------|
| ✅ Proven   | a verified derivation exists (a test), ideally with a worked example |
| 🟡 In reach | the geometric pipeline reduces it to an index-algebra expression, but *closing* it needs a capability still on the deferred list |
| ⬜ Planned  | needs infrastructure not yet built |

## Catalog

| # | Theorem | Statement | Status | Closing needs |
|---|---------|-----------|--------|---------------|
| 1 | cross-with-identity commutes | `a × I = I × a` | ✅ Proven | — |
| 2 | bac-cab | `a × (b × c) = b (a·c) − c (a·b)` | ✅ Proven | — |
| 3 | cross-identity-cross | `a × I × b = b ⊗ a − (a·b) I` | 🟠 Blocked by bug | a rank-2 transpose bug in the cross pipeline (see below) |
| 4 | identity has no axial vector | `vec(I) = 0` | ✅ Proven | — |
| 5 | trace of the identity | `tr(I) = n` (`= 3` in 3D) | ✅ Proven | — |

(`I` is the identity tensor; `×` cross, `⊗` tensor product, `·` dot; `vec`/`tr`
the vector invariant / trace.)

## Notes per theorem

### 1. `a × I = I × a` — ✅ Proven

The first invariant-level theorem proved end to end (vibe 000049 follow-ups).
Derivation: expand both sides in WCS, `distribute_contraction` to push the cross
over the identity dyad `I = Σ_m e_m ⊗ e^m`, `simplify_basis_cross` to turn each
`e_i × e_m` into `ε e^k`, then `canonicalize`.  Both sides reduce to the same
`Σ_i Σ_j Σ_k −ε_{kji} a_k e_j e_i`.  Closing it required, in order: the
contraction-over-⊗ distribution, ε's cyclic symmetry (already in
`canon_symmetry`), and canonical ordering of nested `ExplicitSum` binders
(Fubini).  Test: `BasisFeasibility.CrossWithIdentityCommutes`; Python:
`TestBasisSteps.test_cross_with_identity_commutes`.  Worked example:
`examples/cross_identity.{py,ipynb}`.

### 4. `vec(I) = 0` — ✅ Proven

The vector invariant (axial vector) of a symmetric tensor vanishes; the identity
is symmetric.  Through the basis `I = Σ_i e_i⊗e_i`, so `vec(I) = Σ_i e_i × e_i`,
and each `e_i × e_i = 0` by the antisymmetry of the cross product.  Derivation:
`expand_in_basis` → `expand_dyad_ops` (`vec(e_i⊗e_i) = e_i × e_i`) →
`simplify_basis_cross` (`→ ε_{iik} e_k`) → `unroll_sums` →
**`eval_eps_concrete`** (every term has a repeated index, so ε = 0) →
`fold_arithmetic` → `0`.  `eval_eps_concrete` (the concrete Levi-Civita
evaluator, sibling of `eval_delta_concrete`) was the missing piece.  Test:
`BasisFeasibility.VectorInvariantOfIdentityIsZero`.

### 5. `tr(I) = n` — ✅ Proven

Through the basis, `tr(I) = Σ_i e_i·e_i = Σ_i δ_ii = n` (3 in 3D); reached by the
same coordinate machinery (`expand_dyad_ops` `tr(e_i⊗e_i) = e_i·e_i`,
`simplify_basis_dot`, `unroll_sums`, `eval_delta_concrete`, `fold_arithmetic`).
The closely related `I:I = tr(I) = 3` runs through `expand_double_dot`.

### 2. bac-cab `a × (b × c) = b (a·c) − c (a·b)` — ✅ Proven

`expand_in_basis`, `simplify_basis_cross`, `canonicalize` reduce `a × (b × c)`
to `Σ −ε_{mli} ε_{mkj} a_l b_k c_j e_i` — two Levi-Civita symbols sharing the
summed index `m`.  `contract_eps_pair` (now extended to fire **inside a larger
product**) contracts the ε-pair over `m` into the generalized Kronecker δ-form,
keeping `a,b,c,e`; the δ's then collapse via the **concrete unroll path**
(`expand_products` → `unroll_sums` → `eval_delta_concrete` → `fold_arithmetic`),
so no symbolic δ-substitution is needed.  Both sides reduce to the same
coordinate form (`algebraic_eq`).  Test: `BasisFeasibility.BacCab`; the new
ε-pair-in-product unit test is `ContractEpsPair.FiresInsideProduct`.

The key extension: `try_contract_eps_pair` now flattens the product, finds the
two ε's among the other factors, contracts over the indices **shared by both
ε's** (not all peeled sums), and re-attaches the others — and peels a leading
`Negate`.  Backward-compatible with the bare `ε ⊗ ε` `eps_delta_*` cases.

### 3. cross-identity-cross `a × I × b = b ⊗ a − (a·b) I` — 🟠 Blocked by a bug

Same shape as bac-cab, and the ε-pair machinery now fires — but the engine
reduces `(a × I) × b` to `a ⊗ b − (a·b) I`, the **transpose** of the correct
`b ⊗ a − (a·b) I` (verified by hand and by a concrete `a = e₁, b = e₂` check:
the answer is `e₂ ⊗ e₁ = b ⊗ a`).  bac-cab is rank-1 so it cannot expose this;
`a × I × b` is rank-2 and does.  The reduction of the engine's *own* intermediate
form `Σ −ε_{mlj} ε_{mki} a_l b_k e_j⊗e_i` works out to `b ⊗ a` by hand, so the
δ-symmetric `build_kronecker_det` is not the obvious culprit — the transpose is
somewhere in the rank-2 cross/contract path (a wrong free-slot↔dyad-leg
association, or a leg choice in the cross-with-dyad distribution).  **A genuine
bug to hunt**, not a missing feature.  Until fixed, `a × I × b` (and any rank-2
cross identity) cannot be trusted.

## Capabilities now in place

- **ε-pair contraction within a product** — done (theorem #2): `contract_eps_pair`
  fires on an `ε ⊗ ε` buried in a coordinate polyad, contracting over the shared
  index and keeping the rest.
- **δ-collapse** — the concrete-unroll path (`unroll_sums` →
  `eval_delta_concrete` → `fold_arithmetic`) closes the δ's without a symbolic
  δ-substitution.  A first-class `Σ_p δ^p_a f(p,…) → f(a,…)` step (the
  parametric-RHS gap, vibes 000033 §6 / 000040) would give cleaner *symbolic*
  results but is not required to *prove* these.

## Next

1. **Fix the rank-2 cross transpose bug** (theorem #3) — the one thing standing
   between the engine and trustworthy rank-2 cross identities.
2. Then add theorems as the infrastructure grows; each proven one gets a test
   and, where instructive, a maintained example.  Candidates: Binet–Cauchy, the
   scalar/vector triple-product identities, symmetric/antisymmetric
   decompositions, and the differential-operator theorems of Stage 5.
