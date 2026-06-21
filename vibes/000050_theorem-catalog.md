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
| 3 | cross-identity-cross | `a × I × b = b ⊗ a − (a·b) I` | ✅ Proven | — |
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

### 3. cross-identity-cross `a × I × b = b ⊗ a − (a·b) I` — ✅ Proven

Same shape as bac-cab, but rank 2 — and proving it flushed out a real transpose
bug in `contract_eps_pair`.  The pipeline (expand in WCS → `simplify_basis_cross`
→ `canonicalize`) produces the *correct* intermediate
`Σ −ε_{mlj} ε_{mki} a_l b_k e_j⊗e_i`; the ε-pair contracts over `m` into the
generalized Kronecker δ-form, and the concrete-unroll path closes the δ's — same
recipe as bac-cab.  Test: `BasisFeasibility.CrossIdentityCross` (C++, with a guard
that the result is *not* the transpose `a ⊗ b − (a·b) I`); Python:
`test_cross_identity_cross`.

**The bug (fixed):** after contracting the ε-pair, `try_contract_eps_pair`
re-attached the surviving non-ε factors (`others`) with a forward `for (o:
others) result = o ⊗ result;` loop.  Prepending in forward order **reverses** the
sequence, which is harmless for the commuting coordinates but **swaps the two
non-commuting basis vectors** of the dyad — transposing the result.  bac-cab has
a single free basis vector (rank 1), so the swap was invisible; `a × I × b` has
two (rank 2) and exposed it.  Fix: re-attach in reverse (`others.rbegin()..rend()`)
so the original left-to-right order — left leg before right leg — is preserved.
One line, in `src/derivation.cpp`.

The lesson for future rank-≥2 work: any step that flattens a polyad and rebuilds
it must preserve the order of the non-commuting (basis/dyad-leg) factors; a
"prepend in a loop" is a reversal in disguise.

**A second, independent proof** (`BasisFeasibility.CrossIdentityCrossViaReassembly`;
Python `test_cross_identity_cross_via_reassembly`) routes the same theorem through
the *pattern matcher* and *completeness reassembly* instead of the ε-pair
contraction — a cross-check from another angle (vibe 000053):

1. start from `a × (b × I)`, expand **only** I (as a standalone expression
   spliced back) → `a × (b × (e_i⊗e^i))`;
2. `distribute_contraction` pushes the cross onto the near leg →
   `Σ_i (a × (b × e_i)) ⊗ e^i`;
3. **bac-cab, registered as a reusable `Identity` with subtree pattern
   variables** `x×(y×z) = y(x·z) − z(x·y)`, fires via `apply_identity` on the
   `a×(b×e_i)` subterm → `Σ_i (b(a·e_i) − e_i(a·b)) ⊗ e^i`;
4. `expand_products` distributes `(…)⊗e^i`;
5. **`reassemble_completeness`** folds the resolution of identity:
   `Σ_i (a·e_i)(b⊗e_i) = b⊗a` (shape A) and `Σ_i (a·b) e_i⊗e_i = (a·b) I`
   (shape B) — landing in pure direct notation `b⊗a − (a·b) I` (structural,
   not merely algebraic, equality).

This exercises the subtree matcher firing under a sum/dyad, and closes the
final symbolic fold-back that plain `reassemble` (which needs a literal
coordinate tensor `a_i`) could not.

## Capabilities now in place

- **ε-pair contraction within a product** — done (theorem #2): `contract_eps_pair`
  fires on an `ε ⊗ ε` buried in a coordinate polyad, contracting over the shared
  index and keeping the rest.
- **δ-collapse** — the concrete-unroll path (`unroll_sums` →
  `eval_delta_concrete` → `fold_arithmetic`) closes the δ's without a symbolic
  δ-substitution.  A first-class `Σ_p δ^p_a f(p,…) → f(a,…)` step (the
  parametric-RHS gap, vibes 000033 §6 / 000040) would give cleaner *symbolic*
  results but is not required to *prove* these.
- **Completeness reassembly** — `reassemble_completeness` (vibe 000053) folds
  the resolution of identity where it is *partially contracted*:
  `Σ_i (X·e_i) e_i → X` and `Σ_i (scalars) e_i⊗e_i → (scalars) I`.  This is the
  invariant-level fold-back that plain `reassemble` cannot do (it needs a
  literal coordinate tensor `a_i`, but bac-cab leaves the invariant dot `a·e_i`).
  It retires the symbolic-δ-substitution need for the basis layer's vector
  reassembly — the `a·e_i = a_i` route would need component materialization plus
  δ-substitution; the `a·I = a` route this step takes does not.
- **Subtree-variable identities firing under a sum/dyad** — a theorem encoded as
  an `Identity` with slot-less rank-1 pattern variables (vibe 000051) is applied
  by `apply_identity` on a subterm buried inside `Σ_i (…) ⊗ e_i`.  Reusing a
  proved theorem as a rewrite rule, today, without a first-class `Theorem` type.

## Next

1. Add theorems as the infrastructure grows; each proven one gets a test and,
   where instructive, a maintained example.  Candidates: Binet–Cauchy, the
   scalar/vector triple-product identities, symmetric/antisymmetric
   decompositions, and the differential-operator theorems of Stage 5.
