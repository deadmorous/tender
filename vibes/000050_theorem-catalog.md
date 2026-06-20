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
| 2 | bac-cab | `a × (b × c) = b (a·c) − c (a·b)` | 🟡 In reach | ε-pair contraction *inside a product* + δ-substitution |
| 3 | cross-identity-cross | `a × I × b = b ⊗ a − (a·b) I` | 🟡 In reach | same as #2 |

(`I` is the identity tensor; `×` cross, `⊗` tensor product, `·` dot.)

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

### 2. bac-cab `a × (b × c) = b (a·c) − c (a·b)` — 🟡 In reach

The geometric half works: expand, `distribute_contraction`, two rounds of
`simplify_basis_cross`, `canonicalize` reduce `a × (b × c)` to the index form
`Σ −ε_{mli} ε_{mkj} a_l b_k c_j e_i` — a product of two Levi-Civita symbols
sharing the summed index `m`.  Closing it then needs two things still deferred:
(a) **ε-pair contraction inside a larger product** — `contract_eps_pair` exists
but matches only a bare `ε ⊗ ε`, not `ε ε · (coordinates) · e`; the e-graph +
identity library (`eps_delta_*`, vibe 000046) is the general route, not yet
wired to the basis steps / Python; (b) **δ-substitution** — collapsing the
resulting `δ_{ab} x_a → x_b` against the coordinates, the recognized
parametric-RHS gap (vibes 000033/000040).

### 3. cross-identity-cross `a × I × b = b ⊗ a − (a·b) I` — 🟡 In reach

Same shape as bac-cab: the cross/identity-dyad pipeline reduces it to an ε-ε
index expression; closing needs the same ε-pair-in-product contraction and
δ-substitution.

## The shared blockers (what unlocks the 🟡 rows)

Two deferred capabilities recur and would convert the in-reach theorems to
proven:

- **δ-substitution** as a first-class step/rule: `Σ_p δ^{p}_{a} f(p,…) → f(a,…)`
  — needs a parametric slot + computed RHS (vibe 000033 §6, 000040).  Today it is
  worked around per-case with a data `Identity` (as in the `(e_i×e_j)·e_k = ε_ijk`
  example) or the concrete-unroll path.
- **ε-pair contraction within a product**: applying `eps_delta_*` to an `ε ⊗ ε`
  buried in a polyad of coordinates, via the e-graph saturator — exposing the
  identity library to the basis workflow (and to Python).

Once these land, #2 and #3 (and most cross-product vector identities) follow the
same expand → simplify_basis_cross → ε-δ → δ-substitution → reassemble arc.

## Next

Add theorems as the infrastructure grows; each proven theorem gets a test and,
where instructive, a maintained example.  Candidates after the δ-substitution /
ε-pair work: `tr(I) = n`, `I · a = a` (already a step), Binet–Cauchy, the
scalar/vector triple-product identities, and the differential-operator theorems
of Stage 5.
