# 000075 — strain compatibility: the sign catch and gaps A–C

Start of the strain-compatibility work (task #26).  The user's hand derivation
is `vibes/images/strain-compatibility.png`: `inc ε ≡ ∇×(∇×ε)ᵀ = 0` for
symmetric ε, derived by expanding **only the ∇'s** in the Cartesian frame
(`∇ = e_i ∂_i`, `∂_i e_j = 0`) and applying the a×B×c identity (1) — ε itself
is never expanded in any basis.  The identity's derivation lives in
`../tender-sandbox/identities.ipynb`.

## Tender catches a lost minus sign

Probing the derivation componentwise (Cartesian chart, rank-2 `components`
from vibe 000074 doing the projections):

- The five-term RHS of identity (1) is **correct**:
  `a×B×c = trB(c⊗a − (a·c)I) + (c·B·a)I + (a·c)Bᵀ − c⊗(B·a) − (c·B)⊗a` ✓.
- The page's first equality `a×(c×B)ᵀ = a×B×c` is **wrong**: tender finds
  `a×(c×B)ᵀ = −a×Bᵀ×c` (= `−(c×B×a)ᵀ`).  Dyad check: for B = u⊗v,
  `a×(c×B)ᵀ = (a×v)⊗(c×u)` while `a×B×c = (a×u)⊗(v×c)`.
- Hence for symmetric ε the page's (2) is `−inc ε`, confirmed by tender on
  all 9 components:

      inc ε = −∇∇θ + Δθ·I − (∇∇··ε)I − Δε + 2(∇∇·ε)ˢ,   θ = tr ε

  and `tr(inc ε) = Δθ − ∇∇··ε`, so `inc ε = 0` still yields (3)
  `Δθ = ∇∇··ε` and (4) — every displayed equation is `= 0`, which is why the
  slip was invisible.  User confirmed ("I worked on −inc ε").  Second hand
  error tender has caught (after the 2T_rθ/r factor in vibe 000073).

## Gaps fixed (A–C)

Brute-force verification surfaced three self-preparation gaps:

- **A. `grad(tr ε)` crashed** ("encapsulate: unsupported factor node"): a
  tr/vec/transpose wrapper around an abstract field is opaque once
  `expand_fields` rewrites the field inside to `Σ ε_ij e_i⊗e_j`.  Fix:
  `expand_fields` runs `steps::expand_dyad_ops` when something expanded
  (tr(a⊗b)→a·b …), and `reduce_field`'s fixpoint gains
  `simplify_basis_dot` + `eval_delta_concrete` so the resulting frame dots
  reduce before differentiation.
- **B. `(e_i·I)·e_j` left unreduced** by `components`/`component_matrix` on a
  term like `Δθ·I`.  Fix: both projections run `expand_identity_frame`
  (I → Σ e_k⊗e_k) after `expand_fields`.
- **C. A scalar quotient fences the reduction**: `e_i·(X/2)·e_j` stuck, both
  at the top level and nested (`2·(X/2)`).  Fixes: `distribute_bilinear`
  (chart.cpp) peels `ScalarDiv` — `(X/s) op r → (X op r)/s`; and
  `distribute_contraction`'s fence distributor peels `ScalarDiv` alongside
  `Negate`, re-applying the divisors at the node it fires, and distributes
  over a Sum/Difference the peel exposes (the entry `expand_products` never
  saw it under the quotient).

Acceptance: `inc ε` equals the sign-corrected invariant form **projected as
one whole tensor expression** — `−(∇∇θ − Δθ·I + (∇∇··ε)I + Δε − 2(A+Aᵀ)/2)`
with `A = ∇(∇·ε)` — entry by entry, plus the trace identity.  Test
`test_cartesian_strain_compatibility_invariant_form` keeps it.

## Next (gap D — the real feature, designed together)

Reproduce the derivation *as performed*: ε stays invariant; only ∇ expands.
Needs (i) a ∇-only expansion mode (the mirror of vibe 000073's expand-first —
legal in Cartesian since the frame is constant), (ii) `apply_identity` with an
abstract field as B (operands decorated with ∂_i∂_j), (iii) reassembly of
`e_i`-contracted second derivatives upward into invariant operators
(∇∇θ, Δε, (∇∇·ε)ˢ, ∇∇··ε).  Likely built on **selective expansion**
(vibe 000054): kind/predicate filter first, positional addressing as the
robust follow-up.  Then the cylindrical evaluation of the compatibility
equations as the showcase example.
