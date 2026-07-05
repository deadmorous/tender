# 000076 — strain compatibility, gap D: deriving inc ε *as performed*

Continues vibe 000075.  The componentwise identity `inc ε = ∇×(∇×ε)ᵀ =
−∇∇θ + Δθ·I − (∇∇··ε)I − Δε + 2(∇∇·ε)ˢ` is already *verified* (brute-force
`chart.components`, test `test_cartesian_strain_compatibility_invariant_form`).
Gap D is to *derive* it the way the hand page does: ε never leaves abstract
form; only the ∇'s expand; the a×B×c identity fires **once** on `e_i × ε × e_j`;
the ∂'s are applied and the result **reassembles** into named invariant
operators.

## Where we start (blockers confirmed empirically)

The user's proposed surface: `nabla % transpose(nabla % epsilon)`, expand the
∇'s in WCS, commit the ∂'s to the left, apply a×B×c to the interior, apply the
∂'s and reassemble.  Probing the current code:

1. **`DifferentialExpr` has no tensor algebra.** `nabla % eps` is a Python
   deferred-application shell; `.transpose()` raises `AttributeError`.  So
   `transpose(nabla % eps)` cannot even be *built*.  Minor.
2. **No ∇-only expansion mode.** `chart.rot(eps)` immediately explodes ε into 27
   scalar components in the constant frame (`−(∂_z ε_xy) i i + …`).  Nothing
   expands `∇ = Σ_i e_i ∂_i` while keeping ε an abstract symbol.  Core gap.
3. **Derivative directions are concrete-only.** `FieldDerivDir` carries a
   concrete `CoordinateRef` (chart_id, slot) — no free/abstract index ∂_i tied
   to a summed frame vector.
4. **No abstract ∇ Expr node to reassemble *into*.** ∇∇θ, Δε, (∇∇·ε)ˢ, ∇∇··ε
   have no first-class representation; `chart.grad/div/laplacian` only ever
   produce *expanded* component forms.
5. **Selective expansion (vibe 000054) is still an unimplemented view-note.**
   The concrete-construction route largely sidesteps it (we *build* the
   expansion rather than expand one occurrence of a mixed tree).

## Decisions (user, this session)

Two forks, both taken on the ambitious side:

- **Reassembly target — new abstract ∇ Expr nodes.**  Introduce first-class
  invariant differential-operator node(s) so the derived result is a clean,
  renderable tender expression (`−∇∇θ + Δθ·I − (∇∇··ε)I − Δε + 2(∇∇·ε)ˢ`), not a
  pile of components.  Bonus: once `nabla` ops lower to real Expr nodes,
  `transpose(nabla % eps)` is just an ordinary Expr op — blocker 1 dissolves.
- **Index model — abstract implicit-summation index.**  The identity fires
  **once** on `e_i × ε × e_j` with i,j implicitly summed, faithful to the page.
  Needs a partial derivative that carries a **free** index tied to a summed
  frame vector — a genuine Expr-model extension (finding 3).

## Design sketch — the abstract ∇ node

One node covers everything by composition:

```cpp
enum class DelKind : uint8_t { Grad, Div, Curl };   // ∇⊗ , ∇· , ∇×
struct Del final { DelKind kind; Expr const* operand; };
```

Rank: Grad → r+1, Div → r−1 (r≥1), Curl → r (3D).  Then

| operator | node form |
|---|---|
| θ = tr ε | `Trace(ε)` |
| ∇θ | `Del{Grad, θ}` |
| ∇∇θ | `Del{Grad, Del{Grad, θ}}` |
| Δθ, Δε | `Del{Div, Del{Grad, ·}}` |
| ∇·ε | `Del{Div, ε}` |
| ∇∇·ε | `Del{Grad, Del{Div, ε}}` |
| (∇∇·ε)ˢ | symmetric part of the above |
| ∇∇··ε | `Del{Div, Del{Div, ε}}` |
| inc ε (input) | `Del{Curl, Transpose(Del{Curl, ε})}` |

Render `Del{Div, Del{Grad, x}}` as `Δ x`.  Canon: distribute over Sum/Difference
(∇ is R-linear), pull numeric literals, ∇ of a non-field constant → 0; otherwise
opaque (Leibniz over products is *not* an auto-rule).

## Increment plan (each step keeps the build/tests green)

1. **Abstract ∇ nodes** (`Del`) — factories, rank, structural_eq/cmp/hash,
   render, minimal canon, nf lower/match, Python `nabla`→Expr.  Unblocks the
   starting expression.
2. **Free-index ∂** — `FieldDerivDir` gains an abstract `CountableIndex`
   direction; thread through identity, summation detection, canon, render.
3. **∇-only WCS expansion** (`expand_nabla`) — `Del` → `Σ_i e_i ⊙ ∂_i(operand)`,
   fields kept abstract; Cartesian commits ∂ left (∂_i e_j = 0).
4. **apply a×B×c** once on the free-index `e_i × ε × e_j` (reuse
   `apply_identity`; fix self-prep gaps).
5. **Reassemble** `Σ_i e_i ⊙ ∂_i(X)` (and 2nd-derivative pairs) upward into
   `Del` nodes — reverse of step 3.
6. **Showcase** — Cartesian strain compatibility derived as performed, then the
   cylindrical evaluation of the compatibility equations.

## Status

Decisions settled; tasks #1–#6 tracked.  Starting Increment 1.
