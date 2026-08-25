# 000099 M4 brief — certification to L2

The per-milestone brief for M4 (vibe 000093).  M4 adds no features: for each
challenge below L2, either the surface already handles it (promote) or the
attempt names a concrete gap (file it; fix it only if it is small).

Starting state: **23 challenges — 14 L2, 3 L1, 6 L0.**

## Outcome

**14 L2 / 3 L1 / 6 L0  →  16 L2 / 6 L1 / 1 L0.**  Five challenges promoted,
each its own commit; no feature was added, and no claim was weakened to get a
green.

| # | Result |
|---|---|
| 000017 | **L1** — a 13-shape battery; found that dots, traces and ⊗-polyads fold back but **crosses do not** |
| 000019 | **L1** — the convective identity verifies componentwise |
| 000023 | **L2** — all three spherical equilibrium equations, cot θ terms and all |
| 000022 | **L2** — the Lamé cylinder collapses to the Euler ODE, in both the compact and expanded forms |
| 000018 | **L1** — all three metric forms of `a·b` in a genuinely oblique basis |

**The remaining reds are now capabilities, not omissions** — which was the
point.  Every one names what is missing:

1. **An invariant Leibniz rule group** (∇ over a product) — blocks 000012,
   000013, 000019.  The single most valuable next capability: three
   challenges wait on it, and it is the gateway to tier C.
2. **Fence distribution inside a contraction operand** — blocks 000010; canon
   cannot state `T··ε` when `T` holds a ⊗-product.
3. **ε-reassembly** — `reassemble` folds δ-contractions back to invariants but
   not ε-contractions, so an expanded cross cannot return to `a × b`
   (000017).  This explains a pattern in the suite: challenges 000001 and
   000014 contract the ε-*pair* into δ's first — not style, the only route.
4. **Index raising/lowering** (`g_ij g^jk = δ_i^k`) — blocks 000018's L2.
5. **Inverse chart embeddings** (vibe 000090 approach B) — blocks 000021, the
   last L0.

Findings 1 and 3 are the ones to act on: the Leibniz group unlocks the most,
and ε-reassembly closes an asymmetry in the bridge that every cross-bearing
derivation currently works around.

## The work, ordered by cost

Ordered deliberately: the cheap ones first, so the expensive ones are
attempted with the scoreboard already improved and the remaining gaps
sharply stated rather than vaguely feared.

| # | Challenge | Blocker today | Cost |
|---|---|---|---|
| 000017 | basis round-trip battery | unwritten — the machinery exists | small |
| 000019 | convective term `(u·∇)u` | unattempted | small |
| 000023 | ∇·T spherical vs textbook | unattempted; transcription care | medium |
| 000018 | oblique metric `g_ij a^i b^j` | unattempted | medium |
| 000022 | Lamé thick cylinder → ODE | unattempted | medium |
| 000012 | curl-curl **invariantly** | no Leibniz rule group | large |
| 000013 | ∇ product rules **invariantly** | same | large |
| 000010 | energy `T··ε` invariantly | canon cannot state it | blocked |
| 000021 | cross-chart reverse | vibe 000090 approach B | blocked |

## Rules of engagement

- **No forced wins.**  A challenge that cannot be reached honestly keeps its
  xfail, with the reason sharpened to name the missing capability.  The red is
  the roadmap; a green obtained by weakening the claim is worse than red.
- **L1 before L2.**  An unattempted challenge earns L1 (endpoint verified) on
  the way; L2 only when the derivation runs on the documented surface.
- **A gap fix must be small and general.**  M4 is not the place to build a
  differential rule group — that is M5-adjacent work, and if 000012/000013
  need it, they say so and stay red.
- Every promotion is its own commit, as in M2 increment 4.

## Expected outcome

Promoting the five cheap-to-medium challenges would put the scoreboard at
roughly **19–20 L2**.  The four remaining reds would then be exactly the four
capabilities tender genuinely lacks — a Leibniz rule group, fence
distribution inside a contraction operand, and inverse chart embeddings —
which is a far more useful backlog than "six things not attempted".
