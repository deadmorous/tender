# 000099 M4 brief — certification to L2

The per-milestone brief for M4 (vibe 000093).  M4 adds no features: for each
challenge below L2, either the surface already handles it (promote) or the
attempt names a concrete gap (file it; fix it only if it is small).

Starting state: **23 challenges — 14 L2, 3 L1, 6 L0.**

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
