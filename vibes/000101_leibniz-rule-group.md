# 000101 The Leibniz rule group — brief

The first item carried forward from the completed M0–M4 spine (vibe 000093):
rules for ∇ acting on a product, which three challenges wait on (000012,
000013, 000019) and which the applied-mechanics arc will need immediately.

## Audit: what is statable, and does it fire?

Vibe 000096 kept differential rules out of M2 on the grounds that "the fence
machinery makes this a correctness minefield" and that ∇-bearing terms are
inert in the e-graph.  Measurement says the caution was half right and half
overcautious.

**They fire.**  Statable ∇ rules match and rewrite normally:

| rule | result |
|---|---|
| `∇(fg) = f∇g + g∇f` | **proved**, rule fired 3× |
| `∇·(u×v) = v·(∇×u) − u·(∇×v)` | **proved**, fired 1× |
| `∇×(∇×u) = ∇(∇·u) − Δu` | **proved**, fired 1× |

So the fence does not make ∇ inert.  A rule whose LHS canon accepts behaves
like any other.

**But half of them cannot be *stated*.**  Canon rejects a ⊗-product sitting
inside a *contraction* operand:

| expression | canon |
|---|---|
| `∇(f g)` | OK |
| `∇(u·v)` | OK |
| `∇·(u×v)` | OK |
| `∇×(∇×u)`, `u×(∇×u)`, `Δu` | OK |
| **`∇·(f u)`** | **fails** — "a nested ⊗ inside an operand awaits fence distribution" |
| **`∇×(f u)`** | **fails** — same |

The rule is mechanical: a ⊗ inside a `Dot`/`Cross` operand fails; inside a
gradient it is fine.

## The finding that reorders the backlog

Vibe 000093's carried-forward list had "Leibniz rule group" (item 1) and
"fence distribution inside a contraction operand" (item 3) as independent.
They are not: **the fence gap is a prerequisite for half the Leibniz group**,
and it is the same gap that blocks challenge 000010.  One capability, three
consequences.

## Plan

**Step 1 — ship the statable rules now.**  A `leibniz` group with the rules
canon accepts:

- `grad-product`   `∇(fg) = f∇g + g∇f`
- `div-cross`      `∇·(u×v) = v·(∇×u) − u·(∇×v)`
- `curl-curl`      `∇×(∇×u) = ∇(∇·u) − Δu`
- `grad-dot`       `∇(u·v)` — the Leibniz half, as far as it goes

Each fire-tested at birth, per the vibe-000096 discipline.  DAG placement:
all four are **derived**, each citing nothing (they are provable by component
reduction, which is what their challenges' L1 tests already do).

**Step 2 — see what promotes.**  Expected: 000012 (curl-curl) to L2 outright;
000013 partially (two of its four product rules are statable, two are not);
000019 needs `u×(∇×u)` handling beyond the plain rules.  No forced wins — a
challenge that needs the blocked forms keeps its xfail with the reason
updated to name the fence gap rather than "the Leibniz group".

**Step 3 — record what the fence gap now blocks**, with three challenges
pointing at it, as the argument for doing it next.

## Outcome

Shipped: `leibniz` group with `grad-product`, `div-cross`, `curl-curl`, each
fire-tested, plus a guard test pinning *why* `∇·(fu)` and `∇×(fu)` are absent
(canon cannot state them) so the absence reads as a known gap.

Scoreboard **16 → 17 L2**:

- **000012 curl-curl → L2.**  Now proved *invariantly*, ∇ abstract, no
  coordinate system chosen — where its L1 must expand into Cartesian
  components to check the same claim.
- **000013 split honestly.**  Its L2 now proves the two statable product
  rules invariantly and passes; a *second*, xfailed test carries the two
  blocked ones with the capability named.  Splitting was the right call: a
  single all-or-nothing test would have hidden that half the challenge is
  already reachable.
- **000019 not promoted**, and its reason sharpened.  The group exists now, so
  "needs Leibniz" was no longer true: `(u·∇)u` wants the ε-δ identity *and*
  Leibniz together, which nobody has attempted.  A different, smaller gap
  than the one it claimed.

`grad-dot` (`∇(u·v)`) was dropped from the plan: its right-hand side is
`u×(∇×v) + v×(∇×u) + (u·∇)v + (v·∇)u`, whose last two terms are the very
directional derivatives challenge 000019 cannot state — so the rule would be
unusable even though the left side canonicalizes.  Noted rather than shipped
inert (the vibe-000096 discipline).

## Fence distribution — the prerequisite, and what it unlocked

The gap turned out to be **two** distinct defects wearing one error message.

**1. The fence condition asked about the wrong thing.**  `encapsulate` fenced
a ⊗ operand when *that operand* contained ∇.  But in `∇·(f u)` the ∇ is the
operand's **sibling**: `f⊗u` holds no operator, yet floating the scalar out —
`∇·(f u) → f(∇·u)` — is precisely the Leibniz error.  The barrier in
`distribute_contraction` is about the whole contraction; the fence condition
was about one operand.  Fixed by passing `sibling_operator` down from the
chain site, which knows.  The same fix was needed in the `Cross` arm.

**2. `distribute_contraction` never handled double dots.**  It covered `·`
and `×` only, so `(λ tr(ε) I)··ε` never had its scalars floated out, and canon
then met a ⊗ inside a contraction operand and threw.  This was challenge
000010's blocker all along — *not* the identity, which has been in the library
and firing since M2.

**Scoreboard 17 → 19 L2**, and every one of these had been red for milestones:

- **000010 → L2.**  `T··ε = λ(tr ε)² + 2μ ε··ε` proved invariantly.
- **000013 → L2 (complete).**  Its blocked half — `∇·(fu)`, `∇×(fu)` — became
  statable, so all four product rules are now proved with ∇ abstract.
- **`div-scaled` and `curl-scaled`** joined the `leibniz` group (5 rules).
- **The identity DAG has no open proof obligations left**: `ddot-identity`,
  the one unproven node, is discharged by challenge 000010.

Six tests failed on the fix — every one of them a test asserting the *old*
blocked behaviour, which is what a removed blocker should look like.  Two
needed genuine reframing: the `Unsupported` verb path had no reachable case
left, so it is now exercised through an ill-formed implicit summation (an
Oblique index repeated at the same level), which canon still rightly refuses.

One meta-test gap surfaced and was closed: a challenge could claim
`proves="x"` while node `x` still said `proof=None`, and nothing caught the
one-way declaration.  `test_every_claim_is_acknowledged_by_the_identity` now
walks that direction too.

## Risk

`grad-product` fired **three times** on a one-line proof: its right-hand side
contains further `∇(…)` patterns that re-match.  That is benign here (the
e-graph merges rather than rewrites, and it converged in one pass) but it is
the classic shape of a blow-up.  The benchmark should carry a Leibniz case,
and the group must never be handed to a verb together with everything else
without a budget.
