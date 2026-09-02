# 000110 M5A brief — time, variation, rotation tensors, rigid-body kinematics

The applied-mechanics arc of vibe 000093 (M5A), items 1 and 2 in one brief,
because they are one body of work: the variation δ, the total derivative d/dt
and the angular velocity ω are all *derivations*, and a rotation's angular
velocity turns out to be a construction over whichever derivation is handed to
it — d/dt gives the angular velocity, δ gives the virtual rotation.  Splitting
them would build the same machinery twice.

**Scope, reframed 2026-09-02 (Stepan).**  The definite integral, integration by
parts and the fundamental lemma — increments I4/I5 of this brief's first draft —
**moved to vibe 000111**, where they belong with the continuum arc's
cross-section resultants.  Nothing in finite-dimensional applied mechanics needs
them: the virtual-work route to the equations of motion for a system of points
or rigid bodies is algebraic, and only *Hamilton's* principle (δ∫L dt = 0) wants
an integral.  This brief keeps I1–I3 (done) and gains rotation tensors and
rigid-body kinematics.

**Process note, recorded against this brief's own author.**  Vibe 000093 says:
write the milestone brief, agree it, then implement.  I1–I3 were implemented
while the brief was still being edited, and the first challenge Stepan wanted
(000027) arrived after the code did.  Nothing had to be undone, but the plan
was fitted to the work rather than the other way round.  The rotation
increments below are written *before* any of their code exists, and stay that
way until agreed.

## The headline: measured before planned, and the engine is already here

Vibe 000102 left one open question — **1b: is the variation δ representable as
`Σ c_k ⊗ D_k`?**  It mattered because a "no" was the only surviving argument
for option (B), a *declared* `derivation` trait on operator symbols, with all
the matcher work that implies.

**The answer is yes**, and it took six lines to measure.  Writing `t` and the
chain `q, q̇, q̈` as ordinary coordinate atoms and `L` as a rank-0 field
declared to depend on `(q, q̇, t)`:

```python
d = td.deriv
DDt   = d(t) + qd*d(q) + qdd*d(qd)          # d/dt
delta = dq*d(q) + dqd*d(qd)                 # δ

td.apply_operators(DDt * L)     # ∂ₜL + (∂_q L) q̇ + (∂_q̇ L) q̈
td.apply_operators(delta * L)   # (∂_q L) δq + (∂_q̇ L) δq̇
td.apply_operators(delta * (f*g))  # f (∂_q g) δq + (∂_q f) g δq
td.apply_operators(DDt * q)     # q̇
td.apply_operators(delta * qd)   # δq̇
```

Every line is today's library, unmodified.  So:

- **δ is a derivation of the `Σ c_k ∂_k` form** — the coefficients are the
  variations `δq_k` and the partials are with respect to the generalized
  coordinates.  It is not a coordinate expansion *in space*, which is what made
  1b look doubtful; it is a coordinate expansion in **configuration space**, and
  that is the same shape.
- **d/dt is the same object**, with coefficients `(1, q̇, q̈, …)`.
- Leibniz, the chain rule through declared dependence, n-ary products, and
  `d/dt q = q̇` all come from `apply_operators` with nothing added.
- Therefore **option (B) is not needed**, and vibe 000102's question 1b closes
  in favour of (D).  Challenge 000024 already certifies the mechanism ("Leibniz
  holds for an operator the library has never heard of"); δ is a new instance of
  it, not a new mechanism.

The partial/total distinction, which is the usual source of confusion in
Lagrangian mechanics, needs no representation either: it *is* the declared
dependence.  `∂L/∂q` holds `q̇` and `t` fixed because `q̇` is a separate
declared dependency of `L`, not a function of `q`; `dL/dt` chains because the
d/dt operator says which coordinates move with `t`.

## The one non-obvious invariant: δ and d/dt commute only if the variations ride the chain

Measured, and it failed the first time:

```
δ(dL/dt) − d/dt(δL)  =  (∂_q L) δq̇ + (∂_q̇ L) δq̈       ≠ 0
```

The missing terms are exactly the ones d/dt would produce if it knew that
`δq` is itself a function of time with `d/dt δq = δq̇`.  Adding
`δq̇ ∂_{δq} + δq̈ ∂_{δq̇}` to the d/dt operator makes the two commute — measured
`algebraic_eq → True`.

This is not a workaround; it is the statement that a virtual displacement is a
*field over the motion*, and it is the reason the operators must come from one
factory rather than be assembled ad hoc by each user: **the commuting of δ and
d/dt is a construction invariant**, and hand assembly gets it wrong on the first
try (this brief's author did).

## What is actually missing

For the time half (I1–I3), two things, neither of them a new algebraic
mechanism:

1. **Names.**  A `TensorName` is a single letter or a LaTeX command, so `q̇`,
   `q̈`, `δq` cannot be written.  The measurements above had to call them
   `u, a, p`.  This is the smallest gap and the most immediately visible.
2. **A public surface** minting the chain and the two operators consistently
   (the invariant above).

For the rotation half (I4–I7) the answer is different in kind, and it is what
the measurements section below is for: the *algebra* is present — Leibniz
reaches through a transpose, δ and d/dt are already interchangeable — and what
is missing is the ability to say **"this symbol is orthogonal"** in a way the
whole engine respects, plus the handful of skew-tensor facts that turn a spin
tensor into an angular velocity vector.

## Increments

### I1 — decorated names — **done** (a329af0)

`TensorName` accepts a LaTeX command applied to a braced name, recursively:
`\dot{q}`, `\ddot{q}`, `\delta{q}`, `\delta{\dot{q}}`, `\bar{\sigma}`,
`\hat{n}`.  Nothing else changes: the name is still an opaque atom, rendering
already passes it through (`\boldsymbol{\dot{q}}` for a bolded rank ≥ 1), and
structural identity is unaffected — `q` and `\dot{q}` are simply two names.

*Done when:* `ws.coords(r"\dot{q}")` mints a coordinate that renders as q̇, and
`make_tensor_name` still rejects `qdot`, `\dot q`, `\dot{q}x` and `\dot{}`.

Shipped as specified.  One thing the plan did not anticipate: `NameStr` had to
grow 16 → 32 characters, because the pendulum challenge's own vocabulary
(`\delta{\ddot{\phi}}`) is 20.  Canon and saturation benchmarks unchanged.

### I2 — the time chain and its two operators — **done**

A public factory — `ws.time("t")` → `tender.mechanics.Time`, named by Stepan
from three candidates — that mints `t`, a set of generalized
coordinates to a requested order of derivative, their variations, and returns
the two operators built over all of them:

```python
tm = ws.time("t")
q, qd, qdd = tm.coordinate("q", orders=2)     # q, q̇, q̈  (named by decoration)
L = tm.field("L", 0, deps=[q, qd, tm.t])
ddt, delta = tm.ddt(), tm.variation()
```

The factory owns the invariant: every minted coordinate is registered in the
chain, `ddt` carries `q̇ ∂_q + q̈ ∂_q̇ + δq̇ ∂_{δq} + …` over all of them, and
`variation` carries `δq ∂_q + δq̇ ∂_q̇ + …`.

Three decisions taken while writing it:

- **Operators are built afresh on every call** rather than cached and the chain
  frozen.  Minting after taking an operator is then not an error, it just means
  the operator you hold is older than the chain — documented as "take the
  operator at the point of use".  Freezing would have made the natural notebook
  order (mint, work, mint more) an error for no gain.
- **The chain closes one order beyond what it returns.**  `coordinate("q",
  orders=2)` hands back `q, q̇, q̈` but the operator also knows `q⃛`, so `d/dt q̈`
  is its true successor rather than a silent zero — the one place truncation
  would have produced a *wrong* answer instead of an incomplete one.  `orders`
  caps at 2 because LaTeX has no fourth dot.
- **`tm.field` requires `deps`.**  The core default (depend on all coordinates)
  is a trap here: `dL/dt` of such a field chains through the *variations* and
  sprouts δ terms that mean nothing.  Refusing with that explanation costs one
  line at each call site and removes a whole class of nonsense result.

*Done when:* a challenge asserts `δ(dL/dt) = d/dt(δL)` for `L(q, q̇, t)`, and
`d/dt q = q̇`, `δ q̇ = δq̇`, with every object coming from the factory.

Shipped: challenge **000025** (tier E, L2 performed) carries exactly those,
plus the derivation property of both operators over a product.  Challenge
**000026** (the pendulum's Lagrange equation) is the enumerated red for I4/I5.

### I3 — tensors and fields of time — **done**

Rank ≥ 1 fields of time already differentiate (`d/dt (r·v)` gives the product
rule, measured).  What is missing is the **commuting of ∂ₜ with an abstract ∇**:
`∂ₜ(∇⊗u)` currently raises *"differentiating a ∇ operator is not supported;
expand ∇ in a chart first"*, but `t` is not a chart coordinate, so ∂ₜ should
simply pass through the operator.  Without this, elastodynamics cannot be
written invariantly.

*Done when:* `∂ₜ(∇⊗u) = ∇⊗(∂ₜu)` and `∂ₜ(∇·σ) = ∇·(∂ₜσ)` hold with no chart,
and the refusal survives for a genuine chart coordinate.

Shipped as a **declared bit**, not an inference: `nonspatial` on
`CoordinateRef`, beside `nonneg`, set by `ws.time` on everything it mints.  The
differentiator cannot deduce it — an abstract ∇ belongs to no chart, so ∂ₜ and
∂_r are structurally alike, and the difference is a fact about what the
coordinate *is*.  With the bit, `∂_q ∇ = 0`; without it the refusal stands, and
correctly: `∂_r ∇` picks up the scale factors and the connection.  Both
directions are in challenge 000025, and the pass-through test was checked
against a build with the bit ignored.

### I4 — the rotation tensor

A proper orthogonal `P`: `P·Pᵀ = Pᵀ·P = I`.  Three things have to be settled,
and the measurements below (M1, M3) say the first is the hard one.

*Representation.*  Two candidates, and they are complementary rather than
rival:

- **Declared.**  `P` is an opaque rank-2 symbol carrying an *orthogonality
  property*, the way a rank-2 field can already carry `symmetric=True`.  This
  is how direct-notation mechanics is actually written, and it is what every
  invariant derivation below needs.
- **Constructed.**  `P = e_k ⊗ E_k`, the tensor that carries one orthonormal
  frame onto another (Zhilin's own way of writing it).  Then orthogonality is
  not an axiom but a two-line consequence of frame completeness, and the
  library *proves* what the declared form asserts.

Proposal: ship both, with the constructed form as the **proof route** for the
declared form's rules — the same honest-vs-asserted split the identity DAG
already draws (vibe 000102 Q2).  A `rotation` rule group holds `P·Pᵀ → I`,
`Pᵀ·P → I`, and their proof obligation is a challenge that builds `P` from two
frames and reduces.

*Done when:* `prove_equal(P·Pᵀ, I)` proves for a declared rotation, `(P·a)·(P·b)
= a·b` proves, and a challenge derives both from the constructed form.

### I5 — spin: the angular velocity of a *derivation*

The observation this brief is organised around.  For **any** derivation `D` and
any orthogonal `P`, differentiating `P·Pᵀ = I` gives

```
D(P)·Pᵀ + P·D(P)ᵀ = 0        ⟹        D(P)·Pᵀ  is skew
```

so every derivation has a spin.  `d/dt` gives the angular velocity tensor
`Ω = Ṗ·Pᵀ` and `Ṗ = Ω·P`; `δ` gives the **virtual rotation** `Θ = δP·Pᵀ` and
`δP = Θ·P`.  One construction, and the "better tooling for rotation variations"
this reframing asks for falls out of it rather than being built separately.

Measured: the Leibniz step works today for both operators, transpose included —
`d/dt(P·Pᵀ)` returns `Ṗ·Pᵀ + P·Ṗᵀ` and `δ(Q·Qᵀ)` the matching pair with the ∂_q
chain rule (M2 below).  So the skewness is *derivable*, not to be asserted.

What is missing is the **skew ⇄ axial vector** bridge: `Ω = ω × I`, `Ω·a =
ω × a`, `(a × I)ᵀ = −(a × I)`.  Measured (M5): neither of the last two is in the
rule library, and both come back `exhausted` — not refuted, so they are simply
absent.  They are the natural core of the `rotation` group, and the sign
convention relating `ω` to `vec(Ω)` is to be **measured against the library's
own ε, not assumed** from a textbook.

*Done when:* skewness of `D(P)·Pᵀ` is derived (not declared) for both `d/dt` and
`δ`; `Ṗ = ω × P` and `δP = θ × P` are available; and Poisson's formula
`ė_k = ω × e_k` follows for `e_k = P·E_k`, with challenge 000027 recovered as
its planar, single-angle instance.

### I6 — rigid-body kinematics

With `r = r_C + P·ρ` and `ρ` fixed in the body,

```
v = ṙ_C + ω × (r − r_C)
a = a_C + ε × (r − r_C) + ω × (ω × (r − r_C))          ε ≡ ω̇
```

both by applying `d/dt` twice and folding through I5 — no components, no chart.

*Done when:* the acceleration comes out with its Euler and centripetal terms
from two applications of `tm.ddt()`, invariantly; and the same for a point of a
body whose reference point itself moves.

### I7 — virtual work, and δω

The payoff, and the increment that decides how far the virtual-work principle
carries.  Two parts:

1. **The commuting relation.**  δ and d/dt commute on `P` (I2 guarantees it
   through the coordinate chain), so
   ```
   δΩ − Θ̇  =  Ṗ·δPᵀ − δP·Ṗᵀ  =  Θ·Ω − Ω·Θ
   ```
   — three lines of Leibniz plus orthogonality, and in vector form the
   classical `δω = θ̇ − ω × θ` (sign to be settled by derivation, not by
   memory).  This is the identity that makes rotations usable in a variational
   argument at all, and it is the sharpest test of whether I2's commuting
   invariant was built right.
2. **Virtual work for a finite-DOF system.**  `δA = F·δr_C + M·θ` for a rigid
   body; the equations follow from `δA = 0` for *arbitrary independent* `δr_C`
   and `θ`.  Note what this does **not** need: no integral, no fundamental
   lemma over a domain — for finitely many degrees of freedom the lemma is just
   "the coefficient of each independent arbitrary vector vanishes".  That is the
   whole reason the integral could move to vibe 000111 without stalling M5A.

*Done when:* `δω` is derived; and the plane pendulum's equation of motion comes
out of d'Alembert–Lagrange (virtual work of active and inertial forces) with no
integral anywhere — the finite-DOF sibling of challenge 000026's Hamilton
route.

## What the measurements say (2026-09-02, before any rotation code)

Five things measured against the current library, three of which are
prerequisites rather than nice-to-haves:

- **M1 — a conditional claim is refuted before any rule fires.**  Giving
  `P·Pᵀ = I` as a user rule and asking `prove_equal((P·a)·(P·b), a·b)` returns
  **`refuted`**, with `fired = {}` and one pass: the component decision
  procedure (vibe 000097) expands `P` as an *arbitrary* tensor, finds the two
  sides differ, and short-circuits.  It is right about arbitrary `P` and wrong
  about the question asked.  Every rotation identity is conditional, so this
  blocks I4 outright.  The fix is one of: the refutation abstains when a symbol
  carries user hypotheses; or the hypothesis rides on the *symbol* (a declared
  property the component expander respects) rather than on a rule.  The second
  is the one that also makes `prove_equal` usable without a rule list, and it
  is why I4's "declared" representation is not optional.
- **M2 — Leibniz already reaches through the transpose**, for both operators.
  `d/dt(P·Pᵀ)` and `δ(Q·Qᵀ)` both come out as the correct two-term sums.  I5's
  skewness derivation has no representational obstacle.
- **M3 — in an orthonormal frame `e_i` and `e^i` are distinct atoms that render
  identically.**  `wcs.direction(0)` and `wcs.cobasis(0)` both print bold **i**
  and compare unequal, and no public step bridges them (`simplify`,
  `reduce_frame`, `to_concrete`, `insert_metric`, `contract_metric` all leave
  them apart).  So `expand_identity(I)` does *not* compare equal to
  `i⊗i + j⊗j + k⊗k` built from `direction()` — measured, and it is exactly the
  comparison the constructed `P = e_k ⊗ E_k` route ends on.  Invisible in the
  rendering, which is what makes it a trap rather than a nuisance.
- **M4 — the cross of two concrete frame vectors does not fold** (recorded with
  challenge 000027): `k × i` reduces to the bound sum `−ε_{i13} e_i`, and four
  further public steps are needed to reach `j`.
- **M5 — the axial-vector facts are absent, not wrong.**  `(a × I)·b = a × b`
  and `(a × I)ᵀ = −(a × I)` both come back `exhausted` under the `cross` and
  `dyadic` groups.  They are I5's first content.

M1 and M3 are the two that decide the shape of I4; neither was visible from the
plan, and both were cheap to find.

## Challenges of the group

| # | Claim | Level |
|---|---|---|
| 000025 | d/dt and δ are derivations; ∂ₜ passes through ∇ | L2 |
| 000026 | m l² φ̈ + m g l sin φ = 0 from δ∫L dt = 0 | L0 — moved to **vibe 000111** |
| 000027 | rotating frame: d/dt e_r = ω k × e_r = ω e_φ | L2 |

**000027 is Stepan's**, proposed while this brief was still being written, and
it is the one that most deserved to be here first: it is the smallest statement
in which a *vector* has a time derivative at all, so it is where the algebra and
the mechanics first meet.  It needs the chain rule in time through an angle
`φ = ω t` that is an expression rather than a coordinate; it needs the fixed
frame to be constant in time (which it is because i, j, k are not fields —
nothing had to be declared about `t`); and its middle member is Poisson's
formula for `Ω = ω k`, the planar instance of I5.

Planned for the rotation increments, one per increment as vibe 000093 requires:
`P` preserves lengths and angles (I4) · `D(P)·Pᵀ` is skew for both derivations,
and Poisson's formula in general (I5) · rigid-body velocity and acceleration
(I6) · `δω`, and the pendulum by d'Alembert–Lagrange (I7).

## Open questions — Stepan's, recorded not resolved

**Q1 — how much Lagrange/Appell machinery should tender have?**  Stepan's lean:
perhaps none here, since the Lagrangian and the acceleration energy are *user
inputs*.  If so, tender owes only the operators (∂/∂q, ∂/∂q̇, d/dt — all
shipped in I2) and "Lagrange's equations" is a line the user writes.  The case
for a helper would have to come from a challenge where the *assembly*, not the
algebra, is what hurts.  Recommendation: no helper until such a challenge
exists; revisit after I7's pendulum.

**Q2 — variations of constraint equations.**  For a holonomic `f(q,t) = 0`,
`δf = Σ (∂f/∂q_k) δq_k` is just δ applied, and works today.  The genuinely open
part is *nonholonomic*: the admissibility condition on virtual displacements is
not the variation of anything — it is a declaration about which `δq` are
allowed, and it is what Lagrange multipliers and Appell's quasi-velocities key
on.  Does tender need a first-class constraint object that filters admissible
variations, or is that user bookkeeping?  Recommendation: leave it open until a
Neimark–Fufaev challenge is attempted; deciding it now would be designing
against an imagined problem.

**Q3 — which representation of `P` is the foundation?**  Settled in I4 as
"declared, proved by constructed", but the declaration mechanism itself (trait
on the symbol vs rule in a group) is decided by M1's fix, not by taste.

**Q4 — left or right?**  `Ω = Ṗ·Pᵀ` (spatial) and `Pᵀ·Ṗ` (body) are both wanted
and are not the same tensor.  Names and defaults to be chosen when I5 is
written; the construction is the same either way.

## Order and risk

I1 → I3 are done.  I4 is blocked on the M1 refutation question and nothing
else; I5 depends on I4 and on the axial-vector facts of M5; I6 is
straightforward once I5 exists; I7 is the one that decides whether the
virtual-work principle carries as far as this reframing hopes.

The risk worth naming: **I4's declared property is a hypothesis the library
must not be able to forget.**  A rule that fires is visible; a property that
silently fails to constrain a component expansion produces a confident wrong
answer, which is exactly what M1 measured.  Whatever mechanism I4 picks, the
challenge for it must include a *negative*: a claim about a general rank-2
tensor that is still refuted, so the licence is seen to be attached to the
symbol and not to the shape.

Deliberately not here: Euler angles and other parameterizations (derived from
`P`, per vibe 000093, not foundational); the integral and Hamilton's principle
(vibe 000111); Appell and nonholonomic mechanics (blocked on Q2).
