# 000110 M5A brief — time as an independent variable, and the variation δ

The first increment group of the applied-mechanics arc (vibe 000093 M5A item
1): *time, generalized coordinates q(t), the total derivative d/dt, the
variation δ, and integration by parts in time*.  Per vibe 000093's execution
protocol this is the per-milestone brief: increments, each with a one-line
testable "done when".

The δ machinery is deliberately shared with the continuum arc's Ritz route, and
the definite integral of increment I4 is M5B item 1's cross-section resultant
seen from a different side.  Built once.

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

Four things, none of them a new algebraic mechanism:

1. **Names.**  A `TensorName` is a single letter or a LaTeX command, so `q̇`,
   `q̈`, `δq` cannot be written.  The measurements above had to call them
   `u, a, p`.  This is the smallest gap and the most immediately visible.
2. **A public surface** minting the chain and the two operators consistently
   (the invariant above).
3. **The definite integral** `∫_{t₀}^{t¹} … dt` — no such node exists.
4. **Integration by parts and the fundamental lemma**, which is what turns
   `δS = 0` into the Euler–Lagrange equations.

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

### I3 — tensors and fields of time

Rank ≥ 1 fields of time already differentiate (`d/dt (r·v)` gives the product
rule, measured).  What is missing is the **commuting of ∂ₜ with an abstract ∇**:
`∂ₜ(∇⊗u)` currently raises *"differentiating a ∇ operator is not supported;
expand ∇ in a chart first"*, but `t` is not a chart coordinate, so ∂ₜ should
simply pass through the operator.  Without this, elastodynamics cannot be
written invariantly.

*Done when:* `∂ₜ(∇⊗u) = ∇⊗(∂ₜu)` and `∂ₜ(∇·σ) = ∇·(∂ₜσ)` hold with no chart,
and the refusal survives for a genuine chart coordinate.

### I4 — the definite integral over a named domain

An `Integral` node: integrand, integration variable, and a **named domain**
(`[t₀,t₁]` as an opaque named interval — the same "definite integral over an
unspecified domain as a first-class named quantity" that M5B item 1 needs for
cross-section resultants).  Ships with ANF placement, canon (linearity: a
factor free of the integration variable comes out), render, and a challenge, per
the vibe-000093 working agreement.

*Done when:* `∫(a f + b g) = a ∫f + b ∫g` for `a, b` independent of `t`, δ
commutes with ∫ over a fixed domain, and the node round-trips canon.

### I5 — by parts, endpoints, and the fundamental lemma

- `integrate_by_parts` — `∫ f (d/dt g) dt = [f g] − ∫ (df/dt) g dt`, with the
  boundary term explicit.
- A declaration that a variation vanishes at the endpoints, which kills the
  boundary term (the user's assertion, recorded — cf. vibe 000102 Q2).
- `fundamental_lemma` — from `∫ X_k δq_k dt = 0` for arbitrary `δq_k`, conclude
  `X_k = 0`.

*Done when:* challenge **Lagrange equations of the pendulum**: from
`δ∫L dt = 0` with `L = ½ m l² φ̇² + m g l cos φ`, derive
`m l² φ̈ + m g l sin φ = 0` on the public surface (L2).

## Order and risk

I1 → I2 are independent of I4 → I5 and unblock the whole of item 2 (rotation
tensors) as well, since angular coordinates and their rates use the same chain.
I3 is small and stands alone.  I4 is the only new node kind in the group, and
therefore the only place the vibe-000085 lesson applies: the domain is *data on
the node*, never a presentation wrapper.

Not in this group, and noted so they are not smuggled in: the material
derivative `D/Dt = ∂ₜ + v·∇` is the same operator form, but writing `v·∇`
invariantly runs into the ∇-positionality wall (`apply_operators(v·∇ u)` today
reads the ∇ leftward and gives `(∇·v) u`); challenge 000019 works around it by
expanding ∇ first.  That is a continuum-arc concern, not a time-and-variation
one.
