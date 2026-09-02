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

### I4 — constrained symbols: `unit` and `orthogonal`

Stepan's requirement, and it reframes the increment: orthogonality is a
**property carried by the tensor**, the way symmetry is — and, like symmetry, it
must survive being handed to any part of the engine.  Two kinds are needed and
must be told apart: **proper** (det = +1, a rotation; these form a group under
`·`) and **improper** (det = −1, containing a reflection).

Measured, and it is the fact that decides the mechanism:

- **Symmetry is consumed by *canon*.**  `canonicalize(Sᵀ)` *is* `S` for a field
  declared symmetric — a structural identification of components, so the
  normal form can simply absorb it, and a false claim about a non-symmetric `G`
  is still correctly `refuted`.
- **Orthogonality cannot be.**  `P·Pᵀ = I` is a *quadratic* relation between
  components; no normal form on the index structure expresses it.  Nor can the
  component decision procedure use it — which is exactly why it wrongly refutes
  today (M1).

So the two properties are declared in the same place and consumed in different
ways.  The proposal:

1. **A property on the symbol**, extended to plain tensors as well as fields
   (measured: `symmetric=` is available on `ws.field` only, which is the gap
   behind Stepan's "(should) already have").  Rank-1 gains `unit` (|n| = 1) —
   needed by the reflection and turn forms below, and a constrained symbol of
   exactly the same kind.
2. **The property mints rewrite rules in the Context**, per symbol: `unit n`
   mints `n·n → 1`; `orthogonal P` mints `P·Pᵀ → I` and `Pᵀ·P → I`.  Per-symbol
   rules rather than a schema with a property-restricted pattern variable —
   a derivation has few rotations, and this needs no matcher work.
   `td.rules("rotation", ctx)` then collects whatever that context has declared,
   so the group is *context-derived* rather than static.
2b. **The derivative of a constraint is a constraint** (Stepan): `n·ṅ = 0` and
   the skewness of `Ṗ·Pᵀ` are the same phenomenon, and should be handled the
   same way.  A property is therefore an *equation* first and a rewrite rule
   second — `unit n` is `n·n = 1`, `orthogonal P` is `P·Pᵀ = I` — and for any
   derivation `D` the library can produce `D` of it:
   ```
   D(n·n = 1)   ⟹   n·D(n) = 0          (directed: n·ṅ → 0, n·δn → 0)
   D(P·Pᵀ = I)  ⟹   D(P)·Pᵀ  is skew    (directed: (D(P)·Pᵀ)ᵀ → −D(P)·Pᵀ)
   ```
   Minted as rules alongside the constraint itself, per derivation in play.  The
   pay-off is that I6's skewness stops being a hand-written special case and
   becomes an *instance*: one mechanism, and every future constraint gets its
   differentiated form for free.  It also settles the orientation question the
   general form raises — for `unit` the directed reading is obvious, for
   `orthogonal` it is the skewness rule above.
3. **The refutation abstains** when a claim contains a constrained symbol:
   status `exhausted`, never `refuted`, because the component expansion cannot
   represent the constraint.  Symmetry keeps refuting, because it can.
4. **Propagation**, which is what makes the group structure usable: `Pᵀ` is
   orthogonal with the same sign; a `·`-chain of orthogonal symbols is
   orthogonal with the product of their signs (so rotation·rotation is a
   rotation, rotation·reflection is improper).  Structural, no pattern matching.

Measured support for the pieces: a `Basis` already carries `Handedness` and a
signed cell volume ±1, so "two orthonormal frames of the same orientation" is
checkable rather than assumed.  There is **no `det`** in the library; the sign
rides on the property instead (see the open question below).

*Done when:* `prove_equal(P·Pᵀ, I)` proves for a declared rotation with no rule
list passed by hand; `(P·a)·(P·b) = a·b` proves; the same claim about an
undeclared rank-2 `A` is still `refuted`; and `n·n → 1` holds for a unit vector.

### I5 — the ways to write a rotation

An abstract declared `P` is necessary and **not sufficient** (Stepan).  The
forms that must all be first-class, each of them orthogonal *by construction*:

| Form | | Sign |
|---|---|---|
| `P = e_i ⊗ E_i` | two orthonormal frames | proper iff same handedness |
| `Q = I − 2 n⊗n` | reflection in the plane ⟂ `n`, `\|n\| = 1` | improper |
| `P = n⊗n + (I − n⊗n) cos θ + (n × I) sin θ` | the turn tensor about `n` by `θ` | proper |
| `P = P₁·P₂·…·P_n` | composition — the group property | product of signs |
| `P = exp(a × I)` | the finite-rotation vector | proper |

The last is on the list as a joke that is nearly serious: the three-term form
above is what one actually writes, so `exp` is not needed — but it is the same
tensor, and if an exponential ever arrives it arrives here.

**How they are represented.**  A property lives on a *leaf*, and every form
except the composition is a **sum**, not a leaf.  So each shipped form is a
**constructor that names the tensor, stamps the property, and registers its
defining identity**:

```python
Q = ws.reflection(n)             # a symbol Q, improper-orthogonal
td.apply_identity(Q, defn(Q))    # unfolds to I − 2 n⊗n when you want it
```

which is how one works by hand — you write `Q`, not its formula, until you need
the formula.  Composition needs no constructor: it is recognised structurally
by I4's propagation.

**The escape hatch matters more than the list.**  Stepan: "not sure what else
can appear."  **Settled: check-and-stamp.**  A form the library has never seen
is handed in, *verified* against the constraints already declared, and comes
back as a stamped symbol with its defining identity:

```python
n = ws.vector("n", unit=True)
P = ws.orthogonal_from("P", I - 2*(n*n), proper=False)
```

The verification is the engine proving `P·Pᵀ = I` from the declared
constraints — so the stamp is *earned*, and a form that is not orthogonal is
refused with its residual shown rather than silently accepted.  The five shipped
forms are pre-proved instances of exactly this path, which is why the list not
being exhaustive costs nothing.

The alternative — a recogniser that spots known shapes inline — was declined:
every form but the composition is a **sum**, canon is free to rearrange sums,
and M3 and M4 are both instances of structural recognition being more fragile
than it looks.

One hole, named rather than papered over: **check-and-stamp can verify
orthogonality but not the *sign*.**  `P·Pᵀ = I` holds for both kinds, and
without a `det` nothing distinguishes them for an abstract form.  So `proper=`
is the user's declaration, *recorded as an assertion* in the vibe-000102 Q2
sense — and the day a form arrives whose sign cannot be taken on trust is the
day `det` becomes necessary (Q5).

Each shipped form owes a challenge that its formula really is orthogonal;
measured, the reflection's proof is two rules deep — `(I − 2n⊗n)·(I − 2n⊗n)ᵀ`
expands to `I·I + 4(n·n) n⊗n − 2n⊗(n·I) − 2(I·n)⊗n`, which needs only
`I·a = a` (shipped, `identity-dot`) and `n·n = 1` (I4's unit property).

**Transport, and where the sign finally bites.**  Conjugation by a rotation is
what makes the forms compose, and it needs three rules, the last two of which
hold *only for a proper* `Q` (an improper one flips the sign — a reflection
reverses cross products):

```
Q·(a⊗b)·Qᵀ = (Q·a)⊗(Q·b)          any orthogonal Q
Q·(a × b)  = (Q·a) × (Q·b)        proper Q only
Q·(a × I)·Qᵀ = (Q·a) × I          proper Q only
```

This is the concrete consumer that justifies carrying proper/improper on the
property even though there is no `det` (Q5): without the flag these rules cannot
be minted correctly, and with it they can.  Measured: the first is *already*
canonical for the left factor (`A·(a⊗b) = (A·a)⊗b` proves), while
`(a⊗b)·A = a⊗(Aᵀ·b)` hits the M8 bug below; the two cross rules are absent.

**Polar decomposition** (`A = P·U = V·P`, `U`, `V` symmetric, `P` a rotation)
needs nothing new: *using* the theorem means declaring an abstract rotation and
an abstract symmetric tensor, which is I4.  Proving it is a different matter and
is not in this brief.

*Done when:* each form in the table constructs a tensor that satisfies
`P·Pᵀ = I` through the library's own rules, with the right sign; a composition
of two declared rotations is a rotation without being told; and an unanticipated
form can be checked and stamped by the general path.

### I6 — spin: the angular velocity of a *derivation*

The observation this brief is organised around.  For **any** derivation `D` and
any orthogonal `P`, differentiating `P·Pᵀ = I` gives

```
D(P)·Pᵀ + P·D(P)ᵀ = 0        ⟹        D(P)·Pᵀ  is skew
```

so every derivation has a spin — and a skew tensor is `ω × I`, which is the form
this project writes it in (Stepan: the standalone `Ω` is not used; `ω × I` is).
So the increment's real content is the **axial vector**, not a spin tensor:

```
d/dt :   Ṗ·Pᵀ = ω × I           Ṗ  = ω × P            ω  = −½ (Ṗ·Pᵀ)_×
δ    :   δP·Pᵀ = δo × I         δP = δo × P           δo = −½ (δP·Pᵀ)_×
```

One construction, two derivations.  The "better tooling for rotation
variations" this reframing asks for is not separate machinery — it is the same
machinery handed `δ` instead of `d/dt`.

**Notation: keep the balance of δ.**  The virtual rotation is written `δo`
(Eliseev's notation), not `θ`, so that every term of every equation carries the
same number of δ's — `δP = δo × P` balances, `δP = θ × P` does not.  This is a
real check, not a decoration: an unbalanced equation in a variational
derivation is almost always a mistake, and the eye catches it only if the
notation makes δ visible.  It is also the first place I1's decorated names earn
their keep, since `δo` is `\delta{o}` and the rate of a virtual rotation is
`\delta{\dot{o}}`.

Measured, and the conventions line up with no adjustment needed:

- **M2**: the Leibniz step works today for both operators, transpose included —
  `d/dt(P·Pᵀ)` returns `Ṗ·Pᵀ + P·Ṗᵀ` and `δ(Q·Qᵀ)` the matching pair with the
  ∂_q chain rule.  So the skewness is *derivable*, not to be asserted.
- **M7**: `(a × I)_× = −2a` in tender's own ε and `vec` conventions — measured
  on a concrete vector, so it is decidable rather than a matter of taste.  The
  inversion `ω = −½ (…)_×` is therefore the library's convention already, and
  `(a⊗b)_× = a×b` is what `expand_dyad_ops` does.

What is missing is the rest of the **axial-vector bridge**: `(a × I)·b = a × b`
and `(a × I)ᵀ = −(a × I)`.  Measured (M5): both come back `exhausted` under the
`cross` and `dyadic` groups — absent, not wrong.  They are the first content of
the `rotation` group.  A third is wanted by I8: the commutator
`(a × I)·(b × I) − (b × I)·(a × I) = (a × b) × I`.

*Done when:* skewness of `D(P)·Pᵀ` is derived (not declared) for both `d/dt` and
`δ`; `Ṗ = ω × P` and `δP = δo × P` are available with `ω = −½ (Ṗ·Pᵀ)_×`; and
Poisson's formula `ė_k = ω × e_k` follows for `e_k = P·E_k`, with challenge
000027 recovered as its planar, single-angle instance.

### I7 — rigid-body kinematics

With `r = r_C + P·ρ` and `ρ` fixed in the body,

```
v = ṙ_C + ω × (r − r_C)
a = a_C + ε × (r − r_C) + ω × (ω × (r − r_C))          ε ≡ ω̇
```

both by applying `d/dt` twice and folding through I5 — no components, no chart.

*Done when:* the acceleration comes out with its Euler and centripetal terms
from two applications of `tm.ddt()`, invariantly; and the same for a point of a
body whose reference point itself moves.

### I8 — virtual work, and δω

The payoff, and the increment that decides how far the virtual-work principle
carries.  Two parts:

1. **The commuting relation.**  δ and d/dt commute on `P` (I2 guarantees it
   through the coordinate chain), so with `Ṗ = ω × P` and `δP = δo × P`
   ```
   δ(ω × I) − d/dt(δo × I)  =  Ṗ·δPᵀ − δP·Ṗᵀ  =  (δo × ω) × I
   ```
   — three lines of Leibniz plus orthogonality, then the commutator identity of
   I6 — giving `δω = (δo)˙ − ω × δo`, balanced in δ on both sides (sign to be
   settled by derivation, not by memory).  This is the identity that makes
   rotations usable in a variational argument at all, and it is the sharpest
   test of whether I2's commuting invariant was built right.
2. **Virtual work for a finite-DOF system.**  `δA = F·δr_C + M·θ` for a rigid
   body; the equations follow from `δA = 0` for *arbitrary independent* `δr_C`
   and `δo`.  Note what this does **not** need: no integral, no fundamental
   lemma over a domain — for finitely many degrees of freedom the lemma is just
   "the coefficient of each independent arbitrary vector vanishes".  That is the
   whole reason the integral could move to vibe 000111 without stalling M5A.

*Done when:* `δω = (δo)˙ − ω × δo` is derived; and the plane pendulum's equation of motion comes
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
  is why I4's "declared" representation is not optional.  **Settled by Stepan:
  the property rides on the symbol**, so the abstention is the *other* half of
  the same fix rather than an alternative to it.
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
- **M8 — `refuted` is unsound where a `transpose` survives into the component
  check.**  Found while measuring the transport rules above, and it outranks
  everything else in this brief.  Every one of these *true* identities comes
  back **`refuted`** — the status documented as "a real negative, from a
  decision procedure independent of the rules":
  ```
  tr(Aᵀ) = tr(A)          a·Aᵀ = A·a          a·A = Aᵀ·a
  Aᵀ·Bᵀ = (B·A)ᵀ          (a⊗b)·A = a⊗(Aᵀ·b)
  ```
  Expanded by hand on the public surface, both sides of each are *identical*
  (`tr(Aᵀ)` and `tr(A)` both give `A_ii`; `a·Aᵀ` and `A·a` both give
  `A_ij a_j e_i`).  The cause, read in `src/engine.cpp`: `to_components` never
  calls `expand_dyad_ops` and has no rule for a surviving `Transpose`, so the
  node reaches the comparison as an *opaque atom*; the two sides then differ
  structurally, and `has_residue` — a whitelist naming ε, δ, the metric,
  binders, `Nabla` and `Deriv` — does not list `Transpose`, `Trace` or
  `VectorInvariant`, so the procedure believes it decided.  `(Aᵀ)ᵀ = A` escapes
  only because canon collapses the double transpose first.

  Two fixes, wanted in this order: make `has_residue` conservative about the
  unary invariants (a wrong answer becomes an honest `exhausted`), then teach
  `to_components` to push a transpose through the component form (an honest
  answer becomes the right one).  Every rotation identity in I4–I8 is stated
  with transposes, and I4's abstention design assumes this procedure is
  trustworthy, so this is a **prerequisite**, not a neighbouring bug.
- **M7 — the `vec` convention already agrees with Zhilin's.**  `(a × I)_× = −2a`,
  measured on a concrete vector in WCS, so `ω = −½ (Ṗ·Pᵀ)_×` is the library's
  own convention and nothing has to be adjusted or chosen.
- **M6 — symmetry is a *canonical form*, orthogonality cannot be.**
  `canonicalize(Sᵀ)` is structurally `S` for a symmetric field, and `Gᵀ = G` for
  a non-symmetric one is correctly `refuted` — so the property machinery already
  works end to end where the constraint is a linear identification of
  components.  `P·Pᵀ = I` is quadratic and no index normal form expresses it,
  which is precisely why it needs rules plus abstention rather than canon.  Also
  measured: `symmetric=` is available on `ws.field` but not on `ws.tensor`, and
  a symmetric `S` does not yet give `a·S·b = b·S·a` (`exhausted`).
- **M5 — the axial-vector facts are absent, not wrong.**  `(a × I)·b = a × b`
  and `(a × I)ᵀ = −(a × I)` both come back `exhausted` under the `cross` and
  `dyadic` groups.  They are I5's first content.

M1, M3 and M6 are the ones that decide the shape of I4 and I5; M8 has to be
fixed before any of them.  None was visible from the plan, and all were cheap to
find — M8 in particular fell out of checking one transport rule the conjugation
challenge needs.

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
formula for `Ω = ω k`, the planar instance of I6.

**Stepan's second challenge — the angular velocity of a finite rotation.**  For
a rotation of angle `θ` about a unit axis `n` (the finite rotation vector `θ n`,
Zhilin):

```
ω  =  θ̇ n  +  sin θ · ṅ  +  (1 − cos θ) · n × ṅ
```

Derived, not asserted: differentiate the turn tensor of I5, contract with its
transpose, and read off `−½ (…)_×`.  This is the hardest thing in the brief and
the best single indicator that the rotation machinery is real — everything
before it is definitional, and this is the first result that could come out
*wrong* rather than merely absent.  It also exercises every piece at once: the
turn tensor, the unit constraint on `n` (and `n·ṅ = 0`, its derivative), the
axial-vector bridge, and the time chain.  If it lands, I8's `δω` is the same
derivation with δ in place of d/dt.

**Stepan's third challenge — conjugation rotates the axis.**  With `P(θn)` the
turn tensor of I5 and `Q` any rotation,

```
Q · P(θ n) · Qᵀ  =  P(θ · Q·n)
```

— the same angle about the rotated axis.  *Prove* it at least; *derive* the
right-hand side at best, which is the stronger claim and the more useful one,
since the theorem is what the composition (commutation) formula for rotations
rests on.  The derivation is the three-term form conjugated term by term, and
each term needs one transport rule from I5: `Q(n⊗n)Qᵀ = (Qn)⊗(Qn)`,
`Q·Qᵀ = I`, and `Q(n × I)Qᵀ = (Qn) × I` — the last of which is exactly where
properness is *required*, so this challenge is also the negative test for the
sign: with an improper `Q` the identity is false, and the library should say so.

Planned for the rotation increments, one per increment as vibe 000093 requires:
a declared rotation preserves lengths and angles while an undeclared tensor
does not (I4) · each way of writing a rotation is orthogonal by construction
(I5) · `D(P)·Pᵀ` is skew for both derivations, and Poisson's formula in general
(I6) · Zhilin's `ω(θ, n)` and the conjugation theorem above, as the proof that
the machinery is real ·
rigid-body velocity and acceleration (I7) · `δω`, and the pendulum by
d'Alembert–Lagrange (I8).

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

**Q3 — ~~which representation of `P` is the foundation?~~**  **Settled
(Stepan, 2026-09-02):** the property is carried by the tensor, as symmetry is,
with proper and improper told apart; and an abstract declared `P` is necessary
but not sufficient — the frame-pair, reflection, turn-tensor and composition
forms are all first-class (I5).

**Q5 — ~~is a `det` operator needed now?~~**  **Settled (Stepan): no.**  The
sign rides on the property; each shipped form knows its own, and a composition
multiplies them.  `det` arrives when something cannot be done without it — the
first such thing will be a form whose sign cannot be taken on the user's word
(see I5), and the continuum arc will want one eventually.

**Q6 — ~~how should an unanticipated form be admitted?~~**  **Settled (Stepan):
check-and-stamp** — `ws.orthogonal_from(name, expr, proper=…)`, verified by the
engine against the declared constraints.  See I5.

**Q4 — left or right?**  `Ω = Ṗ·Pᵀ` (spatial) and `Pᵀ·Ṗ` (body) are both wanted
and are not the same tensor.  Names and defaults to be chosen when I5 is
written; the construction is the same either way.

## Order and risk

**I0 — fix M8 first.**  `refuted` claiming a true identity is false is the one
defect in this brief that makes the library actively misleading rather than
merely incomplete, and every rotation statement below is written with
transposes.  Nothing else should start until it is fixed and a challenge pins
it.

I1 → I3 are done.  I4 (constrained symbols) is the foundation and the only
increment with a genuine mechanism question left in it; I5 (the ways to write a
rotation) is mostly constructors once I4 exists; I6 depends on I4 and on the
axial-vector facts of M5; I7 is straightforward once I6 exists; I8 is the one
that decides whether the virtual-work principle carries as far as this
reframing hopes.

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
