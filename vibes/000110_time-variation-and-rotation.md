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

### I4 — constrained symbols: `unit` and `orthogonal` — **mostly done**

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
   rotation, rotation·reflection is improper).

   **Measured after I4b: neither needs any machinery.**  `Pᵀ·(Pᵀ)ᵀ` reduces to
   `Pᵀ·P` by canon's transpose involution, which the minted rule closes; and
   `(P·Q)·(P·Q)ᵀ = I` follows from the per-symbol rules plus the `transpose`
   group.  (The user's route to the first — `Pᵀ = P⁻¹` and `A·A⁻¹ = I` — says
   the same thing; tender has no inverse operator, so `P·Pᵀ = I` is the
   primitive and `Pᵀ = P⁻¹` would be the derived reading.)  What propagation
   *is* still needed for is the **sign**, which no rule computes: that belongs
   with I5's check-and-stamp.

Measured support for the pieces: a `Basis` already carries `Handedness` and a
signed cell volume ±1, so "two orthonormal frames of the same orientation" is
checkable rather than assumed.  There is **no `det`** in the library; the sign
rides on the property instead (see the open question below).

*Done when:* `prove_equal(P·Pᵀ, I)` proves for a declared rotation with no rule
list passed by hand; `(P·a)·(P·b) = a·b` proves; the same claim about an
undeclared rank-2 `A` is still `refuted`; and `n·n → 1` holds for a unit vector.

**Shipped: all but the second.**  `ws.rotation("P")`, `ws.orthogonal("Q",
proper=False)` and `ws.vector("n", unit=True)`; the constraint rides on the
symbol *and* in the context registry from one factory call; the rules are added
to every `prove_equal` automatically; the component procedure abstains when a
constrained symbol appears; and the negatives hold — an undeclared `A·Aᵀ = I` is
still refuted.  Challenge 000030.

Three corrections to the plan above, each measured:

1. **"Per-symbol rules need no matcher work" was wrong.**  A slot-less abstract
   tensor *is* a pattern variable, so the minted `P·Pᵀ → I` first read as "for
   any X, X·Xᵀ = I" and proved the orthogonality of every tensor in sight.  The
   fix is not a property-restricted variable but its dual and simpler cousin: a
   **constrained symbol is literal in a pattern**.  That is why the declaration
   is stamped on the object as well as registered in the context — the matcher
   has no `Context` in hand, and threading one through it would be a far larger
   change than the bit.
2. **The registry must be per context, not shared with children.**  Sharing
   read well in the abstract ("a child must see its parent's declarations") and
   was wrong in practice: every Python `Context` is a child of one hidden
   default, so every declaration became global and a fresh `Workspace`
   inherited the previous one's rotations.  A symbol *name* is exactly what two
   contexts reuse for different objects — the same reasoning the chart
   registries already carry.
3. **`(P·a)·(P·b) = a·b` does not prove**, and the reason is not about
   constraints at all — see I4b.

### I4b — a rule must fire inside a longer chain — **done**

Canon normalises `(P·a)·(P·b)` to the contraction chain `a·Pᵀ·P·b`, and no rule
fires on the interior run `Pᵀ·P`: a two-factor pattern does not match a
contiguous sub-run of a longer chain.  Measured with ordinary symbols and no
constraint anywhere —

```
rule  A·B → I        target  a·A·B·b        fires: nothing
```

— through the e-matcher *and* through directed `apply_identity`, so it is not a
saturation-scheduling accident.  `nf_match.cpp` does have sub-chain machinery
(`rewrite_subchain`, `splice_chain`), which makes this a narrower question than
it looks: why that path does not take this case.

This is vibe 000100's context-blocking problem — the ninth instance became the
tenth — and it now blocks the rotation arc rather than an isolated challenge, so
it stops being a neighbouring concern.  **I4b is the next thing to do**, before
I5: every rotation derivation below is a chain of contractions with a
cancellable interior, and a rule that only fires on a whole term is not much use
in one.

*Done when:* `A·B → I` fires inside `a·A·B·b`; `(P·a)·(P·b) = a·b` proves; and
challenges 000029 and 000030 lose their xfail markers.  **All three.**

It was narrower than vibe 000100's general problem, and the hint above was the
right one: `rewrite_subchain` already matched a pattern run anywhere in a chain,
at any depth, with the join operators checked.  What it refused was the *shape
of the rule*, in two symmetric places:

- **The replacement had to be a chain.**  `A·B → I` replaces a run with one
  atom, and the path bailed before attempting a match.  Now a single tensor
  factor splices in place of the run — which also covers a replacement of a
  different chain kind (a cross-valued right-hand side dropping into a
  contraction chain) as one opaque factor.
- **The pattern had to be a chain.**  `(A·B)ᵀ → Bᵀ·Aᵀ` is one unary factor, so
  it had to be matched against a chain's *elements* rather than as a run of
  them.  A bare subtree variable is excluded — `U → …` would rewrite arbitrary
  subterms, and nobody writes that on purpose.

Both fire through `fire_identity_on_term`, which is the single door used by
directed `apply_identity` *and* by e-graph saturation, so one fix served both.
Each half was checked against a build with it neutered.

One library gap fell out and is fixed with it: `identity-dot` was `I·a = a` with
a **rank-1** variable, so `P·I·Pᵀ` stalled one step from `I`.  The variable is
now unranked — `I·a = a` and `I·A = A` are the same defining property, and the
gate was an accident of how the axiom was first written.

The dividend is the one the plan predicted but could not yet demonstrate:
**rotations compose without being told.**  `(P·Q)·(P·Q)ᵀ = I` proves by
`transpose-product`, `Q-orthogonal`, `identity-dot`, `P-orthogonal` — four
rules, none of them about composition (challenge 000030).

### I5 — the ways to write a rotation — **mostly done**

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

**Commutation, and why it is a *step* and not a rule.**  The conjugation
theorem read the other way is the commutation rule — `Q·P(a) = P(Q·a)·Q` — so
two rotations may be swapped at the price of rotating the axis of the one that
moves.  Two consequences shape the increment:

- It **must be applicable at a chosen site**, because with three or more
  rotations `P₁·P₂·P₃` there are several adjacent pairs and the derivation
  depends on which one is commuted (Stepan).  The library already has the
  surface for that — the paths and `td.at` / `Expr.rewrite_at` of vibe 000054 —
  so the increment's job is to supply the step, not a new addressing mechanism.
  This is the first time that machinery is asked for by a *mechanics* problem
  rather than by a manipulation of an expression's shape.
- It **must not be a saturation rule.**  `P₁·P₂ → P(P₁a₂)·P₁` and its mirror
  are each other's inverse, so an e-graph would loop on them and the cost model
  has no reason to prefer either side.  Commutation is a *directed step the user
  aims*, which is the vibe-000102 Q1 conclusion arriving again: a transformation,
  not a pattern.

**Polar decomposition** (`A = P·U = V·P`, `U`, `V` symmetric, `P` a rotation)
needs nothing new: *using* the theorem means declaring an abstract rotation and
an abstract symmetric tensor, which is I4.  Proving it is a different matter and
is not in this brief.

*Done when:* each form in the table constructs a tensor that satisfies
`P·Pᵀ = I` through the library's own rules, with the right sign; a composition
of two declared rotations is a rotation without being told; and an unanticipated
form can be checked and stamped by the general path.

**Shipped: the reflection, the turn tensor, composition, and check-and-stamp.**
`ws.reflection`, `ws.turn`, `ws.orthogonal_from`, `ws.definition`; both
constructors *verify* on construction and refuse with the residual, so nothing
here is asserted except the sign.  Challenge 000031.

Four things learned in the building, three of them gaps closed on the way:

1. **The verification is a directed reduction, not a saturation.**
   `prove_equal` on the turn tensor exhausted memory; the same facts applied as
   directed rewrites, interleaved with `simplify_scalars`, close in four
   rounds.  It has to be that way for a reason worth keeping: `cos²θ + sin²θ =
   1` is a **step**, not a rule, so no amount of saturation would ever reach
   it.  Vibe 000102's Q1 conclusion — a transformation, not a pattern — for the
   fourth time.
2. **Canon could not transpose a scaled tensor.**  `(2A)ᵀ`, `tr(2A)` and
   `vec(2A)` did not canonicalize *at all*: not being dyads, they kept the
   unary wrapped around a ⊗ node, and `encapsulate` refuses one of those with a
   message about fence distribution — a diagnosis that sends the reader a long
   way from a missing two-line case.  All three unaries are linear over scalar
   multiplication; `expand_dyad_ops` now splits a *scaled single* operand as
   well as a dyad.  Its docstring had claimed "scalar factors pulled through"
   all along.
3. **`I·X = X` had no right-hand companion.**  Canon does not commute a
   contraction chain, so `I·X` and `X·I` are two shapes and one rule cannot
   cover both; the reflection stalled on `n·I`.  `identity-dot-right` is an
   axiom beside `identity-dot`.
4. **`a × a = 0` was not known.**  Canon folds `a×b + b×a` — the antisymmetry —
   but not its degenerate case, because the canonical ordering of a cross has
   nothing to swap when the operands are already equal.  Registered as
   `cross-self`; without it the turn tensor reduced to `I + (…)(n × n)` and
   stopped one step short.

And the four axial-vector rules of M5 are now the `rotation` group —
`skew-transpose`, `skew-dot`, `skew-dot-left`, `skew-product` — measured true
and absent before being written, and the content I6 rests on directly.

**The frame-pair form `P = e_i ⊗ E_i` was the increment's red, and M3 was
fixed next, which cleared it** — see the M3 entry below for what the defect
actually turned out to be.  `ws.frame_rotation(name, frame, reference)` builds
and verifies it, and it is the one form whose *sign* the library settles for
itself: a `Basis` records its handedness as the sign of its cell volume, so
frames of one orientation give a rotation and opposite ones a reflection.

Its reduction needs one thing the others do not — the frame's own knowledge,
`e_i·e_j = δ_ij` and the completeness `Σ e_i⊗e_i = I`.  Neither is an identity
about symbols, so the verifier takes the frames as an argument and runs those
steps alongside the rules.  The reduction ends on the identity *written out on
the frame*, and completeness is what turns that back into `I`.

**Naming (Stepan, 2026-09-03):** the three-term form is `ws.rotation(name, axis,
angle)`, not `ws.turn` — "turn" is reserved for a particular small-rotation
case.  `ws.rotation(name)` with no axis remains the abstract declaration, so
one verb covers "some rotation" and "this rotation".

### I6 — spin: the angular velocity of a *derivation* — **first half done**

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

**Shipped: the skewness, both spins, and the axial vector.**  `tm.rotation`,
`tm.unit_field`, `tm.spin`, `tm.angular_velocity`, `tm.constraint_rules`;
challenge 000032.  Skewness is *derived* — `d/dt(P·Pᵀ)` is the spin plus its
transpose, and `P·Pᵀ` is `I`, whose derivative is zero — and the minted rule is
the citable record of that derivation rather than an assertion standing in for
it.

The claim the increment exists to make is visible in one line:

```
Ṗ·Pᵀ  =  q̇  (∂_q P)·Pᵀ            δP·Pᵀ  =  δq  (∂_q P)·Pᵀ
```

— the same tensor, with `q̇` in one and `δq` in the other.  The virtual rotation
is not separate machinery; it is this construction handed a different
derivation.

Two things had to be true, and neither was in the plan:

- **A turning rotation is a field *and* a constrained symbol, on one object.**
  Without the field dependence `d/dt P` is zero.  Without the constraint riding
  on the *same* object, `∂_t P` and `P` — which share a name — are two pattern
  **variables in one**, so every rule relating them binds the same variable to
  two different factors and never fires.  Measured: the identical rule fired on
  `(A·Bᵀ)ᵀ → −(A·Bᵀ)` and not on the marked form.  Hence
  `make_constrained_field`, and hence `tm.rotation` rather than
  `ws.rotation` for a rotation that moves.
- **δ reaches a rotation only through the generalized coordinates.**  A
  rotation depending on `t` alone has `δP = 0` — correctly, since a variation
  varies the configuration and not the clock — so `tm.rotation` takes `deps`,
  and no rule is minted about a spin that vanishes.

### I6b — `Ṗ = ω × P`, and Poisson — **done**

The guess that this needed `ω` to be a **name** was wrong, and measuring said
so: the step from "the spin is skew" to "the spin is `ω × I`" is a special case
of an **unconditional** identity, so there is no hypothesis to encode and
nothing to name.

```
½(A − Aᵀ)  =  −½ (A_×) × I        for every rank-2 A
```

A skew tensor is its own skew part, so `S = −½(S_×) × I` follows.  Four rules
shipped: that decomposition, its converse (one theorem, two directed rules —
one *extracts* an axial vector, the other *consumes* one), and the rank-2 forms
of `(a × I)·B = a × B` and `(a × B)·c = a × (B·c)`.

`tm.poisson(P)` derives `D(P) = w × P` in three links and returns it as a
citable `Identity`, refusing if a link fails:

```
ω × I      = Ṗ·Pᵀ      skewness (I6) + the decomposition
(ω × I)·P  = ω × P     skew-dot-tensor
(Ṗ·Pᵀ)·P   = Ṗ         orthogonality
```

Poisson's `ė_k = ω × e_k` is then one step from it, exactly as Stepan put it —
`ė_k = Ṗ·E_k = (ω × P)·E_k = ω × (P·E_k)` — and `δP = δo × P` is the *same
call* with `δ` passed instead, which is the arc's claim discharged rather than
asserted.  Challenge 000033.

Three things measured on the way, all of them costs of the current design:

- **`refuted` was unsound once more, and this one was live.**  `½(A − Aᵀ) =
  −½(A_×) × I` came back **refuted** — true, and called false — because
  `to_components` never *distributed*: one side arrived with its coefficient
  factored outside a sum and the other term by term, and the difference of
  shape read as a difference of value.  Fixed in the same reduction loop as I0.
  Two false refutations in two increments, both in `to_components`, both
  "the reduction did not finish and the leftovers were compared".
- **A rule does not reach inside a parenthesised sum.**  After `axial-to-skew`
  fires, the spin's transpose sits inside `½(S − Sᵀ)`, where the skewness rule
  cannot see it — `rewrite_in_factor` does not descend into a `Paren`.  The
  derivation distributes first, which works; the general fix is vibe 000100's,
  and this is another instance for its file.
- **Mixed-operator associativity is rank-conditional.**  Stepan expected the
  fence design to give `(a × B)·c = a × (B·c)`.  Canon does flatten a chain of
  *one* operator — `(A·B)·c` and `A·(B·c)` are structurally one form — but not
  across two, and it is right not to: with a rank-1 middle operand the two
  groupings are not both well-formed (`b·c` is a scalar, and `a ×` a scalar is
  nothing).  A flattening keyed on the operator cannot decide that, so the fact
  is a rank-gated rule.

### I7 — rigid-body kinematics — **done but for the cone**

**Composed rotations first.**  For `P = P₁·P₂`, the transport rule of I5 gives
the angular velocities' composition law directly:

```
Ṗ·Pᵀ = Ṗ₁·P₁ᵀ + P₁·(Ṗ₂·P₂ᵀ)·P₁ᵀ = (ω₁ + P₁·ω₂) × I     ⟹    ω = ω₁ + P₁·ω₂
```

— the second term needing `P₁·(a × I)·P₁ᵀ = (P₁·a) × I`, i.e. properness again.
This is the bridge from the rotation increments to the kinematics: a body whose
orientation is a product of rotations about named axes has an angular velocity
that is a sum of transported ones, and that is what a real problem is written
in.

With `r = r_C + P·ρ` and `ρ` fixed in the body,

```
v = ṙ_C + ω × (r − r_C)
a = a_C + ε × (r − r_C) + ω × (ω × (r − r_C))          ε ≡ ω̇
```

both by applying `d/dt` twice and folding through I5 — no components, no chart.

*Done when:* `ω = ω₁ + P₁·ω₂` is derived for a composition; the acceleration
comes out with its Euler and centripetal terms from two applications of
`tm.ddt()`, invariantly; and the same for a point of a body whose reference
point itself moves.  **All three done**; the rolling cone remains, and it is a
*problem* rather than an identity — it needs the rolling constraint, which is
where I8's admissibility form comes in, so it may belong there.

**Shipped: the composition law, the velocity and the acceleration**, all
invariantly, with challenge 000034 and `tm.reduce` (the directed reduction with
everything a chain knows — the rules a rotation derivation needs come from
three places, and `prove_equal` gathers them while a directed derivation does
not).  `ω = ω₁ + P₁·ω₂` needs the transport rule, which is where the
proper/improper sign finally does work, and the challenge carries the negative:
a reflection's transport flips, and the rotation's sign is not available to it.

The acceleration is derived by differentiating the **velocity**, not the
position twice — which is how one does it by hand, and here it is also what
keeps every rewrite's right-hand side a single term: a rule whose RHS is a sum
cannot be spliced into a chain (an I4b limit), so `P̈ = ε×P + ω×(ω×P)` would not
fire inside `P̈·ρ`.  Going through the velocity never forms that shape.

Four findings, one of them serious:

- **A rule about a derivative rewrote the undifferentiated symbol.**  The
  matcher's literal-atom comparison checked name, rank and slots but *not* the
  applied-derivative marks, so `∂_t P → ω × P` fired on a bare `P` — and then
  on its own output, thirteen times over, which is how it was noticed.  The
  same hole existed for a pattern *variable* carrying marks.  Both fixed; marks
  are part of identity (vibe 000077 step D) and now the matcher agrees.
  Poisson's rule is the first shipped rule whose left-hand side carries a mark,
  which is why nothing caught it earlier.
- **A lone-factor pattern could not reach inside a *cross* chain** — the I4b
  path gated the pattern's chain kind against the target's, and a one-factor
  pattern belongs to no kind.  `∂_t P` sits inside `ω × ∂_t P` in every
  acceleration.
- **Naming ω earns its keep here, where I6b did not need it.**  `ω̇` of a name
  is one mark; `ω̇` of the formula `−½(Ṗ·Pᵀ)_×` is a page, and the second
  derivative stops being readable.  `tm.angular_velocity(P, name=…)` mints it
  as a field and registers the formula as its definition.
- **Two rules can race.**  Adding `cross-dot-assoc-tensor` diverted the turn
  tensor's verification into `a × (b × I)`, a shape `skew-product` no longer
  reached, and the reduction stopped one step short.  `cross-skew` is the exit
  from it: two routes into one place, so both need a way out.  A directed
  reduction is order-sensitive in a way saturation is not, and this is the
  price.

**The challenge for it (Stepan): a cone rolling on a plane** (Zhilin), or
another body whose orientation is composed of two or three rotations about fixed
axes.  It is the right shape for this increment because nothing in it is
abstract: named axes, a composition, a rolling constraint, and an angular
velocity that must come out along the contact line rather than being asserted
to.  Its rolling condition is integrable — nonholonomic in form, effectively
holonomic in substance (Stepan) — so it is a kinematics challenge that happens
to carry a constraint, not a nonholonomic one; see Q2.  It exercises the composition law above, the transport rule, and the time
chain together, and it is the first challenge in the brief that is a *problem*
rather than an identity.

### I8 — virtual work, and δω — **δω started**

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
**Shipped (I8, second half):** `tm.poisson_rules(P)` — Poisson *per
coordinate*, `∂_c P = ĉ × P` with the axis **named** — `tm.coefficients(δA)`,
the `triple-rotate` identity, and challenge 000037.  The virtual displacement
`δr = δr_C + δo × (r − r_C)` is the velocity with δ in place of d/dt, the same
call with a different derivation; the generalized force `Q_q = F·∂_q r_C +
(ρ × F)·q̂` comes out with its moment term rather than having one put in; and
`δA = 0` concludes by equating coefficients, which is the whole of the
fundamental lemma for finitely many degrees of freedom.

Two more instances of the naming and reach lessons, both now familiar:

- **Poisson has to be per coordinate.**  The operator form is a *sum* for
  several coordinates and a *product* (`δq ∂_q P`) even for one, and neither can
  be matched inside a contraction chain.
- **The axis has to be named.**  `ĉ = −½(∂_c P·Pᵀ)_×` mentions `∂_c P`, so the
  rule rewrites its own right-hand side — measured, seven times deep before the
  reduction was stopped.  `q̂` also *reads* as what it is.
- And one new instance of the reach problem: the triple-product rotation proves
  on atoms and **not** with a compound operand — `(q̂ × (P·ρ))·F` against
  `((P·ρ) × F)·q̂` comes back `exhausted`.  Pinned in the challenge as a
  negative so it is not mistaken for a truth.

2. **Virtual work for a finite-DOF system.**  `δA = F·δr_C + M·θ` for a rigid
   body; the equations follow from `δA = 0` for *arbitrary independent* `δr_C`
   and `δo`.  Note what this does **not** need: no integral, no fundamental
   lemma over a domain — for finitely many degrees of freedom the lemma is just
   "the coefficient of each independent arbitrary vector vanishes".  That is the
   whole reason the integral could move to vibe 000111 without stalling M5A.

**δω, so far:** verified for **one** generalized coordinate — challenge 000036,
and the cross term is *proved* zero rather than assumed, `a × a = 0` doing the
work.  With **two** it stalls, and what stalls is not the identity but a
theorem beneath it: the residual is the integrability condition
`∂_r a_q − ∂_q a_r = −a_q × a_r`, which follows from `∂_q∂_r P = ∂_r∂_q P` (canon
already gives that — marks are sorted) once each partial spin can be put in
axial form.  The missing rule is `vec((a×I)·(b×I)) = −(a×b)`, measured true and
absent.

One change of substance came out of it, and it is the kind that should have
been obvious earlier: **the differentiated constraints are minted per
independent variable, not per operator.**  `d/dt P` for a rotation of two
coordinates is `q̇ ∂_q P + ṙ ∂_r P`, so a rule about the whole spin is a rule
about a *sum* — and a multi-term left-hand side is exactly what the matcher
cannot compile, so the rule died the moment anything distributed.  Each partial
spin is skew in its own right (differentiate `P·Pᵀ = I` by one coordinate), and
a sum of skew terms is skew, so the finer statement is both truer and usable.
`tm.poisson` reduces to a fixed point for the same reason: one term per
coordinate, each needing its own pass.

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
- **M3 — two spellings of one frame vector.**  ~~`e_i` and `e^i`~~ — the
  variance reading was wrong, and fixing it started by measuring again:
  `basis(0)` and `cobasis(0)` are the *same object* in an orthonormal frame.
  What differed was the **value symbol** `i` and the **indexed frame vector**
  `e₁`: `wcs.basis(0)` returns the first, `wcs.direction(0)` the second, both
  print bold **i**, and `structural_eq` said they differed.  So
  `expand_identity(I)` did not compare equal to `i⊗i + j⊗j + k⊗k`, and nothing
  on the page said why.

  **Fixed (I5 postscript):** canon folds a *concrete* indexed direction into
  its value symbol, in the direction the renderer had already chosen — a
  difference invisible in the notation is not a distinction the algebra should
  keep.  Symbolic `e_i` is untouched: it is a bound direction that completeness
  and reassembly match on, and it has no value symbol to fold to.  One
  consumer had to follow: `fold_operator` compared a *raw* operator's
  coefficients against a canonicalized target, so it now canonicalizes the
  operator first — the vibe-000060 "steps self-prepare" rule, arriving as a
  consequence.
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

  Two fixes, wanted in this order: make `has_residue` conservative (a wrong
  answer becomes an honest `exhausted`), then make the reduction finish the job
  (an honest answer becomes the right one).  Every rotation identity in I4–I8 is
  stated with transposes, and I4's abstention design assumes this procedure is
  trustworthy, so this is a **prerequisite**, not a neighbouring bug.

  **Done — I0, commit below.**  Implementing it corrected the diagnosis above in
  one respect and widened it in another.  The transpose *was* being pushed
  through by `expand_in_basis`; what survived was the layer beneath — the frame
  dots `i·j` that opening a trace or transpose *creates*, which were produced
  after `simplify_basis_dot` had already run and so were never collapsed.  So
  the fault was **ordering**, not a missing case: `to_components` now iterates
  {`expand_dyad_ops`, cross, dot, canon} to a fixed point, since each pass makes
  work for the others.  And pinning the belt turned up a **second, independent
  false refutation** with no transpose in it at all: a tensor of *unknown rank*
  cannot expand on a frame, so `tr(X·Y)` and `tr(Y·X)` reach the comparison
  whole and differ structurally — the cyclicity of the trace, refuted.  Hence
  the residue rule is stated positively rather than as a list of node kinds: a
  complete reduction leaves *a polynomial in the component symbols and nothing
  else*, so any surviving contraction or invariant operator means it did not
  finish.  Challenge 000028 pins both classes and the negatives.
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

Six more are named below, all Stepan's, and they are the brief's real
acceptance test — the increments are what makes them reachable:

| | Claim | Increment |
|---|---|---|
| ω of a finite rotation | `ω = θ̇ n + sin θ ṅ + (1 − cos θ) n × ṅ` | I6 |
| conjugation | `Q·P(θn)·Qᵀ = P(θ · Q·n)` | I5 |
| commutation of two | `Q·P(a) = P(Q·a)·Q` | I5 |
| commutation of three | the same, *at a chosen site* of `P₁·P₂·P₃` | I5 |
| a cone rolling on a plane | composed rotations about fixed axes | I7 |
| the pendulum, without an integral | d'Alembert–Lagrange | I8 |

**000027 is Stepan's**, proposed while this brief was still being written, and
it is the one that most deserved to be here first: it is the smallest statement
in which a *vector* has a time derivative at all, so it is where the algebra and
the mechanics first meet.  It needs the chain rule in time through an angle
`φ = ω t` that is an expression rather than a coordinate; it needs the fixed
frame to be constant in time (which it is because i, j, k are not fields —
nothing had to be declared about `t`); and its middle member is Poisson's
formula for `Ω = ω k`, the planar instance of I6.

**Stepan's second challenge — the angular velocity of a finite rotation** —
**filed as challenge 000035, and it is red.**  It was listed as I6's
proof-of-realness and then not written; Stepan asked where it had gone, which
is the right question.  Attempting it now stops at two *different* places, one
per route, and neither is about rotations:

- **Invariant route.**  The spin of the turn tensor reduces to within a factor
  of the answer — the residual is `sin θ (cos θ − 1)·[(ṅ × I) + (n×(n×ṅ))×I]`,
  and the bracket vanishes because `n × (n × ṅ) = −ṅ`, which the library
  *proves*.  What it cannot do is fold `ṅ⊗n − n⊗ṅ` back into `(n × ṅ) × I`:
  that needs a rule whose **left-hand side is two terms**, which the matcher
  cannot compile.
- **Concrete route.**  With a genuinely moving axis (`n = cos φ i + sin φ j`,
  φ(t)) everything is decidable in components, and the whole difference
  collapses to one scalar that is identically zero:
  `cos φ (cos²φ + sin²φ − 1)(1 − cos θ)`.  Nothing folds it —
  `simplify_scalars` knows a Pythagorean *pair* but not a pair sharing a factor
  with the rest of a sum, i.e. `cos²φ·X + sin²φ·X → X`.

Both gaps are named capabilities that block other things too, which is what a
red challenge is for.  The pieces the derivation rests on are kept green in the
same challenge, so the red is known to be about the last step.

For
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

**Stepan's fourth and fifth — commutation of two and of three rotations**,
both resting on the conjugation theorem.  The two-rotation case is the theorem
read as a swap; the three-rotation case adds the part that matters, that the
derivation must **select which pair to commute** and reach a stated form, so it
is as much a test of the derivation surface (vibe 000054 paths) as of the
rotation algebra.

Planned for the rotation increments, one per increment as vibe 000093 requires:
a declared rotation preserves lengths and angles while an undeclared tensor
does not (I4) · each way of writing a rotation is orthogonal by construction
(I5) · `D(P)·Pᵀ` is skew for both derivations, and Poisson's formula in general
(I6) · Zhilin's `ω(θ, n)`, the conjugation theorem and the two commutation
derivations above, as the proof that the machinery is real ·
rigid-body velocity and acceleration, and a cone rolling on a plane (I7) ·
`δω`, and the pendulum by d'Alembert–Lagrange (I8).

## Open questions — Stepan's, recorded not resolved

**Q1 — how much Lagrange/Appell machinery should tender have?**  Stepan's lean:
perhaps none here, since the Lagrangian and the acceleration energy are *user
inputs*.  If so, tender owes only the operators (∂/∂q, ∂/∂q̇, d/dt — all
shipped in I2) and "Lagrange's equations" is a line the user writes.  The case
for a helper would have to come from a challenge where the *assembly*, not the
algebra, is what hurts.  Recommendation: no helper until such a challenge
exists; revisit after I7's pendulum.

**Q2 — ~~variations of constraint equations~~ — settled in scope (Stepan):
nonholonomic constraints are limited to those *linear in the generalized
speeds*.**  That restriction is what makes the question answerable, because both
kinds then produce the *same shape*:

```
holonomic      f(q,t) = 0            ⟹   δf  =  Σ (∂f/∂q_k) δq_k = 0
nonholonomic   Σ a_k(q,t) q̇_k + a₀ = 0  ⟹   Σ a_k(q,t) δq_k     = 0
```

— in both cases **a linear form in the variations, equated to zero**.  So there
is no constraint *object* to design: what tender owes is the linear form and the
means to read its coefficients off, and admissibility is the statement that this
form vanishes.

One distinction has to stay visible in the surface, because it is exactly where
a mechanical application of δ gives the wrong answer: **the virtual condition is
not δ of the nonholonomic constraint.**  Varying `Σ a_k q̇_k + a₀` produces `δq̇`
terms; the admissibility condition is instead the Chetaev replacement
`q̇_k → δq_k` with `a₀` dropped.  Holonomic constraints are the special case
where the two happen to coincide.  A step that performs that replacement should
say that is what it is doing, not present itself as a variation.

And in invariant form the same shape appears without generalized coordinates at
all: rolling without slipping is `v_C + ω × ρ = 0` at the contact point, whose
admissibility condition is `δr_C + δo × ρ = 0` — linear in the virtual
quantities, and balanced in δ.

**But the rolling cone is not a nonholonomic example** (Stepan): its rolling
condition is *integrable* — nonholonomic in form, effectively holonomic in
substance.  So I7's challenge exercises the **shape** of the answer above and
none of its difficulty, and this brief must not be read as discharging the
nonholonomic case.  A genuinely nonholonomic challenge — a rolling disc, a
sphere, a skate, the Neimark–Fufaev material of vibe 000093's M5A item 3 — is
still owed, and it is where the Chetaev distinction above stops being a
formality.  Worth noting that the difference between the two is itself a
derivation (is this linear form integrable?), and one tender says nothing about
today.

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

**Q4 — ~~left or right?~~**  **Settled (Stepan): left, and only left.**
`d/dt P = ω × P`, so `ω × I = Ṗ·Pᵀ`.  The body-frame `Pᵀ·Ṗ` is not carried; if
it is ever wanted it is a conjugate of this one, not a second primitive.  I6 is
written this way already.

## Order and risk

**I0 — fix M8 first** (confirmed by Stepan as the prerequisite and the first
thing to do).  `refuted` claiming a true identity is false is the one defect in
this brief that makes the library actively misleading rather than merely
incomplete, and every rotation statement below is written with transposes.
Nothing else starts until it is fixed and a challenge pins it.

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
