# 000102 Leibniz should belong to *derivations*, not to ∇ — a brainstorm

## The objection (user, 2026-08-25)

The `leibniz` group of vibe 000101 hardcodes `∇`.  But ∇ is not special: it is
`e^i ∂_i`, and the fundamental object is the partial derivative `∂_i`.  Rules
that fire on ∇ will not fire on any *other* first-order operator — and thin-rod
mechanics needs exactly that:

```
∇  =  ∇_⊥  +  λ t ∂_s
```

(schematically; the real second term is richer) — `∇_⊥` the projection onto the
cross-section plane, `t` the tangent to the axis, `s` the arc coordinate, and λ
a formal small parameter.  Expressions of the same shape as the `leibniz` group
then appear with `∇_⊥` in place of ∇, and none of the rules fire.

The objection is correct, and it is worth stating what makes it correct: the
Leibniz rule is a property of a **derivation** — a linear map `D` with
`D(ab) = D(a) b + a D(b)` — not of any particular operator.  Every rule in the
group is a *consequence* of

1. `∂_i` being a derivation, and
2. the operator being expandable as `Σ_k v_k ∂_k` in a frame, and
3. bilinearity of `·`, `×`, `⊗`.

For instance `∇·(f u) = e^i·∂_i(f u) = (∂_i f)(e^i·u) + f (e^i·∂_i u) =
u·∇f + f (∇·u)` — the vector rule is the scalar Leibniz plus linearity of the
contraction.  Nothing in it is about ∇.

## What tender already has

**Both ingredients exist**, and the vibe-000101 rules bypass them:

- **∂-Leibniz is implemented**: `apply_operators` (vibe 000077) carries out
  `∂_x(x·f) = f + x ∂_x f` — the derivation property, hardcoded, generic in the
  operand.
- **Frame expansion is implemented**: `chart.expand_nabla` lowers ∇ to
  `Σ_i (1/h_i) e_i ∂_i`.

`examples/navier_lame.py` already performs the whole route — expand ∇, apply
∂-Leibniz, reassemble — which is precisely the derivation the user describes as
"switching to a coordinate form".  So the general capability is present; what is
missing is that it is **chart-bound**, while the M2 rules are chart-free but
operator-specific.  Neither is chart-free *and* operator-generic.

## The measured constraint

An operator slot **cannot be a pattern variable today**.  A rule written with a
rank-1 subtree variable in ∇'s position does not fire on ∇ (measured:
`exhausted`, nothing fired).  `Nabla` is its own node kind and `place_factors`
keeps operators positional, so no ordinary variable reaches that position.

That rules out the cheapest fix and shapes everything below.

## What the rod split actually looks like

Eliseev, *Mechanics of Deformable Bodies* §5.5, eq. (5.2)
(`vibes/images/nabla_split_for_curved_rods.png`):

```
∇  =  R^i ∂/∂q^i  =  ∇⊥ + v⁻¹ t (∂_s − Ω_t D)

∇⊥ ≡ e_α ∂/∂x_α          D ≡ t · x × ∇⊥
v  ≡ R₁×R₂·R₃ = λ⁻¹ + t·Ω×x
```

with `R(x_α, s) = λ⁻¹ r(s) + x`, `x ≡ x_α e_α(s)` (5.1).

Reading it closely settles more than the earlier sketch did:

1. **`∇ = R^i ∂/∂q^i` is stated as the definition** — cobasis times partial.
   The user's claim is not an analogy; it is how the source defines it.
2. **`∇⊥ = e_α ∂/∂x_α` is the *same shape*, summed over fewer indices.**  Not a
   different kind of object — a partial frame expansion (α ranges over the
   cross-section only).
3. **`D ≡ t·x×∇⊥` is a *scalar* operator built from another operator.**
   Expanded, `D = (t·x×e_α) ∂_α` — a scalar derivation whose coefficients are
   scalar *fields*.  Operators compose.
4. **The coefficients are fields, not constants.**  `v⁻¹ = (λ⁻¹ + t·Ω×x)⁻¹`
   varies with `x` and `s`.  This matters and it is benign: scaling a
   derivation by a field keeps it a derivation, since `f∂(ab) = f((∂a)b +
   a(∂b)) = (f∂a)b + a(f∂b)`.  The coefficient rides along; only `∂` acts.
5. **The whole second term is (scalar field)·(vector)·(scalar derivation)**:
   coefficient `v⁻¹`, direction `t`, derivation `(∂_s − Ω_t D)` — itself a
   field-weighted combination of partials.

So every operator here fits one shape:

> **Op = Σ_k  c_k ⊗ D_k**, where `c_k` is a vector-valued field and `D_k` is a
> **scalar derivation** — itself `Σ_i f_i ∂_i` with scalar-field `f_i`.

∇, ∇⊥, `D`, and `v⁻¹t(∂_s − Ω_t D)` are all instances, differing only in their
coefficients and in which partials they sum over.

## Can Leibniz rules be derived at all?  Yes — and the derivation is short

For `Op = Σ_k c_k ⊗ D_k` and any bilinear product `⊙`:

```
Op ⊙ (a b)  =  Σ_k c_k ⊙ D_k(a b)                     linearity of ⊙ in Op
            =  Σ_k c_k ⊙ ( (D_k a) b + a (D_k b) )    D_k is a derivation
            =  Σ_k [ c_k ⊙ (D_k a) b  +  c_k ⊙ a (D_k b) ]   distribute
```

Three ingredients, and tender now has all three: **∂-Leibniz** (`apply_operators`,
vibe 000077), **linearity**, and **distribution of a contraction over the
resulting sums and products** — which is exactly what the fence work of vibe
000101 delivered.  Nothing further is required *in principle*; what is missing
is a representation in which `Op` can be written as `Σ c_k ⊗ D_k` in the first
place, and a way to say "`D_k` is a derivation".

That is the honest answer to "are we able to derive any Leibniz rules at all":
**yes, and by one script rather than per operator** — once the representation
exists.

## Options

**(A) Parameterize the rule factories by operator.**  `ti.leibniz(op)` builds
the five rules for whatever operator symbol it is handed.  Cheap, available
today, no engine work.  But it *asserts* rather than derives — nothing checks
that `op` really is a derivation — and every new operator needs its rules
minted and, to be honest, its own proof obligation in the DAG.

**(B) A `derivation` trait on operators, plus operator-binding pattern
variables.**  Mark a symbol as satisfying Leibniz; let a rule bind any
derivation-tagged operator in the ∇ slot.  Then **one** `product-rule` schema
replaces the five ∇ rules and covers `∇_⊥`, `t ∂_s`, and anything else declared.
This is the design that matches the mathematics.  Cost: matcher work (a new
kind of pattern variable, restricted by trait) and a trait mechanism on
operator nodes.

**(C) Derive per operator, generically.**  Keep the rules as *derived* DAG
nodes, and give the derivation script an operator parameter: expand the
operator in a frame, apply ∂-Leibniz, reassemble.  One script, run once per
operator, producing that operator's rules with a real proof behind each.  This
is the coordinate-form proof the user anticipates, and it fits the DAG: rules
stop being asserted and become theorems.

**(D) Represent the expansion, not the operator.**  Give tender a first-class
notion of "operator expanded on a frame", `Σ_k v_k ∂_k`, so ∇, ∇_⊥ and
`t ∂_s` are *the same kind of thing* differing only in their `v_k`.  Then the
product rules are properties of that kind, proved once.  The most principled,
the most work, and it is what the rod expansion `∇ = ∇_⊥ + λ t ∂_s` wants
anyway — that expansion is an equation between frame-expanded derivations.

## The three designs, as interfaces

### (B) A `derivation` trait, and pattern variables that bind operators

*What the user writes:*

```python
ws = t.Workspace()

# Declare an operator and assert the property that licenses Leibniz.
grad_perp = ws.operator(r"\nabla_\perp", rank=1, derivation=True)
delta     = ws.operator(r"\delta",       rank=0, derivation=True)   # variation

# ONE schema replaces the five ∇ rules — it binds any derivation-tagged operator.
rules = td.rules("leibniz")        # now operator-generic
td.prove_equal(grad_perp @ (f * u),
               f * (grad_perp @ u) + u @ (grad_perp * f), rules)   # fires
td.prove_equal(delta * (a * b), delta(a) * b + a * delta(b), rules)  # also fires
```

*How it works:* a new pattern-variable kind, `derivation_var("D")`, matches an
operator node carrying the trait — the matcher work the measurement above shows
is unavoidable, since ordinary subtree variables cannot reach an operator slot.
The rule library then holds `product-rule`, `div-product`, `curl-product` as
*schemas*, not ∇ instances.

*What it does not do:* nothing checks that the trait is deserved.  `derivation=True`
is the user's assertion, and a wrong one silently produces wrong mathematics.
That is (B)'s real cost, and it is why (C) matters.

### (C) Derive the rules, generically, per operator

*What the user writes:*

```python
cyl, (r, th, z) = ws.cylindrical_chart()

# One script, any operator: expand in the frame, apply ∂-Leibniz, reassemble.
rules, proof = td.derive_leibniz(ws.nabla(), chart=cyl)

proof                     # a Derivation — renders as the table of steps
rules                     # [Identity('grad-product'), Identity('div-scaled'), …]
```

*How it works:* exactly the route `examples/navier_lame.py` already performs by
hand, parameterized by the operator instead of hardcoding ∇.  The returned
identities carry their derivation, so they enter the DAG as **derived nodes
with real proofs** rather than as asserted axioms — which is what the current
five are, their "proof" being a challenge that checks them in components.

*What it does not do:* it produces rules *for one operator at a time*, and it
needs a chart.  It makes the rules honest; it does not make them general.

### (D) Represent the expansion, not the operator

*What the user writes:*

```python
# An operator IS its expansion: vector coefficients ⊗ scalar derivations.
e, x, tng, s = ...                                   # frame, position, tangent, arc
d_alpha = ws.partials(x_1, x_2)                      # ∂/∂x_α
d_s     = ws.partial(s)

grad_perp = ws.vector_derivation(coeffs=e[:2], parts=d_alpha)   # e_α ∂_α
D         = ws.scalar_derivation(coeffs=[tng @ (x % e[a]) for a in (0, 1)],
                                 parts=d_alpha)                 # t·x×∇⊥
nabla     = grad_perp + (1/v) * tng * (d_s - Omega_t * D)       # Eliseev (5.2)
```

and then **no Leibniz rules exist at all** — the product rule is a property of
the *kind* `Σ c_k ⊗ D_k`, proved once, and every operator built this way
inherits it.  `∇`, `∇⊥`, `D`, `δ`, `D/Dt` are one kind of thing.

*How it works:* the six-line derivation above becomes the engine's own
knowledge.  A product rule is not matched, it is *computed* from the operator's
coefficients.

*Why it fits this problem specifically:* the rod equation `∇ = ∇⊥ + v⁻¹t(∂_s −
Ω_t D)` is an **equation between operators**.  Under (B) or (C) it is a fact the
user asserts and the library cannot check or use.  Under (D) it is an ordinary
expression — operators add, scale by fields, and compose — so the asymptotic
substitution the rod theory performs *is* algebra tender can do, rather than
something done on paper before tender sees it.

*What it costs:* a new node kind with its own canonical form, matcher support,
rendering, and the rank bookkeeping (a vector derivation applied via `·`
lowers rank; via `⊗` raises it).  By the CLAUDE.md rule it ships with ANF +
identities + render + a challenge in one increment.  It is the largest of the
three by a wide margin.

## A view

Having read (5.2), I would revise the earlier framing.  (B) and (C) are still
complementary — **(C) supplies the proofs, (B) supplies the reach** — and (C)
is still the step available with no engine work.  But the rod split makes
**(D) the destination**, not an over-engineered alternative:

- Every operator in (5.2) is already of the form `Σ c_k ⊗ D_k`.  The
  representation is not a generalisation we are inventing; it is what the
  source writes down.
- `∇ = ∇⊥ + v⁻¹t(∂_s − Ω_t D)` is an **equation between operators**.  (B) and
  (C) can only take it as given.  (D) makes it manipulable — and the whole
  asymptotic method consists of substituting it and collecting orders in λ, so
  under (B)/(C) the interesting half of rod theory happens outside tender.
- Under (D) the product rules stop being library content, which removes the
  question this vibe is about rather than answering it per operator.

So: **(C) now** — it is small, it makes the existing rules honest, and its
script is the same expansion (D) would internalise, so the work is not thrown
away.  **(D) when the rod arc starts**, designed against (5.2) directly.  **(B)
only if a consumer appears that needs a *declared* derivation with no
expansion behind it** — the variation δ may be exactly that case, since δ is
not `Σ c_k ∂_k` in any coordinate sense, and that is worth checking before
committing to (D) as the single mechanism.

The rod expansion is that second consumer, and there is a third worth noting
now: **the variation δ and the material derivative D/Dt are also derivations**.
`δ(ab) = δ(a)b + a δ(b)` is the backbone of the virtual-work formulation, which
vibe 000093 puts at the head of the applied-mechanics arc.  So the trait in (B)
is not a rod-specific accommodation — it is the same mechanism M5A needs, which
argues for designing it once, with all three consumers in view.

## Q1: does n-ary Leibniz — `D(a·b·c·d)` — come for free?

Not uniformly, and the answer separates the options more sharply than anything
else so far.

**(D): free — and for a reason worth naming.**  Under `Op = Σ_k c_k ⊗ D_k`, the
product rule is *computed*: apply `D_k` to a term, which in the `Nf` model is
already an **n-ary** product `coeff · [scalars] · [tensors]`.  "Differentiate a
term" is then "sum over its factors, differentiating one and leaving the rest",
which has no arity built in.  Two, three, ten factors are the same code path.

This is a direct dividend of the M1 flattening: because a term is n-ary in the
IR rather than a binary tree, poly-linearity is structural rather than
something to implement.

**(B): not free — and the gap is not small.**  A rule is a *pattern*, and a
pattern with two variables matches two factors.  The obvious escape is a "rest"
variable, `D(F * REST) = D(F)*REST + F*D(REST)`, applied repeatedly — but that
requires the pattern to **name** the remaining factors so the right-hand side
can put them back inside a fresh `D`.  Measured: `NfBinding::subtrees` binds one
`Factor const*` per name.  Sub-product matching *carries* the unmatched factors
through unchanged; it cannot hand them to the RHS as a group.

Nor does the recursive route the question suggests work as written:
`D(a·b·c) = D(a·(b·c))` needs a *re-association* that the flat term model has
deliberately normalized away — there is no `(b·c)` grouping to match.  So (B)
would need **variadic (associative) pattern matching**, a real matcher
extension, before n-ary Leibniz is expressible at all.

**(C): depends on what it produces.**  As a *rule factory* it inherits (B)'s
limitation — the identities it emits are ordinary patterns of fixed arity.  As a
*transformation* applied directly to the user's expression (expand the operator,
apply ∂-Leibniz to the term, reassemble) it is n-ary-native for exactly (D)'s
reason.  Worth deciding deliberately: **(C) should be a transformation, not a
rule factory**, and the rules it registers should be understood as *citable
records* of what it proved, not as the mechanism.

This is the strongest argument yet for (D), and against (B) as the primary
mechanism.

## Q2: who says a given expansion really *is* ∇?

The proposal — user-declared equivalence classes, at their own risk — is close,
but tender already has the machinery to do better, and to make the risk
*visible* where declaration is genuinely needed.

**An operator equality is an identity like any other.**  A chart defines ∇
canonically: `∇ = R^i ∂/∂q^i`, from the embedding, with nothing to assert.  Any
other expansion — Eliseev's `∇ = ∇⊥ + v⁻¹t(∂_s − Ω_t D)` — is then a *claim*
about two operators, and claims are what the identity DAG is for:

- Register it as a **derived** node, whose proof obligation is a challenge:
  expand both sides on the chart's own frame and compare.  Same shape as every
  other derived identity, same machinery, same scoreboard.
- Or register it as an **axiom** — the "on your own risk" case — which is
  legitimate but *recorded*: the DAG prints it as an axiom, so an unchecked
  equivalence is visible in the scoreboard rather than living in a user's head.

That is better than a separate equivalence-class mechanism on two counts: there
is nothing new to build, and the honest-vs-asserted distinction is the one the
DAG already draws.

Two related cases fall out of the same framing:

- **The same ∇ in different charts** (WCS vs cylindrical) is not an identity to
  prove but the *cross-chart* question of vibe 000090 — one operator, two
  expansions, related by coordinate reprojection.  Approach A already handles
  the forward direction.
- **Checking is decidable where it matters.**  Two expansions over the same
  chart can be compared by expanding both and reducing — the same component
  decision procedure that backs `Refuted` (vibe 000097).  So the challenge
  proving an operator identity has a mechanical route, not a bespoke one.

## Implementation: (D) was largely already built

Measured before planning any of it, and it reframes the cost entirely.
**(D)'s representation already exists**, in nodes tender has had since vibe
000077:

```
Op = Σ_k c_k ⊗ ∂_k     is   TensorProduct(coefficient, Deriv(coord)), summed
```

- `chart.nabla()` returns exactly that — `Σ_i (1/h_i) e_i ∂_i` as an ordinary,
  composable `Expr`.
- `apply_operators` applies the product rule to it, **generically**: it walks
  the term's factors, so it never mentions ∇.
- A **custom** operator works with no registration whatever.  Measured:
  `grad_perp = e₀ ∂_x + e₁ ∂_y`, hand-built, and `apply_operators` applies
  Leibniz to it correctly.
- **n-ary is free**: `∇(f g h)` produces its three terms per component in one
  pass, confirming the Q1 analysis.

So (D) is not a new node kind with its own canon, matcher and renderer.  It is
**a public surface over machinery that exists** — plus the reassembly gap
(below).  The estimate in "The three designs" above was wrong by a wide margin,
and wrong in the direction that matters: the principled option is also the
cheap one.

**Shipped now:**

- `td.deriv` and `td.apply_operators` **restored to the public surface**.  M3
  demoted them to `tender.steps` on the measurement that no example called
  them — correct at the time, but they are (D)'s building blocks, and that
  demotion would have hidden the mechanism this whole design rests on.
- Challenge **000024**: Leibniz certified for an operator the library has never
  heard of, over four factors, and — discharging the (C) debt — the shipped
  `∇(fg)` rule *derived* from the expansion rather than asserted.
- Cheatsheet section: operators are **built, not named**.

**What is still missing**, and it is one thing: the derived form stays in
components.  `Σ_i c_i ⊗ ∂_i(fg)` cannot be folded back into `f ∇g + g ∇f`
because `reassemble` does not descend into a contraction — **vibe 000100's
problem**, arriving for the ninth time.  Challenge 000024's L2 records it.
Until it is fixed, a derived Leibniz rule cannot be *restated* as the invariant
identity the library ships, which is why those five rules stay asserted for now.

That makes vibe 000100 the blocker for finishing (D), not a parallel concern.

## Open questions

1. ~~Is `∇_⊥` a derivation in the same sense?~~  **Settled by (5.2)**: `∇⊥ ≡
   e_α ∂/∂x_α` is the same shape as `∇ = R^i ∂/∂q^i`, summed over the
   cross-section indices only.  No projection machinery needed.
1b. ~~Is the **variation δ** representable as `Σ c_k ⊗ D_k`?~~  **Settled: yes**
   (vibe 000110, measured).  `δ = Σ_k δq_k ∂/∂q_k` — a coefficient-weighted sum
   of partials after all, once one notices the expansion is over *configuration*
   space rather than physical space, which is what made it look doubtful here.
   `apply_operators` carries it out unmodified, and `d/dt` is the same object
   with coefficients `(1, q̇, q̈, …)`.  So (B)'s trait has no consumer, and the
   argument for keeping it available lapses.
2. ~~Where does the formal small parameter λ live?~~  **Settled (user)**: λ and
   the collection of terms by powers of λ are a **separate work item**, already
   on the M5B roadmap.  Not part of the operator design.
3. ~~Should the five existing ∇ rules survive?~~  **Settled (user)**: they are
   likely to be **removed** unless they end up constituting (B); the direction
   is (D).  Whether or not they survive, **they must all be derived** — (C) —
   rather than asserted.  So (C) is not optional: it is the debt owed on what
   vibe 000101 shipped.
4. **Where a "this is a derivation" mark would live** — elaborated, since the
   question was too terse.  The mark says an operator obeys Leibniz *by
   assertion*, and there are two homes:

   - **On the node**, as a trait beside `WellKnownKind`.  The matcher can then
     test it directly, and it travels with the expression.  But it is C++ (a
     new operator needs a rebuild — exactly what moving the identity library to
     Python was meant to avoid), and being part of a node makes it part of
     *structural identity*: two operators differing only in the flag would
     compare unequal and hash differently, which is a real consequence for
     canon.
   - **In a Python registry keyed by symbol**, beside the identity library.
     Users add operators without rebuilding; structural identity is untouched.
     The cost is that the C++ matcher cannot see it, so "this rule applies only
     to derivations" must be enforced when the rule set is *built*, in Python,
     rather than during matching.

   Given the library-in-Python decision, the registry is the consistent choice.
   But the question mostly **dissolves under (D)**: an operator constructed as
   `Σ c_k ⊗ D_k` is a derivation *by construction*, with nothing to assert.  A
   mark is needed only for operators declared *without* an expansion — which is
   the δ case in 1b.  So: defer it until δ forces the issue, and if it is ever
   needed, put it in the Python registry.

## Status

Brainstorm.  Nothing scheduled.  The immediate, no-engine-work step if we want
one is (C): re-derive the existing ∇ rules through the frame route, so they
become theorems with a generic script rather than asserted identities.
