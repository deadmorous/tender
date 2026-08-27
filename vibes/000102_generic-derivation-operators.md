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

## Open questions

1. ~~Is `∇_⊥` a derivation in the same sense?~~  **Settled by (5.2)**: `∇⊥ ≡
   e_α ∂/∂x_α` is the same shape as `∇ = R^i ∂/∂q^i`, summed over the
   cross-section indices only.  No projection machinery needed.
1b. Is the **variation δ** representable as `Σ c_k ⊗ D_k`?  It is a derivation,
   but not obviously a coefficient-weighted sum of coordinate partials — which
   is the one place (D) might not reach, and therefore the argument for keeping
   (B)'s trait available.
2. Where does the **formal small parameter** λ live?  `∇ = ∇_⊥ + λ t ∂_s` is
   an expansion in λ, and collecting orders in λ is separately on the roadmap
   (vibe 000093 M5B).  Are these one feature or two?
3. Should the five existing ∇ rules survive (B), as a specialization the engine
   can use directly, or be deleted in favour of the schema?  Specializations
   fire faster; two representations of one fact is how drift starts.
4. Does the trait belong on the *node* or on a registry keyed by symbol?  The
   identity library lives in Python; a trait that users can set on their own
   operators probably should too.

## Status

Brainstorm.  Nothing scheduled.  The immediate, no-engine-work step if we want
one is (C): re-derive the existing ∇ rules through the frame route, so they
become theorems with a generic script rather than asserted identities.
