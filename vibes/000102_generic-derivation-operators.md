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

## A view

(B) and (C) are complementary rather than alternatives: **(C) supplies the
proofs, (B) supplies the reach.**  (C) can be done now, with no engine work,
and would immediately make the ∇ rules honest — today they are asserted
identities whose "proof" is a challenge that verifies them in components, when
they are really corollaries of ∂-Leibniz.  (B) is the piece that makes one rule
serve every operator, and it needs matcher work that should not be started
without a second consumer to shape it.

The rod expansion is that second consumer, and there is a third worth noting
now: **the variation δ and the material derivative D/Dt are also derivations**.
`δ(ab) = δ(a)b + a δ(b)` is the backbone of the virtual-work formulation, which
vibe 000093 puts at the head of the applied-mechanics arc.  So the trait in (B)
is not a rod-specific accommodation — it is the same mechanism M5A needs, which
argues for designing it once, with all three consumers in view.

## Open questions

1. Is `∇_⊥` a derivation in the same sense?  It is a *projected* operator —
   `P·∇` — so it satisfies Leibniz, but its interaction with the frame is not
   the plain `e^i ∂_i`.  Does (D)'s representation cover it, or does projection
   need its own treatment?
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
