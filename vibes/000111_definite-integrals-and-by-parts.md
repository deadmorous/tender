# 000111 M5B brief — definite integrals over a named domain, and by parts

Split out of vibe 000110 on 2026-09-02 (Stepan): the definite integral, its
by-parts rule and the fundamental lemma are **continuum-arc** work, not
applied-mechanics work.  The reason is worth stating, because it is what made
the split obvious:

> For a system with finitely many degrees of freedom, the virtual-work
> principle needs no integral.  "δA = 0 for arbitrary independent virtual
> displacements" concludes by *equating coefficients*, not by a lemma over a
> domain.  Only Hamilton's principle — `δ∫L dt = 0` — reaches for one, and it is
> a convenience there rather than a necessity.

So M5A (vibe 000110) can go all the way to the equations of motion of a
constrained system of points and rigid bodies without any of this, while M5B
cannot take its **first** step without it: vibe 000093's M5B item 1 is
"cross-section/domain resultants — definite integrals over an unspecified
domain as first-class named quantities (area, static moments, inertia
moments)", and that gates Saint-Venant.

## Why one node serves both

The action `∫_{t₀}^{t¹} L dt` and the static moment `∫_A y dA` are the same
object: an integrand, a variable of integration, and a **domain that is named
rather than described**.  Neither wants the domain's shape — the whole point of
the semi-inverse method's bookkeeping is that `A`, `S_y`, `I_yy` are symbols
obeying algebraic relations, and the whole point of Hamilton's principle is
that `[t₀, t₁]` is fixed while the path varies.  One node, two vocabularies.

That is the entanglement vibe 000093 predicted ("the same δ machinery serves the
continuum arc's Ritz route — deliberate entanglement exploited once, built
once"), arriving from the other side: it is the *integral*, not δ, that both
arcs share.

## Increments (not yet agreed — this is a placeholder brief)

1. **The node.**  `Integral{integrand, variable, domain}`, the domain an opaque
   named atom.  Ships with ANF placement, canon (linearity: a factor free of the
   integration variable comes out), render, and a challenge — the vibe-000093
   working agreement for a new node kind, and the one place the vibe-000085
   lesson applies: the domain is *data on the node*, never a presentation
   wrapper.
2. **Named resultants.**  Area, static moments, moments of inertia as named
   integrals over a cross-section, with the relations between them.  This is
   M5B item 1 proper and the gate on Saint-Venant.
3. **By parts.**  `∫ f (d/dt g) dt = [f g] − ∫ (df/dt) g dt`, boundary term
   explicit.
4. **Endpoint vanishing.**  A recorded *declaration* that a variation vanishes
   on the boundary — the user's assertion, visible in the derivation, not a
   convention baked into by-parts (cf. vibe 000102 Q2).
5. **The fundamental lemma.**  From `∫ X_k δq_k dt = 0` for arbitrary `δq_k`,
   conclude `X_k = 0`.

## Challenge

**000026** — the plane pendulum from `δ∫L dt = 0` — moves here from vibe 000110
and is this vibe's enumerated red.  Note that the *same equation* is reachable
in M5A by d'Alembert–Lagrange with no integral (vibe 000110 I7), which makes the
pair a good demonstration of what the integral actually buys: not the answer,
but Hamilton's route to it.

## Not decided yet

- Whether the domain carries any structure at all (dimension? a boundary
  operator?) or is a bare name until something needs more.
- Whether `[f g]` (the boundary term) is a node, an `Integral` over a degenerate
  domain, or a named evaluation.
- Whether the by-parts step is directed (a step) or an identity (a rule) — the
  vibe-000102 Q1 answer suggests a *transformation*, since it must work over
  n-ary products.

None of these should be settled before the cross-section material is read;
designing the continuum arc's integral from the time side alone is how it would
come out wrong.
