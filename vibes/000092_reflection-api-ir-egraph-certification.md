# 000092 Reflection — API, internal representation, unused machinery, certification challenges

A stock-taking discussion (user request, 2026-08-02): after ~65 vibes of
incremental growth with multiple direction changes, reflect on (1) the
Python-facing interface, (2) the internal representation, (3) whether all coded
abstractions are actually used, (4) a basis set of "certification" challenges,
and (5) textbooks matching tender's notation, including the aimed-at analytical
mechanics scope.

**Feedback iteration (2026-08-03).** The user reviewed the reflection and
settled the open questions; decisions are folded into the sections below and
marked **DECIDED**. The future-scope section 6 replaces the earlier
"aspirational" tier F with the user's actual roadmap. The implementation plan
is deliberately a separate discussion — vibe 000093 — because the pieces
(IR consolidation, e-graph revival, API unification, certification) are
entangled and need their own ordering argument.

## 1. Python-facing interface

What already works well: the `Workspace` facade (vibe 000070), operator
overloading on `Expr` (`@` dot, `*` ⊗/scaling, `%` cross), charts deriving
their whole geometry from an embedding, `Derivation` as an audit trail, and
unusually good docstrings.

Problems:

**Too many parallel routes to the same mathematics.** Differentiation alone has
four surfaces: `td.partial`/`td.deriv`+`apply_operators`, the
`tender.operators` deferred `DifferentialExpr` layer, `chart.grad/div/rot/
laplacian`, and the core invariant `t.nabla` lowered by `chart.evaluate`
(vibe 000084). `DifferentialExpr` is a Python-side shadow AST that does not
compose with `Expr` (no canonicalize, no mixing in sums). Since vibe 000084
made the core-`nabla` + `chart.evaluate` route work end-to-end, it should be
the one blessed route; `tender.operators` should become sugar over it or be
atticed, and `chart.grad/div/rot` internal.

**The step catalog is the API.** `tender.derivation` exports ~30 steps, many of
which are internals wearing API clothes (`fold_equal_addends_structural`,
`implicitize`, `distribute_contraction`, `expand_dyad_ops`). Docstrings citing
vibe numbers are the tell: these are patches, not vocabulary. Proposal: a small
goal-directed verb layer — `simplify`, `expand(expr, what=...)`, `factor`,
`prove_equal(lhs, rhs)` (returning a derivation, not a bool), `components`,
`invariant` (reassembly) — with the step catalog demoted to
`tender.derivation.steps` for power users.

**Steps are silent.** A step that fires and a step that no-ops look identical;
worse, `apply_identity` with no match still returns the *canonicalized* input,
so it both "does nothing" and changes the tree. This was the vibe-000056
deciding usability failure and it is still structurally present. Every step
should report whether it fired; `Derivation.step` should surface no-ops
(warn or raise unless `optional=True`).

**Namespace split.** Knowing whether a function lives in `t`, `td`, `tb`, or
`tc` is memorization. `Workspace` (or `Expr` methods) could absorb the common
90%: `expr.simplify()`, `expr.expand(frame)`, `chart.components(...)` already
points the right way.

## 2. Internal representation

Good bones: arena-allocated `Expr` under a `Context`, rank/space metadata,
well-known kinds, an ANF that makes equality decidable, the labeled-path
selection layer (vibe 000054).

Structural issues — each has produced a recurring family of vibes:

1. **Dual representation.** The surface tree, the ANF, and (nominally) the
   e-graph are three representations; steps operate on the surface tree and
   call canonicalize ad hoc. This is the root of "load-bearing no-op steps"
   (vibe 000056 #1) and the implicit/explicit summation duality (#2). Fix
   direction: make canonical-implicit form the *invariant at step boundaries*,
   enforced by a single wrapper (the "steps self-prepare" rule of vibe 000062,
   promoted from convention to mechanism), and render directly from it.

2. **Additive/multiplicative structure is not flattened.** `Sum`, `Difference`,
   `Negate`, `ScalarDiv`, scalar-weighted sums, and summation binders are six
   shapes every step must peel. The `expand_double_dot` docstring enumerating
   the shapes it distributes through (vibe 000091) is the smoking gun. A
   flattened n-ary additive node with one rational coefficient per term, and an
   n-ary product node, as the only internal forms (Difference/Negate existing
   only in the renderer) would delete a large fraction of the 4,700-line
   `derivation.cpp`.

3. **Missing traversal combinators.** Each step reimplements "map over addends
   through all wrappers" and "bottom-up to fixpoint", with slightly different
   coverage — hence "worked in one arrangement, not another". A shared
   `map_addends` / `map_factors` / `bottomup_fixpoint` library would make new
   steps small and uniformly total.

4. **Semantic facts encoded syntactically.** The ∇ operator fence carried as a
   `Paren` (vibe 000085) is a presentation node doing semantic work. Operator
   applicativity should be node data the canonicalizer respects natively.

**DECIDED — one interface, one engine.** Items 1–3 of this reflection are one
problem, not three: the interface is unintuitive *because* the engine
underneath is a bag of manually-ordered steps. The direction is: revive the
saturation e-graph, use it extensively and consistently as the reasoning
engine, and revisit the derivation-step catalog on top of it (verbs public,
steps demoted), so the user-facing surface becomes goal-directed and
order-insensitive.

**DECIDED — chart-based coordinate systems are the one way.** There are two
routes to a coordinate system today: the old well-known-basis factories
(`coord_system.cpp`: `tb.wcs/cylindrical/spherical/polar_2d`, re-exported as
`Workspace.cylindrical()` etc.) and the chart route (`ws.chart(reference,
coords, embedding)`), which *derives* the frame, metric, scale factors, and
connection from the embedding. The old route is obsolete except for `wcs()`
(charts still need the orthonormal reference frame): every modern example
builds charts. Plan: `Workspace` grows chart conveniences
(`ws.cylindrical_chart()` …), the curvilinear well-known factories and
`coord_system` retire to the attic once tests migrate.

## 3. Are all abstractions used?

**The e-graph is complete and practically unused.** `nf_egraph` + e-matcher +
`saturate` + ε-weighted cost extraction + the identity library + benchmarks +
the Python binding exist (vibes 000042–000046, 000048), but the only callers
are its own tests. No example uses `td.saturate`; no internal step lowers to
it. The irony: it was built precisely to eliminate manual step ordering — the
exact failure mode that now makes new challenges fail. Decision needed:

- **Revive**: make saturation the backend of `prove_equal` and (bounded)
  `simplify`, seeded with the identity library. This attacks the #1 usability
  problem directly and justifies the investment already made.
- **Attic**: freeze it (move to `attic/`), stop paying maintenance every time
  the ANF changes.

Half-alive is the one wrong answer. **DECIDED: revive.** The e-graph becomes
the engine behind the goal-directed verbs and is used extensively and
consistently; the identity library becomes a first-class, growing asset. The
step catalog is revisited in the same move (vibe 000093).

Smaller findings: `coord_system` (84 lines) was validation scaffolding for
charts and is a candidate for the attic; `polynomial` is genuinely used inside
scalar simplification; `Identity`/`apply_identity` is marginal today but
becomes central if saturation is revived.

## 4. Certification challenge suite

Observation to explain first: new challenges fail because success requires
discovering a magic step sequence. So the suite must certify not just *that* an
endpoint is reachable but *how*: define pass levels

- **L1 (verified)**: endpoint confirmed by component check (`algebraic_eq`
  after chart expansion).
- **L2 (performed)**: the derivation runs in direct notation "as a human would
  write it", using only documented verbs, with no internals and no
  trial-and-error steps.

A challenge counts as certified only at L2. Proposed basis set, in
`challenges/`, CI-run:

**A. Algebra (chart-free)**
- A1 bac-cab `a×(b×c) = b(a·c) − c(a·b)` and the ε-δ contractions behind it.
- A2 `(a×b)·(c×d) = (a·c)(b·d) − (a·d)(b·c)` (Lagrange identity).
- A3 `a×(b×I) = b⊗a − (a·b)I` and the `a×B×c` family (vibes 000056, 000063).
- A4 `tr`/`vec`/`transpose` algebra: `tr(A·B) = tr(B·A)`, `(Aᵀ)ᵀ=A`,
  `vec(skew part) ↔ ×`, `A = sym A + skew A`.
- A5 double-dot invariants: `I··A = tr A`, `A··(b⊗c) = c·A·b`, energy
  `½λ(tr ε)² + μ ε··ε` reduction.

**B. Basis / coordinates**
- B1 round-trip battery: `reassemble(expand_in_basis(x)) == x` over a
  randomized set of shapes (rank ≤ 2, dots/crosses/dyads).
- B2 oblique-basis contraction with co/contravariant components (g_ij raising
  and lowering).

**C. Vector calculus (invariant ∇, any chart)**
- C1 product rules: `∇·(fu)`, `∇×(fu)`, `∇(fg)`, `∇·(a×b)`.
- C2 `∇×∇f = 0`, `∇·(∇×u) = 0`.
- C3 `∇×(∇×u) = ∇(∇·u) − Δu`.
- C4 `∇(u·v)` and `(u·∇)u = ∇(u²/2) − u×(∇×u)`.

**D. Curvilinear**
- D1 grad/div/rot/Δ tables in cylindrical and spherical vs textbook values.
- D2 `Δ(1/r) = 0` (r ≠ 0); `∇R = I`, `Δ r² = 4` (cyl) / `6` (sph) — already
  proven once, keep as regressions.
- D3 cross-chart: `cyl.evaluate(∇ ⊗ cart.position()) = I` (vibe 000090).

**E. Mechanics endpoints**
- E1 Navier–Lamé from `∇·T` (exists — `examples/navier_lame.py`).
- E2 strain compatibility `∇×(∇×ε)ᵀ = 0` incl. ε (vibe 000075/000080).
- E3 Lamé thick-walled cylinder: equilibrium in cyl components reduced to the
  ODE in σ_rr — the rotating-cylinder challenge generalized.
- E4 momentum balance `∇·T + ρb = ρü` projected in cyl and sph.

**F. Future scope** — superseded by section 6: challenges for the new
directions are extracted from the books themselves (the user has most of them
as Russian PDFs, so derivations can be lifted directly as challenge specs).

Development discipline: features are motivated by moving a specific challenge
from failing→L1→L2; the suite is the definition of "tender is a decent
product".

## 6. Future scope — the user's roadmap (feedback iteration)

Ambition: cover the derivations found in the section-5 books. (These are
exactly the books the user learned from as a student of St. Petersburg
Polytechnic University, where Lurie, Eliseev, and Zhilin worked; Gantmacher is
a favorite.) Most are available to the user as Russian PDFs — an immediate,
concrete source of challenges. Today tender covers a small core of general
linear continuum mechanics; the extensions go in two directions.

### 6a. Continuum mechanics (source: Eliseev — the mechanics-of-solids book
and the crack-theory book)

1. **Generalized Saint-Venant problem** — extension, bending, torsion of a
   prismatic body.
2. **2D elasticity** — biharmonic Airy function, then Muskhelishvili complex
   potentials. Requires complex scalars and the Kolosov–Muskhelishvili
   ∂/∂z, ∂/∂z̄ calculus (each potential depends on both z and z̄).
   Deliberately **not** an early step.
3. **Thin curved rods** — two routes: (a) voluntarily constrained strain
   field → Ritz-type approximations; (b) asymptotic analysis with a formal
   small parameter in the coordinate along the rod axis. Prerequisite:
   collecting terms at specific powers of a formal parameter.
4. **Thin shells and plates** — asymptotic analysis leading to a variant of
   Kirchhoff–Love theory. Same formal-small-parameter machinery.
5. **Applied rod/shell/plate problems** — once the governing equations are
   *derived* (3, 4), a stock of practical problems built on them.
6. **Linear fracture mechanics** — stress intensity factors via asymptotic
   analysis near the crack tip.
7. **Later/maybe**: inelasticity and nonlinear kinematics — several
   prerequisites missing; explicitly not among the first steps.

### 6b. Applied mechanics (sources: Gantmacher; Zhilin/Eliseev for rotation;
Neimark & Fufaev for nonholonomic systems)

1. **Finite-DOF constrained systems** from Gantmacher: virtual work principle,
   Lagrange equations, Appell's equations — the certification-tier material.
2. **3D rotation done right.** Gantmacher's gap is the absence of a consistent
   3D-rotation treatment; Zhilin and Eliseev supply it — rotation tensors used
   directly, in conjunction with Euler angles or other angle sets, *or with no
   angles introduced at all*. Tender should support the angle-free
   rotation-tensor calculus natively (orthogonal tensors, angular-velocity
   tensor/vector, Poisson kinematic equations).
3. **Hamiltonian mechanics** — a significant part of Gantmacher, cool to
   support; the user does not feel expert enough to drive its design best, so
   it is queued for later (possibly a future contributor, possibly the user).
4. **Nonholonomic systems** — Neimark & Fufaev as a challenge source
   (quasi-velocities, Appell/Chaplygin-style derivations).

### New capability prerequisites implied by 6a/6b

- **Formal small parameter & series**: a formal ε, truncated expansions,
  `collect_orders(expr, eps)` — gates rods, shells, plates, cracks.
- **Time & variation**: generalized coordinates q(t), total d/dt, the
  variation δ and integration by parts in time — gates Lagrange equations and
  virtual work.
- **Rotation-tensor algebra**: proper-orthogonal tensors as first-class
  citizens — gates rigid-body and rod/shell kinematics.
- **Complex scalars + z/z̄ calculus** — gates 2D elasticity; last in line.

## 5. Textbooks in tender's notation

Tender's notation is the Gibbs dyadic / direct notation as practiced by the
Petersburg school (∇ acting from the left, `a·B·c`, dyads `⊗`, physical bases
`e_i` with scale factors `h_i`). Books that actually follow it:

**Tensor algebra & analysis**
- Gibbs, Wilson — *Vector Analysis* (1901). The origin of dyadics.
- A. I. Lurie — *Theory of Elasticity* (Springer 2005; appendix is a compact
  course in exactly tender's tensor calculus).
- A. I. Lurie — *Nonlinear Theory of Elasticity* (North-Holland 1990).
- P. A. Zhilin — *Векторы и тензоры второго ранга в трёхмерном пространстве*
  (SPb, 2001); also *Рациональная механика сплошных сред*.
- V. V. Eliseev — *Механика деформируемого твёрдого тела* (SPbPU) — pure
  direct notation throughout.
- Borisenko, Tarapov — *Vector and Tensor Analysis with Applications* (Dover)
  — physical components, scale factors h_i.
- J. G. Simmonds — *A Brief on Tensor Analysis* (Springer UTM) — Gibbsian.
- D. A. Danielson — *Vectors and Tensors in Engineering and Physics*.
- N. E. Kochin — *Векторное исчисление и начала тензорного исчисления*.

Caveat: Gurtin (and Gurtin–Fried–Anand) use direct notation but the "west
coast" convention — their `grad u` is tender's `(∇⊗u)ᵀ`; useful content,
conflicting convention.

**Analytical mechanics (constrained finite-DOF systems, Lagrange / Appell /
virtual work) — the aimed-at scope**
- A. I. Lurie — *Analytical Mechanics* (Springer 2002; Аналитическая механика
  1961). The anchor: virtual work, Lagrange equations, quasi-velocities,
  Appell's equations, same school and notation as the elasticity volume.
- F. R. Gantmacher — *Lectures in Analytical Mechanics*.
- Ju. I. Neimark, N. A. Fufaev — *Dynamics of Nonholonomic Systems* (AMS
  1972) — Appell's equations and quasi-coordinates in depth.
- L. A. Pars — *A Treatise on Analytical Dynamics* — includes Gibbs–Appell.
- A. P. Markeev — *Теоретическая механика*.
- V. F. Zhuravlev — *Основы теоретической механики*.
- J. Wittenburg — *Dynamics of Multibody Systems* (Springer) — inertia
  dyadics and constrained systems in direct vector/tensor notation.
- T. R. Kane, D. A. Levinson — *Dynamics: Theory and Applications* — Kane's
  method (essentially Gibbs–Appell), direct vector/dyadic notation.

(Editions/availability quoted from memory; verify before purchasing.)
