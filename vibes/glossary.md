# Glossary

Every abbreviation and piece of project jargon used across the vibes, the
source, and the Python surface, in one place — so a term met months later can
be looked up rather than reverse-engineered.  Unlike the numbered vibes this
file is not a dated discussion: it is a living index, edited in place.

**Rule:** spell a term out at its first use in a vibe, doc comment or
docstring, and add it here in the same commit.  If it is not worth an entry, it
is probably not worth abbreviating.

## Rewriting engine

| Term | Meaning |
|---|---|
| **AC** | *Associative-commutative.*  Standard term-rewriting vocabulary for matching modulo those two laws: the pattern `a·b` should match the target `b·a` when `·` commutes.  "AC matching" means the matcher tries the orders the operator permits instead of comparing operands positionally (vibe 000096 increment 3). |
| **ANF** | *Algebraic normal form* — the canonical form of vibe 000037, the theory-T0 decision procedure `canonicalize` computes.  Largely superseded in the code by `Nf` (below), which is the flat data structure implementing it. |
| **Nf / nf** | *Normal form* — the flattened intermediate representation of vibes 000057/000058 (`src/include/tender/nf.hpp`): an additive set of `Term`s, each `coeff · [scalars] · [tensors]`.  The `tender::nf` namespace. |
| **T0** | The equational theory `canonicalize` decides (vibe 000037): associativity/commutativity of the right operators, α-renaming of dummies, like-term collection, tensor symmetries.  Deliberately *excludes* distributivity and contraction identities — those are rewrite rules for the engine, not normalization. |
| **e-graph** | *Equality graph.*  A data structure holding many equivalent forms of an expression at once, compactly: **e-nodes** (operator applications) grouped into **e-classes** (sets of provably equal forms).  Rewrites merge e-classes rather than replacing terms, so no rewrite has to be undone. |
| **e-node / e-class** | One operator application inside an e-graph / a set of e-nodes known to be equal.  See above. |
| **e-matching** | Pattern matching *inside* an e-graph — a pattern may match any representative of an e-class, not just one syntactic form. |
| **saturation** | Running every rewrite rule everywhere it matches, repeatedly, until nothing new is learned (the *fixed point*) or a budget stops it.  "Equality saturation" is this over an e-graph. |
| **decorated name** | A tensor name written as a LaTeX command applied to a braced name, nesting: `\dot{q}`, `\ddot{\phi}`, `\delta{\dot{q}}`.  One opaque atom — the decoration is notation for the reader, and carries no meaning to the algebra (vibe 000110 I1). |
| **α-renaming** | Consistently renaming bound (dummy) indices, so `Σ_i a_i b_i` and `Σ_j a_j b_j` are recognised as the same expression.  From the λ-calculus notion of α-equivalence. |
| **LHS / RHS** | Left- / right-hand side (of an identity, equation, or rewrite rule). |
| **DAG** | *Directed acyclic graph* — the shape of an expression once identical subtrees are shared (hash-consed) rather than duplicated. |
| **BFS** | *Breadth-first search.* |

## Tensor algebra and coordinates

| Term | Meaning |
|---|---|
| **WCS** | *World Cartesian System* — the fixed orthonormal reference frame (i, j, k) every chart is built over. |
| **CS** | *Coordinate system.* |
| **chart** | A coordinate mapping `q^i ↦ WCS` from which tender derives the whole curvilinear geometry (tangent basis, metric, scale factors, physical frame, connection).  The one supported way to introduce coordinates (vibe 000092). |
| **realm** | Which index conventions apply — `Oblique` (upper/lower distinct), `Orthonormal` (interchangeable, spelled lower by convention), etc. |
| **dyad** | A rank-2 tensor written as `a ⊗ b`. |
| **ε-δ identities** | The Levi-Civita/Kronecker contraction identities (`Σ_i ε^{ijk} ε_{ilm} = δ^j_l δ^k_m − δ^j_m δ^k_l` and friends). |
| **bac-cab** | The mnemonic for the vector triple product `a × (b × c) = b(a·c) − c(a·b)` — "back minus cab". |
| **inc ε** | The *incompatibility* tensor `∇ × (∇ × ε)ᵀ` of a strain field ε. |
| **fence** | A structural barrier canon must not distribute through — notably an operator fence, a `⊗` holding an abstract ∇ that must stay nested with the field it differentiates (vibes 000085, 000096). |

## Mechanics

| Term | Meaning |
|---|---|
| **DOF** | *Degrees of freedom.* |
| **ODE / PDE** | *Ordinary / partial differential equation.* |
| **SIF** | *Stress intensity factor* — the coefficient governing the near-tip stress field in linear fracture mechanics. |
| **FEM** | *Finite element method.* |
| **Navier–Lamé** | The elastic equilibrium operator `μ∇·∇u + (λ+μ)∇(∇·u)`. |
| **Saint-Venant** | Either the generalized extension/bending/torsion problem for a prismatic body, or the strain compatibility equations — context distinguishes them. |
| **Lamé constants** | The isotropic elastic moduli λ and μ. |
| **generalized coordinate** | An independent parameter fixing a mechanical system's configuration (the `q` of Lagrangian mechanics), and its rates `q̇`, `q̈`.  In tender these are ordinary coordinate atoms of a chart-free coordinate group; what makes `q̇` the rate of `q` is the d/dt operator built over them, not the objects (vibe 000110). |
| **rotation tensor / turn tensor** | A proper orthogonal `P` (`P·Pᵀ = I`, det P = 1) carrying one orthonormal frame onto another; the angle-free foundation of rigid-body kinematics in the Zhilin/Eliseev treatment, with Euler angles and other parameterizations *derived* from it rather than underlying it (vibe 000110 I4). |
| **spin (of a derivation)** | `D(P)·Pᵀ` for a derivation `D` and a rotation `P` — skew, because differentiating `P·Pᵀ = I` says so.  `d/dt` gives the **angular velocity** tensor Ω (axial vector ω), δ gives the **virtual rotation** Θ (axial vector θ).  One construction serving both (vibe 000110 I5). |
| **axial vector** | The vector ω of a skew tensor Ω, defined by `Ω·a = ω × a`; equivalently `Ω = ω × I`.  The bridge between the tensor and vector forms of an angular velocity. |
| **Poisson's formula** | `ė = ω × e` for a vector rigidly attached to a rotating frame — the kinematic content of `Ṗ = Ω·P` (challenge 000027 is its single-angle instance). |
| **virtual work** | `δA`, the work of the applied forces on a virtual displacement.  For finitely many degrees of freedom `δA = 0` for arbitrary independent virtual displacements concludes by equating coefficients — no integral, which is why vibe 000110 needs none (vibe 000111 owns the integral). |
| **variation** | The operator δ of the calculus of variations: a derivation (`δ(ab) = δ(a)b + aδ(b)`) sending each generalized coordinate to an independent virtual displacement `δq`.  Measured to be of the ordinary `Σ c_k ∂_k` form — coefficients `δq_k`, partials with respect to the `q_k` — so it needs no mechanism of its own (vibe 000110, settling vibe 000102's question 1b). |

## The derivation surface

| Term | Meaning |
|---|---|
| **catalogue** | `tender.steps` — every step with its category, summary, and what it wants besides the expression (vibe 000106).  The answer to "which step do I need?", as data rather than lore. |
| **primary step** | A step a derivation reaches for *by name*; the rest are the moves those are built from — still importable, but not vocabulary.  Marked `*` in `ts.describe()` (vibe 000106). |
| **need / option / kind** | What a step wants besides the expression.  A **need** is required (`basis`, `chart`, `coord`, `rules`, `level`, `op`, `identity`), an **option** is accepted (`target`, `variance`); the name of each is its **kind**, drawn from a closed list so a tool can supply it from context.  A session adds `ctx`, which no step takes — it is where the rule library is built from. |
| **fires** | A step *fires* when it does work — as distinct from changing the expression, which canonical reordering also does.  A step reports it (`StepResult.fired`) rather than having it inferred from the outside (vibe 000106). |
| **session** | A derivation in progress: `td.explore(expr)` (vibe 000108).  Owns the tree of everything tried, the path currently shown through it, and the arguments found in the user's namespace. |
| **probed / asking entry** | A row in the derivation surface's chooser that was *run* to get there, with the fingerprint delta as its evidence — against one that opens a second list instead (`apply_identity`), because trying every rule of a growing library on every redraw is work nobody asked for (vibe 000108 §11). |
| **binder** | `tender.steps.using(**context)` — the catalogue with the shared arguments already supplied, so a derivation is a list of plain `Expr -> Expr` callables rather than a list of lambdas (vibe 000108 §13). |
| **binding** | One object from the user's namespace, the kind it serves, and *the name that scope calls it* — which is what the emitted script writes, so pasted code leans on the preamble (vibe 000108 §4). |

## Project process

| Term | Meaning |
|---|---|
| **vibe** | A numbered design-discussion note in `vibes/`, `NNNNNN_subject.md`. |
| **challenge** | A certification problem in `challenges/`, one directory each, run by CI and summarised in `SCOREBOARD.md`. |
| **L1 / L2** | Certification levels (vibe 000092 §4).  **L1 verified**: the endpoint is confirmed, typically by reducing to components.  **L2 performed**: the derivation runs in direct notation on the documented public surface, as a human would write it — no internals, no trial-and-error steps. |
| **M0 … M5** | The milestones of the consolidation plan (vibe 000093): M0 harness, M1 IR consolidation, M2 engine revival, M3 API unification, M4 certification to L2, M5 the physics arcs. |
| **IR** | *Intermediate representation* — the internal form an expression is manipulated in, as opposed to the surface syntax users write. |
| **AST** | *Abstract syntax tree.* |
| **DSL** | *Domain-specific language.* |
| **API** | *Application programming interface.* |
| **CI** | *Continuous integration* — the GitHub Actions workflow in `.github/workflows/ci.yml`. |
| **DRY** | *Don't repeat yourself* (CLAUDE.md principle 6). |
| **attic** | `attic/`, where superseded code is frozen rather than deleted. |

## Odds and ends

| Term | Meaning |
|---|---|
| **LCD** | *Lowest common denominator* — in the fraction-collection sense, not the display. |
| **ADL** | *Argument-dependent lookup*, the C++ name-resolution rule that lets `visit(v, e)` find `tender::visit`. |
| **NSDMI** | *Non-static data member initializer* — a C++ default member value (`int x = 3;` inside a struct). |
| **SPbPU** | *St. Petersburg Polytechnic University* (Lurie, Eliseev, Zhilin). |
| **DT** | *Design takeaway* — an observation recorded in a vibe without being scheduled as work (vibe 000072). |
