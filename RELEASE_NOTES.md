# Release notes

Each section covers the commits between two tags, summarized by theme.

## v0.4 — 2026-09-01

*21 commits since v0.3.*

- an interactive derivation surface: a derivation is a list of steps, `ts.using`
  binds their shared arguments, a filtered chooser offers what applies, and a
  session emits the notebook cell that reproduces it
- one identity is applied as a single step that asks for its arguments, instead
  of sixteen entries guessing at them
- the nabla fence is respected throughout reassembly: `reassemble_nabla` reads a
  contracted direction pair as a Laplacian, `contract_delta` contracts through a
  derivative mark, and coefficients pool across the fence
- a component form `a_i = a·e_i` converts back to invariant form, keeping its
  derivative marks through alpha-renaming
- guards where reassembly used to produce nonsense: it refuses components that
  carry derivative marks, and a scalar denominator no longer ends a summation

## v0.3 — 2026-08-30

*86 commits since v0.2.*

- a challenge suite with a scoreboard and a CI freshness check: 23 textbook-scale
  derivations, each certified at a level, so a gap in the library is a named red
  entry rather than a hunch
- the equality-saturation engine became the user-facing engine — the verbs
  `prove_equal` and `simplify`, budgets in time and memory, a trace, refutation
  of false claims, and an extraction cost driven by the caller's intent
- the identity library moved to Python as named rule groups (Leibniz among them),
  with a dependency graph that keeps a proof from citing what it has yet to prove
- a public surface worth teaching: named charts, `ws.nabla`, internal steps
  demoted to `tender.steps`, unsupported input reported instead of raised,
  notebook display for derivations, and a cheatsheet that matches
- steps report for themselves — a catalogue, `applicable` / `why_not` / `explain`,
  and refusals that say what was wrong
- index-directed folding, one `reassemble` with a target, and fence distribution:
  the bridge between component and invariant form went from twelve steps to four

## v0.2 — 2026-08-03

*12 commits since v0.1.*

- selective expansion: address a sub-expression by path, rewrite or extract just
  that part, and read the paths off a labeled LaTeX view of the expression
- `expand_double_dot` distributes over scaled and signed sums
- the consolidation plan that shapes the following releases: the e-graph as the
  one engine, charts as the only coordinate systems, a verb-level API, and a
  certification suite

## v0.1 — 2026-07-13

*409 commits, of which 322 since `attempt_01`.*

The library as it stands was built here, on a second attempt that archived the
first.

- a rebuilt expression core with a flat algebraic normal form (Nf) underneath —
  canonicalization, a pattern matcher with subtree variables, implicit Einstein
  summation, and an e-graph with equality saturation
- vector bases and coordinate systems: orthonormal and oblique bases with a
  metric, basis-aware indices, expansion into components and reassembly back to
  invariant form
- coordinate charts derived entirely from an embedding — tangent and physical
  bases, metric, scale factors, connection coefficients — and scalar fields with
  partial differentiation and a targeted simplifier
- first-class differential operators: `∂` and an invariant `∇`, Leibniz by
  construction, `grad` / `div` / `rot` / Laplacian built from them, and
  `chart.evaluate` to express any invariant `∇`-expression in any chart
- worked endpoints that prove the whole slice: BAC-CAB, `a × I × b`, strain
  compatibility, cylindrical equilibrium, and the Navier–Lamé reduction
- a Python front end over the C++20 core, LaTeX rendering, notebook examples,
  a README and an API cheatsheet

## attempt_01 — 2026-06-12

*87 commits.*

The first attempt, kept as a tag and archived in `attic/`. It reached a working
expression tree, rational arithmetic, an index and slot system, derivations with
named steps, an identity library with pattern matching, Python bindings, and
proofs over the Kronecker delta and Levi-Civita symbol — then was restarted
because the expression grammar needed rethinking.
