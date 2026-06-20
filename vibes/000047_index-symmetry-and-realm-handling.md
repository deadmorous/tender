# 000047 Index symmetry and realm handling for the identity library

A design decision that precedes filling the gaps surfaced while reviewing the
Stage-2 identity library (vibe 000046). It governs **all four** index identities
(`delta_contraction`, `delta_trace`, `eps_delta_1`, `eps_delta_2`), not just the
one the question was first asked about.

## The problem

Each library identity is built in **one realm** (`Realm::Oblique`) with **one
fixed slot order** (δ as upper-then-lower; ε as a fixed index sequence). The
matcher is exact on both axes:

- `match_slot` (identity.cpp:71–85) rejects on any per-slot `level`, `realm`, or
  `space` mismatch.
- `match_node` for a `TensorObject` walks slots **positionally**
  (identity.cpp:162) — slot order is never relaxed.

So the rules silently fail to fire on:

1. **A different realm** — an `Oblique` LHS will not match an `Orthonormal`
   target. The rule is realm-specific.
2. **A reordered object** — `δ_a{}^p` (lower-then-upper) will not match a
   `δ^p_a` (upper-then-lower) pattern, even though the Kronecker delta is
   symmetric in value. For ε the same applies to any non-identical index
   permutation, with the added wrinkle of an antisymmetric **sign**.

`canonicalize` does no per-object slot normalization today (the `well_known ==
Delta`/`LeviCivita` handling in derivation.cpp is all in bespoke step functions
like `contract_delta`, not in the canonical form), so these are distinct
canonical e-nodes that never unify.

## Realm: parameterize the builder — decision (a)

In ~99% of expressions every index shares one realm, so we do **not** need
cross-realm matching. The builders take the realm explicitly:

> `delta_contraction(ctx, space, realm)`, …, and the e-graph saturates with the
> instance(s) the problem actually uses.

Rejected: **realm-coercion in canonicalize** (normalizing Orthonormal to a
single level so one rule covers it). More invasive, entangled with vibe-28 realm
semantics, and unnecessary given the single-realm reality. If genuinely-mixed
expressions ever arise, revisit then.

(Orthonormal has a deeper question — whether up/down is even a distinction there,
metric being the identity. Parameterizing by realm lets the Orthonormal builder
choose its own canonical slot shape without forcing a decision now.)

## Symmetry / slot order: canonicalize from the trait generators — decision (c)

`TensorTraits` already carries the mechanism: `symmetry` (`SymmetrySpec`,
value-preserving permutation generators) and `antisymmetry` (`AntisymmetrySpec`,
sign-flipping generators). These **are** the generators of the symmetry
identities. **Prerequisite (step 0):** they are currently left default-empty by
`make_delta` (expr.cpp:118) and `make_levi_civita` (expr.cpp:151); both paths
below require populating them first:

- δ: symmetry generator = the transposition (0 1) — order-2 group {id, swap}
  (the swap also swaps levels U↔L; value-preserving).
- ε rank-3: full S₃ — even perms {id, (012), (021)} = cyclic shifts are the
  symmetry generators; an odd transposition is the antisymmetry generator.

Two ways to consume the generators were weighed.

### (d) — emit symmetry/antisymmetry rewrite **rules**, rejected

Symmetry rules are **pure expanders**: they never lower cost, only add e-nodes
that are equal-up-to-sign.

- δ: each δ → 1 extra same-class e-node. Benign (linear).
- ε: even perms add 3 e-nodes to ε's class; odd perms add 3 `(−1)·ε` e-nodes to
  the **negated** class. **One ε ≈ 6 e-nodes over 2 classes.**

Because the e-matcher searches every e-node of a class, a rule with `d` deltas
and `e` epsilons then enumerates up to **2^d · 6^e** orientation combinations —
≈ **36×** for the two-ε `eps_delta_*`. These rules fire on every δ/ε in every
intermediate term and never make reductive progress, compounding with the
productive rules. This is exactly the AC/commutativity explosion that drove AC
and α into the canonical form (vibes 37/42) rather than running them as rules.

### (c) — orbit-canonical normal form, chosen

Collapse each symmetry family to **one** representative e-node at canonicalize
time: enumerate the (tiny, ≤ |group|) orbit from the stored generators, pick the
lexicographically-minimal slot arrangement (reusing the slot ordering
canonicalize already uses — DRY), and for an antisymmetric reorder extract a ±1
**sign** into the term coefficient. The ANF already hosts that sign (signed
coefficients / `Negate`-as-−1, no `Difference`), so a lone ε that canonicalizes
to −ε_min becomes `Negate(ε_min)` folded into the enclosing coefficient.

This answers the three guiding considerations directly:

1. The generators in `TensorTraits` are precisely the input — one generic
   orbit-walk, no per-tensor hand-coding.
2. "More to implement (cyclic shift for rank-3 ε)" **dissolves**: cyclic shift
   is not a special case, it is just an element of ε's generated orbit. δ-swap,
   ε-cyclic, ε-transposition, and any future symmetric tensor share one code
   path.
3. No combinatorial blow-up: the orbit collapses to a single e-node, match
   enumeration is unchanged from today, and the O(|orbit|) walk is paid **once**
   per leaf at canonicalize, not repeatedly during saturation.

It is the architecturally consistent move: structural invariants the matcher
should not rediscover belong in the canonical form — AC → canonical, α →
canonical, now symmetry/antisymmetry → canonical, each consuming data the system
already stores. Because both the identity LHS and the target pass through the
same `canonicalize`, the positional matcher then "just works" for every order.

## Net effect across the library

With (a) + (c), all four identities become realm-parameterized and
order-insensitive uniformly: `delta_contraction` matches reversed δ slots,
`delta_trace` is unaffected, and `eps_delta_1`/`eps_delta_2` match any index
ordering and sign of the input ε's.

## Next / implementation order

1. Populate δ/ε `symmetry`+`antisymmetry` generators in the builders (shared
   prerequisite; testable in isolation against the orbit they generate).
2. Generic generator-driven orbit-canonicalization of `TensorObject` in
   `canonicalize`, with sign extraction into the term coefficient (the form-
   touching change — do first per the previous discussion).
3. Add the `realm` parameter to the `identities::*` builders.

Then the Stage-2 exercise resumes on a matcher that no longer cares about realm
spelling or index order, and the distribution-as-a-rule work (vibe 000040)
remains the next *engine* item before Stage 3 bases (vibe 000036).

## Implemented

All three steps landed together.

- **Realm parameter** — `identities::{delta_contraction,delta_trace}(ctx, space,
  realm)` and `eps_delta_{1,2}(ctx, realm)`. Tests: `RealmOrthonormalContraction`
  (an Orthonormal rule contracts an Orthonormal target) and
  `RealmMismatchDoesNotFire` (an Oblique rule leaves an Orthonormal target
  untouched — `match_slot` is realm-exact).
- **Generators populated** in the builders (`expr.cpp`): δ carries one symmetry
  generator, the swap `(0 1)` with `same_level_only = false` (the symmetry holds
  across the U/L levels); ε (only at `n == 3`, the exercised dimension) carries
  two antisymmetry generators, the adjacent transpositions `(0 1)` and `(1 2)` —
  totally antisymmetric, so the even permutations arise from the closure and no
  separate symmetry generators are needed.
- **Orbit canonicalization** in `canon` (`derivation.cpp`, `canon_symmetry`):
  BFS closure of the slot sequence under the generators (tiny orbit, linear
  search), pick the orbit-minimal arrangement via the same `(level, realm,
  space, index)` key as `expr_cmp`, fold a −1 from an antisymmetric reorder into
  a leading `Negate` (the ANF's signed-coefficient slot), and return scalar 0
  when an arrangement is reached with both signs. Hooked into the `TensorObject`
  arm of `canon`, so it runs on every leaf at canonicalize time. Tests
  (`SymmetryCanon`): δ slot-swap equal, ε cyclic-shift equal, ε transposition
  flips sign (`ε^{ijk}+ε^{jik}=0`), ε repeated index is 0.

### Two notes from implementation

- **Sign-conflict ⇒ 0 is reached only by *concrete* repeats.** A repeated
  *countable* index (ε^{iik}) is rejected earlier by `materialize_implicit_sums`
  as an ill-formed Oblique same-level pair, before `canon` runs — so the zero
  branch fires on concrete-index repeats (the post-evaluation case), which is
  what the test uses.
- **`same_level_only` is carried but not yet consulted** by `canon_symmetry`.
  Both populated generators set it `false` (level-agnostic), so behaviour is
  correct today; honouring it is deferred until a partially-symmetric tensor
  (symmetric only among same-level slots) is introduced. Likewise ε's generators
  exist only for `n == 3`; other ranks stay symmetry-less until needed.

396 C++ / 98 Python pass; `identities.cpp`/`expr.cpp`/`derivation.cpp` coverage
maintained.

### Refinement: rank-2 ε and no silent gaps

A review of `make_levi_civita` surfaced two points. (1) The rank-3 *cyclic
shift* needs **no** separate `symmetry` generator: it is an even permutation
already in the closure of the two antisymmetry transpositions (sign +1), and
`canon_symmetry` walks that closure — declaring it would only re-derive orbit
nodes it already reaches. (2) The original `if (n == 3)` left antisymmetry
**silently empty** for every other rank, which is a latent bug for the
practically-important rank 2 (a rank-2 ε *is* antisymmetric). Replaced with a
`switch`: rank 2 carries its single transposition `(0 1)`, rank 3 unchanged, any
other rank **throws** rather than building a symmetry-less ε. The resolution now
runs before slot construction (fail fast). Tests: `MakeLeviCivita.Rank2` asserts
one antisymmetry generator and empty symmetry; `MakeLeviCivita.UnsupportedRankThrows`
covers rank 4. 407 C++ / 98 Python pass.

### Orthonormal level convention

In the Orthonormal realm upper and lower are interchangeable, so the library
spells every Orthonormal index **lower** by default (`level_for(realm, l)` in
`identities.cpp` lowers `Upper→Lower` only for Orthonormal; Oblique unchanged).
`delta_contraction(…, Orthonormal)` therefore yields `δ_pa δ_pb`, not `δ^p_a
δ^p_b`. This is a *construction* convention, not a `canonicalize` coercion
(consistent with rejecting (b)): matching is level-exact, so Orthonormal targets
must be spelled lower too. Tests: `OrthonormalRuleIsLowerSpelled` (the lower rule
does not fire on an upper-spelled Orthonormal target) plus the updated
`RealmOrthonormalContraction`. 398 C++ / 98 Python; identities.cpp 100%, overall
95%.
