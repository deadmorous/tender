# 000095 M1 brief — IR consolidation, increment breakdown

## Progress record

- **Increment 1 DONE** (badcc82): `tender::view` combinators
  (signed_addends/sum_of over `nf::additive_flatten`, skeleton-preserving
  `map_additive_leaves` with no-op pointer reuse, canonical `map_nf_terms`,
  guarded `fixpoint`) + `expand_unary` ported.  Byte-identical.
- **Increment 2 DONE**: `view::distribute_bilinear` — the one bilinear
  distributor (Sum/Diff always; Negate, α-renamed binders, ScalarDiv,
  scaled-additive as options) with the **left-operand-first normative peel
  order**.  Ported: `dd_expand` (the whole vibe-000091 cascade deleted; only
  the dyad rule remains), `distribute_any`/`expand_products` (options all
  off), and the signed-addends family (`fold_equal_addends_structural`,
  `collect_terms`, `factor_common` now consume `view::signed_addends`;
  `collect_signed_addends` deleted).  Order note: dd_expand's old
  per-shape interleave was replaced by the left-first order — no test,
  example, or challenge output changed (860 C++ + 346 Python green,
  scoreboard unchanged).  `derivation.cpp` 4697 → 4569 (−128);
  `nf_view.cpp` 211 shared, unit-tested lines (17 tests).
  **Deliberately not ported**: `distribute_contraction` — its additive
  peeling is entangled with the vibe-000085/000088 operator-fence barrier;
  touching it belongs with increment 4's fence work, not a mechanical port.
  `flatten_factors`/`product_of` stay in derivation.cpp: single definitions,
  not duplicated peels; unifying them is cosmetic.
- **Increment 4 DONE — M1 COMPLETE**: ∇ fence as explicit data.
  `nf::ParenKind { Grouping, OperatorFence }` on `nf::Paren`; the one
  creation site (`encapsulate`'s TensorProduct-with-∇ arm, where fence-ness
  is *defined* via `contains_nabla`) stamps `OperatorFence`; downstream the
  kind is data — included in `equal`/`compare`/`hash`, required by the
  Nf matcher, carried through pattern instantiation, and keyed into the
  e-graph's `NfENode` congruence and rebuilt at extraction.  4 new tests
  incl. the raise→re-lower kind-stability round trip.  Audit note: the only
  content-sniffing ever present was at the creation site (legitimate — the
  definition); consumers never sniffed, they just couldn't distinguish.
  **Observation filed for M2**: `place_factors`' `has_operator` scans only
  top-level factors for `Deriv`/`Nabla` — an operator buried inside a fence
  factor does not make the sibling scalars positional; whether that is a
  latent scope bug deserves a dedicated look when the e-graph work touches
  term placement.  871 C++ + 349 Python green; scoreboard unchanged.
- **Increment 3 DONE**: step contract + fired/no-op reporting.
  `apply_identity` no-match now returns the input pointer untouched (was:
  canonicalized input — the vibe-000056 §1 offender); `expand_products` no
  longer rebuilds identical products (exposed by the new inert-input pointer
  tests over 9 public steps); Python `Derivation.step` records
  `(name, fired)` per step (`steps` property), warns `NoOpStep` on a no-op
  unless `optional=True`, and takes a `label=`.  **The reporting caught two
  dead steps in challenge 000001's own L2 route on its first run** (a second
  `simplify_basis_cross` and an `expand_products` — both no-ops all along);
  the route is now 7 steps, all fired, asserted as such.  867 C++ + 349
  Python tests green; scoreboard unchanged.

The per-milestone brief for M1 (vibe 000093), grounded in a code audit of
`expr.hpp`, `nf.hpp`, `nf_lower.{hpp,cpp}`, and `derivation.cpp`.

## The audit finding that reshapes M1

Vibe 000093 sketched M1.1 as "introduce flattened n-ary additive/
multiplicative forms".  **That form already exists**: `nf::Nf` (vibes
000057/000058) is exactly it — an additive set of `Term`s, each
`coeff · [sorted scalars] · [positional tensors]` with the sign carried in a
signed `Rational` coeff, contraction chains stored flat, and per-index sum
modes as data.  `canonicalize` = `raise(canonicalize_nf(e))`: lower to Nf,
canonicalize there, raise back to a *binary* `Expr`.

So the dual-representation problem (vibe 000092 §2.1–2.2) is precisely
located: **steps consume the raised binary tree and re-peel it by hand.**
Evidence:

- `expand_unary` (derivation.cpp) peels `Sum`/`Difference`/`Negate`/
  `ScalarDiv` around every unary op; `dd_expand` carries the vibe-000091
  shape list; each peel is a slightly different subset.
- Signed-addend flattening exists in **three** copies:
  `collect_signed_addends` (derivation.cpp), `nf::additive_flatten`
  (nf_lower.cpp), and the per-step peels.
- The ∇ operator fence is carried through canon as a `Paren` factor
  (nf_lower.cpp `encapsulate`) — a *grouping* node doing *semantic* barrier
  work (vibe 000085), distinguishable only by inspecting its contents.

M1 is therefore not "build a new IR" — it is **close the gap between the Nf
and the steps**: one flattening implementation, steps written against the
flat view, the fence made explicit, and steps that report what they did.

## Increments

Each keeps the build, all 318 unit tests, and the 23-challenge suite green;
the scoreboard must not change (M1 is internal — any accidental challenge
promotion or regression is a stop-and-look signal).

1. **Term-view combinators** — one new module (`src/nf_view.{hpp,cpp}` or a
   derivation-internal header), two levels:
   - *Surface level* (no canonicalization, order-preserving — required by the
     vibe-000080 "don't re-canonicalize mid-route" caveat):
     `for_each_signed_addend(e, f)` handling `Sum`/`Difference`/`Negate`/
     `ScalarDiv`-by-scalar/scalar-weighted groups/summation binders in one
     place, and a matching rebuilder.  Implemented on top of (or replacing)
     `nf::additive_flatten`; `collect_signed_addends` and the per-step peels
     become calls into it.
   - *Canonical level*: `map_nf_terms(ctx, e, f)` — lower, transform each
     `nf::Term` (or the whole term set), raise; plus `bottomup_fixpoint`.
   *Done when:* combinators unit-tested; `expand_unary`'s hand peeling is
   replaced by the surface combinator with byte-identical rendered outputs
   across the full suite.

2. **Port the worst offenders.**  Candidates, chosen by measured
   duplication: `dd_expand`'s additive handling (the vibe-000091 shape
   list), `distribute_contraction`, `expand_products`,
   `fold_equal_addends` (whose like-term collection is arguably just
   canon + implicitize — if the port shows it *is*, fold it into that and
   say so).  At least three ported.
   *Done when:* suite + challenges green with unchanged rendered outputs
   (or a listed, justified improvement); the per-step shape enumerations are
   deleted; the net `derivation.cpp` line change is recorded in this vibe.

3. **Step contract + fired/no-op reporting** (the vibe-000056 §1 fix):
   - C++ pointer contract: every public step returns its *input pointer*
     when nothing changed (several steps already do — make it universal and
     assert it in tests).
   - `apply_identity` on no-match returns the input **unchanged** — today it
     returns the canonicalized input, so it simultaneously "does nothing"
     and rewrites the tree.  Canonicalization on match stays.
   - Python: `Derivation.step` records the step name and whether it fired
     (`result is not input` / structural check), warns on a no-op unless
     `step(fn, optional=True)`; history rendering can show the fired flags.
   *Done when:* a no-op step in a `Derivation` warns; an `apply_identity`
   no-match preserves its input (new unit test); challenge 000001's L2
   narration shows per-step fired status.

4. **∇ fence as explicit data** (the vibe-000085 cleanup): the operator
   fence gets its own representation — a `kind` on `nf::Paren`
   (grouping vs operator-fence) or a dedicated `Fence` factor — created
   where `encapsulate` today wraps the fence, consulted where code today
   sniffs a Paren's contents for a ∇.  Bare `Paren` goes back to meaning
   only "opaque grouping".
   *Done when:* the vibe-000085 regression tests are green, rendering is
   unchanged, and no code path infers fence-ness by content inspection.

## Deliberately out of scope

No verb API, no e-graph work (M2), no public-surface or naming changes (M3),
no challenge promotions (M4).  The single user-visible change is the
`Derivation` no-op warning of increment 3 — everything else is internal.
`Negate`/`Difference` remain in the `Expr` variant as construction/render
forms; removing them from the surface language is not needed once no step
peels them by hand, and touching the surface belongs to M3.

## Risks

- Ported steps changing output *shape* while staying algebraically equal:
  acceptable only if the rendered form is unchanged or strictly cleaner;
  each such case gets a line in this vibe when it happens.
- `fold_equal_addends` may collapse into canon + implicitize; if so, keep
  the public name as a thin alias (public surface changes are M3's).
- The fence change touches canon's most delicate area (vibes 000085/000088);
  its increment lands last, alone, with the whole challenge suite as the
  gate.
