# What the L2 derivation should be, and what blocks it

## The two candidate routes

**Via the invariant** — reassemble `Σ a_i b_j g^ij` back to `a·b`, then
re-expand covariantly to get `g_ij a^i b^j`.

*Tested: does not work.*  `reassemble` leaves the expression untouched.  Its
pattern is "a coordinate tensor times a polyad of **basis vectors**", and in
the metric-contracted form the basis vectors are gone — absorbed into `g^ij`
by `simplify_basis_dot`.  So this is blocked by a *different* gap from the one
the challenge currently names.

**Directly, by index gymnastics** — the textbook derivation:

```
g^ij a_i b_j  =  g^ij (g_ik a^k)(g_jl b^l)      lower both indices
              =  (g^ij g_ik) g_jl a^k b^l        regroup
              =  δ^j_k g_jl a^k b^l              inverse metric
              =  g_kl a^k b^l                    contract δ
```

Three ingredients, of which tender already has the last.

## Why the direct route is the right one — not merely the shorter one

The round-trip would be a *weaker* proof even if it worked.  Both component
forms were produced **from `a·b`** by construction, so folding one back and
re-expanding recovers the other trivially: it demonstrates nothing about the
metric.  The content of this claim is exactly the index gymnastics — that
`g_ij` and `g^ij` are mutually inverse, and that raising and lowering are
consistent.  A round-trip through the invariant *bypasses* the thing being
claimed.

Compare challenge 000015, whose L2 legitimately cites a library identity
verified independently by its own L1.  That is citation.  This would be
evasion.

## What is missing, in order

0. **The metric is not exposed to Python at all.**  `make_metric` /
   `WellKnownKind::Metric` exist in C++, but `tender.metric(...)` does not,
   so a rule mentioning `g` cannot even be *written* on the Python surface.
   This is blocker zero and the smallest — it is the same binding shape as
   `tender.delta`.

1. **`metric-inverse`** — `Σ_j g^ij g_jk = δ^i_k`.  A clean **axiom**, exactly
   parallel to `delta-contraction`: it is what "reciprocal basis" means.

2. **`metric-lower` / `metric-raise`** — `a_i = g_ij a^j`.  Arguably an axiom
   (the definition of covariant components), but note it is *derivable* inside
   tender: expand `a` on the covariant frame and dot with `gᵢ`, giving
   `a_i = a·gᵢ = a^j (gⱼ·gᵢ) = g_ij a^j`.  Deriving it would make it a
   `derived` node with a real proof, which is the better outcome.

With those, the L2 is a short narrated `Derivation` — lower, regroup,
contract the inverse metric, contract δ — in the style of challenge 000014.

## Placement in the identity DAG

- `metric-inverse` — **axiom**, tag `basis`.
- `metric-lower` — **derived**, cites nothing (proved from the definition of
  components, like the other component-level nodes), proof = this challenge.

## Resolved (vibe 000103)

The plan below was followed, with one change of shape worth recording.

Blocker zero — no `tender.metric` — is fixed; `metric(realm, space, level0,
level1, idx0, idx1)` is now bound alongside `delta`.

The other two turned out to be **one operation**, not two, and it belongs as a
*step* rather than as DAG rules.  `contract_metric` spends a metric to move an
index: the surviving index is `g`'s other index, at `g`'s other level.  Read
three ways that is raising (`g^{ip} a_p → a^i`), lowering (`g_{ip} a^p → a_i`),
and the inverse pair — because raising `g_{pk}`'s lower index gives `g^i{}_k`,
and a `g` whose slots straddle the divide *is* the Kronecker δ (`g^i{}_j =
e^i·e_j`).  So `metric-inverse` never needed postulating as a separate axiom; it
falls out of what raising does, which is a better outcome than the planned
axiom.  `insert_metric(level)` is the same operation run backwards, paying a
metric to move an index the other way — the "lower both indices" move the
textbook derivation opens with.

Modelled on `contract_delta`, which is the precedent: a step for the derivation
to use, with the DAG node reserved for a rule the e-graph fires.  No rule was
added, so no node was.

The derivation is then three lines and never leaves component form:

    g^ij a_i b_j  →  a^i b_i  →  g_ij a^i b^j
              raise a      lower b

One caveat found while testing: contracting moves whichever factor the step
reaches first, so `a^m b_m` and `a_m b^m` are both "the mixed form" and are not
structurally equal.  Converting between *those two* also needs the metric —
which is the fact this challenge is about, so it is fitting rather than
awkward.

## Correction: why reassembly really fails (and what would fix it)

The note above blamed the metric absorbing the basis vectors.  Measurement
says the cause is more basic, and the metric is only half of it:

| expression | reassembles? |
|---|---|
| `a_i e^i` | **yes** → `a` |
| `(a_i e^i) · b` | no — *even with the other side already invariant* |
| `(a_i e^i) · (b_j e^j)` | no |
| the same shape in an **orthonormal** basis | no |

So `reassemble` never folds *inside a contraction operand*.  It recognises a
whole term of the form (coordinate tensor) × (polyad of basis vectors), and
nothing else.  The WCS round-trip of challenge 000017 works only because
`simplify_basis_dot` + `contract_delta` **eliminate the basis vectors
altogether**, leaving `a_i b_i` — recognisable by a different path.  In an
oblique basis `g^ij` is not δ, so nothing contracts away and the term never
reaches that shape.

The fix is therefore two independent pieces (**both since built** — (b) in
vibe 000103's first commit, (a) as `contract_metric`/`insert_metric` — though
the L2 uses the direct route below, not this one):

**(a) `g^ij = e^i·e^j`** — the definition of the metric, and the exact inverse
of what `simplify_basis_dot` does.  Reintroduces the basis vectors:
`a_i b_j g^ij  →  a_i b_j (e^i·e^j)`.

**(b) reassembly that descends into a contraction**, folding each operand
independently: `(a_i e^i)·(b_j e^j) → a·b`.

Both are needed for the oblique case, and (b) is worth having on its own —
it would let the orthonormal round-trip succeed *without* the δ detour,
which is a simplification of the bridge rather than another special case.
Together they are exactly the derivation

    g^ij a_i b_j = (e^i·e^j) a_i b_j = (a_i e^i)·(b_j e^j) = a·b .

## Separately: the reassembly gap

`reassemble` not recognising the metric-contracted form is worth recording on
its own account.  It is a second instance of the asymmetry challenge 000017
found: the bridge folds *some* contracted forms back to invariants (δ) and not
others (ε — and now `g`).  A general "recognise a contracted component form"
capability would close all three at once, and is probably the more valuable
target than three separate patches.
