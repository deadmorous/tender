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

The fix is therefore two independent pieces:

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
