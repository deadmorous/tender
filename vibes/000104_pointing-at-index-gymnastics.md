# 000104 Pointing at index gymnastics

Vibe 000054 gave us positional addressing — paths, `.at`, `.find`, `td.at` — so
a step can be applied to *part* of an expression instead of all of it.  Vibe
000103 gave us index-algebra steps that fold and unfold contracted components.
The two do not compose, and challenge 000018 is where that first bit.

## Where it came up

`contract_metric` moves whichever factor it reaches first.  So

    g^ij a_i b_j   →   a^i b_i

raises `a`, and there is no way to ask for `b` instead — even though "raise the
other one" is an ordinary thing to want mid-derivation, and the whole point of
the metric is that both moves are available.  The natural reach is for vibe
000054: point at the part, apply the step there.

## Measured: three ways to point, three different failures

Starting from `Σ_j Σ_i a_i b_j g^{ij}`:

| what you point at | path | result |
|---|---|---|
| the metric `g^{ij}` | `[0,0,1]` | **no-op** — the partner is outside the subtree |
| the target `b_j` | `[0,0,0,1]` | **no-op** — the metric is outside the subtree |
| the smallest subtree holding both | `[0,0]` | **fires, and strands a binder**: `Σ_i Σ_? a^i b_i` |

The third is the interesting one.  It fires, it raises the right thing — and it
produces a `Σ_?`, a binder for an index that no longer exists, because the
binder lives *above* the addressed subtree and `at` cannot shed what it cannot
see.  The renderer cannot even name it.

It is also not the control we wanted: pointing there still raised `a`, the same
factor the whole-expression call chose.

## It is not about the metric

The same thing happens to `contract_delta`, which has been in the library since
long before any of this.  From `Σ_j Σ_k Σ_i δ_{kj} a_k b_j c_i e_i`:

```
whole expression   →  a_j b_j c_i e_i                     ✓
td.at(…, [0,0,0])  →  Σ_j Σ_? Σ_i a_j b_j c_i e_i         ✗ stranded
td.at(…, [0,0])    →  Σ_j Σ_? a_j b_j c_i e_i             ✗ stranded
```

So **every index-algebra step is unsafe under `td.at`**, and has been.  Nothing
caught it because the challenges call these steps on whole expressions.

A second symptom sits alongside: an extracted subterm re-renders with fresh
index names, because its dummies are bound above it.  `x.at([0,0,0,0,0])` on the
δ example prints `δ_{ji} a_j b_i` where the parent says `δ_{kj} a_k b_j`.  The
part is not wrong, but reading it is already misleading — the names are the
renderer's, not the expression's.

## The diagnosis

**The unit of address and the unit of the operation do not coincide.**

Vibe 000054 addresses *subtrees*.  Index gymnastics acts on *index clusters* —
a set of factors joined by a shared index, plus the binder standing above them.
A cluster is not a subtree: its members are scattered across a product, and the
binder that owns it is an ancestor of all of them.  Naming a subtree can
therefore include the binder but not the factors (too high), or the factors but
not the binder (too low), and there is no path that lands on the cluster.

This is vibe 000103's finding again, on the other side of the mirror.  There the
problem was *matching* by shape rather than by index; here it is *pointing* by
shape rather than by index.  The fix was the same shape of fix — read the index
structure — and it should be here too.

## What would fix it

Two directions, and they are complementary rather than alternative.

**Point by index.**  `td.at_index(expr, index, step)`, or a `target=` argument
on the steps themselves.  The index names the cluster exactly, which is the
whole argument of vibe 000103; the step then finds its own factors and its own
binder, as it already does.  For the motivating case the user says "raise the
index `b` carries", which is what they actually mean.  This needs a way to *name*
an index on the Python surface — `.find` returns paths to objects, and an index
is not an object.  The labeled view (`tender.render.labeled`) already prints
index names, so the naming is half-built.

**Point at a factor, and let the step widen.**  Keep paths as the surface, but
have index-algebra steps accept a path as *a member of* the cluster rather than
as the whole of it, then walk out to the binder themselves.  Cheaper, reuses
`.find(kind="Metric")` and friends, and reads naturally: "spend *this* metric".

Either way the stranded binder must go: a step that consumes an index has to be
able to drop the binder for it, which means the addressing machinery must hand
the step a writable view that reaches up to the binder, not a detached subtree.

**Minimum honest fix, whatever else happens:** `td.at` should refuse — or the
steps should refuse — rather than silently emit `Σ_?`.  Producing an expression
with a binder over nothing is worse than doing nothing, and it is what happens
today.

## Postscript: a neighbouring bug, found the same way

Challenge 000019's derivation wanted the same trip home, and `reassemble_nabla`
answered it with a silently wrong result: `Σ_i (u·e_i) ⊗ ∂_i u` — which is
`(u·∇)u` — came back as a bare **`u`**, the derivative gone.

The cause is a cousin of the addressing problem.  That term has *two*
field-carrying factors, because the frame vector is contracted into a `u` that
does not own its ∂-mark.  The single-operand classifier assigned `operand = f`
in a loop, so the second factor overwrote the first and the first was dropped
without trace.  There is already a safety valve for terms with ≥2 ∂-*marked*
factors; this shape has only one mark, so it slipped past.

It now leaves the term alone.  Folding it properly — to `u·(∇⊗u)` — needs a leg
rule for the directional derivative, where the frame vector belongs to a ∂ on a
*different* factor than the one it is contracted with.  That is worth building,
and it is the same "which factor does this index belong to" question this vibe
is about.

## Resolved

Both halves, and the split turned out to matter: one was a correctness bug, the
other the feature.

**The stranded binder was a bug, and a quiet one.**  `td.at` now compares which
of the *enclosing* binders' indices the addressed part carried before the step
against after.  An index it used and no longer does was summed away — δ
contracted it, a metric spent it — so that binder is dropped along with it.  A
binder that was *already* vacuous is left alone: `Σ_m X` with X free of m is
dim·X, not X, so dropping one the step did not consume would change the value.
That distinction is the whole of the fix.

There is one case it refuses: the consumed index is still carried *outside* the
addressed part.  Then the binder is still doing work for that occurrence — and
worse, if the step *summed* over the index (which the contractions do), the
result is already wrong, because the outside occurrence was not summed with it.
No splice can repair that, so `at` raises and says to widen the address.

Measured, on the case this vibe opened with:

| addressed at | before | after |
|---|---|---|
| `[0,0]` | `Σ_j Σ_? a_j b_j c_i e_i` | `Σ_j a_j b_j c_i e_i` ✓ |
| `[0,0,0]` | `Σ_j Σ_? Σ_i …` | ✓ |
| `[0,0,0,0]` | (stranded) | ✓ |

— and each now equals the whole-expression result, so the answer no longer
depends on where the caller happened to point.

**The targeting was the feature, and a path could not carry it.**  `raise the
*other* one` is an ordinary thing to want, and this vibe's measurement showed
why no path expresses it: the factors sharing an index are scattered across a
product, and no subtree holds just `b` and `g`.  So the metric steps take an
optional `target` naming the tensor whose index is to move:

```python
td.contract_metric(x, target="a")   # g^ij a_i b_j → a^i b_i
td.contract_metric(x, target="b")   #              → a_i b^i
td.insert_metric(m, Level.Upper, target="a")
```

A **name**, not a path, and deliberately: a name survives canonicalization,
which these steps perform internally (they self-prepare), while a path does
not.  It also reads the way the mathematics is spoken.  A target that is not
present, or names a factor already at the level asked for, is a no-op rather
than a wrong move.

What is *not* done: the general "point by index" surface.  It was not needed —
naming the factor answers the question a person actually asks — and an index has
no stable handle across canonicalization to name it by.  The larger addressing
design in this vibe stands unbuilt, and now without a motivating case.

## Status

Design proposal, unscheduled.  The `Σ_?` stranding is a live (if latent) bug
rather than a missing feature, and is worth fixing on its own account before the
addressing question is settled.
