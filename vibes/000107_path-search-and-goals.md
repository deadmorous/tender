# 000107 Path search, and how to say what you want

Vibe 000106 built the prerequisites and measured that the search space had
become small.  This note is the milestone that uses them.  It was prototyped
before being written, and the surprise is where the difficulty sits: **the
search is nearly free; saying what you want is the work.**

## 1. The search already works

A plain breadth-first search whose successor function is `applicable` and whose
visited set is keyed on the canonical form — thirty lines — rediscovers the
corpus's own derivations:

| goal | route found | nodes | time |
|---|---|---|---|
| `a×(b×c)` → `(a·c)b − (a·b)c` (challenge 000001) | `expand_in_basis → reduce_frame → contract_eps_pair → reduce_frame → reassemble` | 19 | 0.1 s |
| `(a×b)·(c×d)` → Lagrange (challenge 000014) | *the same five steps* | 17 | 6.4 s |

Two things stand out.

**The routes are the ones the challenges spell out by hand.**  Not an equivalent
route — the same one.  That is the acceptance test vibe 000106 proposed, passing
on its first two cases.

**Both problems found the same route.**  The step redesign did not merely shrink
the space; it made it *canonical* enough that different problems of the same
kind take the same path.  That is worth more than the node counts.

### The cost is per node, not the number of nodes

17 nodes took 6.4 s where 19 nodes took 0.1 s.  The difference is expression
*size*: each node runs all 39 catalogued steps, and on a four-vector expansion
each of those is slow.  So the lever is not a better search algorithm but **less
work per node**:

- `wants` as a pre-filter — it can reject 9 of 39 steps on the fingerprint alone
  (vibe 000106 §4b), without running them;
- primaries only — 17 steps rather than 39, and the measured branching drops
  from 4–10 to 2–5;
- and the obvious: do not re-run `applicable` on states already seen.

None of that is research.  It is the reason this milestone is worth starting.

## 1a. Correction: the corpus measured the wrong thing

**The framing below is biased, and the bias is instructive.**

Everything measured here came from the challenges, and a challenge is a
*certification*: both sides are known by construction, because that is what
"certify" means.  So the corpus reports what tender is **tested with**, and I
read it as what tender is **used for**.  Those are not the same population, and
the difference inverts the conclusion.

In real use — the user's account, and the reason the library exists — a
derivation **starts with a left-hand side and does not know the right**:

> start with an invariant tensor expression and end up with expressions of its
> components in a certain coordinate frame.  Or some other form.  Maybe collect
> terms at powers of a small parameter.  The right-hand side is usually not
> known when a derivation starts.

That relegates §2(a) and §2(b) — meet-in-the-middle, and reaching a named target
— to what they actually are: **verification**.  Useful, and worth having, but
they answer a question the user has already answered.

And it promotes §2(c), *reach a form*, from "blocked, and instructively" to
**the case that matters**.  Which is uncomfortable, because it is the one this
note has the least to say about: its goal vocabulary is three prose exit
conditions and a fingerprint that cannot yet express one of them.

The honest position is therefore not "design the search" but "we do not yet know
how a user states what they want".  Guessing at that vocabulary from the
certification suite is what produced this note's inversion; guessing again from
first principles would be no better.

## 2. Four ways to say what you want, and only two of them work

The corpus and the existing API between them already contain four goal shapes.

### (a) Meet in the middle — the dominant one, and it needs no goal at all

The commonest pattern in the challenges is not "reach X"; it is **reduce both
sides until they agree**.  53 of the assertions are `algebraic_eq(reduced_lhs,
reduced_rhs)`.

That shape needs nothing written down: run the search from both ends and stop
when the frontiers intersect.  Measured, and it is *cheaper* than the directed
search:

```
a·b  vs  b·a                    met in 4 nodes:  both sides expand_in_basis → reduce_frame
a×(b×c) vs (a·c)b − (a·b)c      met in 7 nodes:  both sides expand_in_basis → reduce_frame → to_concrete
```

19 nodes directed, 7 nodes meeting — and the meeting route is exactly what
challenge 000001's L1 does by hand.  **The hardest part of goal specification is
avoidable for the commonest case**, which is a good place to start.

### (b) Reach a named target — works today

`goal = algebraic_eq(·, target)`.  Both searches above.  Nothing new is needed.

### (c) Reach a *form* — blocked, and instructively

"Get me to components", "get me to numbers", "bring it home" are the goals a
person actually states, and vibe 000106 §2 already gave every bridge step a
checkable **exit condition**:

| step | exits when |
|---|---|
| `reduce_frame` | no basis-vector product remains |
| `to_concrete` | no symbolic index remains |
| `reassemble` | nothing further folds |

Those *are* the goal vocabulary, already written and already documented.  But
they are prose, not predicates.  Measured: `to_concrete`'s own exit condition
**cannot be expressed** against today's fingerprint, because `index_slots`
counts concrete indices too —

```
a_i b_i                        index_slots 2
a_x b_x + a_y b_y + a_z b_z    index_slots 6      ← "no symbolic index" is not index_slots == 0
```

So the fingerprint needs a `dummy_indices` / `concrete_indices` split, and
probably more of the same kind.  **The goal language and the fingerprint
co-design**: vibe 000106 deliberately held the fingerprint open pending real
cases, and this is the first real case telling it what to become.

### (d) Prefer a reading — exists, for the other engine

`engine_simplify(lhs, rules, prefer="fewest_crosses")` (vibe 000097) is already
intent-as-a-goal: *state what you want, not the answer*, expressed as a cost over
a saturated graph.  Challenge 000001 uses it and calls it "discovered by intent".

The analogue for path search is a cost to minimise rather than a predicate to
satisfy — "reach the form with fewest ε", "with no basis vectors".  Worth
having, and it should reuse `PREFER`'s vocabulary rather than invent a second
one.

## 3. Proposed surface

```python
td.reach(expr, goal, **context)     # directed
td.meet(lhs, rhs, **context)        # bidirectional; no goal to state
```

with `goal` accepting, in rising order of ambition:

- an **`Expr`** — reach something algebraically equal to this;
- a **name** — `"components"`, `"concrete"`, `"invariant"`: the step exit
  conditions of §2(c), promoted to a vocabulary;
- a **callable** — the escape hatch, for a goal nobody anticipated;
- a **`prefer=`** cost, as (d).

Both return a `Derivation` — which already exists, already records which steps
fired, and already renders — so a found route is inspectable and re-runnable
rather than magic.  A search that cannot reach its goal should say what it
*did* reach and what was in the way, reusing the step reports of vibe 000106
§4b.

## 4. What could go wrong

- **A correct route nobody would write.**  Search optimises for reachability, not
  legibility.  Mitigation: the corpus routes are the acceptance test, and cost
  can prefer shorter/primary-only routes.
- **The engine inside the search.**  Vibe 000106 §6 settled that saturation is
  *one of the steps*.  A budget matters: `engine_simplify` at every node would
  dominate the cost.
- **Goal creep into a proof language.**  "Reach a state where `prove_equal`
  finishes" is expressible and tempting; it is also where a search stops being
  a search.  Worth a boundary before, not after.

## 5. Order of work

1. **`meet`** — the dominant case, needs no goal language, already measured to
   work and to be cheap.
2. **Fingerprint fields** the exit conditions need (`dummy_indices` first), so
   §2(c) becomes expressible.
3. **`reach`** with `Expr` and named goals.
4. **Cost/`prefer`**, reusing the e-graph's vocabulary.
5. Per-node cost work (`wants` pre-filter) — only when it is measurably the
   limit, which on today's numbers it already is for large expressions.

## Status

**Postponed — waiting on data from real use.**

The engineering is ready: the prerequisites (vibe 000106) are done, and §1 shows
the search itself is close to free.  What is missing is the thing this note set
out to design — a vocabulary for *stating what you want* — and §1a explains why
it cannot be settled from here: the only corpus available is a certification
suite, which contains no examples of the case that dominates real use.

So the next input is not more design.  It is **using tender on concrete
problems** and recording what the goals actually turn out to be — "components in
this frame", "collected at powers of ε", and the ones nobody has thought to name
yet.  Those are the data the goal vocabulary should be fitted to.

Kept as measured and still true, whenever this resumes:

- the search rediscovers the corpus routes, in tens of nodes;
- the cost is per node, not node count, and the levers for it are known;
- `meet` is cheap and needs no goal language, so it remains the right *first*
  thing to build — for verification, which is what it serves;
- the fingerprint and the goal language co-design, and the fingerprint should not
  be extended until a real goal asks for a field.
