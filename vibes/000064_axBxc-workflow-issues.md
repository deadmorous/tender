# 000064 — a × B × c workflow issues (user playthrough)

The cross-removal + reassembly machinery (vibe 000063) now closes `a × B × c`
end-to-end *in a hand-tuned step order*.  This note collects the rough edges the
**user** hits while driving it interactively — places where the workflow is
counter-intuitive, a step misfires, or output is awkward.  Each issue is one
concrete `# STEPS` snippet dropped into the template below; we reproduce it, log
what goes wrong, and once the list settles we plan the fixes.

## The template

```python
import tender as t
import tender.derivation as td
import tender.basis as tb
from tender import Level, Realm
from IPython.display import Math, display

def disp(x):
    display(Math(x.latex()))

ctx = t.Context()
frame = tb.wcs(ctx)

co = tb.Variance.Covariant
contra = tb.Variance.Contravariant

I = t.identity(ctx)

a = t.tensor("a", 1, ctx)
b = t.tensor("b", 1, ctx)
B = t.tensor("B", 2, ctx)
c = t.tensor("c", 1, ctx)

id_axIxb = td.Identity(
    "axIxb",
    a%I%b,
    b*a - (a@b)*I)

id_axIxb_inv = td.Identity(
    "axIxb_inv",
    b*a,
    a%I%b + (a@b)*I)

a_B_c = a % B % c

x = a_B_c

# STEPS

disp(x)
```

A CLI-runnable mirror lives in the scratchpad (`harness.py`): same preamble,
`disp` prints LaTeX, `run("<steps>")` execs the snippet against the shared env.

## Issues

### #1 — Two terms collapse to one after the final `apply_identity` — NOT a bug

```python
x = tb.expand_in_basis(x, frame, co)
x = td.apply_identity(id_axIxb_inv)(x)
x = tb.simplify_basis_cross(x, frame)
x = td.apply_identity(id_axIxb)(x)
```

User's worry: before the last step there are **two** terms, after it just
**one** — too good to be true?

**Verdict: legitimate, smart, not a bug.**  The single survivor is numerically
*exactly* `(a×B)×c` (checked against the direct triple product).

Mechanism — the `I`-bookkeeping round-trips away:

- `id_axIxb_inv` (`b⊗a = a×I×b + (a·b)I`) is the dyad identity; applied to the
  basis dyad `e_k⊗e_j` of the expanded `B` it injects a `+(a·b)I` term.  After
  `simplify_basis_cross` the two pre-final terms are: **(1)** a scalar·I term
  `[B a c · εε] I` (that injected `(a·b)I`), and **(2)** an `a×I×b`-shaped term.
- The final `id_axIxb` (`a×I×b = b⊗a − (a·b)I`) rewrites term (2) into a dyad
  `b⊗a` **minus** a matching `(a·b)I`.  That `−(a·b)I` is exactly the negative of
  term (1), so `canonicalize` cancels them, leaving only the dyad
  `−B_nm ε_jnl ε_imk a_l c_k e_j e_i`.

`simplify_basis_cross` only touched the cross parts, so the `+(a·b)I` from the
inverse and the `−(a·b)I` from the forward identity cancel by construction.  The
exact numeric match is the proof: a wrongly-dropped `I` term would have shifted
the result by precisely that term.  (A pleasant payoff of reusing the proven
`a×I×b` identity in both directions.)

### #2 — `contract_eps_pair` leaves explicit `Σ` in its output

```python
x = tb.expand_in_basis(x, frame, co)
x = td.apply_identity(id_axIxb_inv)(x)
x = tb.simplify_basis_cross(x, frame)
x = tb.simplify_basis_dot(x, frame)
x = td.contract_delta(x)
x = td.contract_eps_pair(x)
```

User's report: the result's first term carries explicit sums:

```
Σ_i Σ_m Σ_k Σ_l  −B_{im} a_k c_l I (δ_{im} δ_{kl} − δ_{il} δ_{km})  −  (uncrossed a×I×b term)
```

everywhere else the system uses *implicit* Einstein summation, so the bare `Σ`s
are jarring.

**Diagnosis — forced by the Sum-scope boundary (vibe 000052), not a bug in the
result.**  The ε-pair contraction produces a δ-determinant `(δδ − δδ)`, which is
a **Sum**.  The surviving summed indices `i,m,k,l` sit *inside* that Sum factor,
and the implicit-summation convention only recognises a repeated index within
one multiplicative term — never across a `+`.  So `implicitize` (which
`contract_eps_pair` already runs on its output) cannot strip those binders and is
forced to leave them explicit.  Confirmed: `expand_products` alone does **not**
clear them (canonicalize re-materialises but never strips back to implicit), but
the next `contract_delta` distributes the determinant *and* contracts the δ's,
giving a fully implicit result `−tr(B)(a·c) I + (…)I − (a×I×b term)`.

So this is the same iterate-`contract_eps_pair`→`contract_delta` shape as vibe
000063; the dangling explicit `Σ` is just the intermediate the user sees if they
stop on `contract_eps_pair`.

**Candidate fixes (for the planning phase):**
1. Have `contract_eps_pair` distribute its δ-determinant (`expand_products`)
   *before* the final `implicitize`, so each emitted term is a single product and
   the binders strip — output becomes implicit-clean (cost: the one term splits
   into the determinant's several terms; the δ's still await `contract_delta`).
2. Leave it and document the pairing, treating the explicit `Σ` as a normal
   between-steps intermediate.

Relates to the broader vibe 000052 note: an index that should sum stays
explicit/bound precisely when it straddles a `+`; the principled fix is
distribute-then-implicitize.

### #3 — `reassemble` leaks explicit `Σ` onto the terms it can't fold — a real bug

```python
x = tb.expand_in_basis(x, frame, co)
x = td.apply_identity(id_axIxb_inv)(x)
x = tb.simplify_basis_cross(x, frame)
x = tb.simplify_basis_dot(x, frame)
x = td.contract_delta(x)
x = td.contract_eps_pair(x)
x = td.contract_delta(x)
x = tb.reassemble(x, frame)
```

User's report: now the **second** (third) term carries explicit sums.  The first
two terms fold cleanly, but the still-uncrossed `a×I×b` term comes out as

```
−a·c tr(B) I  +  a·(Bᵀ·c) I  +  Σ_l Σ_k Σ_n Σ_m Σ_i Σ_j −B_{mn} ε_{jnk} ε_{iml} a_l c_k (e_j × (I × e_i))
```

**Diagnosis: a real bug — `reassemble` does not `implicitize` its output.**  Its
self-prep `canonicalize` *materialises* every implicit Einstein sum into explicit
`ExplicitSum` binders.  It then folds the first two terms (→ clean invariants)
but cannot fold the third (it still holds a cross `e_j × (I × e_i)`, not a basis
dyad).  Because a fold *did* fire, the no-op guard doesn't return the original
input — and `reassemble` returns `out` **without** `implicitize`, so the
materialised `Σ`s leak onto the unfolded third term.  That term is a single
product with each index appearing exactly twice, so it is fully implicitizable —
`reassemble` simply never strips it back.

The derivation steps already do this right: `contract_eps_pair` /
`contract_delta` return `implicitize(ctx, out)` (derivation.cpp:2178, 2590);
`reassemble` and `reassemble_completeness` return bare `out`
(basis.cpp:1186, 1209).

**Fix (clear, small):** make both return
`out == prepped ? e : steps::implicitize(ctx, out)` — `steps::implicitize` is
already public (derivation.hpp:194) and basis.cpp already uses `steps::`.  This
matches the self-prepare contract (vibe 000062): a step must not leak its
internal materialisation onto the caller's expression.  (Distinct from #2: there
the `Σ` is *forced* by a Sum-scope boundary and cannot be implicitized; here the
term is a clean product and the `Σ` is purely a missing-implicitize leak.)

### #4 — opposite terms don't cancel; no way to finish clean

```python
# … the full both-directions run (id_axIxb_inv … reassemble … id_axIxb … reassemble)
```

User's report: the result has **five** terms; the 1st should cancel the 4th and
the 2nd the 3rd, and the 5th carries explicit sums:

```
(a·Bᵀ)·c I  −  a·c tr(B) I  +  a·c tr(B) I  −  a·(Bᵀ·c) I  +  Σ… −B ε ε a c e_j e_i
```

**Findings:**
- The four `I`-terms **are** pairwise equal-and-opposite and **do** cancel — a
  single `canonicalize` collapses all four to zero, leaving only the dyad term
  (verified).  This includes 1≡4 (`(a·Bᵀ)·c ≡ a·(Bᵀ·c)`, which `canonicalize`'s
  dot/associativity normalisation equates) and 2≡3 (`±a·c tr(B)`).
- They are left visible because `reassemble` returns **raw** output: not folded
  (opposite addends never combine) and not implicitized (#3, the `Σ` on term 5).
- The bind: `canonicalize` cancels the four but **re-materialises** explicit `Σ`
  on the surviving dyad; `implicitize` would strip them — **but `implicitize` is
  not exposed in the Python API** (`td` has no implicitize/simplify/normalize).
  So from Python the user cannot reach a clean canonical-*and*-implicit result.

(Aside — math, not a workflow bug: after the four cancel, the lone survivor
`−B ε_jnl ε_imk a c e_j e_i` is exactly the issue-#1 ε-dyad whose two ε's share
NO index — the dead-end from vibe 000063.  This both-directions path circles back
to it, so it does not by itself finish the reduction.  Not what the user is
flagging here, but worth noting the path is non-terminating for the invariant
form.)

**Candidate fixes (planning phase):**
1. Land #3 (`implicitize` on `reassemble`/`reassemble_completeness` return) —
   necessary, kills the term-5 `Σ`.
2. Give the user a way to *finish*: expose `implicitize` to Python, and/or add a
   combined `simplify`/`finalize` = `canonicalize` then `implicitize` (fold
   equal-and-opposite terms, then strip the re-materialised `Σ`).  Without a
   canonicalize-then-implicitize the two needs (fold vs. implicit form) fight.
3. Optionally have `reassemble` fold its own output (run `fold_equal_addends`
   before returning) so terms it emits combine with siblings without a separate
   canonicalize — but the materialise/implicitize tension (fix 2) is the real
   gap, since cancellation across *two* reassemble calls only completes at the
   end anyway.

### #5 — render runs out of index names (`dummy_name: index out of range`)

```python
# … reassemble, then re-expand and cross again:
x = tb.expand_in_basis(x, frame, co)
x = tb.simplify_basis_cross(x, frame)
```

User's report: throws `IndexError: IndexSpace::dummy_name: index out of range`;
"maybe we should think how to represent more indices."

**Diagnosis: a display-only limit — the computation is fine.**  The full
pipeline runs; only `x.latex()` throws.  `space_3d()` carries a **fixed 17-name
schema** (`i j k l m n p q r s t u v w x y z`, `o` skipped); the renderer hands
each distinct dummy index the next unused schema name
(`render.cpp:45 space->dummy_name(pos++)`), and the 18th distinct index throws
`out_of_range`.  The expression itself is valid (indices are integer ids
internally; nothing downstream uses `dummy_name`).

**Why this term needs > 17 names — a compounding cause:** the 2nd
`expand_in_basis` re-expanded the *already-reassembled* invariants
(`tr(B) → tr(B_xw e_x e_w)`, `a·(Bᵀ·c) → a_u e_u·((B_ts e_t e_s)ᵀ·c_r e_r)`, …)
back into coordinates, on top of the cross term's own sum indices — well past 17
distinct dummies.  So invariants the user had just folded get blown back open.

**Candidate fixes (planning phase):**
1. **Unbounded names (the real fix).**  Make `dummy_name(n)` *generate* beyond
   the base schema instead of indexing into it: `schema[n % S]` with a numeric
   subscript `⌊n / S⌋` (omitted when 0) — i.e. `i … z`, then `i_1 j_1 … z_1`,
   `i_2 …`.  Readable, infinite, per-space (3D letters, 2D/4D Greek).  Keeps
   `dummy_name` total; render no longer throws.
2. **Reduce index pressure (orthogonal, helps but not sufficient):** selective
   `expand_in_basis` (vibe 000054) so already-invariant subterms aren't
   re-expanded; and landing #3/#4 so stray explicit `Σ` don't carry extra
   live dummies into render.

### #6 — `reassemble` "does nothing" while its prep silently simplifies 5 → 1

```python
# … a longer run ending:
x = td.contract_delta(x)
x = tb.reassemble(x, frame)   # ← appears to do nothing
```

Input to the last step (5 terms): one ε-dyad plus four coordinate `I`-terms
`B_ii a_k c_k e_l e_l`, `−B_jk a_k c_j e_l e_l`, `−B_jj a_k c_k e_i e_i`,
`+B_jk a_k c_j e_i e_i`.  `reassemble` returns it **bit-for-bit unchanged**
(`structural_eq(before, after) == True`).

**Root cause — the self-prep canonicalisation is computed and then thrown away:**
1. `reassemble`'s self-prep `canonicalize` **cancels the four `I`-terms** (they
   are pairwise equal-and-opposite — verified: terms 2–5 sum to zero, and the
   whole 5-term expression equals `(a×B)×c`), collapsing `prepped` to the single
   ε-dyad term.
2. That ε-dyad is *unfoldable* (its two ε's share no index — the issue-#1
   dead-end), so the fold leaves `prepped` untouched: `out == prepped`.
3. The no-op guard `return out == prepped ? e : out` then returns the **original
   5-term `e`** — discarding the 5→1 cancellation the prep performed.

So a real simplification is available but hidden; the step looks idle.  The guard
(from vibe 000062, meant to preserve pointer identity for `EXPECT_EQ` no-op
tests) over-fires: it returns `e` not only on a true no-op but whenever the
*fold* is a no-op, even if the *prep* (canonicalize) did real work
(`prepped != e`).

**Candidate fix (unifies with #3/#4):** reassemble should surface the prepared
result.  Return `steps::implicitize(ctx, out)` and decide the no-op by comparing
that against the *original* `e` (e.g. `structural_eq`), returning `e` only when
genuinely unchanged — not when `out == prepped` but `prepped != e`.  Then this
run yields the single clean ε-dyad (#3's `implicitize` keeps it sum-free).

## The reassemble-output cluster (#3, #4, #6)

Three symptoms, one underlying defect: **`reassemble`/`reassemble_completeness`
return the wrong thing on the prep-changed-it / fold-didn't path.**  Today:
`return out == prepped ? e : out`.  This (a) never `implicitize`s, leaking the
materialised `Σ` (#3); (b) reverts to the raw `e`, discarding canonicalisation —
both the cancellation of opposite terms (#4) and the 5→1 collapse (#6).  Proposed
single fix:

```cpp
auto const* result = steps::implicitize(ctx, out);
return structural_eq(result, e) ? e : result;
```

plus exposing `implicitize` (and/or a `simplify` = canonicalize→implicitize) to
Python so a derivation can be *finished* (#4).

### #7 — "every path dead-ends on a share-nothing ε-pair" — expansion-order artifact, NOT gap 2

User's overarching worry after #1–#6: *every* sequence they tried ended on an
ε-pair with no common index (the vibe-000063 dead-end), so maybe the regroup
(gap 2) is necessary, or there's a bug — they couldn't find a cross-free
invariant route.

**Resolution: cross-free invariant form IS reachable; gap 2 is NOT needed.**  The
missing ingredient is one line — **expand the inserted `I` (a second
`expand_in_basis`) right after the dyad identity, *before* `simplify_basis_cross`:**

```python
x = tb.expand_in_basis(x, frame, co)
x = td.apply_identity(id_axIxb_inv)(x)   # inserts an abstract I:  e_j × I × e_k
x = tb.expand_in_basis(x, frame, co)     # ← expand that I into Σ_m e_m ⊗ e_m
x = tb.simplify_basis_cross(x, frame)
x = tb.simplify_basis_dot(x, frame)
# then iterate: contract_eps_pair → simplify_basis_dot → contract_delta  (to ε-free)
x = tb.reassemble(x, frame)
```

This reaches `(a·Bᵀ)·c I − a·c tr(B) I + a·c Bᵀ + tr(B) c⊗a − c⊗(B·a) −
(Bᵀ·c)⊗a` (ε-count 0; verified — the last term still prints with a stray `Σ`,
which is exactly bug #3).

**Why the order is decisive.**  The dyad identity inserts an *abstract* `I` in
the middle of `e_j × I × e_k`.  Only when that `I` becomes `Σ_m e_m ⊗ e_m` do its
basis vectors supply the **shared dummy** that couples the ε's: expanding it
*before* the cross step yields the four-ε term `ε_qrp ε_pjn ε_lmk ε_jki` (sharing
`p, j, k`), which the N-ε `contract_eps_pair` grinds to ε-free coordinate form.
If the `I` is never expanded (#1–#5) or expanded *too late* — after
`simplify_basis_cross` has already turned the outer crosses into ε's without it
and `contract_eps_pair` has committed them (#6) — the ε's never acquire that
dummy and you are left with the share-nothing pair.  The dead-end is an artifact
of *expansion order*, not a missing capability; the gap-2 regroup is one
alternative route, not a prerequisite.

This matches the established recipe (vibe 000063 §"Gap 1 done", and the
`BasisFeasibility.CrossTensorCross` test): dyad identity → **expand the I** →
cross → dot → (eps-pair/dot/delta)* → reassemble.

**Workflow lesson worth surfacing to users:** "expand a freshly *introduced* `I`
before reducing crosses."  Candidate ergonomic fixes — a guard/warning when
`simplify_basis_cross` meets a cross around an unexpanded `I`, or selective
expand (vibe 000054) to expand just that `I`, or a small composite step
"insert-dyad-identity-and-expand."

### #8 — one `reassemble` isn't enough; a sign between sum binders blocks the fold (+ misleading render)

```python
# … the working recipe, ending in a single reassemble:
x = td.contract_delta(x)
x = tb.reassemble(x, frame)
```

Output — five terms fold to invariants, but the second-from-end stays raw:

```
−a·c tr(B) I + a·(Bᵀ·c) I + a·c Bᵀ − c⊗(B·a)
   + Σ_j −Σ_i B_{ji} c_j e_i ⊗ a            ← coordinates + basis vector + double Σ
   + tr(B) c⊗a
```

User's three observations: (a) one reassemble is not enough; (b) that term still
has coordinates and basis vectors with a *double explicit sum*; (c) it renders
`Σ_j −Σ_i …` which "reads like a difference `Σ_j − Σ_i`" — the sum looks
body-less, but its body is actually `−Σ_i …`.

**#8a — the fold bug.**  `reassemble`'s peel loop (basis.cpp:1110–1132) peels
*all contiguous* `ExplicitSum`s, then *one* `Negate`.  This term is
`Σ_j · Negate(Σ_i · …)` — a sign **between** the binders.  So it collects only
`j`; the `i` binder stays trapped under the `Negate`, `flatten_product` sees the
opaque `Σ_i(…)` as a single factor, and nothing folds.  Verified: a bare
`canonicalize` reshapes `Σ_j −Σ_i …` → `Σ_j Σ_i −…` (sums adjacent, sign in the
body), after which a second reassemble folds it to `−(Bᵀ·c)⊗a`.  That is the only
reason a second pass is needed (#9).

**Fix (small):** peel `ExplicitSum` and `Negate` **interleaved** in one loop,
accumulating the summed indices and a running sign:

```cpp
std::vector<int> summed; bool negated = false; Expr const* body = node;
for (;;) {
    if (auto* es = std::get_if<ExplicitSum>(&body->node)) {
        if (es->bound) return node;
        summed.push_back(es->index.id); body = es->body;
    } else if (auto* n = std::get_if<Negate>(&body->node)) {
        negated = !negated; body = n->operand;
    } else break;
}
```

(applies to both `reassemble` and `reassemble_completeness`).  Then this term
folds in **one** pass.

**#8b — the render bug.**  `ExplicitSum(j, Negate(ExplicitSum(i, body)))` renders
as `\sum_{j} -\sum_{i} body`, ambiguous between `(Σ_j) − (Σ_i body)` and
`Σ_j(−Σ_i body)`.  A `Negate` whose operand is an `ExplicitSum` (or a sum whose
body is a `Negate`) should be parenthesised, e.g. `\sum_{j}\left(-\sum_{i}
body\right)`, or the sign hoisted.  Largely hidden once forms are kept canonical
(`Σ_j Σ_i −body` reads cleanly), but render should be robust to the raw shape.

### #9 — a second `reassemble` finishes the job (workaround for #8a)

```python
# … as #8, then one more:
x = tb.reassemble(x, frame)
```

Yields the clean final invariant

```
(a·Bᵀ)·c I − a·c tr(B) I + a·c Bᵀ + tr(B) c⊗a − c⊗(B·a) − (Bᵀ·c)⊗a
```

— numerically `= (a×B)×c` (vibe 000063).  The need for the second pass is purely
#8a: the first reassemble can't peel the sign-between-binders term, the canonicalise
*inside* the second pass repairs the nesting, and then it folds.  **With #8a fixed,
a single reassemble reaches this directly.**  (A `reassemble`-to-fixpoint loop
would also paper over it, but fixing the peel is the right move.)

## Summary so far

| # | symptom | verdict | fix |
|---|---|---|---|
| 1 | two terms → one after final identity | correct (I-terms cancel by design) | none |
| 2 | explicit `Σ` after `contract_eps_pair` | forced by Sum-scope boundary (vibe 052) | distribute-then-implicitize, or document |
| 3 | explicit `Σ` on terms `reassemble` can't fold | **bug** — missing `implicitize` on return | reassemble-output fix (below) |
| 4 | opposite terms uncancelled; can't finish clean | raw output + no Python `implicitize` | reassemble-output fix + expose `implicitize`/`simplify` |
| 5 | render `dummy_name` out of range | **display limit** — fixed 17-name schema | generate subscripted overflow names; (also selective expand, vibe 054) |
| 6 | `reassemble` no-op hides a 5→1 simplification | **bug** — guard discards prep's canonicalisation | reassemble-output fix (compare vs original `e`) |
| 7 | every path dead-ends on share-nothing ε-pair | **not a bug** — expansion-order artifact; gap 2 not needed | expand inserted `I` *before* `simplify_basis_cross`; (ergonomics: warn / selective-expand / composite step) |
| 8a | one `reassemble` leaves a `Σ_j −Σ_i …` term unfolded | **bug** — peel loop stops at a sign between binders | peel `ExplicitSum`/`Negate` interleaved (basis.cpp:1110) |
| 8b | `Σ_j −Σ_i …` renders as a confusing "difference" | **render bug** — negated sub-sum not disambiguated | parenthesise `Negate(ExplicitSum …)` / hoist sign (render.cpp) |
| 9 | need a *second* `reassemble` to finish | workaround for 8a (2nd pass's canonicalize repairs nesting) | fixing 8a makes one pass enough |

## Resolution (implemented)

All four real code defects are fixed; #1/#2/#7 stay as documented (not bugs).
Commits, in order:

1. **#8a — interleaved peel** (`Fix reassemble peel: sign between sum
   binders`).  `reassemble`'s peel loop now walks `ExplicitSum`/`Negate`
   interleaved, tracking a running sign.  Subtlety found while testing: a
   *standalone* `Σ_j −Σ_i …` is adjacency-normalised by reassemble's self-prep
   `canonicalize` (→ `Σ_j Σ_i −…`), so it folded even before the fix.  The bug
   only bites when the term is an addend **inside a larger Sum**: `rewrite_tree`
   is bottom-up, so the fold callback first sees the *inner* `Σ_i …` (with `j`
   still free, can't fold), then the full `Σ_j −(Σ_i …)` — where the interleaved
   peel is what collects both binders past the sign.  Guards:
   `Reassemble.SignBetweenSumBinders` (fold shape) and the integration test
   `BasisFeasibility.CrossTensorCrossOnePass` (one reassemble ⇒ no `ExplicitSum`
   ⇒ closed form; fails without the fix).

2. **#3/#4/#6 — reassemble output** (`Reassemble surfaces prep + strips
   materialized Σ`).  Both `reassemble` and `reassemble_completeness` now return
   `steps::implicitize(ctx, out)`, reporting a no-op (returning the original `e`)
   only when the fold did not fire (`out == prepped`) **and**
   `structural_eq(prepped, e)`.  Note: the planned `prepped == e` *pointer* guard
   was wrong — `canonicalize` is not pointer-idempotent (it rebuilds even an
   already-canonical `a`), so a `structural_eq` test is required.  The two
   `Σ_i δ_ii` no-op tests now assert equality to the self-prep
   (`implicitize(canonicalize(·))`) rather than pointer-identity, since
   reassemble legitimately canonicalises a non-pattern;
   `Reassemble.SurfacesPrepCancellation` guards the #6 prep-surfacing directly.

3. **#4 — Python API** (`Expose implicitize and simplify to Python`).
   `td.implicitize` (binds `steps::implicitize`) and `td.simplify` =
   `implicitize(canonicalize(·))`, the "finish the derivation" composite.

4. **#8b — render** (`Parenthesize negated sub-sum in render`).  `sum_str` now
   wraps a `Negate` body whose operand is itself a sum, so `Σ_j −Σ_i …` prints as
   `Σ_j (−Σ_i …)`.

5. **#5 — index names** (`Unbounded dummy index names via subscript overflow`).
   `dummy_name(n)` wraps `schema[n % S]` with subscript `⌊n / S⌋`; only a
   negative index throws.

**Combined effect on the user's workflow:** with #7's recipe (expand the inserted
`I` before `simplify_basis_cross`), a **single** `reassemble` now yields a clean,
fully-implicit invariant directly — no leftover `Σ`, no manual `canonicalize`,
and re-expanding/rendering the result no longer hits the name limit.  One minor
residual (not one of the logged issues, not fixed): `reassemble` is not
idempotent — a second pass reassociates dots via `canonicalize`
(`a·(Bᵀ·c) → (a·Bᵀ)·c`); the forms are algebraically equal.
