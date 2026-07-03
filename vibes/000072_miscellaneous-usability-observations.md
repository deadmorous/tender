# 000072 — Miscellaneous usability observations

A running log of small usability issues and rough edges found while exercising
tender interactively.  These are collected first, then triaged into fixes.
Deferred items carried over from vibe 000071 (common-denominator fraction
combination + symmetric-dyad factoring for the 3-term ∇∇f form) are tracked
there; this vibe is for the *miscellaneous* findings.

## Shared preamble

All observations below assume this preamble:

```python
import tender as t
import tender.basis as tb
import tender.chart as tc
import tender.derivation as td
from tender.operators import nabla, d, laplacian
from IPython.display import Math, display

ws = t.Workspace()


def disp(*exprs):
    for e in exprs:
        display(Math(e.latex()))


I = ws.identity()
cart_basis = ws.wcs()
```

## Observations

### Obs 1 — `express` on the Cartesian chart renames `i, j, k` → `e_x, e_y, e_z`

Appendix:

```python
x, y, z = ws.coords("x", "y", "z")
cart = ws.chart(cart_basis, [x, y, z], [x, y, z])   # embedding == coords ⇒ Cartesian
R = cart.radius_vector()
disp(cart.express(R))
```

Output:

```
x e_x + y e_y + z e_z
```

whereas `R` itself prints as `x i + y j + z k`.

The two names refer to the **same three vectors**:

- The **reference basis** of the WCS names its vectors `i, j, k` (via
  `vector_symbol`).
- The **physical frame** that `physical_frame()` / `express` builds names them
  generically `e` subscripted by coordinate → `e_x, e_y, e_z`.

For a genuinely curvilinear chart the `e`-with-subscript naming is exactly what
we want (`e_r, e_θ, e_z`).  For the Cartesian chart it is jarring: `express(R)`
mathematically equals `R` but prints under different symbols.

Open question / candidate fix: when a chart's physical frame coincides with its
reference basis (orthonormal, embedding == coords, unit scale factors), reuse
the reference basis's `vector_symbol` names instead of minting `e_<coord>`.
Or, more generally, let the frame inherit the reference basis's symbol when the
frame vector *is* the reference vector.

### Obs 2 — `physical_frame()` cache keyed by `chart_id` never invalidates (**bug**)

Building a chart, calling `physical_frame()`, then building a *different* chart
that **reuses the same `chart_id`** returns the **stale** first frame.

Reproduction:

```python
# spherical-flavored chart on chart_id=2
r  = ws.coordinate("r",        chart_id=2, slot=0, nonneg=True)
th = ws.coordinate(r"\theta",  chart_id=2, slot=1)
ph = ws.coordinate(r"\varphi", chart_id=2, slot=2)
sph = ws.chart(cart_basis, [r, th, ph],
               [r*t.sin(th)*t.cos(ph), r*t.sin(th)*t.sin(ph), r*t.cos(th)])
sph.physical_frame()                       # binds chart_id=2 → spherical frame

# reuse chart_id=2 for a cylindrical chart
r2, th2, z2 = (ws.coordinate("r", chart_id=2, slot=0, nonneg=True),
               ws.coordinate(r"\theta", chart_id=2, slot=1),
               ws.coordinate("z", chart_id=2, slot=2))
cyl = ws.chart(cart_basis, [r2, th2, z2], [r2*t.cos(th2), r2*t.sin(th2), z2])
cyl.physical_frame()   # dirs come back e_r, e_θ, e_φ  ← STALE spherical frame!
```

Cause: `physical_frame` (src/chart.cpp) early-returns the cached frame via
`ctx.chart_frame(chart_id)` with no check that the *chart* (coords + embedding)
matches the one that populated the cache.  This is what makes a cylindrical CS
"look spherical, with a φ coordinate" when a `chart_id` is accidentally reused.

Mitigation today: a fresh `chart_id` per chart — automatic if you build
coordinates with `ws.coords(...)`.  Candidate fix: key the cache on chart
*identity* (coords + embedding), or at least detect a mismatched rebind and
recompute (and warn), rather than silently returning the stale frame.

### Obs 3 — `slot=` is redundant with the coordinate list order

A coordinate atom's identity is `(chart_id, slot)`, carried on the atom itself.
`CoordinateChart(reference, coords, embedding)` does **not** stamp slots back
onto the atoms in `coords`; the differentiator, mixed-partial sorting, and the
connection matcher all key on `(chart_id, slot)` independently of the chart
object.  So `slot=` is load-bearing at the low-level `tender.coordinate` API,
yet fully derivable from the position in the `coords` list.  `ws.coords(...)`
already auto-assigns it; the manual `slot=` only survives on the raw API.

Candidate: have `CoordinateChart` (or a helper) assign/validate slots from the
list order so the raw API stops requiring hand-kept `slot=`.

### Obs 4 — `nonneg` is reachable from `ws.coords`

Confirmed working, not a defect — recorded for completeness.  `nonneg=True` for
a radius goes through the facade as the set of non-negative names:

```python
r, th, z = ws.coords("r", r"\theta", "z", nonneg=("r",))   # licenses √(r²)→r
```

### Obs 5 — coordinates cannot be shared across charts (by design, today)

Reusing the Cartesian `z` atom inside a cylindrical chart does not work: the
Cartesian `z` is `(chart_id=cart, slot=2)` while the cylindrical chart needs
`(chart_id=cyl, slot=2)`.  Same display name, different identity ⇒ a different
coordinate; the shared atom would keep belonging to the Cartesian chart and the
cylindrical differentiator would not treat it as that chart's third coordinate.
Each chart mints its own coordinate atoms.  Deliberate coordinate sharing is
part of the deferred multi-chart "tree of CS" design (vibe 000071), not yet
supported.

**Amendment (worth pursuing later).** In the common case the cylindrical `z`
is *mathematically identical* to the WCS `z`: the embedding
`(r cosθ, r sinθ, z)` passes the reference `z` straight through, so it is not
merely a same-named coordinate but the very same axis (barring a
rotation/translation between the charts, which is a separate story).  Forcing a
distinct `(chart_id=cyl, slot=2)` atom there is artificial.  A future
coordinate-sharing model should let a chart declare that one of its coordinates
*is* a coordinate of the reference (an identity embedding component), so:

- `∂/∂z` is recognised as the same operator in both charts,
- fields depending on the shared `z` need not be restated per chart,
- and no stale-frame / duplicate-atom bookkeeping is needed for it.

This is the concrete first instance motivating the "tree of CS" work: shared
coordinates along identity embedding directions.

### Recommended cylindrical setup (facade)

```python
r, th, z = ws.coords("r", r"\theta", "z", nonneg=("r",))
cyl = ws.chart(cart_basis, [r, th, z], [r * t.cos(th), r * t.sin(th), z])
```

`ws.coords` mints a fresh `chart_id` and fills slots by position, sidestepping
Obs 2 (stale-cache collision) and Obs 3 (manual `slot=`) entirely.

### Obs 6 — `radius_vector()` returns the WCS form ⇒ operators produce mixed-frame results

```python
r, th, z = ws.coords("r", r"\theta", "z", nonneg=("r",))
cyl = ws.chart(cart_basis, [r, th, z], [r * t.cos(th), r * t.sin(th), z])
R = cyl.radius_vector()
disp(cyl.gradient(R))
```

gives a mixed-frame dyad that will not fold:

```
cos θ e_r i + sin θ e_r j − sin θ e_θ i + cos θ e_θ j + e_z k
```

The **left** vectors are the cylindrical frame (`e_r, e_θ, e_z`, produced by ∇),
the **right** vectors are WCS (`i, j, k`, carried by `R`).  ∇R should be `I`, but
a dyad with one leg in each frame cannot collapse.

Root cause: `cyl.radius_vector()` returns the **WCS** representation
`r cosθ i + r sinθ j + z k` (natural — it is assembled directly from the
embedding `x,y,z`).  The operators (`∇ = Σ (1/h_i) e_i ∂_i`) work intrinsically
in the chart's own frame, so they emit `e_r,e_θ,e_z` and never touch `R`'s WCS
legs.

The intrinsic form is what folds.  `cyl.express(R) = r e_r + z e_z`, and:

```python
cyl.gradient(cyl.express(R))     # → I   (collect_terms → 1·I)
```

Candidate fixes (pick one; both discussable):

1. **`radius_vector()` returns the intrinsic form** for a curvilinear chart —
   `r e_r + z e_z` — so the position vector lands in the chart's own frame like
   every other operator result.  (The WCS form remains available via
   `to_reference`.)  This matches the vibe-000071 principle: *stay in the chosen
   CS, don't return to WCS.*
2. **Operators `express` their input into the chart frame as a prep step**, so a
   WCS-expressed field is projected onto `e_i` before differentiation.  More
   general (handles any WCS input, not just `radius_vector`), but heavier and
   changes operator semantics for mixed-frame inputs.

Leaning toward (1) as the primary fix (least surprising, cheap), with (2) as a
possible robustness layer later.

### Obs 7 — gradient of the intrinsic radius vector folds cleanly to `I` (no issue)

Confirms the working path from Obs 6 — recorded as a positive baseline:

```python
r, th, z = ws.coords("r", r"\theta", "z", nonneg=("r",))
cyl = ws.chart(cart_basis, [r, th, z], [r * t.cos(th), r * t.sin(th), z])
R = cyl.radius_vector()
X = cyl.gradient(cyl.express(R))     # express(R) = r e_r + z e_z
disp(X)                              # → I
```

Once the position vector is in the chart's own frame, `∇R` folds to `I` with no
leftover terms.  This is the behaviour Obs 6's fix (1) would make the default.

### Obs 8 — `cart.express` reduces mixed-frame `∇R` to the identity's components but can't fold — blocked by Obs 1

Trying to "cure" the mixed-frame `∇R` (Obs 6) by re-expressing it in WCS:

```python
x, y, z = ws.coords("x", "y", "z")
cart = ws.chart(cart_basis, [x, y, z], [x, y, z])

r, th, z = ws.coords("r", r"\theta", "z", nonneg=("r",))
cyl = ws.chart(cart_basis, [r, th, z], [r * t.cos(th), r * t.sin(th), z])
R = cyl.radius_vector()
disp(cart.express(cyl.gradient(R)))
```

gives

```
i e_x + j e_y + k e_z
```

This is *exactly* the identity tensor `Σ_k unit_k ⊗ unit_k` — the mixed cyl/WCS
dyad (Obs 6) has been fully reduced to its identity components.  But it will not
fold to `I` because the **left** legs came out named `i, j, k` (reference) and
the **right** legs `e_x, e_y, e_z` (Cartesian physical frame).  Same three
vectors, two symbol sets ⇒ the resolution-of-identity fold cannot match them.

Proof the algebra is right — force one consistent naming with `to_reference`:

```python
cyl.to_reference(cyl.gradient(R))                 # → i i + j j + k k
tb.fold_resolution_of_identity(..., cart_basis)   # → I
```

So this is a second, stronger motivation for fixing **Obs 1**: if the Cartesian
chart's physical frame reused the reference `vector_symbol` names (`i, j, k`),
then `cart.express(∇R)` would print `i i + j j + k k` and fold straight to `I`.
The naming split is not merely cosmetic — it *blocks a fold*.

Related asymmetry worth noting: `cart.express` left the two dyad legs in two
different frames (left → reference `i,j,k`, right → physical `e_x,e_y,e_z`)
rather than putting both legs in the Cartesian physical frame.  Whether
`express` on a rank-2 object should drive *all* legs into the target frame
uniformly is worth a closer look once Obs 1 is resolved.

## Design takeaways

### DT 1 — An invariant, chart-bound position vector (not a global constant like `I`)

Motivated by Obs 6 and Obs 8: the mixed-frame pain comes from `radius_vector()`
returning a coordinate (WCS) expansion, so operators differentiate a field whose
legs live in a different frame than the `∇` they produce.  The cleaner primitive
is an **invariant position vector** you can write down without picking a basis —
analogous to how `I` is written without a basis — and expand/differentiate on
demand.  With that, `∇R = I` falls out intrinsically in any chart, no mixed-frame
trap.

**But the `I` analogy only goes halfway.**  `I` is genuinely universal: identical
components (δ) in every basis, one global constant.  The position vector is
**not** a single global thing — and continuum mechanics is the reason:

- There is no single "the" radius vector.  There is the position in the
  **reference** configuration (`r`) and in the **actual/current** configuration
  (`R`); they are *distinct* vectors related by the motion `R = χ(r, t)`, and in
  the usual reading they range over different bodies/spaces.
- The central kinematic object is what connects them: the **deformation
  gradient** `F = ∂R/∂r = Grad R` — `∇` of one configuration's position vector
  taken with respect to *another* configuration's coordinates.  It is emphatically
  **not** `I`.

**Recommendation.**  Do *not* add a global `R` constant.  Add a **chart-bound
invariant position atom** — e.g. `chart.position()` (working name; `radius()`
also fine) — that returns an invariant, configuration-tagged vector: the thing
you differentiate.  Keep the existing `radius_vector()` as its explicit
WCS/coordinate expansion.  Consequences:

- `chart.position()` differentiated w.r.t. that chart's own coords → `I`,
  intrinsically and cleanly (fixes the Obs 6/8 ergonomics without relying on
  `express`/`to_reference` gymnastics).
- `r ≡ reference_chart.position()` and `R ≡ actual_chart.position()` are just two
  such atoms — the reference/actual pair drops out naturally.
- `F = Grad_r R` is a **cross-configuration (cross-chart) gradient**, i.e. exactly
  the deferred cross-chart differentiation (vibe 000071 P7 F3 / "tree of CS").
  A chart-bound invariant position atom is a natural prerequisite / stepping
  stone toward the deformation-gradient machinery.

Status: design takeaway, not scheduled.  Interacts with DT-level items in vibe
000071 (cross-chart differentiation) and Obs 1 (frame naming) here.

*(more observations to be appended as they are found)*
