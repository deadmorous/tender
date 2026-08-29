# `tender.operators` — the deferred ∇ / ∂ DSL

Retired in the M3 clean-up (vibes 000093 §5 / 000098), which had flagged it as
the one genuinely redundant route on the Python surface.

## What it was

A small DSL — `nabla`, `d(coord)`, `laplacian` — for writing differential
expressions before choosing a chart:

```python
from tender.operators import nabla, d, laplacian
expr = nabla @ (nabla * g)      # symbolic ∇·∇g
expr.evaluate(cart)             # …components, once a chart is chosen
```

## Why it went

It built a Python-side tree that was **not** a `tender.Expr`.  So a
`DifferentialExpr` could not be canonicalized, mixed into a sum with an ordinary
expression, handed to the engine verbs, or matched by a rule — only
`.evaluate(chart)` turned it into something the rest of the library could work
with.  Every capability the core gained after it was written (the e-graph verbs,
the identity DAG, selective application, the fold table) was unreachable from
it.

The core route does the same job with real expressions, and deferral falls out
for free because `t.nabla()` *is* an ordinary expression that no step is obliged
to lower:

```python
nabla = t.nabla(ctx=ws.ctx)
expr  = nabla @ (nabla * g)     # a tender.Expr; renders Δg
cart.evaluate(expr)             # …components, once a chart is chosen
```

| DSL | core |
|---|---|
| `nabla * f`, `nabla @ v`, `nabla % v` | the same operators on `t.nabla()` |
| `laplacian(f)` | `t.laplacian(f)` |
| `d(x) * f` | `td.deriv(x) * f`, then `td.apply_operators` |
| `nabla.along(v) * T` | `v @ (nabla * T)` |
| `expr.evaluate(chart)` | `chart.evaluate(expr)` |
| `nabla.at(chart)` | `chart.nabla()` |

## What moved where

`python/tender/operators.py` and the seven tests that exercised the DSL surface
are here.  The other 36 tests from `python/tests/test_operators.py` were about
the ∇ node all along; they stayed in the live tree as
`python/tests/test_nabla.py`, together with three of the seven ported onto the
core route (parenthesised operands, the directional derivative `(v·∇)R = v`, and
`chart.nabla()` reproducing the gradient).

Nothing here is built or run by CI.
