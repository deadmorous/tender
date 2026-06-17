# 000040 delta-trace is not blocked — the real limit is a parametric/computed RHS

A correction to vibes 000033 §6 and 000039. Both record `delta-trace`
(`δ^p_p = n`) as an *open limitation* — "no way to force two slots of the same
delta to match the same bound index". That claim describes the **rejected**
slot-less / named-tensor-variable approach of vibe 000033 §4.1. Under the index
matcher we actually shipped (vibe 000039), it is false: `delta-trace` already
works through the generic engine.

## Why it works now

Vibe 000033 worried there was no way to force two slots of one delta to share
the same bound index. But our matcher binds the **binder itself as a pattern
variable** and reuses its id in both slots. Matching `ExplicitSum{p, δ^p_p}`:

- the `ExplicitSum` case does `try_bind(p → q)` against the target's binder `q`;
- slot 0 of the body does `try_bind(p → q)` again; slot 1 does `try_bind(p → q)`
  a third time.

`try_bind` (`identity.cpp:53`) requires consistency with the existing binding,
so two *different* target indices in the two slots fail the second `try_bind`.
The "same-index constraint" vibe 000033 said we lacked is exactly what
binder-as-variable plus consistency checking provides for free. Verified:

- `Σ_p δ^p_p` (3D) → `3`;
- `Σ_p δ^p_A` (one free index) → unchanged (correctly does **not** fire).

So `delta-trace` joins `delta-contraction` and `eps-delta-2` as already handled.

## The residual limitation (a feature, not a block)

What `delta-trace` *cannot* do today is be **dimension-polymorphic**. The
identity is written per concrete `IndexSpace`:

- `match_slot` (`identity.cpp:71`) requires the slot's `IndexSpace` to be equal,
  so a 3D-written pattern never matches a 2D delta — sound, but space-specific;
- the RHS is a fixed `ScalarLiteral`, so the dimension `n` is baked in.

Verified: a `delta-trace-3d` identity applied to a 2D trace does not fire; a
separate `delta-trace-2d` correctly yields `2`. So you register one identity per
space (`δ^p_p = 2, 3, 4`), and they are mutually sound — none misfires across
dimensions.

A single space-agnostic `δ^p_p = dim(space of p)` needs two new capabilities,
which are the **same** ones the general `delta-substitution`
(`Σ_p δ^p_A f(p,…) → f(A,…)`, vibe 000039) will want:

1. **parametric pattern slots** — let a slot's `IndexSpace` be a bound variable
   so the pattern matches any space;
2. **a computed RHS** — a rule carries a `binding → Expr` closure instead of a
   fixed tree, so `n` is read from the matched space's dimension.

## How / when

This is not a matcher unblock; it is a richer **rule type**. It belongs in the
e-graph's rewrite-rule interface (vibe 000034), which already needs more than a
bare `Identity` (cost-driven, applied under saturation). Giving a rule a
make-RHS-from-binding hook costs little there and subsumes both
dimension-polymorphic `delta-trace` and the `delta-substitution` remainder rule
(vibe 000039's "remainder variable, not higher-order unification"). Until then,
per-space constant identities cover the real cases with no loss of soundness.

## Net

Nothing is blocked for starting the e-graph. The one item previously flagged as
genuinely blocked turns out to work; the residual is a "parametric slot +
computed RHS" capability to design *into* the rule type rather than retrofit.
