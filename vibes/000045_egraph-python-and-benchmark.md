# 000045 Saturation in Python, and a benchmark (Stage 1, part 2 — slice 4)

Fourth slice of the e-graph (vibe 000034, items #9 + #10): the Python surface
for saturation, and a benchmark harness.

## td.saturate

`td.saturate(expr, rules, max_iterations=30)` builds an e-graph in `expr`'s
context, adds `expr`, applies the `rules` (a list of `td.Identity`) to a fixed
point, and returns the cheapest extracted expression. No manual step ordering,
unlike a linear `Derivation`:

```python
rule = td.Identity("delta-contraction",
                   tender.explicit_sum(p, dul(p, a) * dul(p, b), ctx=ctx),
                   dll(a, b))
td.saturate(tender.explicit_sum(q, dul(q, m) * dul(q, n), ctx=ctx), [rule])
# → δ_{mn}
```

Binding: `_core._saturate(expr, lhss, rhss, max_iterations)` takes the rules as
two parallel `std::vector<PyExpr>` (a rule's name is just a label, unused by
saturation), constructs `Identity`s, runs `EGraph::saturate`, and returns
`extract(find(root))`. All expressions must share one `Context` (the e-graph and
its allocations live there). Python tests: contraction, the nested-in-a-sum case
(rewritten via congruence with no ordering), and a no-match identity returning
the canonical input.

## Benchmark

`benchmarks/egraph_saturate_bench.cpp` — the first wired-up benchmark. The
`TENDER_BUILD_BENCHMARKS` option existed but nothing was built; this adds
`benchmarks/CMakeLists.txt` and `add_subdirectory(benchmarks)`. Per CLAUDE.md
principle 7 (minimise dependencies) it is **framework-free**: a `steady_clock`
loop, no Google Benchmark.

It reports the *deterministic* shape of saturation — passes to the fixed point,
e-node and e-class counts — which is the primary signal (timing depends on build
type and is informational). On the two index identities:

```
delta-contraction   passes=2  nodes=6  classes=4   ~7 us/op
eps-delta-2         passes=2  nodes=8  classes=7   ~7 us/op
```

Both settle in two passes (one to rewrite, one to confirm the fixed point),
confirming the size-reducing rule set converges immediately, as vibe 000034
predicts. Built with `-O2` regardless of the surrounding build type so the
timing is meaningful; not registered with CTest.

## Status

The e-graph (vibe 000034) is now complete end to end: core (042), e-matcher
(043), saturate (044), and now Python + benchmark (045). Remaining is item #8 —
a *curated rule set* (distribution, and the parametric-slot + computed-RHS rule
type of vibe 000040) — which is better grown against real use. That use is
**Stage 2** of the roadmap (vibe 000038): exercise the engine on coordinate-level
identities (ε-δ, generalized Kronecker, traces, determinant identities), letting
the needed rules and any ANF/engine revisions surface from concrete examples.

## Tests

386 C++ / 98 Python pass (+3 Python saturate tests). Benchmark builds and runs.
egraph.cpp remains at 100% line coverage (the benchmark is outside the coverage
filter, which is `src/` only).
