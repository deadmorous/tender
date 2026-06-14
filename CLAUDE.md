# tender

Tensor algebra library for computational mechanics, using direct (coordinate-free) tensor notation.

## Coding principles

1. **Incremental growth.** The system must be alive (buildable, testable) at every step. No dead branches; no stubs that don't compile.
2. **Test everything.** Unit tests for every module. Functional and integration tests when a full slice is needed. Mock components to isolate units.
3. **Check test coverage.** Gaps are bugs-in-waiting.
4. **Benchmark performance-sensitive operations.** Any operation that could be called in a tight loop gets a benchmark from the start.
5. **Maintain feasibility examples.** A small set of end-to-end examples prove the system works as a whole. They evolve with the system and are never deleted.
6. **Keep code DRY.** Don't repeat yourself — shared logic lives in one place.
7. **Minimise dependencies.** Reinventing a small wheel is fine; it keeps the codebase stable and predictable. Large dependencies are taken on only when the alternative is unreasonable.
8. **Prefer aggregation over inheritance.** When struct A needs the behaviour of struct B, give A a field of type B rather than inheriting from B, unless there is a genuine is-a relationship and virtual dispatch is intended. Aggregation avoids slicing, makes data layout explicit, and keeps types independently testable.

## Before committing

- **Run clang-format** on all changed C++ files before every commit.
  A `.clang-format` file is at the repo root.  Format everything under `src/`,
  `tests/`, `benchmarks/`, and `python/`:
  ```bash
  find src tests benchmarks python -name "*.cpp" -o -name "*.hpp" \
    | xargs clang-format -i
  ```
- **Strip Jupyter notebooks** — no outputs or execution counts in the repo.
  Use `examples/strip_notebooks.sh` (or `nbstripout <file>`) before staging
  any `.ipynb`.  The CI `notebooks-clean` job enforces this.

## Conversation log

All notable design discussions and decisions are recorded in `vibes/` as Markdown files.
Each distinct topic gets its own file, named `NNNNNN_subject.md` where NNNNNN is a zero-padded sequential number reflecting the order the topic was first discussed.
